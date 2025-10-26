/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : PID + ADC + RPLIDAR + MPU6500
  * @description    : 智能移动机器人主控程序
  *                   功能包括：
  *                   - PID电机控制系统
  *                   - 激光雷达数据处理
  *                   - MPU6500姿态传感器数据读取
  *                   - 蓝牙通信            // 设置新数据标志，稍后发送
            has_new_lidar_data = 1;*                   - ADC电压监测
  ******************************************************************************
  */
/* USER CODE END Header */

// 系统头文件包含
#include "main.h"      // 主头文件
#include "adc.h"       // ADC外设头文件
#include "dma.h"       // DMA外设头文件
#include "tim.h"       // 定时器外设头文件
#include "usart.h"     // USART外设头文件
#include "gpio.h"      // GPIO外设头文件
#include "i2c.h"       // I2C外设头文件
#include "mpu6500.h"   // MPU6500传感器头文件
#include <string.h>    // 字符串处理库
#include <stdio.h>     // 标准输入输出库
#include <math.h>      // 数学函数库
#include <stdint.h>    // 标准整数类型库
#include <stdlib.h>

// 电机控制相关宏定义
#define MAX_SPEED 350                   // 最大电机速度（PWM值）
#define SPEED_STEP 120                  // 速度调整步长
#define SPEED_UPDATE_INTERVAL 30        // 速度更新间隔（毫秒）
#define TURN_DURATION 1150              // 转向持续时间（毫秒）
#define TURN_SPEED 300                  // 转向速度（PWM值）

#define MAX_PWM                700
#define MIN_PWM                60 

// 电机速度补偿系数（根据实际测试调整）
#define MOTOR_LEFT_FORWARD_COMPENSATION 0.95f
#define MOTOR_RIGHT_BACKWARD_COMPENSATION 0.95f

// 运动方向定义
#define DIR_STOP 0x00                      // 停止
#define DIR_FORWARD 0x01                   // 前进
#define DIR_BACKWARD 0x02                  // 后退
#define DIR_LEFT 0x03                      // 左转
#define DIR_RIGHT 0x04                     // 右转
#define DIR_UTURN 0x05 

// 雷达命令定义
#define CMD_LIDAR_SCAN 0xA5

// 编码器和PID控制相关定义
#define PPR 360                         // 编码器每转脉冲数（Pulse Per Revolution）
#define SAMPLE_TIME_MS 100              // 采样时间间隔（毫秒）
// 编码器计数倍率：若TIM在编码器模式使用四边计数，则每转有效计数为 PPR * 4
#define ENCODER_COUNT_MULTIPLIER 4

// 激光雷达数据处理相关定义
#define DMA_BUFFER_SIZE 256             // DMA缓冲区大小
#define DATA_PACKET_SIZE 5              // 激光雷达数据包大小
#define ANGLE_FILTER_THRESHOLD 1.0f     // 角度过滤阈值
#define MIN_VALID_DISTANCE 50.0f        // 最小有效距离（毫米）
#define MAX_VALID_DISTANCE 12000.0f     // 最大有效距离（毫米）
#define LIDAR_TIMEOUT_THRESHOLD 500     // 激光雷达超时阈值（毫秒）

//1
// 精准控距相关
// 在宏定义部分添加比例因子
#define CODE_DISTANCE_CM 170.0f        // 代码中设定的距离
#define ACTUAL_DISTANCE_CM 70.0f       // 实际行走的距离
#define DISTANCE_RATIO (ACTUAL_DISTANCE_CM / CODE_DISTANCE_CM)  // 比例因子 ≈ 0.41176
#define WHEEL_DIAMETER_CM 20.42f                    // 轮径（cm）
#define WHEEL_CIRCUMFERENCE_CM (3.1415926535f * WHEEL_DIAMETER_CM)  // 周长
#define PULSES_PER_CM ((PPR * ENCODER_COUNT_MULTIPLIER) / WHEEL_CIRCUMFERENCE_CM)  // 每cm脉冲数

// 新增转向控制参数
#define TURN_TIME_UNIT_MS      5        // 转向时间单位（毫秒）
#define TURN_PWM               300      // 转向PWM值
#define TURN_MAX_DEG           180.0f   // 最大转向角度（度）
#define PI_F                   3.1415926f  // 圆周率
#define WHEEL_BASE_CM          13.0f    // 两轮中心距（cm）

// 新增运动状态机
typedef enum {
    MOT_IDLE = 0,           // 空闲，停止
    MOT_FWD_DIST,           // 前进指定距离
    MOT_BWD_DIST,           // 后退指定距离  
    MOT_TURN_LEFT_ANGLE,     // 左转状态 - 按角度/位移控制
    MOT_TURN_RIGHT_ANGLE     // 右转状态 - 按角度/位移控制
} motion_state_t;

/**
 * @brief PID控制器结构体
 * 用于实现电机速度的PID闭环控制
 */
typedef struct {
    float Kp;           // 比例系数
    float Ki;           // 积分系数
    float Kd;           // 微分系数
    float setpoint;     // 目标值
    float integral;     // 积分累计值
    float last_error;   // 上次误差值
    float output;       // 输出值
    float max_output;   // 最大输出限制
    float min_output;   // 最小输出限制
} PID_Controller;

/**
 * @brief PID控制算法计算函数
 * @param pid PID控制器指针
 * @param measured 测量值
 * @return 控制输出值
 */
float PID_Compute(PID_Controller* pid, float measured) {
    // 计算当前误差
    float error = pid->setpoint - measured;
    // 积分累计
    pid->integral += error;
    // 计算微分
    float derivative = error - pid->last_error;
    // PID控制算法
    pid->output = pid->Kp * error + pid->Ki * pid->integral + pid->Kd * derivative;
    // 更新上次误差
    pid->last_error = error;

    // 输出限幅处理
    if (pid->output > pid->max_output) pid->output = pid->max_output;
    if (pid->output < pid->min_output) pid->output = pid->min_output;

    return pid->output;
}

// PID控制器实例化
PID_Controller pidA = {0.3f, 0.05f, 0.02f, 0, 0, 0, 0, 100, -100};  // 最大修正±100
PID_Controller pidB = {0.3f, 0.05f, 0.02f, 0, 0, 0, 0, 100, -100};  // 最大修正±100


// 电机转速变量
float rpmA = 0.0f;  // 电机A转速（RPM）
float rpmB = 0.0f;  // 电机B转速（RPM）

// 蓝牙通信相关变量
uint8_t bluetooth_rx_data = 0;          // 蓝牙接收数据缓冲
uint8_t target_direction = DIR_STOP;    // 目标运动方向
uint8_t current_direction = DIR_STOP;   // 当前运动方向
uint32_t current_speed = 0;             // 当前速度
uint32_t target_speed = 0;              // 目标速度
uint32_t last_speed_update_time = 0;    // 上次速度更新时间

// 蓝牙连接状态管理
uint8_t bluetooth_connected = 0;        // 蓝牙连接状态标志
uint8_t connection_announced = 0;       // 连接通知发送标志
uint8_t connection_msg[] = "Connected\r\n";  // 连接成功消息

// 转向控制变量（仅保留定时转向）
uint8_t turning_direction = DIR_STOP;   // 转向方向（用于定时转向）

// 定时转向相关
uint8_t use_timed_turning = 0;           // 是否使用定时转向（基于TURN_DURATION或可调值）
uint32_t turn_start_time = 0;            // 定时转向开始时间（ms）
uint32_t turn_time_ms = TURN_DURATION;   // 可调的转向时长（默认使用宏）

// 编码器相关变量
uint32_t lastEncoderA = 0;               // 电机A上次编码器值
uint32_t lastEncoderB = 0;               // 电机B上次编码器值
// （运行时字符设置功能已移除）
// ...existing code...
char uart_buf[100];                     // UART输出缓冲区

// 激光雷达数据处理相关变量
uint8_t lidar_dma_buffer[DMA_BUFFER_SIZE];      // 激光雷达DMA接收缓冲区
uint8_t lidar_dataBuffer[DATA_PACKET_SIZE];     // 激光雷达数据包缓冲区
uint8_t lidar_dataIndex = 0;                    // 数据包索引
uint32_t lidar_rxIndex = 0;                     // 接收数据索引
volatile uint8_t lidar_process_packet = 0;      // 数据包处理标志
float lastLidarAngle = 0.0f;                    // 上次激光雷达角度
uint32_t lastLidarRxTime = 0;                   // 上次激光雷达接收时间

//2
// 精准控距相关变量
volatile int32_t accumulated_pulses = 0;           // 累积脉冲数
uint8_t distance_control_active = 0;              // 是否激活距离控制

// 角度转向相关变量
static motion_state_t motion = MOT_IDLE;  // 当前运动状态
static float turn_target_sumabs_cm = 0.0f;  // 转向目标位移量(cm)
static float turn_accum_sumabs_cm  = 0.0f;  // 转向已累计位移量(cm)
volatile uint32_t target_pulses = 0;  // 目标脉冲数

//3
// 在全局变量区域添加
float speedA_cmps = 0.0f;  // 电机A线速度
float speedB_cmps = 0.0f;  // 电机B线速度

// 编码器增量计算函数
int32_t get_encoder_delta(uint32_t current, uint32_t last) {
    int32_t delta = (int32_t)(current - last);
    // 处理32位计数器溢出
    if (delta > 0x7FFFFFFF) delta -= 0xFFFFFFFF;
    else if (delta < -0x7FFFFFFF) delta += 0xFFFFFFFF;
    return delta;
}

// UART句柄外部声明
extern UART_HandleTypeDef huart1;  // 蓝牙通信UART
extern UART_HandleTypeDef huart6;  // 激光雷达UART

// 调试相关全局变量
static uint32_t last_debug_time = 0;       // 调试信息输出时间（统一2秒）

// 函数声明
void SystemClock_Config(void);                                    // 系统时钟配置
void ControlMotor(uint8_t direction, uint32_t speed);            // 电机控制函数
void UpdateSpeedRamp(void);                                       // 速度渐变更新函数
void SendConnectionNotification(void);                            // 发送连接通知函数
void ProcessLidarData(uint8_t* data);                            // 激光雷达数据处理函数
void SendLidarToBluetooth(float angle, float distance, uint8_t quality); // 激光雷达数据发送函数

/**
 * @brief 转向运动状态管理函数
 * @param dCntA 左轮编码器计数变化量
 * @param dCntB 右轮编码器计数变化量
 * 处理基于编码器的转向角度控制
 */
static void HandleMotionManager(int32_t dCntA, int32_t dCntB) {
    // 计算单个采样周期内的轮子位移量
    float ds_cm = (float)3.14f * 8.7f / (float)(360 * 3);  // 每个计数对应的位移(cm)
    float sA = fabsf((float)dCntA) * ds_cm;  // 左轮位移量(cm)
    float sB = fabsf((float)dCntB) * ds_cm;  // 右轮位移量(cm)

    // 转向停止判断：按两轮位移绝对值之和达到阈值即停止
    if (motion == MOT_TURN_LEFT_ANGLE || motion == MOT_TURN_RIGHT_ANGLE) {
        float ds_sum = sA + sB;
        
        // 防止编码器计数跳变溢出：如果本次位移过大(>10cm)，则忽略不计
        if (ds_sum < 10.0f) {
            turn_accum_sumabs_cm += ds_sum;  // 累计有效位移量
        }
        
        // 判断是否达到目标转向角度
        if (turn_accum_sumabs_cm >= turn_target_sumabs_cm) {
            motion = MOT_IDLE;      // 清除转向状态
            target_direction = DIR_STOP;
            current_direction = DIR_STOP;
            ControlMotor(DIR_STOP, 0);  // 停止电机
            
            // 重置转向变量
            turn_accum_sumabs_cm = 0;
            turn_target_sumabs_cm = 0;
            
        }  
    }
}

// 在合适的位置添加连接状态管理
void SendConnectionNotification(void) {
    if (bluetooth_connected && !connection_announced) {
        // 发送连接成功消息
        //uint8_t connect_msg[] = "ROBOT_CONNECTED\r\n";
        //HAL_UART_Transmit(&huart1, connect_msg, sizeof(connect_msg)-1, 1000);
        connection_announced = 1;
        
        // 发送就绪状态
        //uint8_t ready_msg[] = "READY_FOR_COMMANDS\r\n";
        //HAL_UART_Transmit(&huart1, ready_msg, sizeof(ready_msg)-1, 1000);
    }
}

static void MotorTurnLeftPWM(int pwm) {
    if (pwm < 0) pwm = 0;
    if (pwm > MAX_SPEED) pwm = MAX_SPEED;

    // 左轮后退，右轮前进 - 实现原地左转
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwm);  // 左轮后退
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);    // 左轮前进停止
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, pwm);  // 右轮前进  
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);    // 右轮后退停止
}

static void MotorTurnRightPWM(int pwm) {
    if (pwm < 0) pwm = 0;
    if (pwm > MAX_SPEED) pwm = MAX_SPEED;

    // 左轮前进，右轮后退 - 实现原地右转
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);    // 左轮后退停止
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwm);  // 左轮前进
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);    // 右轮前进停止
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, pwm);  // 右轮后退
}

/**
 * @brief UART接收完成回调函数
 * @param huart UART句柄指针
 * 处理来自蓝牙和激光雷达的数据接收
 */
// 在 HAL_UART_RxCpltCallback 函数中替换命令解析部分
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart1) {
        if (!bluetooth_connected) bluetooth_connected = 1;

        // 新的命令格式：第一个字节是命令类型，第二个字节是参数
        static uint8_t command_buffer[2];
        static uint8_t command_index = 0;
        
        command_buffer[command_index++] = bluetooth_rx_data;
        
        if (command_index == 2) {
            uint8_t cmd_type = command_buffer[0];
            uint8_t arg = command_buffer[1];
            command_index = 0;
            
            // 处理运动命令
            if (cmd_type >= 0x01 && cmd_type <= 0x04) {
                target_direction = cmd_type;
                
                if (cmd_type == DIR_STOP) {
                    // 停止所有动作
                    target_speed = 0;
                    motion = MOT_IDLE;
                    distance_control_active = 0;
                    ControlMotor(DIR_STOP, 0);
                    current_direction = DIR_STOP;
                } 
                else if (cmd_type == DIR_FORWARD || cmd_type == DIR_BACKWARD) {
										// 前进/后退命令，参数是距离(cm)
										current_direction = cmd_type;
										target_speed = MAX_SPEED;
										accumulated_pulses = 0;
										distance_control_active = 1;
										
										// 根据实际测量比例进行换算
										float distance_cm = (float)arg;  // 上位机发送的目标距离
										float actual_target_cm = distance_cm / DISTANCE_RATIO;  // 换算为代码中的目标距离
										
										// 计算目标脉冲数
										target_pulses = (uint32_t)(actual_target_cm * PULSES_PER_CM);
										
										ControlMotor(current_direction, 0);
								}
                else if (cmd_type == DIR_LEFT || cmd_type == DIR_RIGHT) {
                    // 转向命令，参数是角度(度)
                    float angle_deg = (float)arg;
                    
                    // 设置角度转向参数
                    turn_target_sumabs_cm = WHEEL_BASE_CM * (angle_deg * PI_F / 180.0f);
                    turn_accum_sumabs_cm = 0.0f;
                    
                    // 设置对应的转向状态
                    if (cmd_type == DIR_LEFT) {
                        motion = MOT_TURN_LEFT_ANGLE;
                        MotorTurnLeftPWM(TURN_PWM);
                    } else {
                        motion = MOT_TURN_RIGHT_ANGLE;
                        MotorTurnRightPWM(TURN_PWM);
                    }
                    
                    current_direction = cmd_type;
                }
            }
            // 处理雷达扫描命令
            else if (cmd_type == 0xA5 && arg == 0x20) {
                // 雷达扫描命令 - 确保雷达正在工作
                // 已经在主循环中持续发送数据，这里可以什么都不做
            }
        }
        
        HAL_UART_Receive_IT(&huart1, &bluetooth_rx_data, 1);
    } 
    else if (huart == &huart6) {
        HAL_UART_Receive_DMA(&huart6, lidar_dma_buffer, DMA_BUFFER_SIZE);
    }
}

/**
 * @brief 电机控制函数
 * @param direction 运动方向
 * @param speed 运动速度
 * 根据方向和速度控制四个电机的PWM输出
 */
void ControlMotor(uint8_t direction, uint32_t speed) {
    // 原有的补偿计算
    static float left_compensation = 0.95f;   
    static float right_compensation = 0.96f;  
    
    int32_t pwmA = (int32_t)(speed * left_compensation);
    int32_t pwmB = (int32_t)(speed * right_compensation);

    // PID微调 - 只在直行时启用距离控制
    if ((direction == DIR_FORWARD || direction == DIR_BACKWARD) && distance_control_active) {
        // 计算PID修正值（基于速度误差）
        float correctionA = PID_Compute(&pidA, speedA_cmps);
        float correctionB = PID_Compute(&pidB, speedB_cmps);
        
        // 应用PID微调到补偿后的PWM
        pwmA += (int32_t)correctionA;
        pwmB += (int32_t)correctionB;
    }

    // PWM限幅 - 使用参考代码的限幅值
    if (pwmA > (int32_t)MAX_SPEED) pwmA = MAX_SPEED;
    if (pwmB > (int32_t)MAX_SPEED) pwmB = MAX_SPEED;
    if (pwmA < 200) pwmA = 200;  // 确保最小PWM值足够启动电机
    if (pwmB < 200) pwmB = 200;

    switch(direction) {
        case DIR_STOP:
            // 停止所有电机 - 使用参考代码的实现
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
            break;
            
        case DIR_FORWARD:
            // 前进 - 使用参考代码的实现
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);      // 左轮后退停止
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwmA);   // 左轮前进
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, pwmB);   // 右轮前进
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);      // 右轮后退停止
            break;
            
        case DIR_BACKWARD:
            // 后退 - 使用参考代码的实现
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwmA);   // 左轮后退
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);      // 左轮前进停止
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);      // 右轮前进停止
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, pwmB);   // 右轮后退
            break;
    }
}

/**
 * @brief 速度渐变更新函数
 * 实现平滑的速度变化，避免急加速或急减速
 */
void UpdateSpeedRamp(void) {
    uint32_t current_time = HAL_GetTick();
    
    // 检查是否到达速度更新时间间隔
    if (current_time - last_speed_update_time >= SPEED_UPDATE_INTERVAL) {
        last_speed_update_time = current_time;

        // 处理方向变化
        if (target_direction != current_direction) {
            // 前进/后退方向切换
            if ((target_direction == DIR_FORWARD || target_direction == DIR_BACKWARD) && 
                (current_direction == DIR_STOP || current_direction == DIR_FORWARD || current_direction == DIR_BACKWARD)) {
                current_direction = target_direction;
                target_speed = MAX_SPEED;
                ControlMotor(current_direction, 0);
            }
            // 转向操作
            else if (target_direction == DIR_LEFT || target_direction == DIR_RIGHT || target_direction == DIR_UTURN) {
                target_speed = (target_direction == DIR_UTURN) ? TURN_SPEED : TURN_SPEED;
                current_direction = target_direction;
                ControlMotor(current_direction, 0);
            }
            // 停止操作
						else if (target_direction == DIR_STOP) {
								target_speed = 0;
								if (current_speed > 0) {
										// 渐进减速 - 确保左右轮同步减速
										if (current_speed > SPEED_STEP * 2) {
												current_speed -= SPEED_STEP;
										} else {
												// 当速度很低时直接停止，避免不同步
												current_speed = 0;
										}
										ControlMotor(current_direction, 0);
								} else {
										// 完全停止 - 强制同步停止
										current_direction = target_direction;
										ControlMotor(DIR_STOP, 0);  // 使用DIR_STOP确保完全停止
								}
						}
        }
        // 速度渐变调整
        else if (current_direction != DIR_STOP) {
            if (current_speed < target_speed) {
                current_speed += SPEED_STEP;  // 渐进加速
            } else if (current_speed > target_speed) {
                current_speed -= SPEED_STEP;  // 渐进减速
            }
            ControlMotor(current_direction, 0);
        }
    }
}

// 激光雷达最新数据存储全局变量
float latest_lidar_angle = 0.0f;      // 最新激光雷达角度
float latest_lidar_distance = 0.0f;   // 最新激光雷达距离
uint8_t latest_lidar_quality = 0;     // 最新激光雷达质量
uint8_t has_new_lidar_data = 0;       // 新数据标志

/**
  * @brief 处理激光雷达数据
  * @param data 原始数据指针
  * 解析激光雷达数据包，提取角度、距离和质量信息
  */
void ProcessLidarData(uint8_t* data) {
    // 激光雷达数据结构体（紧凑打包）
    typedef struct {
        uint8_t sync_quality;  // 同步位和质量
        uint16_t angle_q6;     // 角度（Q6格式）
        uint16_t distance_q2;  // 距离（Q2格式）
    } __attribute__((packed)) LidarDataPacket;

    LidarDataPacket* pkt = (LidarDataPacket*)data;
    
    // 检查数据包有效性（同步位和角度最低位）
    if((pkt->sync_quality & 0x80) && (pkt->angle_q6 & 0x01)) {
        uint8_t quality = pkt->sync_quality & 0x7F;  // 提取质量值
        float angle = (pkt->angle_q6 >> 1) / 64.0f; // 转换为角度（度）
        
        // 确保角度在0-360度范围内
        while (angle >= 360.0f) angle -= 360.0f;
        while (angle < 0.0f) angle += 360.0f;
        
        float distance = pkt->distance_q2 / 4.0f;  // 转换为距离（mm）
        
        // 检查距离和质量的有效性
        // 允许 quality == 0 也传输（上位机可能需要查看质量为0的点）
        if(distance >= MIN_VALID_DISTANCE && 
           distance <= MAX_VALID_DISTANCE && 
           quality >= 0) {
            // 保存最新有效数据（供定时发送使用）
            latest_lidar_angle = angle;
            latest_lidar_distance = distance;
            latest_lidar_quality = quality;
            has_new_lidar_data = 1;  // 设置新数据标志，主循环负责发送，避免在中断/解析处直接阻塞串口
            lastLidarAngle = angle;   // 更新上次角度
        }
    }
    
    // 处理完成后重置DMA
    __HAL_DMA_DISABLE(huart6.hdmarx); // 禁用DMA
    huart6.hdmarx->Instance->NDTR = DMA_BUFFER_SIZE; // 重置DMA计数器
    __HAL_DMA_ENABLE(huart6.hdmarx);  // 重新启用DMA
}

/**
 * @brief 发送激光雷达数据到蓝牙
 * @param angle 角度值
 * @param distance 距离值
 * @param quality 质量值
 * 格式化激光雷达数据并通过蓝牙发送
 */
void SendLidarToBluetooth(float angle, float distance, uint8_t quality) {
    char buffer[64];
    // 格式化激光雷达数据
    int len = snprintf(buffer, sizeof(buffer), 
                      "A:%.2f°, D:%.2fmm, Q:%d\r\n", 
                      angle, distance, quality);
    // 通过蓝牙发送数据
    HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
}

/**
 * @brief 主函数
 * 系统初始化和主循环
 */
int main(void) {
    // HAL库初始化
    HAL_Init();
    
    // 系统时钟配置
    SystemClock_Config();
    
    // 外设初始化
    MX_GPIO_Init();         // GPIO初始化
    MX_DMA_Init();          // DMA初始化
    MX_ADC1_Init();         // ADC初始化
    MX_TIM2_Init();         // 定时器2初始化（编码器A）
    MX_TIM3_Init();         // 定时器3初始化（PWM输出）
    MX_TIM4_Init();         // 定时器4初始化（编码器B）
    MX_USART1_UART_Init();  // USART1初始化（蓝牙通信）
    MX_USART6_UART_Init();  // USART6初始化（激光雷达通信）
    MX_I2C1_Init();         // I2C1初始化（MPU6500通信）

    // 启动定时器编码器模式
    HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);  // 启动编码器A
    HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL);  // 启动编码器B
    
    // 启动PWM输出
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);  // 电机A正转PWM
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);  // 电机A反转PWM
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);  // 电机B正转PWM
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);  // 电机B反转PWM

    // 启动蓝牙接收中断
    HAL_UART_Receive_IT(&huart1, &bluetooth_rx_data, 1);

    // MPU6500初始化
    MPU6500_Data mpu_data;
    if (MPU6500_Init(&hi2c1) != 0) {
        // MPU6500 init failed, but no output
    } else {
        // MPU6500 OK, but no output
    }

    // 等待系统稳定
    HAL_Delay(500);
    
    // 启动激光雷达
    uint8_t startCmd[] = {0xA5, 0x20};  // 激光雷达启动命令
    HAL_UART_Transmit(&huart6, startCmd, sizeof(startCmd), 100);
    HAL_UART_Receive_DMA(&huart6, lidar_dma_buffer, DMA_BUFFER_SIZE);
		
    // 通知激光雷达启动（移除输出）

    // 主循环
    while (1) {
        // 更新速度渐变控制
        UpdateSpeedRamp();
			
				//4
				SendConnectionNotification();

				// 在主循环的编码器读取部分替换为：
				uint32_t encoderA = (uint32_t)__HAL_TIM_GET_COUNTER(&htim2);
				uint32_t encoderB = (uint32_t)__HAL_TIM_GET_COUNTER(&htim4);

				// 使用改进的编码器增量计算
				int32_t deltaA = get_encoder_delta(encoderA, lastEncoderA);
				int32_t deltaB = get_encoder_delta(encoderB, lastEncoderB);
				lastEncoderA = encoderA;
				lastEncoderB = encoderB;

				// 计算转速（RPM） - 使用绝对增量近似速度
				float rpmA = (float)abs(deltaA) / PPR * (60.0f / (SAMPLE_TIME_MS / 1000.0f));
				float rpmB = (float)abs(deltaB) / PPR * (60.0f / (SAMPLE_TIME_MS / 1000.0f));

				// 转换为线速度（cm/s）
				speedA_cmps = rpmA * 20.42f / 60.0f;
				speedB_cmps = rpmB * 20.42f / 60.0f;

				// 新增：转向运动状态管理
				HandleMotionManager((int32_t)deltaA, (int32_t)deltaB);

				// 精准距离控制逻辑
				if (distance_control_active) {
						// 使用已计算的deltaA和deltaB的绝对值
						accumulated_pulses += abs((int32_t)deltaA) + abs((int32_t)deltaB);
						// 检查是否达到目标距离（使用动态的 target_pulses）
						if (accumulated_pulses >= target_pulses) {  // 修改这里
								// 达到目标，停止
								target_direction = DIR_STOP;
								current_direction = DIR_STOP;
								target_speed = 0;
								current_speed = 0;
								distance_control_active = 0;
								ControlMotor(DIR_STOP, 0);
								
						}
				}
			
        // 激光雷达数据处理
        uint32_t currentRxIndex = DMA_BUFFER_SIZE - __HAL_DMA_GET_COUNTER(huart6.hdmarx);
        if (currentRxIndex != lidar_rxIndex) {
            // 计算新数据长度
            uint32_t dataLength = (currentRxIndex > lidar_rxIndex) ?
                                  (currentRxIndex - lidar_rxIndex) :
                                  (DMA_BUFFER_SIZE - lidar_rxIndex + currentRxIndex);
            
            // 处理新数据
            for (uint32_t i = 0; i < dataLength; i++) {
                uint8_t data = lidar_dma_buffer[(lidar_rxIndex + i) % DMA_BUFFER_SIZE];
                
                // 查找数据包起始标志（同步位）
                if(lidar_dataIndex == 0) {
                    if(data & 0x80) {  // 同步位为1，表示新扫描开始
                        lidar_dataBuffer[lidar_dataIndex++] = data;
                    }
                } else {
                    lidar_dataBuffer[lidar_dataIndex++] = data;
                    // 完整数据包处理
                    if(lidar_dataIndex >= DATA_PACKET_SIZE) {
                        ProcessLidarData(lidar_dataBuffer);
                        lidar_dataIndex = 0;
                    }
                }
            }
            lidar_rxIndex = currentRxIndex;  // 更新接收索引
        }
        
        // 激光雷达数据超时处理
        if(lidar_dataIndex > 0 && (HAL_GetTick() - lastLidarRxTime > LIDAR_TIMEOUT_THRESHOLD)) {
            lidar_dataIndex = 0;  // 重置数据索引
        }

    // 不在这里清除 has_new_lidar_data，改为在调试输出处按扇区选择性发送

        // ADC电压监测
        HAL_ADC_Start(&hadc1);
        HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
        uint32_t adc_val = HAL_ADC_GetValue(&hadc1);
        float voltage = (adc_val * 3.3f * 11.0f) / 4096.0f;  // 电压转换（考虑分压电阻）

        // PID控制器目标值设置
        // 在主循环的PID设置部分，应该调用PID计算：
				// PID控制器目标值设置
				// PID控制器目标值设置 - 简化版本
				if (current_direction == DIR_FORWARD || current_direction == DIR_BACKWARD) {
						if (distance_control_active) {
								pidA.setpoint = 45.0f;  // 目标速度45cm/s
								pidB.setpoint = 45.0f;  // 目标速度45cm/s
						}
				} else {
						pidA.setpoint = 0;
						pidB.setpoint = 0;
						// 可选：重置积分项
						pidA.integral = pidB.integral = 0;
				}
				
				// PID计算和PWM应用
				if (current_direction == DIR_FORWARD || current_direction == DIR_BACKWARD) {
						if (distance_control_active) {
								// 使用PID计算修正值
								float correctionA = PID_Compute(&pidA, speedA_cmps);  // 注意：这里measured是speedA_cmps
								float correctionB = PID_Compute(&pidB, speedB_cmps);
								
								// 计算最终PWM值
								uint32_t pwmA = target_speed + (uint32_t)correctionA;
								uint32_t pwmB = target_speed + (uint32_t)correctionB;
								
								// PWM限幅
								if (pwmA > MAX_SPEED) pwmA = MAX_SPEED;
								if (pwmB > MAX_SPEED) pwmB = MAX_SPEED;
								if (pwmA < MIN_PWM) pwmA = MIN_PWM;
								if (pwmB < MIN_PWM) pwmB = MIN_PWM;
								
								// 应用PWM - 根据方向设置不同的通道
								if (current_direction == DIR_FORWARD) {
										__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwmA);  // 左轮前进
										__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, pwmB);  // 右轮前进
								} else if (current_direction == DIR_BACKWARD) {
										__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwmA);  // 左轮后退  
										__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, pwmB);  // 右轮后退
								}
						}
				}

        // MPU6500姿态数据读取
        MPU6500_ReadData(&hi2c1, &mpu_data);
        
        // 调试信息输出（雷达数据发送）
        uint32_t current_time = HAL_GetTick();
        if (current_time - last_debug_time >= 5) {  // 每5ms发送一次
            last_debug_time = current_time;
            if (has_new_lidar_data) {
                char buf[48];
                int len = snprintf(buf, sizeof(buf), "A:%.2f,D:%.2f,Q:%u\r\n", latest_lidar_angle, latest_lidar_distance, latest_lidar_quality);
                // snprintf 返回值可能大于 sizeof(buf)-1，当len>=sizeof(buf)时需要截断为可发送长度
                uint16_t send_len = (len > 0) ? (uint16_t)len : 0;
                if (send_len >= sizeof(buf)) send_len = sizeof(buf) - 1;
                if (send_len > 0) {
                    HAL_UART_Transmit(&huart1, (uint8_t*)buf, send_len, 50); // 提高超时到50ms以避免IO被截断
                }
                has_new_lidar_data = 0;
            }
        }

        // （已移除按编码器脉冲停止的逻辑，前进/后退仅由停止命令或速度渐变控制停止）
    }
}

/**
 * @brief 系统时钟配置函数
 * 配置系统时钟为180MHz
 */
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    
    // 使能PWR时钟
    __HAL_RCC_PWR_CLK_ENABLE();
    
    // 配置电压调节器
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    
    // 配置HSI振荡器
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    
    // 配置PLL
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 8;      // 分频系数
    RCC_OscInitStruct.PLL.PLLN = 180;    // 倍频系数
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;  // 主输出分频
    RCC_OscInitStruct.PLL.PLLQ = 2;      // USB等外设分频
    RCC_OscInitStruct.PLL.PLLR = 2;      // 系统时钟分频
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();
    
    // 启用过驱动模式
    if (HAL_PWREx_EnableOverDrive() != HAL_OK) Error_Handler();
    
    // 配置时钟域
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;     // 系统时钟源选择PLL
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;           // AHB时钟不分频
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;            // APB1时钟4分频
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;            // APB2时钟2分频
    
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) Error_Handler();
}

/**
 * @brief 错误处理函数
 * 当系统发生错误时调用此函数
 */
void Error_Handler(void) {
    __disable_irq();  // 禁用中断
    while (1) {       // 无限循环等待
    }
}
