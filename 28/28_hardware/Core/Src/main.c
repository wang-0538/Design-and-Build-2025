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
  *                   - 蓝牙通信控制
  *                   - ADC电压监测
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
#include <stdlib.h>    // 标准库，包含abs函数

// 电机控制相关宏定义
#define MAX_SPEED 400                   // 最大电机速度（PWM值）
#define SPEED_STEP 120                  // 速度调整步长
#define SPEED_UPDATE_INTERVAL 30        // 速度更新间隔（毫秒）
#define TURN_DURATION 1050              // 转向持续时间（毫秒）
#define TURN_SPEED 300                  // 转向速度（PWM值）

#define MAX_PWM                700
#define MIN_PWM                60 

// 电机速度补偿系数（根据实际测试调整）
#define MOTOR_LEFT_FORWARD_COMPENSATION 0.95f
#define MOTOR_RIGHT_BACKWARD_COMPENSATION 0.95f

// 运动方向定义
#define DIR_STOP 0                      // 停止
#define DIR_FORWARD 1                   // 前进
#define DIR_BACKWARD 2                  // 后退
#define DIR_LEFT 3                      // 左转
#define DIR_RIGHT 4                     // 右转
#define DIR_UTURN 5                     // 掉头（U型转向）

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
// 精准控距相关
#define TARGET_DISTANCE_CM 180.0f                    // 目标距离（cm）
#define WHEEL_DIAMETER_CM 20.42f                    // 轮径（cm）
#define WHEEL_CIRCUMFERENCE_CM (3.1415926535f * WHEEL_DIAMETER_CM)  // 周长
#define PULSES_PER_CM ((PPR * ENCODER_COUNT_MULTIPLIER) / WHEEL_CIRCUMFERENCE_CM)  // 每cm脉冲数
#define TARGET_PULSES ((uint32_t)(TARGET_DISTANCE_CM * PULSES_PER_CM))  // 目标脉冲数
// 障碍检测相关
#define OBSTACLE_DISTANCE_THRESHOLD_MM 700 // 70cm = 700mm
#define OBSTACLE_CHECK_DURATION_MS 3000     // 检测时间窗口（毫秒），增加到3000ms确保充分扫描所有方向
// 目标方向视角：要求分别检测四个方向90度视角，误差15度 -> 半宽 = 45 + 15 = 60
#define SECTOR_HALF_WIDTH_DEG 45.0f       // 半宽45°，总90°扇区

//1
// 电机控制相关宏定义
// 新增转向控制参数
#define TURN_TIME_UNIT_MS      5        // 转向时间单位（毫秒）
#define TURN_PWM               300      // 转向PWM值
#define TURN_MAX_DEG           180.0f   // 最大转向角度（度）
#define PI_F                   3.1415926f  // 圆周率
#define WHEEL_BASE_CM          13.0f    // 两轮中心距（cm）

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
PID_Controller pidA = {0.8f, 0.01f, 0.1f, 0, 0, 0, 0, MAX_SPEED, 0};  // 电机A的PID控制器
PID_Controller pidB = {0.8f, 0.01f, 0.1f, 0, 0, 0, 0, MAX_SPEED, 0};  // 电机B的PID控制器

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

//2
// 新增运动状态机
typedef enum {
    MOT_IDLE = 0,           // 空闲，停止
    MOT_FWD_DIST,           // 前进指定距离
    MOT_BWD_DIST,           // 后退指定距离  
    MOT_TURN_LEFT_ANGLE,     // 左转状态 - 按角度/位移控制
    MOT_TURN_RIGHT_ANGLE     // 右转状态 - 按角度/位移控制
} motion_state_t;

static motion_state_t motion = MOT_IDLE;  // 当前运动状态

// 新增转向控制变量
static float turn_target_sumabs_cm = 0.0f;  // 转向目标位移量(cm) - 计算公式: WHEEL_BASE * theta
static float turn_accum_sumabs_cm  = 0.0f;  // 转向已累计位移量(cm) - 根据编码器数据累计

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

// UART句柄外部声明
extern UART_HandleTypeDef huart1;  // 蓝牙通信UART
extern UART_HandleTypeDef huart6;  // 激光雷达UART

// 调试相关全局变量
static uint32_t last_debug_time = 0;       // 调试信息输出时间（统一2秒）
// 精准控距状态
volatile int32_t accumulated_pulses = 0;           // 累积脉冲数
uint8_t distance_control_active = 0;              // 是否激活距离控制

// 函数声明
void SystemClock_Config(void);                                    // 系统时钟配置
void ControlMotor(uint8_t direction, uint32_t speed);            // 电机控制函数
void UpdateSpeedRamp(void);                                       // 速度渐变更新函数
void SendConnectionNotification(void);                            // 发送连接通知函数
void ProcessLidarData(uint8_t* data);                            // 激光雷达数据处理函数
void SendLidarToBluetooth(float angle, float distance, uint8_t quality); // 激光雷达数据发送函数
void StartObstacleCheck(void);
void FinalizeObstacleCheck(void);
void SendLidarDataToBluetooth(void);                    // 发送雷达数据到蓝牙
static void HandleMotionManager(int32_t dCntA, int32_t dCntB);
void SendLidarDataToBluetooth(void);
static void MotorTurnLeftPWM(int pwm);
static void MotorTurnRightPWM(int pwm);

/**
 * @brief UART接收完成回调函数
 * @param huart UART句柄指针
 * 处理来自蓝牙和激光雷达的数据接收
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart1) {  // 蓝牙UART数据接收
        // 标记蓝牙已连接
        if (!bluetooth_connected) bluetooth_connected = 1;

        if (bluetooth_rx_data >= '0' && bluetooth_rx_data <= '4') {
            uint8_t cmd = bluetooth_rx_data - '0';
            target_direction = cmd;
            if (cmd == DIR_STOP) {
                // 停止所有动作
                target_speed = 0;
								motion = MOT_IDLE; 
                use_timed_turning = 0;
                ControlMotor(DIR_STOP, 0);
                current_direction = DIR_STOP;
                // 发起一次障碍检测窗口
                StartObstacleCheck();
            } else if (cmd == DIR_FORWARD || cmd == DIR_BACKWARD) {
                // 启动精准距离控制（前进/后退70cm）
                current_direction = cmd;
                target_speed = MAX_SPEED;
                accumulated_pulses = 0;  // 重置累积脉冲
                distance_control_active = 1;  // 激活距离控制
                ControlMotor(current_direction, target_speed);
            } else if (cmd == DIR_LEFT || cmd == DIR_RIGHT) {
								// 使用角度转向，默认90度转向
								float angle_deg = 90.0f;  
								
								// 设置角度转向参数
								turn_target_sumabs_cm = WHEEL_BASE_CM * (angle_deg * PI_F / 180.0f);
								turn_accum_sumabs_cm  = 0.0f;
								
								// 设置对应的转向状态
								if (cmd == DIR_LEFT) {
										motion = MOT_TURN_LEFT_ANGLE;
										MotorTurnLeftPWM(TURN_PWM);  // 启动左转
								} else {
										motion = MOT_TURN_RIGHT_ANGLE;
										MotorTurnRightPWM(TURN_PWM); // 启动右转
								}
								
								current_direction = cmd;
						}
        }

        // 重新启动蓝牙接收中断
        HAL_UART_Receive_IT(&huart1, &bluetooth_rx_data, 1);
    } 
    else if (huart == &huart6) {  // 激光雷达UART数据接收
        // 重新启动激光雷达DMA接收
        HAL_UART_Receive_DMA(&huart6, lidar_dma_buffer, DMA_BUFFER_SIZE);
        // 检查数据包起始标志
        if (lidar_dma_buffer[0] == 0xA5) lidar_process_packet = 1;
    }
}

/**
 * @brief 电机控制函数
 * @param direction 运动方向
 * @param speed 运动速度
 * 根据方向和速度控制四个电机的PWM输出
 */
void ControlMotor(uint8_t direction, uint32_t speed) {
		// 在开始任何直行操作前，强制清除转向状态
    if (direction == DIR_FORWARD || direction == DIR_BACKWARD) {
        // 重置所有转向相关变量
        motion = MOT_IDLE;
        use_timed_turning = 0;
        turn_accum_sumabs_cm = 0;
        turn_target_sumabs_cm = 0;
        
        // 重置PID控制器状态
        pidA.integral = 0;
        pidA.last_error = 0;
        pidB.integral = 0; 
        pidB.last_error = 0;
    }
    switch(direction) {
        case DIR_STOP:  // 停止所有电机 - 带制动补偿版本
				// 首先停止所有电机
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
				__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
				
				// 制动补偿：根据运动方向给右轮短暂反向制动
				if (current_direction == DIR_FORWARD) {
						// 前进时停止，给右轮短暂反向制动
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 80);  // 右轮反转制动
						HAL_Delay(15);  // 制动时间20ms
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);   // 停止制动
				} 
				else if (current_direction == DIR_BACKWARD) {
						// 后退时停止，给右轮短暂正向制动  
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 80);  // 右轮正转制动
						HAL_Delay(20);
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
				}
				// 转向时通常不需要特殊制动补偿
				
				// 重置所有运动状态
				motion = MOT_IDLE;
				distance_control_active = 0;
				accumulated_pulses = 0;
				turn_accum_sumabs_cm = 0;
				turn_target_sumabs_cm = 0;
				break;
				
				case DIR_FORWARD: {  // 前进运动控制
						// 分别调整左右轮补偿系数
						static float left_compensation = 0.95f;   // 可调整
						static float right_compensation = 0.99f;  // 可调整
						
						int32_t pwmA = (int32_t)(speed * left_compensation);
						int32_t pwmB = (int32_t)(speed * right_compensation);

						// PWM值限幅处理
						if (pwmA > (int32_t)MAX_SPEED) pwmA = MAX_SPEED;
						if (pwmB > (int32_t)MAX_SPEED) pwmB = MAX_SPEED;
						if (pwmA < 200) pwmA = 200;
						if (pwmB < 200) pwmB = 200;

						// 设置电机PWM输出
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, (uint32_t)pwmA);
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, (uint32_t)pwmB);
						__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
						
						// 调试输出实际PWM值
						static uint32_t last_pwm_debug = 0;
						if (HAL_GetTick() - last_pwm_debug > 1000) {
								//snprintf(uart_buf, sizeof(uart_buf), "[PWM] Left:%d, Right:%d\r\n", pwmA, pwmB);
								HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);
								last_pwm_debug = HAL_GetTick();
						}
						break;
				}

        case DIR_BACKWARD: {  // 后退运动控制
            // 分别调整左右轮后退补偿系数
						static float left_backward_compensation = 1.00f;   // 左轮后退补偿系数
						static float right_backward_compensation = 0.95f;  // 右轮后退补偿系数
						
						// 应用后退补偿系数
						int32_t pwmA = (int32_t)(speed * left_backward_compensation);   // 左轮PWM
						int32_t pwmB = (int32_t)(speed * right_backward_compensation);  // 右轮PWM

            // 无航向修正，保持原始后退控制

            // PWM值限幅处理
            if (pwmA > (int32_t)MAX_SPEED) pwmA = MAX_SPEED;
            if (pwmB > (int32_t)MAX_SPEED) pwmB = MAX_SPEED;
            if (pwmA < 200) pwmA = 200;  // 确保最小PWM值足够启动电机
            if (pwmB < 200) pwmB = 200;

            // 设置电机PWM输出（后退方向）
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, (uint32_t)pwmA);  // 左轮（基准）
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, (uint32_t)pwmB);  // 右轮（补偿）
            break;
        }

        case DIR_UTURN:  // U型转向控制
            // 与左转相同的电机控制方式
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, TURN_SPEED);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, TURN_SPEED);
            __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
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
                ControlMotor(current_direction, target_speed);
            }
            // 转向操作
            else if (target_direction == DIR_LEFT || target_direction == DIR_RIGHT || target_direction == DIR_UTURN) {
                target_speed = (target_direction == DIR_UTURN) ? TURN_SPEED : TURN_SPEED;
                current_direction = target_direction;
                ControlMotor(current_direction, target_speed);
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
										ControlMotor(current_direction, current_speed);
								} else {
										// 完全停止 - 强制同步停止
										current_direction = target_direction;
										ControlMotor(DIR_STOP, 0);  // 使用DIR_STOP确保完全停止
										
										// 完全停止后触发一次障碍检测窗口
										StartObstacleCheck();
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
            ControlMotor(current_direction, current_speed);
        }
    }
}

/**
 * @brief 发送连接通知函数
 * 当蓝牙连接建立时发送确认消息
 */
void SendConnectionNotification(void) {
    if (bluetooth_connected && !connection_announced) {
        HAL_UART_Transmit(&huart1, connection_msg, sizeof(connection_msg)-1, 1000);
        connection_announced = 1;
    }
}

// 激光雷达最新数据存储全局变量
float latest_lidar_angle = 0.0f;      // 最新激光雷达角度
float latest_lidar_distance = 0.0f;   // 最新激光雷达距离
uint8_t latest_lidar_quality = 0;     // 最新激光雷达质量
uint8_t has_new_lidar_data = 0;       // 新数据标志
// 障碍检测状态
volatile uint8_t obstacle_check_active = 0;   // 是否在进行障碍检测
uint32_t obstacle_check_start = 0;           // 检测开始时间（ms）
uint8_t obstacle_front = 0;
uint8_t obstacle_back = 0;
uint8_t obstacle_left = 0;
uint8_t obstacle_right = 0;
uint8_t obstacle_report_sent = 0;
// 添加检测计数器，用于调试
uint32_t obstacle_front_count = 0;
uint32_t obstacle_back_count = 0;
uint32_t obstacle_left_count = 0;
uint32_t obstacle_right_count = 0;

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
        if(distance >= MIN_VALID_DISTANCE && 
           distance <= MAX_VALID_DISTANCE && 
           quality > 0) {
            // 保存最新有效数据
            latest_lidar_angle = angle;
            latest_lidar_distance = distance;
            latest_lidar_quality = quality;
            has_new_lidar_data = 1;  // 设置新数据标志
            lastLidarAngle = angle;   // 更新上次角度
            // 如果正在进行障碍检测，则根据角度分类到相应扇区并设置障碍标志（半宽 SECTOR_HALF_WIDTH_DEG）
            if (obstacle_check_active) {
                // 计算角度差到四个正交方向（0=front,90=right,180=back,270=left）
                float d_front = fabsf(fmodf(fabsf(angle - 0.0f), 360.0f));
                if (d_front > 180.0f) d_front = 360.0f - d_front;
                float d_right = fabsf(fmodf(fabsf(angle - 90.0f), 360.0f));
                if (d_right > 180.0f) d_right = 360.0f - d_right;
                float d_back = fabsf(fmodf(fabsf(angle - 180.0f), 360.0f));
                if (d_back > 180.0f) d_back = 360.0f - d_back;
                float d_left = fabsf(fmodf(fabsf(angle - 270.0f), 360.0f));
                if (d_left > 180.0f) d_left = 360.0f - d_left;
                // 若在扇区内并且距离小于阈值则判定有障碍
                if (d_front <= SECTOR_HALF_WIDTH_DEG && distance <= OBSTACLE_DISTANCE_THRESHOLD_MM) obstacle_front = 1;
                if (d_right <= SECTOR_HALF_WIDTH_DEG && distance <= OBSTACLE_DISTANCE_THRESHOLD_MM) obstacle_right = 1;
                if (d_back  <= SECTOR_HALF_WIDTH_DEG && distance <= OBSTACLE_DISTANCE_THRESHOLD_MM) obstacle_back = 1;
                if (d_left  <= SECTOR_HALF_WIDTH_DEG && distance <= OBSTACLE_DISTANCE_THRESHOLD_MM) obstacle_left = 1;
            }
        }
    }
    
    // 处理完成后重置DMA
    if(lidar_process_packet) {
        lastLidarRxTime = HAL_GetTick();  // 更新最后接收时间
        __HAL_DMA_DISABLE(huart6.hdmarx); // 禁用DMA
        huart6.hdmarx->Instance->NDTR = DMA_BUFFER_SIZE; // 重置DMA计数器
        __HAL_DMA_ENABLE(huart6.hdmarx);  // 重新启用DMA
        lidar_process_packet = 0;          // 清除处理标志
    }
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
                      "[LIDAR] Angle:%.2f°, Dist:%.2fmm, Quality:%d\r\n", 
                      angle, distance, quality);
    // 通过蓝牙发送数据
    HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
}

/**
 * @brief 开始一次障碍检测窗口
 * 将在 OBSTACLE_CHECK_DURATION_MS 时间内收集激光点以判断四周障碍
 */
void StartObstacleCheck(void) {
    obstacle_front = obstacle_back = obstacle_left = obstacle_right = 0;
    obstacle_report_sent = 0;
    obstacle_check_start = HAL_GetTick();
    obstacle_check_active = 1;
    // 移除调试打印
}

/**
 * @brief 结束障碍检测并通过蓝牙发送结果
 * 格式：[Obstacle]前后左右顺序的 1/0 序列，例如 "[Obstacle]1001"
 */
void FinalizeObstacleCheck(void) {
    if (!obstacle_check_active) return;
    obstacle_check_active = 0;
    char msg[32];
    // 顺序：前 后 左 右
    int len = snprintf(msg, sizeof(msg), "[Obstacle]%c%c%c%c\r\n", 
                       obstacle_front ? '1' : '0',
                       obstacle_back ? '1' : '0',
                       obstacle_left ? '1' : '0',
                       obstacle_right ? '1' : '0');
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, len, 100);
    obstacle_report_sent = 1;
}

/**
 * @brief 发送雷达数据到蓝牙
 * 发送角度和距离信息到蓝牙设备
 */
void SendLidarDataToBluetooth(void) {
    char lidar_buffer[64];
    
    // 格式化雷达数据：角度和距离
    int len = snprintf(lidar_buffer, sizeof(lidar_buffer),
                      "[RADAR] Angle:%.1f°, Distance:%.1fmm\r\n",
                      latest_lidar_angle, latest_lidar_distance);
    
    // 通过蓝牙发送数据
    if (len > 0) {
        HAL_UART_Transmit(&huart1, (uint8_t*)lidar_buffer, len, 100);
    }
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
        //strcpy(uart_buf, "MPU6500 init failed!\r\n");
    } else {
        //strcpy(uart_buf, "MPU6500 OK\r\n");
    }
    HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);

    // 等待系统稳定
    HAL_Delay(500);
    
    // 启动激光雷达
    uint8_t startCmd[] = {0xA5, 0x20};  // 激光雷达启动命令
    HAL_UART_Transmit(&huart6, startCmd, sizeof(startCmd), 100);
    HAL_UART_Receive_DMA(&huart6, lidar_dma_buffer, DMA_BUFFER_SIZE);
		
    // 通知激光雷达启动
    //snprintf(uart_buf, sizeof(uart_buf), "LIDAR: Starting scan...\r\n");
    HAL_UART_Transmit(&huart1, (uint8_t *)uart_buf, strlen(uart_buf), 100);

    // 主循环
    while (1) {
        // 发送蓝牙连接通知
        SendConnectionNotification();
        
        // 更新速度渐变控制
        UpdateSpeedRamp();

    // 读取编码器数据并计算电机转速
    uint32_t encoderA = (uint32_t)__HAL_TIM_GET_COUNTER(&htim2);  // 读取编码器A计数
    uint32_t encoderB = (uint32_t)__HAL_TIM_GET_COUNTER(&htim4);  // 读取编码器B计数

    // 计算编码器增量（使用无符号差值以处理计数器回绕）
    uint32_t deltaA = encoderA - lastEncoderA;
    uint32_t deltaB = encoderB - lastEncoderB;
    lastEncoderA = encoderA;
    lastEncoderB = encoderB;
			
		// 新增：转向运动状态管理
    HandleMotionManager((int32_t)deltaA, (int32_t)deltaB);

    // 计算转速（RPM） - 使用绝对增量近似速度（不区分方向）
    float rpmA = (float)deltaA / PPR * (60.0f / (SAMPLE_TIME_MS / 1000.0f));
    float rpmB = (float)deltaB / PPR * (60.0f / (SAMPLE_TIME_MS / 1000.0f));
			
        // 转换为线速度（cm/s）
        float speedA_cmps = rpmA * 20.42f / 60.0f;  // 轮径约20.42cm
        float speedB_cmps = rpmB * 20.42f / 60.0f;
			
        // 累积脉冲（前进/后退时都递增）
        if (distance_control_active) {
            // 使用已计算的deltaA和deltaB的绝对值
            accumulated_pulses += abs((int32_t)deltaA) + abs((int32_t)deltaB);
            // 检查是否达到目标距离
            if (accumulated_pulses >= TARGET_PULSES) {
                // 达到目标，停止并触发障碍检测
                target_direction = DIR_STOP;
                current_direction = DIR_STOP;
                target_speed = 0;
								current_speed = 0;  // 确保当前速度也清零
                distance_control_active = 0;  // 关闭距离控制
                ControlMotor(DIR_STOP, 0);
                StartObstacleCheck();  // 触发障碍检测
                // 可选：发送通知
                //snprintf(uart_buf, sizeof(uart_buf), "[Distance] Reached 70cm, stopping\r\n");
                HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);
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
                
                // 查找数据包起始标志
                if(lidar_dataIndex == 0) {
                    if(data == 0xA5) {  // 激光雷达数据包起始标志
                        lidar_dataBuffer[lidar_dataIndex++] = data;
                    }
                } else {
                    lidar_dataBuffer[lidar_dataIndex++] = data;
                    // 完整数据包处理
                    if(lidar_dataIndex >= DATA_PACKET_SIZE) {
                        ProcessLidarData(lidar_dataBuffer);
                        lidar_dataIndex = 0;
											
												// 新增：处理完数据包后立即发送到蓝牙
												if (has_new_lidar_data) {
														SendLidarDataToBluetooth();
														has_new_lidar_data = 0;  // 清除标志
												}
                    }
                }
            }
            lidar_rxIndex = currentRxIndex;  // 更新接收索引
        }
        
        // 激光雷达数据超时处理
        if(lidar_dataIndex > 0 && (HAL_GetTick() - lastLidarRxTime > LIDAR_TIMEOUT_THRESHOLD)) {
            lidar_dataIndex = 0;  // 重置数据索引
        }

        // ADC电压监测
        HAL_ADC_Start(&hadc1);
        HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
        uint32_t adc_val = HAL_ADC_GetValue(&hadc1);
        float voltage = (adc_val * 3.3f * 11.0f) / 4096.0f;  // 电压转换（考虑分压电阻）

        // PID控制器目标值设置
        if (current_direction == DIR_FORWARD || current_direction == DIR_BACKWARD) {
            pidA.setpoint = target_speed;  // 设置电机A目标速度
            pidB.setpoint = target_speed;  // 设置电机B目标速度
        } else {
            pidA.setpoint = 0;
            pidB.setpoint = 0;
            pidA.integral = pidB.integral = 0;  // 清零积分项
        }

        // MPU6500姿态数据读取
        MPU6500_ReadData(&hi2c1, &mpu_data);
        
        // （已移除航向积分与航向PID相关代码）
        
        // 统一每2秒输出一次调试信息
        uint32_t current_time = HAL_GetTick();
        if (current_time - last_debug_time >= 2000) {
            // （航向信息已移除）
            
            // 输出电机转速信息
            //snprintf(uart_buf, sizeof(uart_buf), "[Motor] LeftRPM:%.1f, RightRPM:%.1f r/min\r\n", rpmA, rpmB);
            HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);
            
            // 输出线速度信息
            //snprintf(uart_buf, sizeof(uart_buf), "[Speed] Left:%.1f, Right:%.1f cm/s\r\n", speedA_cmps, speedB_cmps);
            HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);
            
            last_debug_time = current_time;
					  // 输出雷达数据信息
					  if (has_new_lidar_data) {
                int len = snprintf(uart_buf, sizeof(uart_buf),
                                   "[LIDAR] Angle:%.2f°, Dist:%.2fmm, Quality:%u\r\n",
                                   latest_lidar_angle, latest_lidar_distance,
                                   (unsigned)latest_lidar_quality);
                if (len > 0) {
                    // 只在四个正交方向附近发送（方便带宽低的蓝牙调试）
                    float a = latest_lidar_angle;
                    //float tol = 8.0f; // 扇区容差（度），可调
                    // 计算最小角差函数
                    float da_front = fabsf(fmodf(fabsf(a - 0.0f), 360.0f));
                    float da_right = fabsf(fmodf(fabsf(a - 90.0f), 360.0f));
                    float da_back = fabsf(fmodf(fabsf(a - 180.0f), 360.0f));
                    float da_left = fabsf(fmodf(fabsf(a - 270.0f), 360.0f));
                }
                // 避免每2秒重复打印同一帧；已经发送或被过滤后都可以清除标志
                has_new_lidar_data = 0;
            }
            // 输出距离控制状态
            if (distance_control_active) {
                int len = snprintf(uart_buf, sizeof(uart_buf), "[Distance] Accumulated: %d / %u pulses\r\n", accumulated_pulses, TARGET_PULSES);
                HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, len, 100);
            }
        }

        // 如果正在进行障碍检测且超时则汇总并发送
        if (obstacle_check_active) {
            if (HAL_GetTick() - obstacle_check_start >= OBSTACLE_CHECK_DURATION_MS) {
                FinalizeObstacleCheck();
            }
        }
    }
}

//3
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
            //use_timed_turning = 0;  // 清除定时转向标志
            target_direction = DIR_STOP;
            current_direction = DIR_STOP;
            //target_speed = 0;
            ControlMotor(DIR_STOP, 0);  // 停止电机
            
            // 发送转向完成消息
            //snprintf(uart_buf, sizeof(uart_buf), "[AngleTurn] Completed: %.1fcm\r\n", turn_accum_sumabs_cm);
            HAL_UART_Transmit(&huart1, (uint8_t*)uart_buf, strlen(uart_buf), 100);
        }  
    }
}

static void MotorTurnLeftPWM(int pwm)
{
    if (pwm < 0) pwm = 0;
    if (pwm > MAX_PWM) pwm = MAX_PWM;

    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pwm);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, pwm);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
}

static void MotorTurnRightPWM(int pwm)
{
    if (pwm < 0) pwm = 0;
    if (pwm > MAX_PWM) pwm = MAX_PWM;

    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pwm);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, pwm);
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
