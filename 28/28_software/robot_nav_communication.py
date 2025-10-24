"""
================================================================================
智能机器人导航系统 - 串口通信模块
================================================================================
实现蓝牙/串口通信协议
"""

import serial
import time
from datetime import datetime
from typing import Tuple, Optional
from robot_nav_core import MovementCommand, LaserScanData, TargetPose, Pose2D, logger

# ============================================================================
# 串口通信配置
# ============================================================================

class SerialCommConfig:
    """串口通信配置"""
    def __init__(self, port: str = "COM7", baudrate: int = 115200, timeout: float = 0.5):
        self.port_name = port
        self.baud_rate = baudrate
        self.timeout_seconds = timeout
        self.min_command_gap = 3.0  # 命令最小间隔（秒）
        self.retry_count = 3
        self.ack_byte_value = 0xAA

# ============================================================================
# 串口通信接口
# ============================================================================

class HardwareCommunicator:
    """硬件串口通信接口"""
    
    # 协议字节定义
    PKT_HEADER = 0x55
    PKT_FOOTER = 0xFF
    DIR_FORWARD = 0x01
    DIR_BACKWARD = 0x02
    DIR_LEFT_TURN = 0x03
    DIR_RIGHT_TURN = 0x04
    
    def __init__(self, config: SerialCommConfig):
        self.config = config
        self.serial_connection = serial.Serial(
            config.port_name,
            config.baud_rate,
            timeout=config.timeout_seconds
        )
        
        self.last_command_timestamp: Optional[datetime] = None
        self.last_transmitted_packet: Optional[bytes] = None
        
        logger.info(f"串口已连接: {config.port_name} @ {config.baud_rate}bps")
    
    def transmit_grid_movement(self, command: MovementCommand, 
                               max_retries: int = 3) -> Tuple[bool, Optional[LaserScanData]]:
        """
        发送网格化移动指令
        
        Args:
            command: 移动指令
            max_retries: 最大重试次数
        
        Returns:
            (成功标志, 雷达扫描数据)
        """
        # 强制命令间隔
        self._enforce_command_interval()
        
        logger.info(f"发送网格指令: {command.execution_phase}")
        
        # 构建数据包
        if command.execution_phase == "rotate":
            packet = self._build_rotation_packet(command.rotation_degrees)
        elif command.execution_phase == "forward":
            packet = self._build_forward_packet(command.forward_distance)
        else:
            logger.error(f"未知执行阶段: {command.execution_phase}")
            return False, None
        
        self.last_transmitted_packet = packet
        
        # 重试机制
        for attempt in range(max_retries):
            logger.info(f"尝试 {attempt + 1}/{max_retries}")
            
            # 清空输入缓冲
            self.serial_connection.reset_input_buffer()
            time.sleep(0.1)
            
            # 发送数据包
            self.serial_connection.write(packet)
            self.last_command_timestamp = datetime.now()
            
            # 等待响应
            timeout = 8.0 if attempt == 0 else 5.0
            success, scan_data = self._wait_for_acknowledgment(timeout)
            
            if success:
                logger.info("指令执行成功")
                return True, scan_data
            else:
                logger.warning(f"第 {attempt + 1} 次尝试失败")
                if attempt < max_retries - 1:
                    time.sleep(2.0)  # 重试前等待
        
        logger.error("所有尝试均失败")
        return False, None
    
    def _build_rotation_packet(self, angle_degrees: float) -> bytes:
        """构建旋转指令数据包"""
        if angle_degrees > 0:
            direction = self.DIR_LEFT_TURN
        else:
            direction = self.DIR_RIGHT_TURN
            angle_degrees = abs(angle_degrees)
        
        angle_int = int(max(0, min(360, round(angle_degrees)))) & 0xFFFF
        left_ratio = 0x0016  # 22
        right_ratio = 0x0016  # 22
        
        packet = bytes([
            self.PKT_HEADER,
            direction,
            (left_ratio >> 8) & 0xFF,
            left_ratio & 0xFF,
            (right_ratio >> 8) & 0xFF,
            right_ratio & 0xFF,
            (angle_int >> 8) & 0xFF,
            angle_int & 0xFF,
            self.PKT_FOOTER
        ])
        
        hex_string = ' '.join(f'{b:02X}' for b in packet)
        logger.debug(f"旋转数据包: {hex_string} (角度={angle_int}°)")
        
        return packet
    
    def _build_forward_packet(self, distance_meters: float) -> bytes:
        """构建前进指令数据包"""
        direction = self.DIR_FORWARD
        left_ratio = 0x0016
        right_ratio = 0x0016
        
        # 距离转协议单位（厘米）
        distance_value = int(max(0, min(0xFFFF, round(distance_meters * 100))))
        
        packet = bytes([
            self.PKT_HEADER,
            direction,
            (left_ratio >> 8) & 0xFF,
            left_ratio & 0xFF,
            (right_ratio >> 8) & 0xFF,
            right_ratio & 0xFF,
            (distance_value >> 8) & 0xFF,
            distance_value & 0xFF,
            self.PKT_FOOTER
        ])
        
        hex_string = ' '.join(f'{b:02X}' for b in packet)
        logger.debug(f"前进数据包: {hex_string} (距离={distance_meters:.2f}m)")
        
        return packet
    
    def _enforce_command_interval(self):
        """强制命令间隔"""
        if self.last_command_timestamp:
            elapsed = (datetime.now() - self.last_command_timestamp).total_seconds()
            if elapsed < self.config.min_command_gap:
                wait_time = self.config.min_command_gap - elapsed
                logger.debug(f"等待命令间隔: {wait_time:.1f}秒")
                time.sleep(wait_time)
    
    def _wait_for_acknowledgment(self, timeout_seconds: float) -> Tuple[bool, Optional[LaserScanData]]:
        """
        等待下位机确认信号和雷达数据
        
        Returns:
            (成功标志, 雷达扫描数据)
        """
        start_time = datetime.now()
        angle_list = []
        range_list = []
        ack_received = False
        
        logger.debug(f"等待响应（超时{timeout_seconds}秒）...")
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            try:
                if self.serial_connection.in_waiting > 0:
                    # 读取一个字节检查ACK
                    byte_data = self.serial_connection.read(1)
                    if len(byte_data) > 0:
                        if byte_data[0] == self.config.ack_byte_value:
                            logger.debug("收到ACK(0xAA)")
                            ack_received = True
                            time.sleep(0.2)
                            
                            if self.serial_connection.in_waiting == 0:
                                return True, None
                            continue
                        else:
                            # 尝试解析为文本数据
                            try:
                                self.serial_connection.reset_input_buffer()
                                time.sleep(0.05)
                                
                                if self.serial_connection.in_waiting > 0:
                                    line = self.serial_connection.readline().decode('utf-8').strip()
                                    if line and ',' in line:
                                        parts = line.split(',')
                                        if len(parts) == 2:
                                            angle = float(parts[0])
                                            distance = float(parts[1])
                                            angle_list.append(angle)
                                            range_list.append(distance)
                                            logger.debug(f"雷达数据: angle={angle:.3f}, dist={distance:.3f}")
                            except:
                                pass
                
                # 检查是否收到足够数据
                if len(angle_list) >= 4:
                    logger.info(f"收到完整雷达数据({len(angle_list)}条)")
                    return True, LaserScanData(angle_list, range_list)
                
            except Exception as e:
                logger.debug(f"读取异常: {e}")
                time.sleep(0.05)
        
        # 超时处理
        if ack_received or len(angle_list) > 0:
            scan = LaserScanData(angle_list, range_list) if angle_list else None
            return True, scan
        
        logger.warning("响应超时")
        return False, None
    
    def transmit_continuous_setpoint(self, setpoint: TargetPose, 
                                     current_pose: Pose2D) -> None:
        """
        发送连续空间控制指令
        
        Args:
            setpoint: 目标位姿
            current_pose: 当前位姿
        """
        import math
        
        # 计算距离
        dx = setpoint.x - current_pose.x
        dy = setpoint.y - current_pose.y
        distance_m = math.hypot(dx, dy)
        distance_value = int(max(0, min(0xFFFF, round(distance_m * 100))))
        
        # 计算转向
        target_heading = math.atan2(dy, dx)
        angle_diff = (target_heading - current_pose.theta + math.pi) % (2 * math.pi) - math.pi
        
        # 判断前进/后退
        if abs(angle_diff) > math.pi / 2:
            direction = self.DIR_BACKWARD
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        else:
            direction = self.DIR_FORWARD
        
        # 计算左右轮速比
        base_ratio = 22
        if abs(angle_diff) < math.radians(3):
            left_ratio = right_ratio = base_ratio
        else:
            turn_intensity = min(abs(angle_diff) / (math.pi / 2), 1.0)
            ratio_reduction = int(base_ratio * turn_intensity * 0.25)
            
            if angle_diff > 0:  # 左转
                left_ratio = max(1, base_ratio - ratio_reduction)
                right_ratio = base_ratio
            else:  # 右转
                left_ratio = base_ratio
                right_ratio = max(1, base_ratio - ratio_reduction)
        
        # 构建数据包
        packet = bytes([
            self.PKT_HEADER,
            direction,
            (left_ratio >> 8) & 0xFF,
            left_ratio & 0xFF,
            (right_ratio >> 8) & 0xFF,
            right_ratio & 0xFF,
            (distance_value >> 8) & 0xFF,
            distance_value & 0xFF,
            self.PKT_FOOTER
        ])
        
        hex_string = ' '.join(f'{b:02X}' for b in packet)
        logger.debug(f"连续控制数据包: {hex_string}")
        self.serial_connection.write(packet)
    
    def shutdown(self):
        """关闭串口"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info(f"串口 {self.config.port_name} 已关闭")

if __name__ == "__main__":
    print("串口通信模块测试")
    
    # 测试配置
    config = SerialCommConfig()
    logger.info(f"通信配置: {config.port_name} @ {config.baud_rate}bps")

