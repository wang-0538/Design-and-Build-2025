import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
import math
from bleak import BleakClient, BleakScanner
import time
from dataclasses import dataclass
from typing import List, Tuple

# ========== 蓝牙参数 ==========
ROBOT_ADDRESS = "C4:25:02:08:02:21" # 我们小车的蓝牙mac地址
UART_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# ========== 模拟模式参数 ==========
SIMULATION_MODE = False # 设置为True启用模拟模式
SIMULATION_SPEED = 1.0  # 模拟速度倍数（1.0=正常速度，2.0=2倍速）

# ========== 迷宫参数 ==========
ARENA_SIZE_CM = 280.0
ROBOT_RADIUS_CM = 15.0  # 小车半径

# SLAM地图参数（基于雷达扫描的连续地图）
MAP_RES_CM = 5.0  # 5cm分辨率的占据栅格
MAP_SIZE = 200    # 100×100格子 = 500cm×500cm覆盖范围

# SLAM地图更新参数（调优以获得更清晰的建图）
L_OCC = 1.0   # 障碍物增强（1.2 → 2.0）
L_FREE = -0.7    # 自由空间减弱（-1.3 → -0.7）避免过度擦除
L_MAX = 10.0     # 增大范围（4.0 → 10.0）
L_MIN = -10.0    # 增大范围（-4.0 → -10.0）

CENTER_TOL_CM = 2.0

# ========== 起点终点（可调）==========
# 30cm网格，位置为30, 60, 90, 120, 150, 180, 210, 240, 270
# 起点：第8行第8列网格点 (240, 240)
# 终点：第1行第6列网格点 (180, 30)
START_X_CM = 240.0  # 第8列网格点
START_Y_CM = 240.0  # 第8行网格点
EXIT_X_CM = 180.0   # 第6列网格点
EXIT_Y_CM = 30.0    # 第1行网格点

# ========== 小车命令 ==========
CMD_STOP = 0x00
CMD_FORWARD = 0x01
CMD_BACKWARD = 0x02
CMD_TURN_LEFT = 0x03
CMD_TURN_RIGHT = 0x04

# ========== 雷达扫描命令（常见格式）==========
# 不同雷达可能使用不同的命令格式，需要根据实际雷达型号调整
CMD_SCAN_FORMATS = {
    'default': bytes([0xA5, 0x20]),           # 原始格式
    'rplidar_scan': bytes([0xA5, 0x20]),      # RPLidar 扫描命令
    'rplidar_stop': bytes([0xA5, 0x25]),      # RPLidar 停止命令
    'rplidar_reset': bytes([0xA5, 0x40]),     # RPLidar 复位命令
    'ascii_scan': b'SCAN\r\n',                # ASCII 格式
    'ascii_start': b'START\r\n',              # ASCII 启动
    'simple': bytes([0x20]),                  # 简单单字节
    'custom1': bytes([0xA5, 0x5A]),           # 自定义1
    'custom2': bytes([0x55, 0xAA]),           # 自定义2
}

# 当前使用的命令格式
CMD_SCAN = CMD_SCAN_FORMATS['default']

# ========== SLAM地图（与仿真完全一致）==========
class SlamMap:
    def __init__(self):
        self.size = MAP_SIZE
        self.resolution = MAP_RES_CM
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self.origin = (self.size // 2, self.size // 2)

    def world_to_grid(self, x_cm, y_cm):
        gx = int(round(x_cm / self.resolution)) + self.origin[0]
        gy = int(round(y_cm / self.resolution)) + self.origin[1]
        return gx, gy

    def grid_to_world(self, gx, gy):
        x_cm = (gx - self.origin[0]) * self.resolution
        y_cm = (gy - self.origin[1]) * self.resolution
        return x_cm, y_cm

    def is_valid(self, gx, gy):
        return 0 <= gx < self.size and 0 <= gy < self.size

    def get_occupancy_prob(self, gx, gy):
        if not self.is_valid(gx, gy):
            return 0.5
        return 1.0 / (1.0 + np.exp(-self.log_odds[gy, gx]))

    def is_occupied(self, gx, gy, th=0.8):
        return self.get_occupancy_prob(gx, gy) > th

    def is_free(self, gx, gy, threshold=0.45):
        return self.get_occupancy_prob(gx, gy) < threshold

    def is_unknown(self, gx, gy):
        if not self.is_valid(gx, gy):
            return True
        return abs(self.log_odds[gy, gx]) < 0.1

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def update_scan(self, robot_x, robot_y, robot_theta, scan_data, debug=False):
        """
        更新地图
        scan_data: [(angle_deg, distance_mm, quality)]
        注意：真实雷达返回mm，需要转换为cm
        """
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        
        # 统计信息
        total = len(scan_data)
        filtered_quality = 0
        filtered_distance = 0
        filtered_near = 0
        filtered_range = 0
        used = 0
        
        for angle_deg, distance_mm, quality in scan_data:
            # 降低质量阈值，接受更多数据
            if quality < 5:  # 10 → 5
                filtered_quality += 1
                continue
            
            if distance_mm < 100 or distance_mm > 12000:
                filtered_distance += 1
                continue
            
            distance_cm = distance_mm / 10.0  # mm转cm
            
            # 过滤过近的点（小车半径+余量）
            if distance_cm < (ROBOT_RADIUS_CM - 5.0):  # 15-5=10cm
                filtered_near += 1
                continue
            
            # 小车雷达角度转世界角度
            world_angle = robot_theta - angle_deg
            
            # 将雷达极坐标转换为世界坐标
            hit_x = robot_x + distance_cm * math.cos(math.radians(world_angle))
            hit_y = robot_y + distance_cm * math.sin(math.radians(world_angle))
            
            # 检查是否在地图范围内
            if not (0 <= hit_x <= ARENA_SIZE_CM and 0 <= hit_y <= ARENA_SIZE_CM):
                filtered_range += 1
                continue
            
            hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)
            
            if not self.is_valid(hit_gx, hit_gy):
                filtered_range += 1
                continue
            
            line_pts = self.bresenham_line(robot_gx, robot_gy, hit_gx, hit_gy)
            
            # 沿线标free（减少标记点数，避免过度擦除）
            for (gx, gy) in line_pts[:-2]:  # 只到终点前2个格子（减少擦除）
                if self.is_valid(gx, gy):
                    self.log_odds[gy, gx] = np.clip(
                        self.log_odds[gy, gx] + L_FREE, L_MIN, L_MAX)
            
            # 终点标occupied（强化障碍物）
            if line_pts:
                # 只标记终点和前1个格子
                for offset in range(max(0, len(line_pts)-2), len(line_pts)):
                    if offset < len(line_pts):
                        gx, gy = line_pts[offset]
                        if self.is_valid(gx, gy):
                            # 根据质量调整增量
                            weight = 1.0 + (quality / 50.0)  # 质量越高，权重越大
                            self.log_odds[gy, gx] = np.clip(
                                self.log_odds[gy, gx] + L_OCC * weight, L_MIN, L_MAX)
                            used += 1
        
        if debug:
            print(f"    [建图统计] 总:{total}, 用:{used}({used/max(1,total)*100:.0f}%), "
                  f"质量过滤:{filtered_quality}, 距离:{filtered_distance}, "
                  f"过近:{filtered_near}, 越界:{filtered_range}")

    def is_occupied_inflated(self, gx, gy, inflation_cm=8.0, occ_th=0.70):
        """检查是否被障碍物占据（带膨胀）
        小车半径15cm，膨胀8cm安全边距，总计23cm，平衡安全性和可达性
        """
        if not self.is_valid(gx, gy):
            return True
        
        prob = self.get_occupancy_prob(gx, gy)
        
        # 未知区域保守处理：如果概率接近0.5且log_odds很小，视为可能有障碍
        if abs(self.log_odds[gy, gx]) < 0.1:
            return False  # 完全未知区域不视为障碍（让探索继续）
        
        # 降低阈值，更容易判定为障碍
        if prob > occ_th:
            return True
        
        # 增大膨胀半径
        r = int(round(inflation_cm / self.resolution))
        if r <= 0:
            r = 2  # 至少膨胀2个格子
        
        for ix in range(gx - r, gx + r + 1):
            for iy in range(gy - r, gy + r + 1):
                if not self.is_valid(ix, iy):
                    # 地图边界外视为障碍
                    if (ix - gx) ** 2 + (iy - gy) ** 2 <= r ** 2:
                        return True
                    continue
                if (ix - gx) ** 2 + (iy - gy) ** 2 <= r ** 2:
                    prob_nb = self.get_occupancy_prob(ix, iy)
                    # 完全未知的邻居不算
                    if abs(self.log_odds[iy, ix]) < 0.1:
                        continue
                    # 降低阈值
                    if prob_nb > occ_th:
                        return True
        return False

    def is_line_passable(self, x0, y0, x1, y1):
        """检查路径是否可通行
        道路宽70cm，小车直径30cm，安全通行需要留8cm边距
        """
        # 端点检查
        for (x, y) in [(x0, y0), (x1, y1)]:
            gx, gy = self.world_to_grid(x, y)
            if self.is_occupied_inflated(gx, gy, inflation_cm=8.0, occ_th=0.70):
                return False
        
        # 计算路径长度
        length = math.hypot(x1 - x0, y1 - y0)
        
        # 密集采样（每10cm一个点）
        samples = max(3, int(length / 10.0))
        
        for i in range(samples + 1):
            t = i / samples
            sx = x0 + (x1 - x0) * t
            sy = y0 + (y1 - y0) * t
            gx, gy = self.world_to_grid(sx, sy)
            # 路径检查使用适中的安全边距
            if self.is_occupied_inflated(gx, gy, inflation_cm=8.0, occ_th=0.72):
                return False
        
        return True


# ========== 随机地图生成器 ==========
class RandomMapGenerator:
    def __init__(self, arena_size=280.0, grid_size=70.0):
        self.arena_size = arena_size
        self.grid_size = grid_size
        self.grids_x = int(arena_size / grid_size)
        self.grids_y = int(arena_size / grid_size)
        
    def generate_maze(self, seed=None):
        """生成随机迷宫"""
        if seed is not None:
            np.random.seed(seed)
        
        walls = []
        
        # 外边界
        walls.extend([
            (0, 0, self.arena_size, 0),      # 下边界
            (0, self.arena_size, self.arena_size, self.arena_size),  # 上边界
            (0, 0, 0, self.arena_size),      # 左边界
            (self.arena_size, 0, self.arena_size, self.arena_size),  # 右边界
        ])
        
        # 随机生成内部墙壁
        for i in range(1, self.grids_x-1):
            for j in range(1, self.grids_y-1):
                x = i * self.grid_size
                y = j * self.grid_size
                
                # 30%概率生成水平墙
                if np.random.random() < 0.3:
                    walls.append((x, y, x + self.grid_size, y))
                
                # 30%概率生成垂直墙
                if np.random.random() < 0.3:
                    walls.append((x, y, x, y + self.grid_size))
        
        return walls

# ========== 模拟雷达数据生成器 ==========
class SimulatedLidar:
    def __init__(self, arena_size=280.0, use_random_map=True, map_seed=42):
        self.arena_size = arena_size
        self.use_random_map = use_random_map
        
        if use_random_map:
            # 使用随机地图
            map_gen = RandomMapGenerator(arena_size)
            self.walls = map_gen.generate_maze(map_seed)
            print(f"🗺️ 生成随机地图，种子={map_seed}")
        else:
            # 使用固定地图
            self.walls = [
                # 外边界
                (0, 0, arena_size, 0),      # 下边界
                (0, arena_size, arena_size, arena_size),  # 上边界
                (0, 0, 0, arena_size),      # 左边界
                (arena_size, 0, arena_size, arena_size),  # 右边界
                
                # 内部障碍物（模拟迷宫）
                (70, 70, 140, 70),          # 水平墙1
                (140, 70, 140, 140),        # 垂直墙1
                (140, 140, 210, 140),       # 水平墙2
                (210, 140, 210, 210),       # 垂直墙2
                (70, 210, 140, 210),        # 水平墙3
                (140, 210, 140, 280),       # 垂直墙2
            ]
    
    def generate_scan_data(self, robot_x, robot_y, robot_theta):
        """生成模拟雷达扫描数据"""
        scan_data = []
        
        # 生成360度扫描数据
        for angle in range(0, 360, 2):  # 每2度一个点
            # 计算雷达在世界坐标系中的角度
            world_angle = robot_theta - angle
            
            # 射线追踪找到最近障碍物
            distance = self._raycast(robot_x, robot_y, world_angle)
            
            if distance > 0:
                # 添加一些噪声
                noise = np.random.normal(0, 0.02)  # 2%的噪声
                distance = max(10, distance * (1 + noise))
                
                # 质量值（距离越远质量越低）
                quality = max(5, int(100 - distance / 20))
                
                scan_data.append((angle, distance * 10, quality))  # 转换为mm
        
        return scan_data
    
    def _raycast(self, start_x, start_y, angle_deg):
        """射线追踪找到最近障碍物"""
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # 射线步进
        step = 1.0  # 1cm步进
        max_dist = 1200  # 最大12米
        
        for i in range(int(max_dist / step)):
            x = start_x + i * step * dx
            y = start_y + i * step * dy
            
            # 检查边界
            if x < 0 or x > self.arena_size or y < 0 or y > self.arena_size:
                return i * step
            
            # 检查与墙壁的碰撞
            for wall in self.walls:
                if self._point_in_wall(x, y, wall):
                    return i * step
        
        return max_dist
    
    def _point_in_wall(self, x, y, wall):
        """检查点是否在墙内（简化版）"""
        x1, y1, x2, y2 = wall
        wall_thickness = 5.0  # 墙厚5cm
        
        # 水平墙
        if abs(y1 - y2) < 1:
            return (min(x1, x2) - wall_thickness <= x <= max(x1, x2) + wall_thickness and
                    abs(y - y1) <= wall_thickness)
        # 垂直墙
        else:
            return (min(y1, y2) - wall_thickness <= y <= max(y1, y2) + wall_thickness and
                    abs(x - x1) <= wall_thickness)

# ========== 工具函数 ==========
def snap_to_grid(x_cm, y_cm, grid_res=10.0):
    """将坐标吸附到网格点（用于A*搜索）"""
    gx = round(x_cm / grid_res) * grid_res
    gy = round(y_cm / grid_res) * grid_res
    return gx, gy

def snap_to_grid_center(x_cm, y_cm, grid_size=70.0):
    """将坐标吸附到格子中心（70cm格子）"""
    # 格子中心位置：35, 105, 175, 245, 315...
    gx = round(x_cm / grid_size) * grid_size + grid_size / 2
    gy = round(y_cm / grid_size) * grid_size + grid_size / 2
    
    # 确保在边界内
    gx = max(grid_size / 2, min(gx, ARENA_SIZE_CM - grid_size / 2))
    gy = max(grid_size / 2, min(gy, ARENA_SIZE_CM - grid_size / 2))
    
    return gx, gy


# ========== 机器人控制器（真实小车）==========
@dataclass
class RobotCommand:
    cmd_type: str
    value: float

class RobotController:
    def __init__(self):
        self.client = None
        self.x, self.y = START_X_CM, START_Y_CM
        self.theta = 270.0
        self.commands_history = []
        
        # 雷达数据缓冲
        self.scan_buffer = []
        self.scan_complete = False
        self.rx_buffer = bytearray()
        self.use_notify = True
        
        # 模拟模式
        self.simulation_mode = SIMULATION_MODE
        if self.simulation_mode:
            self.simulated_lidar = SimulatedLidar(ARENA_SIZE_CM, use_random_map=True, map_seed=42)
            print("🤖 模拟模式已启用")
        
    async def _discover_services(self, address):
        """连接设备并发现所有服务和特征"""
        print(f"\n正在连接 {address} 以发现服务...")
        try:
            temp_client = BleakClient(address, timeout=15.0)
            await temp_client.connect()
            print("✓ 临时连接成功，正在读取服务...")
            
            # 获取所有服务（转换为列表）
            services = list(temp_client.services)
            print(f"\n找到 {len(services)} 个服务：")
            print("="*60)
            
            for service in services:
                print(f"\n📦 服务 UUID: {service.uuid}")
                print(f"   描述: {service.description}")
                
                # 列出所有特征
                char_count = 0
                for char in service.characteristics:
                    char_count += 1
                    props = []
                    props_lower = [p.lower() for p in char.properties]  # 转换为小写列表
                    
                    if "read" in props_lower:
                        props.append("读")
                    if "write" in props_lower or "write-without-response" in props_lower:
                        props.append("写")
                    if "notify" in props_lower:
                        props.append("通知")
                    if "indicate" in props_lower:
                        props.append("指示")
                    
                    props_str = ", ".join(props) if props else "无"
                    
                    # 高亮UART相关特征（同时支持写和通知）
                    has_write = any(p in props_lower for p in ["write", "write-without-response"])
                    has_notify = "notify" in props_lower
                    is_uart = has_write and has_notify
                    highlight = " ⭐ 可能是UART" if is_uart else ""
                    
                    print(f"   ├─ 特征 {char_count}: {char.uuid}{highlight}")
                    print(f"   │  描述: {char.description}")
                    print(f"   │  属性: {props_str}")
                    print(f"   │  原始: {char.properties}")
            
            print("="*60)
            await temp_client.disconnect()
            print("✓ 服务发现完成，已断开临时连接\n")
            
        except Exception as e:
            print(f"⚠️ 发现服务时出错: {e}")
            import traceback
            traceback.print_exc()
    
    async def connect(self):
        """连接蓝牙"""
        if self.simulation_mode:
            print("🤖 模拟模式：跳过蓝牙连接")
            print("✓ 模拟连接成功")
            return True
        
        # 先扫描附近的蓝牙设备
        print("正在扫描附近的蓝牙设备...")
        print("="*60)
        target_found = False
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            if devices:
                print(f"找到 {len(devices)} 个蓝牙设备：\n")
                for i, device in enumerate(devices, 1):
                    # 标记目标设备
                    is_target = device.address.upper() == ROBOT_ADDRESS.upper()
                    marker = " ← 目标设备" if is_target else ""
                    print(f"{i:2d}. 地址: {device.address}")
                    print(f"    名称: {device.name or '(未知)'}")
                    
                    # 安全地获取RSSI（有些设备可能没有这个属性）
                    rssi = getattr(device, 'rssi', None)
                    if rssi is not None:
                        print(f"    RSSI: {rssi} dBm{marker}")
                    else:
                        print(f"    RSSI: (不可用){marker}")
                    print()
                    
                    if is_target:
                        target_found = True
                        # 发现目标设备的服务
                        await self._discover_services(device.address)
            else:
                print("⚠️ 未发现任何蓝牙设备")
        except Exception as e:
            print(f"⚠️ 扫描设备时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*60)
        print(f"正在连接目标设备 {ROBOT_ADDRESS}...")
        
        for attempt in range(3):
            try:
                if self.client and self.client.is_connected:
                    await self.client.disconnect()
                    await asyncio.sleep(1)
                
                self.client = BleakClient(ROBOT_ADDRESS, timeout=20.0)
                await self.client.connect()
                print(f"✓ 连接成功 (尝试 {attempt+1}/3)")
                
                await asyncio.sleep(1.0)
                
                try:
                    await self.client.start_notify(UART_UUID, self._notification_handler)
                    print("✓ UART通知已启用")
                    self.use_notify = True
                except Exception as e:
                    print(f"通知失败: {e}，切换到轮询模式")
                    self.use_notify = False
                
                await asyncio.sleep(0.5)
                return True
                
            except Exception as e:
                print(f"连接失败 ({attempt+1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(3)
        
        raise Exception("无法连接蓝牙设备")
    
    async def disconnect(self):
        if self.simulation_mode:
            print("🤖 模拟模式：跳过蓝牙断开")
            return
        
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("蓝牙已断开")
    
    def _notification_handler(self, sender, data):
        """处理雷达数据通知"""
        self._process_data(data)
    
    def _process_data(self, data):
        """解析雷达数据"""
        self.rx_buffer.extend(data)
        
        while b'\n' in self.rx_buffer:
            line_end = self.rx_buffer.index(b'\n')
            line_bytes = self.rx_buffer[:line_end]
            self.rx_buffer = self.rx_buffer[line_end+1:]
            
            try:
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if line and line.startswith('A:'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        angle = float(parts[0].split(':')[1])
                        distance = float(parts[1].split(':')[1])
                        quality = int(parts[2].split(':')[1])
                        self.scan_buffer.append((angle, distance, quality))
                        
                        # 检测扫描完成
                        if len(self.scan_buffer) > 500 and angle < 20.0:
                            self.scan_complete = True
            except:
                pass
    
    async def scan_lidar(self):
        """执行雷达扫描（优化等待时间）"""
        print("  [扫描] 开始...")
        
        if self.simulation_mode:
            # 模拟模式：生成模拟雷达数据
            print("  [模拟] 生成雷达数据...")
            await asyncio.sleep(0.5 / SIMULATION_SPEED)  # 模拟扫描时间
            
            scan_data = self.simulated_lidar.generate_scan_data(self.x, self.y, self.theta)
            print(f"  [模拟] 生成 {len(scan_data)} 个数据点")
            return scan_data
        
        # 真实模式
        self.scan_buffer = []
        self.scan_complete = False
        self.rx_buffer = bytearray()
        
        await self.send_command(CMD_SCAN)
        
        # 轮询模式
        if not self.use_notify:
            read_task = asyncio.create_task(self._poll_read())
        
        # 优化的等待策略
        timeout = 15  # 减少到15秒
        start = time.time()
        last_count = 0
        no_data_time = 0
        
        while (time.time() - start) < timeout:
            await asyncio.sleep(0.3)
            
            current_count = len(self.scan_buffer)
            
            # 显示进度
            if current_count != last_count:
                print(f"    {current_count} 点...", end='\r')
                last_count = current_count
                no_data_time = 0
            else:
                no_data_time += 0.3
            
            # 智能结束条件
            if self.scan_complete:
                print(f"\n  [扫描] 完成一圈，{current_count} 个点")
                break
            
            # 如果有足够数据且超过3秒没新数据，提前结束
            if current_count > 100 and no_data_time > 3.0:
                print(f"\n  [扫描] 数据稳定，{current_count} 个点")
                break
            
            # 如果超过10秒且有数据，提前结束
            if current_count > 50 and (time.time() - start) > 10:
                print(f"\n  [扫描] 超时但有数据，{current_count} 个点")
                break
        
        if not self.use_notify:
            read_task.cancel()
        
        if len(self.scan_buffer) == 0:
            print(f"  [扫描] 警告：未收到数据！")
        
        return self.scan_buffer.copy()
    
    async def _poll_read(self):
        """轮询读取（备用）"""
        try:
            while True:
                try:
                    data = await self.client.read_gatt_char(UART_UUID)
                    if data:
                        self._process_data(data)
                except:
                    pass
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
    
    async def send_command(self, cmd, arg=0, with_response=False):
        """发送命令
        
        Args:
            cmd: 命令（可以是单字节、bytes对象等）
            arg: 参数（仅用于运动命令）
            with_response: 是否等待响应（有些设备需要）
        """
        if self.simulation_mode:
            # 模拟模式：跳过蓝牙通信
            return True
        
        if not self.client or not self.client.is_connected:
            print("    错误：蓝牙未连接")
            return False
        
        try:
            if isinstance(cmd, bytes):
                data = cmd
            else:
                data = bytes([cmd, arg])
            
            await self.client.write_gatt_char(UART_UUID, data, response=with_response)
            await asyncio.sleep(0.2)
            return True
        except Exception as e:
            print(f"    发送失败: {e}")
            return False
    
    async def test_lidar_commands(self):
        """测试不同的雷达命令格式"""
        print("\n" + "="*60)
        print("测试雷达命令")
        print("="*60)
        
        for name, cmd in CMD_SCAN_FORMATS.items():
            print(f"\n测试命令格式: {name}")
            print(f"命令内容: {cmd.hex()} (ASCII: {cmd})")
            
            # 清空缓冲区
            self.scan_buffer = []
            self.scan_complete = False
            self.rx_buffer = bytearray()
            
            # 发送命令
            await self.send_command(cmd, with_response=False)
            
            # 等待2秒看是否有数据
            print("等待雷达响应...")
            await asyncio.sleep(2.0)
            
            if len(self.scan_buffer) > 0:
                print(f"✓ 成功！收到 {len(self.scan_buffer)} 个数据点")
                print(f"  建议使用: CMD_SCAN = CMD_SCAN_FORMATS['{name}']")
                return name
            elif len(self.rx_buffer) > 0:
                print(f"⚠️ 收到原始数据但未解析: {len(self.rx_buffer)} 字节")
                print(f"  内容: {self.rx_buffer[:100]}")
            else:
                print(f"✗ 无响应")
            
            # 稍微等待
            await asyncio.sleep(0.5)
        
        print("\n" + "="*60)
        print("所有命令格式都未收到响应")
        print("可能的原因:")
        print("1. 雷达需要特殊的初始化序列")
        print("2. UART_UUID不正确")
        print("3. 雷达使用完全不同的协议")
        print("4. 需要先发送电机启动命令")
        print("="*60)
        return None
    
    async def execute_command(self, cmd):
        """执行命令"""
        self.commands_history.append(cmd)
        
        if cmd.cmd_type == 'scan':
            return await self.scan_lidar()
        
        elif cmd.cmd_type == 'forward':
            dist_cm = cmd.value
            # 限制单次前进距离，避免累积误差导致撞墙
            dist_cm = min(dist_cm, 30.0)  # 最多一次前进30cm（更安全）
            arg = int(min(255, max(1, dist_cm)))
            print(f"  [前进] {dist_cm:.0f}cm")
            await self.send_command(CMD_FORWARD, arg)
            
            # 优化等待时间：根据距离动态调整
            wait_time = dist_cm / 20.0 + 0.8  # 假设20cm/s，基础等待0.8s
            if self.simulation_mode:
                wait_time = wait_time / SIMULATION_SPEED  # 模拟模式加速
            await asyncio.sleep(wait_time)
            
            # 更新位置（连续坐标）
            new_x = self.x + dist_cm * math.cos(math.radians(self.theta))
            new_y = self.y + dist_cm * math.sin(math.radians(self.theta))
            if 0 <= new_x <= ARENA_SIZE_CM and 0 <= new_y <= ARENA_SIZE_CM:
                self.x, self.y = new_x, new_y
            
        elif cmd.cmd_type == 'turn_left':
            angle = cmd.value
            # 直接传递角度值（度），不进行数值转换
            arg = int(angle)
            print(f"  [左转] {angle:.0f}°")
            await self.send_command(CMD_TURN_LEFT, arg)
            
            # 优化转向等待时间
            wait_time = angle / 90.0 + 0.5  # 90°约需1.5秒
            if self.simulation_mode:
                wait_time = wait_time / SIMULATION_SPEED  # 模拟模式加速
            await asyncio.sleep(wait_time)
            self.theta = (self.theta + angle) % 360.0
            
        elif cmd.cmd_type == 'turn_right':
            angle = cmd.value
            # 直接传递角度值（度），不进行数值转换
            arg = int(angle)
            print(f"  [右转] {angle:.0f}°")
            await self.send_command(CMD_TURN_RIGHT, arg)
            
            # 优化转向等待时间
            wait_time = angle / 90.0 + 0.5
            if self.simulation_mode:
                wait_time = wait_time / SIMULATION_SPEED  # 模拟模式加速
            await asyncio.sleep(wait_time)
            self.theta = (self.theta - angle + 360) % 360.0
    
    def save_commands(self, filename="robot_commands.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Robot Commands\n")
            f.write(f"# Total: {len(self.commands_history)}\n\n")
            for i, cmd in enumerate(self.commands_history):
                f.write(f"{i+1:3d}. {cmd.cmd_type.upper():12s} {cmd.value:6.1f}\n")


# ========== 探索器（与仿真完全一致）==========
class Explorer:
    def __init__(self, robot, slam_map):
        self.robot = robot
        self.slam_map = slam_map
        self.visited_targets = set()
        self.planning_failures = 0  # 连续规划失败次数
        self.max_failures = 5       # 最多允许5次连续失败（更宽容）

    def find_frontiers(self):
        frontiers = []
        xmin_g, ymin_g = self.slam_map.world_to_grid(0, 0)
        xmax_g, ymax_g = self.slam_map.world_to_grid(ARENA_SIZE_CM, ARENA_SIZE_CM)
        xmin_g = max(0, xmin_g)
        xmax_g = min(self.slam_map.size-1, xmax_g)
        ymin_g = max(0, ymin_g)
        ymax_g = min(self.slam_map.size-1, ymax_g)

        for gy in range(ymin_g, ymax_g+1):
            for gx in range(xmin_g, xmax_g+1):
                if not self.slam_map.is_unknown(gx, gy):
                    continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = gx + dx, gy + dy
                    if self.slam_map.is_valid(nx, ny) and self.slam_map.is_free(nx, ny):
                        frontiers.append((gx, gy))
                        break
        return frontiers

    def cluster_frontiers(self, frontiers, min_size=3):
        if not frontiers:
            return []
        S = set(frontiers)
        used = set()
        clusters = []
        for p in frontiers:
            if p in used:
                continue
            q = [p]
            used.add(p)
            cur = []
            while q:
                x, y = q.pop(0)
                cur.append((x, y))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    nb = (x+dx, y+dy)
                    if nb in S and nb not in used:
                        used.add(nb)
                        q.append(nb)
            if len(cur) >= min_size:
                clusters.append(cur)
        return clusters

    def select_target(self, clusters):
        if not clusters:
            return None
        rgx, rgy = self.slam_map.world_to_grid(self.robot.x, self.robot.y)
        ex_g, ey_g = self.slam_map.world_to_grid(EXIT_X_CM, EXIT_Y_CM)

        dist_to_exit = math.hypot(rgx - ex_g, rgy - ey_g)
        if dist_to_exit < 20:
            print(f"  接近终点！距离={dist_to_exit*5:.0f}cm，尝试直达")
            return (ex_g, ey_g)

        best, best_cost = None, float('inf')
        for cl in clusters:
            cx = int(round(sum(p[0] for p in cl)/len(cl)))
            cy = int(round(sum(p[1] for p in cl)/len(cl)))
            penalty = 50.0 if (cx, cy) in self.visited_targets else 0.0
            d = math.hypot(cx - rgx, cy - rgy)
            goal_bias = 0.2 * math.hypot(cx - ex_g, cy - ey_g)
            cost = d / max(1.0, math.log(len(cl)+1)) + penalty + goal_bias
            if cost < best_cost:
                best_cost, best = cost, (cx, cy)
        if best:
            self.visited_targets.add(best)
        return best

    def plan_path(self, target_xy, use_relaxed_check=False):
        """A*路径规划（固定90度转向，30cm网格路径）"""
        target_x, target_y = target_xy
        
        # 使用30cm网格进行A*搜索（匹配实际移动距离）
        GRID_SIZE = 30.0  # 30cm网格，匹配实际移动距离
        
        # 起点和终点吸附到网格
        start = snap_to_grid(self.robot.x, self.robot.y, GRID_SIZE)
        goal = snap_to_grid(target_x, target_y, GRID_SIZE)
        
        if start == goal:
            return [start, goal]
        
        # A*搜索
        openq = [(0, start)]
        came = {}
        g = {start: 0}
        
        while openq:
            _, cur = heapq.heappop(openq)
            
            if cur == goal:
                # 重建路径
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            
            cx, cy = cur
            
            # 只使用4方向邻居（正交方向，适合90度转向）
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx = cx + dx * GRID_SIZE
                ny = cy + dy * GRID_SIZE
                
                # 检查边界
                if not (0 <= nx <= ARENA_SIZE_CM and 0 <= ny <= ARENA_SIZE_CM):
                    continue
                
                # 检查路径可通行性（可选择宽松检查）
                if use_relaxed_check:
                    # 宽松检查：只检查终点是否安全
                    gx, gy = self.slam_map.world_to_grid(nx, ny)
                    if self.slam_map.is_occupied_inflated(gx, gy, inflation_cm=5.0, occ_th=0.8):
                        continue
                else:
                    # 正常检查
                    if not self.slam_map.is_line_passable(cx, cy, nx, ny):
                        continue
                
                nb = (nx, ny)
                move_cost = GRID_SIZE  # 固定移动成本（70cm）
                tg = g[cur] + move_cost
                
                if nb not in g or tg < g[nb]:
                    g[nb] = tg
                    h = math.hypot(nx - goal[0], ny - goal[1])
                    heapq.heappush(openq, (tg + h, nb))
                    came[nb] = cur
        
        # 如果正常路径规划失败，尝试宽松检查
        if not use_relaxed_check:
            print("  [备用] 尝试宽松路径规划...")
            return self.plan_path(target_xy, use_relaxed_check=True)
        
        return []  # 无路径

    def path_to_commands(self, path_points):
        """将路径转换为命令（固定90度转向）"""
        if len(path_points) < 2:
            return []
        
        cmds = []
        cur_theta = self.robot.theta
        cur_x, cur_y = self.robot.x, self.robot.y
        
        for i in range(1, len(path_points)):
            target_x, target_y = path_points[i]
            
            # 计算目标方向
            dx = target_x - cur_x
            dy = target_y - cur_y
            dist = math.hypot(dx, dy)
            
            if dist < 1.0:  # 太近，跳过
                continue
            
            target_theta = math.degrees(math.atan2(dy, dx))
            
            # 固定90度转向：将目标角度四舍五入到最近的90度
            target_theta_90 = round(target_theta / 90.0) * 90.0
            if target_theta_90 == 360.0:
                target_theta_90 = 0.0
            
            # 计算转向角度（只能是0, 90, 180, 270度）
            dth = (target_theta_90 - cur_theta + 360.0) % 360.0
            if dth > 180.0:
                dth -= 360.0
            
            # 执行90度转向
            if abs(dth) > 5.0:  # 大于5度才转
                if dth > 0:
                    # 左转90度
                    cmds.append(RobotCommand('turn_left', 90.0))
                else:
                    # 右转90度
                    cmds.append(RobotCommand('turn_right', 90.0))
                cur_theta = target_theta_90
            
            # 前进（格子中心路径，每段30cm更安全）
            MAX_SEGMENT = 30.0  # 更安全的移动距离
            remaining = dist
            
            while remaining > 0:
                segment = min(remaining, MAX_SEGMENT)
                cmds.append(RobotCommand('forward', segment))
                remaining -= segment
                
                # 更新虚拟位置
                cur_x += segment * math.cos(math.radians(cur_theta))
                cur_y += segment * math.sin(math.radians(cur_theta))
                
                # 长距离中途扫描
                if remaining > MAX_SEGMENT:
                    cmds.append(RobotCommand('scan', 0.0))
        
        return cmds

    def check_exit_reached(self):
        dist = math.hypot(self.robot.x - EXIT_X_CM, self.robot.y - EXIT_Y_CM)
        return dist <= 20.0
    
    def should_return_to_start(self):
        """判断是否需要返回起点"""
        return self.planning_failures >= self.max_failures
    
    def plan_path_to_start(self):
        """规划回到起点的路径"""
        print(f"\n⚠️ 连续{self.planning_failures}次规划失败，返回起点重新开始...")
        target_xy = (START_X_CM, START_Y_CM)
        path = self.plan_path(target_xy)
        return path
    
    def reset_exploration(self):
        """重置探索状态（返回起点后）"""
        print("🔄 已返回起点，重置探索状态")
        self.visited_targets.clear()  # 清除已访问记录
        self.planning_failures = 0    # 重置失败计数
        # 注意：不清空地图，保留已建立的地图信息
    
    def is_near_start(self, threshold_cm=30.0):
        """检查是否接近起点"""
        dist = math.hypot(self.robot.x - START_X_CM, self.robot.y - START_Y_CM)
        return dist <= threshold_cm


# ========== 主控制（逐步探索模式）==========
async def main():
    print("="*60)
    if SIMULATION_MODE:
        print("🤖 模拟模式SLAM迷宫探索（逐步探索）")
    else:
        print("真实小车SLAM迷宫探索（逐步探索）")
    print("="*60)
    
    robot = RobotController()
    slam_map = SlamMap()
    explorer = Explorer(robot, slam_map)
    
    try:
        await robot.connect()
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.ion()
        
        max_steps = 100  # 增加步数，因为现在是逐步探索
        
        for step in range(max_steps):
            print(f"\n{'='*40}")
            print(f"步骤 {step+1}")
            print(f"位置: ({robot.x:.0f}, {robot.y:.0f}) 角度: {robot.theta:.0f}°")
            
            # 1. 扫描当前环境
            print("📡 扫描环境...")
            scan_data = await robot.execute_command(RobotCommand('scan', 0.0))
            if scan_data:
                slam_map.update_scan(robot.x, robot.y, robot.theta, scan_data, debug=True)
                print(f"✓ 更新地图，收到 {len(scan_data)} 个数据点")
            
            # 2. 检查是否到达终点
            if explorer.check_exit_reached():
                draw_map(ax1, ax2, robot, slam_map)
                plt.pause(0.1)
                print("\n🎉 *** 到达终点！***")
                break
            
            # 3. 寻找前沿点
            print("🔍 寻找前沿点...")
            fronts = explorer.find_frontiers()
            clusters = explorer.cluster_frontiers(fronts)
            print(f"前沿={len(fronts)}, 簇={len(clusters)}")
            
            if not clusters:
                draw_map(ax1, ax2, robot, slam_map)
                plt.pause(0.1)
                print("🏁 探索完成，无更多前沿点")
                break
            
            # 4. 选择下一个探索目标
            target_g = explorer.select_target(clusters)
            if not target_g:
                print("⚠️ 无法选择目标")
                explorer.planning_failures += 1
                continue
            
            target_world = slam_map.grid_to_world(*target_g)
            print(f"🎯 目标: ({target_world[0]:.0f}, {target_world[1]:.0f})")
            
            # 5. 规划到目标的路径
            print("🗺️ 规划路径...")
            path = explorer.plan_path(target_world)
            if not path or len(path) < 2:
                print("⚠️ 路径规划失败")
                explorer.planning_failures += 1
                
                # 检查是否需要返回起点
                if explorer.should_return_to_start():
                    print("\n🔄 连续规划失败，尝试返回起点...")
                    return_path = explorer.plan_path_to_start()
                    
                    if return_path and len(return_path) >= 2:
                        print("执行返回起点...")
                        return_cmds = explorer.path_to_commands(return_path)
                        for cmd in return_cmds:
                            await robot.execute_command(cmd)
                            await asyncio.sleep(0.2)
                        
                        if explorer.is_near_start(threshold_cm=40.0):
                            print("✓ 成功返回起点")
                            explorer.reset_exploration()
                        else:
                            print("⚠️ 返回起点失败，继续尝试")
                            explorer.reset_exploration()
                    else:
                        print("❌ 无法返回起点，终止探索")
                        break
                continue
            
            # 6. 重置失败计数
            explorer.planning_failures = 0
            print(f"✓ 路径规划成功，长度: {len(path)} 个点")
            
            # 7. 绘制地图和路径
            draw_map(ax1, ax2, robot, slam_map, path)
            plt.pause(0.1)
            
            # 8. 执行单步移动（关键改进：只执行一步）
            print("🚶 执行单步移动...")
            if len(path) >= 2:
                # 只移动到路径的下一个点
                next_point = path[1]  # 跳过当前位置
                current_point = (robot.x, robot.y)
                
                # 计算移动方向和距离
                dx = next_point[0] - current_point[0]
                dy = next_point[1] - current_point[1]
                distance = math.hypot(dx, dy)
                target_angle = math.degrees(math.atan2(dy, dx))
                
                # 计算需要转向的角度
                angle_diff = (target_angle - robot.theta + 360) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                
                # 执行转向（90度固定）
                if abs(angle_diff) > 5:  # 大于5度才转向
                    if angle_diff > 0:
                        print(f"  [左转] 90°")
                        await robot.execute_command(RobotCommand('turn_left', 90.0))
                    else:
                        print(f"  [右转] 90°")
                        await robot.execute_command(RobotCommand('turn_right', 90.0))
                
                # 执行前进（格子中心路径，单步30cm更安全）
                step_distance = min(distance, 30.0)  # 单步最多30cm（更安全）
                if step_distance > 5.0:  # 大于5cm才前进（降低阈值）
                    print(f"  [前进] {step_distance:.0f}cm")
                    await robot.execute_command(RobotCommand('forward', step_distance))
                else:
                    print("  [跳过] 距离太近")
            else:
                print("⚠️ 路径太短，跳过")
            
            # 9. 短暂等待
            await asyncio.sleep(0.3)
        
        # 保存命令
        robot.save_commands()
        print(f"\n📊 总命令: {len(robot.commands_history)}")
        
        plt.ioff()
        plt.show()
        
    finally:
        await robot.disconnect()

def draw_map(ax1, ax2, robot, slam_map, path=None):
    """绘制地图（基于雷达扫描的连续地图）"""
    # SLAM概率地图视图
    ax1.clear()
    ax1.set_title("SLAM建图（雷达扫描）", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, ARENA_SIZE_CM)
    ax1.set_ylim(0, ARENA_SIZE_CM)
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.grid(True, alpha=0.3)
    
    # 显示概率地图
    prob = 1.0 / (1.0 + np.exp(-slam_map.log_odds))
    extent = [-slam_map.origin[0]*slam_map.resolution,
             (slam_map.size-slam_map.origin[0])*slam_map.resolution,
             -slam_map.origin[1]*slam_map.resolution,
             (slam_map.size-slam_map.origin[1])*slam_map.resolution]
    ax1.imshow(prob, cmap='gray_r', extent=extent, origin='lower', 
               vmin=0, vmax=1, alpha=0.8)
    
    # 起点和终点
    ax1.add_patch(Circle((START_X_CM, START_Y_CM), 8, 
                         color='green', alpha=0.6, label='起点'))
    ax1.add_patch(Circle((EXIT_X_CM, EXIT_Y_CM), 8, 
                         color='red', alpha=0.6, label='终点'))
    
    # 机器人（圆形+朝向箭头）
    ax1.add_patch(Circle((robot.x, robot.y), ROBOT_RADIUS_CM, 
                         color='blue', alpha=0.5, linewidth=2, 
                         edgecolor='darkblue', label='小车'))
    arrow_len = 20
    ax1.arrow(robot.x, robot.y,
             arrow_len * math.cos(math.radians(robot.theta)),
             arrow_len * math.sin(math.radians(robot.theta)),
             head_width=8, head_length=8, fc='darkblue', ec='darkblue')
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # 路径规划视图
    ax2.clear()
    ax2.set_title("路径规划（连续空间）", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, ARENA_SIZE_CM)
    ax2.set_ylim(0, ARENA_SIZE_CM)
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.grid(True, alpha=0.3)
    
    # 显示自由/障碍区域（简化视图）
    prob_map = 1.0 / (1.0 + np.exp(-slam_map.log_odds))
    ax2.imshow(prob_map, cmap='RdYlGn_r', extent=extent, origin='lower',
               vmin=0, vmax=1, alpha=0.6)
    
    # 规划路径
    if path and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax2.plot(path_x, path_y, 'c-', linewidth=3, alpha=0.7, label='规划路径')
        ax2.plot(path_x, path_y, 'co', markersize=5)
    
    # 机器人当前位置
    ax2.add_patch(Circle((robot.x, robot.y), ROBOT_RADIUS_CM, 
                         color='blue', alpha=0.6, linewidth=2, 
                         edgecolor='darkblue'))
    ax2.arrow(robot.x, robot.y,
             20 * math.cos(math.radians(robot.theta)),
             20 * math.sin(math.radians(robot.theta)),
             head_width=8, head_length=8, fc='darkblue', ec='darkblue')
    
    # 起点终点
    ax2.plot(START_X_CM, START_Y_CM, 'go', markersize=10, label='起点')
    ax2.plot(EXIT_X_CM, EXIT_Y_CM, 'r*', markersize=15, label='终点')
    
    ax2.legend(loc='upper right', fontsize=9)

async def test_mode():
    """测试模式：测试雷达命令"""
    print("="*60)
    print("雷达命令测试模式")
    print("="*60)
    
    robot = RobotController()
    
    try:
        await robot.connect()
        
        # 测试所有命令格式
        result = await robot.test_lidar_commands()
        
        if result:
            print(f"\n✓ 找到有效的命令格式: {result}")
            print(f"\n请在代码中修改:")
            print(f"  CMD_SCAN = CMD_SCAN_FORMATS['{result}']")
        else:
            print("\n⚠️ 未找到有效的命令格式")
            print("\n建议:")
            print("1. 检查雷达型号和数据手册")
            print("2. 尝试使用串口调试工具测试雷达")
            print("3. 确认UART_UUID是否正确（当前: {})".format(UART_UUID))
            print("4. 检查雷达是否需要供电或初始化")
        
    finally:
        await robot.disconnect()

async def simulation_test():
    """模拟模式测试"""
    print("="*60)
    print("模拟模式测试")
    print("="*60)
    
    robot = RobotController()
    slam_map = SlamMap()
    
    try:
        await robot.connect()
        
        # 测试扫描
        print("\n测试雷达扫描...")
        scan_data = await robot.scan_lidar()
        print(f"收到 {len(scan_data)} 个数据点")
        
        # 测试建图
        print("\n测试SLAM建图...")
        slam_map.update_scan(robot.x, robot.y, robot.theta, scan_data)
        print("建图完成")
        
        # 测试运动
        print("\n测试运动命令...")
        await robot.execute_command(RobotCommand('forward', 20.0))
        await robot.execute_command(RobotCommand('turn_left', 90.0))
        await robot.execute_command(RobotCommand('forward', 15.0))
        
        print(f"\n最终位置: ({robot.x:.1f}, {robot.y:.1f}), 角度: {robot.theta:.1f}°")
        print("✓ 模拟模式测试完成")
        
    finally:
        await robot.disconnect()

async def step_by_step_test():
    """逐步探索测试"""
    print("="*60)
    print("🤖 逐步探索测试")
    print("="*60)
    
    robot = RobotController()
    slam_map = SlamMap()
    explorer = Explorer(robot, slam_map)
    
    try:
        await robot.connect()
        
        # 测试几步探索
        for step in range(5):
            print(f"\n--- 步骤 {step+1} ---")
            print(f"位置: ({robot.x:.0f}, {robot.y:.0f}) 角度: {robot.theta:.0f}°")
            
            # 扫描
            scan_data = await robot.scan_lidar()
            if scan_data:
                slam_map.update_scan(robot.x, robot.y, robot.theta, scan_data, debug=True)
                print(f"✓ 更新地图，收到 {len(scan_data)} 个数据点")
            
            # 寻找前沿
            fronts = explorer.find_frontiers()
            clusters = explorer.cluster_frontiers(fronts)
            print(f"前沿={len(fronts)}, 簇={len(clusters)}")
            
            if clusters:
                target_g = explorer.select_target(clusters)
                if target_g:
                    target_world = slam_map.grid_to_world(*target_g)
                    print(f"目标: ({target_world[0]:.0f}, {target_world[1]:.0f})")
                    
                    # 规划路径
                    path = explorer.plan_path(target_world)
                    if path and len(path) >= 2:
                        print(f"路径长度: {len(path)} 个点")
                        
                        # 执行单步移动
                        next_point = path[1]
                        current_point = (robot.x, robot.y)
                        dx = next_point[0] - current_point[0]
                        dy = next_point[1] - current_point[1]
                        distance = math.hypot(dx, dy)
                        target_angle = math.degrees(math.atan2(dy, dx))
                        
                        angle_diff = (target_angle - robot.theta + 360) % 360
                        if angle_diff > 180:
                            angle_diff -= 360
                        
                        if abs(angle_diff) > 5:
                            if angle_diff > 0:
                                print(f"  [左转] 90°")
                                await robot.execute_command(RobotCommand('turn_left', 90.0))
                            else:
                                print(f"  [右转] 90°")
                                await robot.execute_command(RobotCommand('turn_right', 90.0))
                        
                        step_distance = min(distance, 30.0)  # 格子中心路径，30cm更安全
                        if step_distance > 5.0:
                            print(f"  [前进] {step_distance:.0f}cm")
                            await robot.execute_command(RobotCommand('forward', step_distance))
                        else:
                            print("  [跳过] 距离太近")
                    else:
                        print("⚠️ 路径规划失败")
                else:
                    print("⚠️ 无法选择目标")
            else:
                print("⚠️ 无前沿点")
            
            await asyncio.sleep(0.5)
        
        print(f"\n最终位置: ({robot.x:.1f}, {robot.y:.1f}), 角度: {robot.theta:.1f}°")
        print("✓ 逐步探索测试完成")
        
    finally:
        await robot.disconnect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # 测试模式：测试雷达命令
            print("启动测试模式...")
            asyncio.run(test_mode())
            sys.exit(0)
        elif sys.argv[1] == 'sim':
            # 模拟模式测试
            print("启动模拟模式测试...")
            asyncio.run(simulation_test())
            sys.exit(0)
        elif sys.argv[1] == 'step':
            # 逐步探索测试
            print("启动逐步探索测试...")
            asyncio.run(step_by_step_test())
            sys.exit(0)
        elif sys.argv[1] == 'help':
            print("""
使用方法:
  python slam_real.py              # 运行完整SLAM探索
  python slam_real.py test         # 测试雷达命令格式
  python slam_real.py sim          # 模拟模式测试
  python slam_real.py step         # 逐步探索测试
  python slam_real.py help         # 显示帮助
  
模式说明:
  - 正常模式: 连接真实小车进行SLAM探索
  - 测试模式: 自动尝试多种雷达命令格式
  - 模拟模式: 使用模拟雷达数据进行测试
  - 逐步探索: 测试走一步探一步的探索逻辑
            """)
            sys.exit(0)
    
    # 正常模式：完整探索
    if SIMULATION_MODE:
        print("""
🤖 模拟模式SLAM迷宫探索系统

使用方法:
  python slam_real.py              # 运行模拟探索
  python slam_real.py sim          # 模拟模式测试
  python slam_real.py help         # 显示帮助
        """)
    else:
        print("""
真实小车SLAM迷宫探索系统

使用方法:
  python slam_real.py              # 运行完整探索
  python slam_real.py test         # 测试雷达命令
  python slam_real.py help         # 显示帮助
        """)
    
    asyncio.run(main())



