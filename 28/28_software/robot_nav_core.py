"""
================================================================================
智能机器人导航系统 - 核心模块
作者: [你的名字]
版本: 2.0
日期: 2025
================================================================================

功能概述:
1. SLAM实时建图与定位
2. 多场景地图支持（未知/蛇形/随机）
3. 路径规划（A*算法）
4. 蓝牙串口通信
5. 可视化监控

技术栈:
- Python 3.8+
- NumPy (数值计算)
- Matplotlib (可视化)
- PySerial (串口通信)
"""

from __future__ import annotations
import math
import time
import random
import logging
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Iterable
from pathlib import Path
import numpy as np

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("RobotNav")

# ============================================================================
# 类型定义
# ============================================================================
GridCoord = Tuple[int, int]  # 网格坐标 (row, col)
WorldPos = Tuple[float, float]  # 世界坐标 (x, y) 米
Obstacle = Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

# ============================================================================
# 配置参数类
# ============================================================================

@dataclass
class EnvironmentSettings:
    """环境地图配置"""
    wall_thickness: float = 0.005      # 墙体厚度(米)
    border_thickness: float = 0.01     # 边界厚度(米)

@dataclass
class SnakeMazeSettings:
    """蛇形迷宫配置"""
    arena_size: float = 1.80           # 场地尺寸(米)
    cell_size: float = 0.45            # 格子大小(米)
    wall_count: int = 4                # 横墙数量
    gap_size: int = 1                  # 缺口格数

@dataclass
class RandomMazeSettings:
    """随机迷宫配置"""
    arena_size: float = 1.80
    cell_size: float = 0.45
    wall_count: int = 12               # 墙段数量
    random_seed: Optional[int] = None
    segment_min_cells: int = 1         # 墙段最小长度
    segment_max_cells: int = 2         # 墙段最大长度
    horizontal_bias: float = 0.5       # 水平墙概率

@dataclass
class ExplorationMazeSettings:
    """探索地图配置"""
    grid_dimension: int = 4            # 网格维度 4x4
    cell_size: float = 0.70            # 格子大小70cm
    arena_size: float = 2.80           # 总尺寸2.8米

@dataclass
class PathPlanSettings:
    """路径规划配置"""
    sampling_step: float = 0.03        # 采样步长(米)
    resampling_step: float = 0.05      # 重采样间距
    equality_threshold: float = 1e-6   # 相等判定阈值
    segment_threshold: float = 1e-9    # 线段判定阈值

@dataclass
class LaserSettings:
    """激光雷达配置"""
    beam_count: int = 360              # 光束数量
    max_range: float = 3.0             # 最大量程(米)
    raycast_epsilon: float = 1e-6      # 光线投射精度

@dataclass
class GridMapSettings:
    """栅格地图配置"""
    xy_resolution: float = 0.03        # XY分辨率(米)
    log_odds_free: float = -0.4        # 空闲对数几率
    log_odds_occupied: float = +0.85   # 占据对数几率
    log_odds_min: float = -4.0         # 对数几率下限
    log_odds_max: float = +4.0         # 对数几率上限
    hit_margin: float = 1e-3           # 命中容差
    prob_free_threshold: float = 0.35  # 空闲概率阈值
    prob_occ_threshold: float = 0.65   # 占据概率阈值
    gray_free: float = 0.9             # 空闲灰度值
    gray_occupied: float = 0.0         # 占据灰度值
    gray_unknown: float = 1.0          # 未知灰度值

@dataclass
class VehicleSettings:
    """机器人车体配置"""
    body_radius: float = 0.15          # 车体半径(米) - 用于A*膨胀
    rotation_angle: float = math.radians(18)  # 旋转角度
    max_velocity: float = 0.35         # 最大线速度(m/s)
    control_timestep: float = 0.1      # 控制时间步长(秒)
    timestep_guard: float = 1e-3       # 时间步保护值
    manual_fwd_cone: float = math.pi / 6  # 手动前进锥角
    manual_step_dist: float = 0.10     # 手动步进距离

@dataclass
class TrajectorySettings:
    """轨迹控制配置"""
    lookahead_distance: float = 0.30   # 前瞻距离
    angular_gain: float = 1.8          # 角度增益
    avoidance_gain: float = 1.2        # 避障增益
    avoidance_radius: float = 0.35     # 避障半径
    max_velocity: float = 0.30         # 最大速度
    repulsion_gain: float = 1.0        # 斥力增益
    manual_lookahead: float = 0.25     # 手动前瞻距离

@dataclass
class LocalizationSettings:
    """定位融合配置"""
    enabled: bool = True               # 是否启用融合
    alpha: float = 0.1                 # 融合权重
    max_translation: float = 0.20      # 最大平移容差(米)
    max_rotation_deg: float = 20.0     # 最大旋转容差(度)
    min_point_count: int = 50          # 最小点数阈值
    max_rmse: float = 0.05             # 最大RMSE(米)
    snap_translation: float = 0.02     # 平移捕捉阈值
    snap_rotation_deg: float = 2.0     # 旋转捕捉阈值

@dataclass
class DisplaySettings:
    """可视化配置"""
    figure_size: Tuple[float, float] = (14, 10)  # 图形尺寸(英寸)
    robot_arrow_length: float = 0.10   # 机器人箭头长度
    robot_arrow_head: Tuple[float, float] = (0.03, 0.03)  # 箭头头部尺寸
    ogm_arrow_length: float = 0.10     # OGM箭头长度
    ogm_arrow_head: Tuple[float, float] = (0.03, 0.03)
    lidar_alpha: float = 0.2           # 激光束透明度
    lidar_linewidth: float = 0.5       # 激光束线宽
    thumbnail_size: Tuple[float, float] = (3, 3)  # 缩略图尺寸
    pause_duration: float = 0.01       # 刷新暂停时长

@dataclass
class DataLogSettings:
    """数据记录配置"""
    log_level: int = logging.INFO
    log_format: str = "[%(levelname)s] %(message)s"
    pose_csv_file: str = "pose_log.csv"
    lidar_csv_file: str = "lidar_log.csv"

@dataclass
class ApplicationSettings:
    """应用配置"""
    goal_tolerance: float = 0.10       # 到达目标容差(米)
    control_mode: str = "GOALSEEKING"  # GOALSEEKING | MANUAL
    map_type: str = "UNKNOWN"          # UNKNOWN | UNKNOWNSIM | RANDOM | SNAKE
    entrance_coord: GridCoord = (1, 1) # 入口坐标(默认)
    snake_goal_coord: Optional[GridCoord] = None
    random_goal_coord: GridCoord = (3, 3)
    exploration_goal_coord: GridCoord = (4, 4)

@dataclass
class MasterConfiguration:
    """主配置容器"""
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    snake_maze: SnakeMazeSettings = field(default_factory=SnakeMazeSettings)
    random_maze: RandomMazeSettings = field(default_factory=RandomMazeSettings)
    exploration_maze: ExplorationMazeSettings = field(default_factory=ExplorationMazeSettings)
    path_planning: PathPlanSettings = field(default_factory=PathPlanSettings)
    laser: LaserSettings = field(default_factory=LaserSettings)
    grid_map: GridMapSettings = field(default_factory=GridMapSettings)
    vehicle: VehicleSettings = field(default_factory=VehicleSettings)
    robot: VehicleSettings = field(default_factory=VehicleSettings)  # 添加robot别名
    trajectory: TrajectorySettings = field(default_factory=TrajectorySettings)
    localization: LocalizationSettings = field(default_factory=LocalizationSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    data_log: DataLogSettings = field(default_factory=DataLogSettings)
    application: ApplicationSettings = field(default_factory=ApplicationSettings)

# 全局默认配置
DEFAULT_CONFIG = MasterConfiguration()

# ============================================================================
# 基础数据结构
# ============================================================================

@dataclass
class Pose2D:
    """二维位姿"""
    x: float
    y: float
    theta: float  # 航向角(弧度)

@dataclass
class TargetPose:
    """目标位姿"""
    x: float
    y: float
    theta: float

@dataclass
class MovementCommand:
    """网格化移动指令"""
    rotation_degrees: float  # 旋转角度(度)
    forward_distance: float  # 直行距离(米)
    target_coord: GridCoord  # 目标格子
    execution_phase: str = "rotate"  # 执行阶段: 'rotate' 或 'forward'

@dataclass
class LaserScanData:
    """激光扫描数据"""
    angles: List[float]      # 角度列表(弧度)
    ranges: List[float]      # 距离列表(米)

@dataclass
class NavigationGoal:
    """导航目标"""
    grid_coord: GridCoord

# ============================================================================
# 工具函数
# ============================================================================

def sample_rectangle_boundary(x_min: float, x_max: float, 
                              y_min: float, y_max: float, 
                              step: float) -> Iterable[Tuple[float, float]]:
    """采样矩形边界点"""
    x = x_min
    while x <= x_max:
        yield (x, y_min)
        yield (x, y_max)
        x += step
    y = y_min
    while y <= y_max:
        yield (x_min, y)
        yield (x_max, y)
        y += step

def convert_obstacles_to_points(obstacles: List[Obstacle], 
                                step: float) -> Tuple[List[float], List[float]]:
    """将障碍物转换为点云"""
    x_points: List[float] = []
    y_points: List[float] = []
    for (x_min, x_max, y_min, y_max) in obstacles:
        for x, y in sample_rectangle_boundary(x_min, x_max, y_min, y_max, step):
            x_points.append(x)
            y_points.append(y)
    return x_points, y_points

def resample_trajectory(path_points: List[Tuple[float, float]], 
                       interval: float, 
                       equal_eps: float, 
                       segment_eps: float) -> List[Tuple[float, float]]:
    """重采样路径点"""
    if not path_points:
        return []
    if len(path_points) == 1:
        return path_points[:]
    
    # 去除重复点
    cleaned = [path_points[0]]
    for pt in path_points[1:]:
        if abs(pt[0] - cleaned[-1][0]) > equal_eps or abs(pt[1] - cleaned[-1][1]) > equal_eps:
            cleaned.append(pt)
    
    if len(cleaned) < 2:
        return cleaned
    
    # 计算累计弧长
    arc_lengths = [0.0]
    for i in range(1, len(cleaned)):
        dx = cleaned[i][0] - cleaned[i-1][0]
        dy = cleaned[i][1] - cleaned[i-1][1]
        arc_lengths.append(arc_lengths[-1] + math.hypot(dx, dy))
    
    total_length = arc_lengths[-1]
    if total_length < segment_eps:
        return cleaned
    
    # 均匀重采样
    resampled: List[Tuple[float, float]] = []
    s = 0.0
    segment_idx = 0
    
    while s <= total_length and segment_idx < len(cleaned) - 1:
        # 找到包含s的线段
        while arc_lengths[segment_idx + 1] < s and segment_idx + 1 < len(arc_lengths) - 1:
            segment_idx += 1
        
        seg_length = arc_lengths[segment_idx + 1] - arc_lengths[segment_idx]
        if seg_length <= segment_eps:
            resampled.append(cleaned[segment_idx])
            s += interval
            continue
        
        # 线性插值
        t = (s - arc_lengths[segment_idx]) / seg_length
        x = cleaned[segment_idx][0] * (1 - t) + cleaned[segment_idx + 1][0] * t
        y = cleaned[segment_idx][1] * (1 - t) + cleaned[segment_idx + 1][1] * t
        resampled.append((x, y))
        s += interval
    
    resampled.append(cleaned[-1])
    return resampled

def grid_to_world_position(grid_coord: GridCoord, cell_size: float) -> Tuple[float, float]:
    """
    网格坐标转世界坐标（中心点）
    注意: 网格坐标假定从(1,1)开始，对应物理坐标(0.5*cell_size, 0.5*cell_size)
    """
    col, row = grid_coord
    x = (col - 0.5) * cell_size
    y = (row - 0.5) * cell_size
    return (x, y)

# ============================================================================
# PythonRobotics 导入（可选）
# ============================================================================
PYTHONROBOTICS_PATH = os.environ.get("PYTHONROBOTICS_PATH", None)
if PYTHONROBOTICS_PATH and PYTHONROBOTICS_PATH not in sys.path:
    sys.path.insert(0, PYTHONROBOTICS_PATH)

HAS_ASTAR = False
HAS_ICP = False

# 初始化为None，避免导入错误
external_icp_matching = None

try:
    from PathPlanning.AStar.a_star import AStarPlanner  # type: ignore
    HAS_ASTAR = True
except Exception as e:
    logger.debug(f"A*导入失败: {e}")

try:
    from SLAM.ICPMatching.icp_matching import icp_matching as external_icp_matching  # type: ignore
    HAS_ICP = True
except Exception as e:
    logger.debug(f"ICP导入失败: {e}")
    external_icp_matching = None  # 导入失败时设为None

if __name__ == "__main__":
    print("机器人导航核心模块已加载")
    print(f"A*算法可用: {HAS_ASTAR}")
    print(f"ICP算法可用: {HAS_ICP}")

