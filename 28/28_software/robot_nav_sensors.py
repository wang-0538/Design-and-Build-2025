"""
================================================================================
智能机器人导航系统 - 传感器和SLAM模块
================================================================================
包含:
1. 激光雷达模拟器
2. 占据栅格地图(OGM)
3. ICP扫描匹配
4. 位姿融合
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from robot_nav_core import (
    Pose2D, LaserScanData, LaserSettings, GridMapSettings,
    LocalizationSettings, HAS_ICP, external_icp_matching, logger
)
from robot_nav_maps import BaseEnvironmentMap

# ============================================================================
# 激光雷达模拟器
# ============================================================================

class LaserRangeFinder:
    """激光雷达模拟器"""
    
    def __init__(self, config: LaserSettings):
        self.config = config
        self._beam_angles = np.linspace(0, 2 * np.pi, config.beam_count, endpoint=False)
        self._cos_table = np.cos(self._beam_angles)
        self._sin_table = np.sin(self._beam_angles)
    
    def _compute_ray_box_intersection(self, origin_x: float, origin_y: float,
                                      dir_x: float, dir_y: float,
                                      box_x_min: float, box_x_max: float,
                                      box_y_min: float, box_y_max: float) -> List[float]:
        """计算光线与矩形的交点距离"""
        intersections: List[float] = []
        eps = self.config.raycast_epsilon
        
        # X方向边界
        if abs(dir_x) > eps:
            t1 = (box_x_min - origin_x) / dir_x
            if t1 > 0:
                y_at_t1 = origin_y + t1 * dir_y
                if box_y_min <= y_at_t1 <= box_y_max:
                    intersections.append(t1)
            
            t2 = (box_x_max - origin_x) / dir_x
            if t2 > 0:
                y_at_t2 = origin_y + t2 * dir_y
                if box_y_min <= y_at_t2 <= box_y_max:
                    intersections.append(t2)
        
        # Y方向边界
        if abs(dir_y) > eps:
            t3 = (box_y_min - origin_y) / dir_y
            if t3 > 0:
                x_at_t3 = origin_x + t3 * dir_x
                if box_x_min <= x_at_t3 <= box_x_max:
                    intersections.append(t3)
            
            t4 = (box_y_max - origin_y) / dir_y
            if t4 > 0:
                x_at_t4 = origin_x + t4 * dir_x
                if box_x_min <= x_at_t4 <= box_x_max:
                    intersections.append(t4)
        
        return intersections
    
    def perform_scan(self, robot_pose: Pose2D, environment: BaseEnvironmentMap) -> LaserScanData:
        """执行激光扫描"""
        cos_heading = math.cos(robot_pose.theta)
        sin_heading = math.sin(robot_pose.theta)
        
        range_measurements: List[float] = []
        
        for beam_idx in range(len(self._beam_angles)):
            # 世界坐标系下的光线方向
            dir_x = cos_heading * self._cos_table[beam_idx] - sin_heading * self._sin_table[beam_idx]
            dir_y = sin_heading * self._cos_table[beam_idx] + cos_heading * self._sin_table[beam_idx]
            
            min_distance = self.config.max_range
            
            # 与所有障碍物求交
            for obstacle in environment.obstacle_boxes:
                intersections = self._compute_ray_box_intersection(
                    robot_pose.x, robot_pose.y, dir_x, dir_y, *obstacle
                )
                if intersections:
                    nearest = min(intersections)
                    if nearest > 0:
                        min_distance = min(min_distance, nearest)
            
            # 限制在最大量程内
            measured_range = min_distance if min_distance < self.config.max_range else self.config.max_range
            range_measurements.append(measured_range)
        
        return LaserScanData(angles=list(self._beam_angles), ranges=range_measurements)

# ============================================================================
# 占据栅格地图
# ============================================================================

class SpatialOccupancyMap:
    """空间占据栅格地图 (基于对数几率更新)"""
    
    def __init__(self, config: GridMapSettings, 
                 x_min: float, y_min: float, x_max: float, y_max: float):
        self.config = config
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.resolution = config.xy_resolution
        
        # 计算栅格尺寸
        grid_width = int((x_max - x_min) / self.resolution)
        grid_height = int((y_max - y_min) / self.resolution)
        
        # 对数几率网格 (log-odds)
        self.log_odds_grid = np.zeros((grid_height, grid_width), dtype=float)
    
    def world_to_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        grid_x = int((x - self.x_min) / self.resolution)
        grid_y = int((y - self.y_min) / self.resolution)
        return grid_x, grid_y
    
    def update_with_scan(self, robot_pose: Pose2D, scan_data: LaserScanData) -> None:
        """使用激光扫描更新地图"""
        grid_height, grid_width = self.log_odds_grid.shape
        max_valid_range = max(scan_data.ranges) if scan_data.ranges else 0.0
        margin = self.config.hit_margin
        
        for relative_angle, measured_distance in zip(scan_data.angles, scan_data.ranges):
            # 限制在有效范围内
            clipped_range = min(measured_distance, max_valid_range)
            
            # 计算激光束终点
            global_angle = robot_pose.theta + relative_angle
            end_x = robot_pose.x + clipped_range * math.cos(global_angle)
            end_y = robot_pose.y + clipped_range * math.sin(global_angle)
            
            origin_gx, origin_gy = self.world_to_grid_coords(robot_pose.x, robot_pose.y)
            target_gx, target_gy = self.world_to_grid_coords(end_x, end_y)
            
            if not (0 <= origin_gx < grid_width and 0 <= origin_gy < grid_height):
                continue
            
            # Bresenham光线追踪
            ray_cells = self._bresenham_line_trace(origin_gx, origin_gy, target_gx, target_gy)
            
            if not ray_cells:
                continue
            
            # 判断是否命中障碍物
            hit_obstacle = (measured_distance < max_valid_range - margin)
            
            # 更新沿途格子为自由
            cells_to_update = ray_cells[:-1] if hit_obstacle else ray_cells
            for cell_x, cell_y in cells_to_update:
                if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                    self.log_odds_grid[cell_y, cell_x] += self.config.log_odds_free
                    self.log_odds_grid[cell_y, cell_x] = np.clip(
                        self.log_odds_grid[cell_y, cell_x],
                        self.config.log_odds_min,
                        self.config.log_odds_max
                    )
            
            # 更新终点为占据
            if hit_obstacle:
                end_cell_x, end_cell_y = ray_cells[-1]
                if 0 <= end_cell_x < grid_width and 0 <= end_cell_y < grid_height:
                    self.log_odds_grid[end_cell_y, end_cell_x] += self.config.log_odds_occupied
                    self.log_odds_grid[end_cell_y, end_cell_x] = np.clip(
                        self.log_odds_grid[end_cell_y, end_cell_x],
                        self.config.log_odds_min,
                        self.config.log_odds_max
                    )
    
    def _bresenham_line_trace(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法"""
        cells = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            if error2 >= dy:
                error += dy
                x += sx
            if error2 <= dx:
                error += dx
                y += sy
        
        return cells
    
    def generate_probability_image(self) -> np.ndarray:
        """生成概率地图图像（用于可视化）"""
        # 对数几率转概率
        probability = 1.0 / (1.0 + np.exp(-self.log_odds_grid))
        
        # 转换为灰度图
        image = np.full_like(probability, self.config.gray_unknown, dtype=float)
        image[probability <= self.config.prob_free_threshold] = self.config.gray_free
        image[probability >= self.config.prob_occ_threshold] = self.config.gray_occupied
        
        return image

# ============================================================================
# ICP扫描匹配
# ============================================================================

class ICPScanMatcher:
    """ICP扫描匹配器"""
    
    def __init__(self, laser_config: LaserSettings):
        self.laser_config = laser_config
        self.icp_available = HAS_ICP
    
    def scan_to_cartesian_points(self, pose: Pose2D, scan: LaserScanData) -> np.ndarray:
        """将扫描数据转换为世界坐标系点云"""
        points_x = []
        points_y = []
        max_range = self.laser_config.max_range
        
        for angle, distance in zip(scan.angles, scan.ranges):
            if distance <= 0 or not math.isfinite(distance) or distance >= max_range - 1e-6:
                continue
            
            global_angle = pose.theta + angle
            x = pose.x + distance * math.cos(global_angle)
            y = pose.y + distance * math.sin(global_angle)
            points_x.append(x)
            points_y.append(y)
        
        if not points_x:
            return np.zeros((0, 2), dtype=float)
        
        return np.column_stack([points_x, points_y])
    
    def match_scans(self, previous_points: np.ndarray, 
                   current_points: np.ndarray,
                   previous_pose: Pose2D) -> Tuple[Optional[Pose2D], Optional[float], int]:
        """
        ICP扫描匹配
        
        Returns:
            (估计位姿, RMSE, 匹配点数)
        """
        if not self.icp_available:
            return None, None, 0
        
        try:
            # 调用外部ICP算法
            result = external_icp_matching(previous_points.T, current_points.T)
            
            # 解析返回结果
            if isinstance(result, tuple) and len(result) == 3:
                if all(isinstance(v, (int, float)) for v in result):
                    yaw_delta, tx, ty = float(result[0]), float(result[1]), float(result[2])
                else:
                    rotation_matrix, translation = result[0], result[1]
                    yaw_delta = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    tx, ty = float(translation[0]), float(translation[1])
            else:
                return None, None, 0
            
            # 计算新位姿
            estimated_x = previous_pose.x + tx
            estimated_y = previous_pose.y + ty
            estimated_theta = (previous_pose.theta + yaw_delta) % (2 * math.pi)
            
            # 计算RMSE
            try:
                cos_yaw = math.cos(yaw_delta)
                sin_yaw = math.sin(yaw_delta)
                rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                
                transformed_prev = previous_points @ rotation.T + np.array([tx, ty])
                
                if current_points.shape[0] > 0 and transformed_prev.shape[0] > 0:
                    # 最近邻距离
                    dist_squared = ((transformed_prev[:, None, :] - current_points[None, :, :]) ** 2).sum(axis=2)
                    min_dist_squared = dist_squared.min(axis=1)
                    rmse = float(np.sqrt(min_dist_squared.mean()))
                    point_count = int(transformed_prev.shape[0])
                else:
                    rmse = None
                    point_count = 0
            except Exception:
                rmse = None
                point_count = 0
            
            return Pose2D(estimated_x, estimated_y, estimated_theta), rmse, point_count
        
        except Exception as e:
            logger.debug(f"ICP匹配失败: {e}")
            return None, None, 0

# ============================================================================
# 位姿融合器
# ============================================================================

class PoseFusionModule:
    """位姿融合模块（融合原始位姿和ICP估计）"""
    
    def __init__(self, config: LocalizationSettings):
        self.config = config
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """计算角度差（归一化到[-π, π]）"""
        diff = (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi
        return diff
    
    def fuse_poses(self, raw_pose: Pose2D, icp_pose: Optional[Pose2D],
                  rmse: Optional[float], point_count: int) -> Optional[Pose2D]:
        """
        融合位姿
        
        Args:
            raw_pose: 原始位姿（里程计/传感器）
            icp_pose: ICP估计位姿
            rmse: ICP匹配RMSE
            point_count: 匹配点数
        
        Returns:
            融合后的位姿（如果融合失败则返回None）
        """
        if not self.config.enabled or icp_pose is None:
            return None
        
        cfg = self.config
        
        # 门控条件1: 最小点数
        if point_count < cfg.min_point_count:
            return None
        
        # 门控条件2: RMSE阈值
        if rmse is None or rmse > cfg.max_rmse:
            return None
        
        # 门控条件3: 平移和旋转差异
        translation_diff = math.hypot(icp_pose.x - raw_pose.x, icp_pose.y - raw_pose.y)
        rotation_diff = abs(self._angle_difference(icp_pose.theta, raw_pose.theta))
        
        if translation_diff > cfg.max_translation or math.degrees(rotation_diff) > cfg.max_rotation_deg:
            return None
        
        # 捕捉模式：差异很小时直接使用ICP
        if translation_diff < cfg.snap_translation and math.degrees(rotation_diff) < cfg.snap_rotation_deg:
            return icp_pose
        
        # 加权融合
        alpha = max(0.0, min(1.0, cfg.alpha))
        fused_x = (1 - alpha) * raw_pose.x + alpha * icp_pose.x
        fused_y = (1 - alpha) * raw_pose.y + alpha * icp_pose.y
        fused_theta = raw_pose.theta + alpha * self._angle_difference(icp_pose.theta, raw_pose.theta)
        fused_theta = (fused_theta + 2 * math.pi) % (2 * math.pi)
        
        return Pose2D(fused_x, fused_y, fused_theta)

# ============================================================================
# 虚拟环境仿真器（用于UNKNOWNSIM模式）
# ============================================================================

class VirtualEnvironmentSimulator:
    """虚拟环境仿真器 - 提供位姿、雷达，执行网格化旋转/直行"""
    
    def __init__(self, laser_config: LaserSettings, 
                 environment: 'BaseEnvironmentMap', 
                 initial_pose: Pose2D):
        """
        初始化仿真器
        
        Args:
            laser_config: 激光雷达配置
            environment: 环境地图
            initial_pose: 初始位姿
        """
        self.laser_config = laser_config
        self.environment = environment
        self.lidar = LaserRangeFinder(laser_config)
        self.current_pose = initial_pose
        logger.info("虚拟环境仿真器初始化完成")
    
    def get_current_pose(self) -> Pose2D:
        """获取当前位姿"""
        return self.current_pose
    
    def get_sensor_reading(self) -> LaserScanData:
        """获取雷达扫描数据"""
        return self.lidar.perform_scan(self.current_pose, self.environment)
    
    def execute_rotation_deg(self, angle_degrees: float) -> None:
        """
        执行原地旋转
        
        Args:
            angle_degrees: 旋转角度（度）
        """
        angle_rad = math.radians(angle_degrees)
        new_theta = (self.current_pose.theta + angle_rad) % (2 * math.pi)
        self.current_pose = Pose2D(self.current_pose.x, self.current_pose.y, new_theta)
        logger.debug(f"旋转 {angle_degrees}° → 新朝向 {math.degrees(new_theta):.1f}°")
    
    def execute_forward_move(self, distance_m: float) -> bool:
        """
        执行直行（带墙体碰撞检测）
        
        Args:
            distance_m: 前进距离（米）
        
        Returns:
            是否成功前进
        """
        # 计算目标位置
        target_x = self.current_pose.x + distance_m * math.cos(self.current_pose.theta)
        target_y = self.current_pose.y + distance_m * math.sin(self.current_pose.theta)
        
        # 1. 边界检查
        if not (0 <= target_x <= self.environment.arena_size and 
                0 <= target_y <= self.environment.arena_size):
            logger.warning(f"前进失败：超出边界 ({target_x:.3f}, {target_y:.3f})")
            return False
        
        # 2. 墙体碰撞检测
        if self._check_wall_collision(self.current_pose.x, self.current_pose.y, 
                                     target_x, target_y):
            logger.warning(f"前进失败：墙体碰撞 "
                         f"({self.current_pose.x:.3f},{self.current_pose.y:.3f}) → "
                         f"({target_x:.3f},{target_y:.3f})")
            return False
        
        # 3. 前进成功
        self.current_pose = Pose2D(target_x, target_y, self.current_pose.theta)
        logger.debug(f"前进 {distance_m}m → 新位置 ({target_x:.3f}, {target_y:.3f})")
        return True
    
    def _check_wall_collision(self, x1: float, y1: float, 
                             x2: float, y2: float) -> bool:
        """
        检查从(x1,y1)到(x2,y2)的路径是否与墙体碰撞
        
        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
        
        Returns:
            是否发生碰撞
        """
        dx = x2 - x1
        dy = y2 - y1
        
        # 检查每个墙体矩形
        for wall in self.environment.obstacle_boxes:
            min_x, max_x, min_y, max_y = wall
            
            # 使用线段与矩形相交检测
            if self._line_intersects_rectangle(x1, y1, dx, dy, 
                                              min_x, max_x, min_y, max_y):
                return True
        return False
    
    def _line_intersects_rectangle(self, x0: float, y0: float, 
                                   dx: float, dy: float,
                                   min_x: float, max_x: float,
                                   min_y: float, max_y: float) -> bool:
        """
        检测线段是否与矩形相交
        
        Args:
            x0, y0: 线段起点
            dx, dy: 线段方向向量
            min_x, max_x, min_y, max_y: 矩形边界
        
        Returns:
            是否相交
        """
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return False  # 零长度线段
        
        # 计算参数化线段与矩形边界的交点参数t
        intersections: List[float] = []
        
        # 垂直边界 (x = min_x, x = max_x)
        for x_boundary in [min_x, max_x]:
            if abs(dx) > 1e-9:
                t = (x_boundary - x0) / dx
                if 0 <= t <= 1:  # 线段范围内
                    y = y0 + t * dy
                    if min_y <= y <= max_y:  # y在矩形范围内
                        intersections.append(t)
        
        # 水平边界 (y = min_y, y = max_y)
        for y_boundary in [min_y, max_y]:
            if abs(dy) > 1e-9:
                t = (y_boundary - y0) / dy
                if 0 <= t <= 1:  # 线段范围内
                    x = x0 + t * dx
                    if min_x <= x <= max_x:  # x在矩形范围内
                        intersections.append(t)
        
        # 如果有交点，则发生碰撞
        return len(intersections) > 0

if __name__ == "__main__":
    print("传感器和SLAM模块测试")
    
    from robot_nav_core import DEFAULT_CONFIG
    from robot_nav_maps import UnknownExplorationMap
    
    # 测试激光雷达
    lidar = LaserRangeFinder(DEFAULT_CONFIG.laser)
    env = UnknownExplorationMap(DEFAULT_CONFIG.environment, DEFAULT_CONFIG.exploration_maze)
    test_pose = Pose2D(1.0, 1.0, 0.0)
    scan = lidar.perform_scan(test_pose, env)
    print(f"激光扫描: {len(scan.ranges)}个光束")
    
    # 测试OGM
    ogm = SpatialOccupancyMap(DEFAULT_CONFIG.grid_map, 0, 0, 2.8, 2.8)
    ogm.update_with_scan(test_pose, scan)
    print(f"OGM栅格尺寸: {ogm.log_odds_grid.shape}")
    
    # 测试ICP
    matcher = ICPScanMatcher(DEFAULT_CONFIG.laser)
    print(f"ICP可用: {matcher.icp_available}")

