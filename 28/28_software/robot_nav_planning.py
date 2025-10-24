"""
================================================================================
智能机器人导航系统 - 路径规划和控制模块
================================================================================
包含:
1. A*全局路径规划
2. 轨迹跟踪控制器
3. DFS探索控制器
4. 手动控制器
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from robot_nav_core import (
    GridCoord, Pose2D, TargetPose, MovementCommand, LaserScanData, NavigationGoal,
    PathPlanSettings, VehicleSettings, TrajectorySettings,
    grid_to_world_position, convert_obstacles_to_points, resample_trajectory,
    HAS_ASTAR, logger
)
from robot_nav_maps import BaseEnvironmentMap

if HAS_ASTAR:
    from PathPlanning.AStar.a_star import AStarPlanner  # type: ignore

# ============================================================================
# A*全局路径规划器
# ============================================================================

class GlobalPathPlanner:
    """A*全局路径规划器"""
    
    def __init__(self, environment: BaseEnvironmentMap, 
                 grid_step: float, robot_radius: float):
        self.environment = environment
        self.grid_step = grid_step
        self.robot_radius = robot_radius
        
        # 缓存障碍物点云
        self.obstacle_points: Tuple[List[float], List[float]] = ([], [])
        self.cspace_obstacle_points: Tuple[List[float], List[float]] = ([], [])
        
        if not HAS_ASTAR:
            raise ImportError("需要PythonRobotics A*库。请设置PYTHONROBOTICS_PATH环境变量。")
    
    def compute_path(self, start_coord: GridCoord, goal_coord: GridCoord,
                    planning_config: PathPlanSettings) -> List[Tuple[float, float]]:
        """
        计算从起点到终点的路径
        
        Args:
            start_coord: 起点网格坐标
            goal_coord: 终点网格坐标
            planning_config: 规划配置
        
        Returns:
            路径点列表 [(x, y), ...]
        """
        try:
            # 将障碍物转换为点云
            obstacle_x, obstacle_y = convert_obstacles_to_points(
                self.environment.obstacle_boxes,
                step=planning_config.sampling_step
            )
            self.obstacle_points = (obstacle_x, obstacle_y)
            
            # 计算起点和终点的世界坐标
            start_x, start_y = grid_to_world_position(start_coord, self.environment.cell_size)
            goal_x, goal_y = grid_to_world_position(goal_coord, self.environment.cell_size)
            
            # 创建A*规划器
            planner = AStarPlanner(
                obstacle_x, obstacle_y,
                planning_config.sampling_step,
                self.robot_radius
            )
            
            # 执行规划
            path_x, path_y = planner.planning(start_x, start_y, goal_x, goal_y)
            
        except Exception as e:
            raise RuntimeError(f"A*规划失败: {e}")
        
        if path_x is None or path_y is None or len(path_x) == 0:
            raise RuntimeError("A*返回空路径")
        
        # 提取C空间障碍物格子（用于可视化）
        try:
            cspace_x: List[float] = []
            cspace_y: List[float] = []
            for ix in range(planner.x_width):
                for iy in range(planner.y_width):
                    if planner.obstacle_map[ix][iy]:
                        world_x = planner.calc_grid_position(ix, planner.min_x)
                        world_y = planner.calc_grid_position(iy, planner.min_y)
                        cspace_x.append(world_x)
                        cspace_y.append(world_y)
            self.cspace_obstacle_points = (cspace_x, cspace_y)
        except Exception:
            self.cspace_obstacle_points = ([], [])
        
        # 反转路径（A*返回从终点到起点）
        path_x_list = list(path_x)[::-1]
        path_y_list = list(path_y)[::-1]
        
        # 组合为点对
        path_points = [(x, y) for x, y in zip(path_x_list, path_y_list)]
        
        # 重采样
        resampled_path = resample_trajectory(
            path_points,
            interval=planning_config.resampling_step,
            equal_eps=planning_config.equality_threshold,
            segment_eps=planning_config.segment_threshold
        )
        
        logger.debug(f"A*路径: {len(resampled_path)}个点")
        return resampled_path

# ============================================================================
# 轨迹跟踪控制器
# ============================================================================

class TrajectoryFollowController:
    """轨迹跟踪控制器（用于连续空间导航）"""
    
    def __init__(self, robot_config: VehicleSettings, 
                 controller_config: TrajectorySettings,
                 environment: BaseEnvironmentMap,
                 global_path: List[Tuple[float, float]]):
        self.robot_cfg = robot_config
        self.ctrl_cfg = controller_config
        self.environment = environment
        self.path = global_path[:]
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到[-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def _compute_lookahead_point(self, current_pose: Pose2D) -> Tuple[float, float]:
        """计算前瞻点"""
        if not self.path:
            return (current_pose.x, current_pose.y)
        
        # 找到最近路径点
        distances = [math.hypot(px - current_pose.x, py - current_pose.y) 
                    for px, py in self.path]
        closest_idx = int(np.argmin(distances))  # type: ignore
        
        # 沿路径前瞻
        accumulated_length = 0.0
        idx = closest_idx
        
        while accumulated_length < self.ctrl_cfg.lookahead_distance and idx < len(self.path) - 1:
            dx = self.path[idx + 1][0] - self.path[idx][0]
            dy = self.path[idx + 1][1] - self.path[idx][1]
            accumulated_length += math.hypot(dx, dy)
            idx += 1
        
        return self.path[idx]
    
    def _compute_repulsive_force(self, scan: LaserScanData) -> Tuple[float, float]:
        """计算激光雷达斥力"""
        radius = self.ctrl_cfg.avoidance_radius
        gain = self.ctrl_cfg.repulsion_gain
        
        repulsion_x = 0.0
        repulsion_y = 0.0
        
        for relative_angle, distance in zip(scan.angles, scan.ranges):
            if distance <= 0 or distance > radius:
                continue
            
            # 斥力强度随距离衰减
            strength = gain * max(0.0, (1.0 / distance - 1.0 / radius))
            repulsion_x += -math.cos(relative_angle) * strength
            repulsion_y += -math.sin(relative_angle) * strength
        
        return (repulsion_x, repulsion_y)
    
    def compute_setpoint(self, current_pose: Pose2D, 
                        scan: LaserScanData) -> TargetPose:
        """
        计算控制目标点
        
        Args:
            current_pose: 当前位姿
            scan: 激光扫描数据
        
        Returns:
            目标位姿
        """
        # 1. 计算前瞻目标
        lookahead_x, lookahead_y = self._compute_lookahead_point(current_pose)
        
        # 2. 转换到机器人坐标系
        dx = lookahead_x - current_pose.x
        dy = lookahead_y - current_pose.y
        
        cos_theta = math.cos(current_pose.theta)
        sin_theta = math.sin(current_pose.theta)
        
        robot_frame_x = cos_theta * dx + sin_theta * dy
        robot_frame_y = -sin_theta * dx + cos_theta * dy
        
        # 3. 计算斥力
        repulsion_x, repulsion_y = self._compute_repulsive_force(scan)
        
        # 4. 组合前瞻和斥力
        combined_x = robot_frame_x + self.ctrl_cfg.avoidance_gain * repulsion_x
        combined_y = robot_frame_y + self.ctrl_cfg.avoidance_gain * repulsion_y
        
        # 5. 计算期望航向
        desired_heading_robot = math.atan2(combined_y, combined_x)
        desired_heading_world = (current_pose.theta + desired_heading_robot) % (2 * math.pi)
        
        return TargetPose(x=lookahead_x, y=lookahead_y, theta=desired_heading_world)

# ============================================================================
# DFS探索控制器
# ============================================================================

class DepthFirstSearchExplorer:
    """深度优先搜索探索控制器"""
    
    def __init__(self, environment: BaseEnvironmentMap, 
                 start_coord: GridCoord, goal_coord: GridCoord):
        self.environment = environment
        self.start_coord = start_coord
        self.goal_coord = goal_coord
        self.cell_size = environment.cell_size
        self.grid_size = environment.grid_dimension
        
        # 探索状态
        self.visited_cells: set = set()
        self.exploration_stack: List[GridCoord] = [start_coord]
        self.current_coord = start_coord
        self.reached_goal = False
        
        # 当前待执行指令
        self.pending_command: Optional[MovementCommand] = None
        
        # 墙体地图 {(coord1, coord2): has_wall}
        self.wall_database: Dict[Tuple[GridCoord, GridCoord], bool] = {}
        
        logger.info(f"DFS探索器: 起点={start_coord}, 终点={goal_coord}, "
                   f"网格{self.grid_size}x{self.grid_size}")
    
    def update_wall_map_from_sensor(self, current_coord: GridCoord, 
                                    sensor_distances: LaserScanData) -> None:
        """
        根据传感器数据更新墙体数据库
        
        Args:
            current_coord: 当前格子坐标
            sensor_distances: 雷达扫描数据（假定包含4个方向的距离）
        """
        if not sensor_distances.ranges or len(sensor_distances.ranges) < 4:
            logger.debug(f"传感器数据不足，跳过墙体更新")
            return
        
        col, row = current_coord
        wall_threshold = self.cell_size * 0.8
        
        # 定义四个方向及其邻居
        directions_and_neighbors = [
            ('right', (col+1, row), sensor_distances.ranges[0]),
            ('up', (col, row+1), sensor_distances.ranges[1]),
            ('left', (col-1, row), sensor_distances.ranges[2]),
            ('down', (col, row-1), sensor_distances.ranges[3])
        ]
        
        logger.debug(f"更新格子{current_coord}的墙体:")
        for direction, neighbor_coord, distance in directions_and_neighbors:
            has_wall = distance < wall_threshold
            
            # 双向记录
            self.wall_database[(current_coord, neighbor_coord)] = has_wall
            self.wall_database[(neighbor_coord, current_coord)] = has_wall
            
            logger.debug(f"  {direction} -> {neighbor_coord}: 距离={distance:.3f}m, "
                        f"墙体={'是' if has_wall else '否'}")
    
    def _check_wall_between(self, coord1: GridCoord, coord2: GridCoord) -> bool:
        """检查两个格子之间是否有墙"""
        return self.wall_database.get((coord1, coord2), False)
    
    def _get_accessible_neighbors(self) -> List[Tuple[GridCoord, str]]:
        """获取可访问的未访问邻居格子"""
        col, row = self.current_coord
        accessible: List[Tuple[GridCoord, str]] = []
        
        # 定义四个方向
        neighbors_and_directions = [
            ((col+1, row), 'right'),
            ((col, row+1), 'up'),
            ((col-1, row), 'left'),
            ((col, row-1), 'down')
        ]
        
        for neighbor_coord, direction in neighbors_and_directions:
            n_col, n_row = neighbor_coord
            
            # 边界检查
            if not (1 <= n_col <= self.grid_size and 1 <= n_row <= self.grid_size):
                continue
            
            # 访问检查
            if neighbor_coord in self.visited_cells:
                continue
            
            # 墙体检查
            if not self._check_wall_between(self.current_coord, neighbor_coord):
                accessible.append((neighbor_coord, direction))
                logger.debug(f"发现可达格子 {neighbor_coord} (方向:{direction})")
        
        return accessible
    
    def _calculate_required_rotation(self, current_heading: float, 
                                    target_direction: str) -> float:
        """计算需要旋转的角度（度）"""
        direction_headings = {
            'right': 0.0,
            'up': math.pi / 2,
            'left': math.pi,
            'down': -math.pi / 2
        }
        
        target_heading = direction_headings.get(target_direction, 0.0)
        angle_diff = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
        degrees = math.degrees(angle_diff)
        
        # 量化为90度的倍数
        if abs(degrees) < 45:
            return 0.0
        elif 45 <= degrees < 135:
            return 90.0
        elif degrees >= 135 or degrees <= -135:
            return 180.0
        else:
            return -90.0
    
    def get_next_exploration_command(self, current_pose: Pose2D) -> Optional[MovementCommand]:
        """
        获取下一个探索指令
        
        Args:
            current_pose: 当前位姿
        
        Returns:
            移动指令（如果探索完成则返回None）
        """
        # 如果有待执行指令，继续执行
        if self.pending_command is not None:
            if self.pending_command.execution_phase == "rotate":
                # 旋转完成，进入前进阶段
                self.pending_command.execution_phase = "forward"
                logger.debug(f"旋转完成，准备前进 {self.pending_command.forward_distance:.2f}m")
                return self.pending_command
            elif self.pending_command.execution_phase == "forward":
                # 前进完成，清除指令
                logger.debug(f"前进完成，到达格子 {self.pending_command.target_coord}")
                self.pending_command = None
        
        # 更新当前格子
        col = min(int(current_pose.x / self.cell_size) + 1, self.grid_size)
        row = min(int(current_pose.y / self.cell_size) + 1, self.grid_size)
        self.current_coord = (col, row)
        self.visited_cells.add(self.current_coord)
        
        logger.debug(f"位姿({current_pose.x:.3f},{current_pose.y:.3f},{math.degrees(current_pose.theta):.1f}°) "
                    f"→ 当前格子={self.current_coord}, 已访问={len(self.visited_cells)}个")
        
        # 检查是否到达目标
        if self.current_coord == self.goal_coord:
            self.reached_goal = True
            logger.info("已到达目标格子！")
            return None
        
        # 获取可访问的邻居
        accessible_neighbors = self._get_accessible_neighbors()
        
        # DFS: 探索第一个可达邻居
        if accessible_neighbors:
            neighbor_coord, direction = accessible_neighbors[0]
            rotation_deg = self._calculate_required_rotation(current_pose.theta, direction)
            distance = self.cell_size
            
            self.exploration_stack.append(neighbor_coord)
            
            command = MovementCommand(
                rotation_degrees=rotation_deg,
                forward_distance=distance,
                target_coord=neighbor_coord,
                execution_phase="rotate"
            )
            self.pending_command = command
            logger.debug(f"探索 {neighbor_coord} (方向:{direction}, 旋转{rotation_deg}°)")
            return command
        
        # 回溯
        if len(self.exploration_stack) > 1:
            self.exploration_stack.pop()
            previous_coord = self.exploration_stack[-1]
            
            # 计算回溯方向
            dx = previous_coord[0] - self.current_coord[0]
            dy = previous_coord[1] - self.current_coord[1]
            
            if dx == 1:
                direction = 'right'
            elif dx == -1:
                direction = 'left'
            elif dy == 1:
                direction = 'up'
            elif dy == -1:
                direction = 'down'
            else:
                direction = 'right'
            
            rotation_deg = self._calculate_required_rotation(current_pose.theta, direction)
            distance = self.cell_size
            
            command = MovementCommand(
                rotation_degrees=rotation_deg,
                forward_distance=distance,
                target_coord=previous_coord,
                execution_phase="rotate"
            )
            self.pending_command = command
            logger.debug(f"回溯到 {previous_coord} (方向:{direction}, 旋转{rotation_deg}°)")
            return command
        
        logger.debug("无路可走且无法回溯")
        return None

# ============================================================================
# 手动控制器
# ============================================================================

class ManualKeyboardController:
    """手动键盘控制器"""
    
    def __init__(self, robot_config: VehicleSettings):
        self.config = robot_config
        self.last_key_press: Optional[str] = None
    
    def on_key_event(self, event):
        """键盘事件处理"""
        if event.key in ["up", "down", "left", "right", "q"]:
            self.last_key_press = event.key
    
    def compute_next_pose(self, current_pose: Pose2D, scan: LaserScanData,
                         environment: BaseEnvironmentMap, goal: NavigationGoal) -> Pose2D:
        """
        根据键盘输入计算下一个位姿
        
        Args:
            current_pose: 当前位姿
            scan: 激光扫描
            environment: 环境地图
            goal: 导航目标
        
        Returns:
            下一个位姿
        """
        if self.last_key_press is None:
            return current_pose
        
        key = self.last_key_press
        self.last_key_press = None
        
        if key == 'q':
            return current_pose
        
        # 检查前方是否有障碍
        cone_angle = self.config.manual_fwd_cone
        forward_ranges = [dist for angle, dist in zip(scan.angles, scan.ranges)
                         if (angle <= cone_angle or angle >= 2 * math.pi - cone_angle)]
        
        can_move_forward = (not forward_ranges) or min(forward_ranges) > self.config.body_radius
        step_distance = self.config.manual_step_dist
        
        # 处理按键
        if key == "left":
            return Pose2D(current_pose.x, current_pose.y,
                         (current_pose.theta + self.config.rotation_angle) % (2 * math.pi))
        
        elif key == "right":
            return Pose2D(current_pose.x, current_pose.y,
                         (current_pose.theta - self.config.rotation_angle) % (2 * math.pi))
        
        elif key == "up" and can_move_forward:
            new_x = current_pose.x + step_distance * math.cos(current_pose.theta)
            new_y = current_pose.y + step_distance * math.sin(current_pose.theta)
            return Pose2D(new_x, new_y, current_pose.theta)
        
        elif key == "down" and can_move_forward:
            new_x = current_pose.x - step_distance * math.cos(current_pose.theta)
            new_y = current_pose.y - step_distance * math.sin(current_pose.theta)
            return Pose2D(new_x, new_y, current_pose.theta)
        
        return current_pose

if __name__ == "__main__":
    print("路径规划和控制模块测试")
    
    from robot_nav_core import DEFAULT_CONFIG
    from robot_nav_maps import UnknownExplorationMap
    
    # 测试环境
    env = UnknownExplorationMap(DEFAULT_CONFIG.environment, DEFAULT_CONFIG.exploration_maze)
    
    # 测试DFS探索器
    explorer = DepthFirstSearchExplorer(env, (1, 1), (4, 4))
    test_pose = Pose2D(0.35, 0.35, math.pi/2)
    print(f"DFS探索器已创建，当前格子: {explorer.current_coord}")

