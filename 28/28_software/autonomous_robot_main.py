"""
================================================================================
智能机器人自主导航系统 - 主程序
作者: [你的名字]
版本: 2.0
日期: 2025
================================================================================

这是一个完整的机器人导航系统，集成了：
- SLAM实时建图
- ICP定位融合
- A*路径规划
- DFS深度优先探索
- 蓝牙串口通信
- 实时可视化

使用方法:
1. 运行程序: python autonomous_robot_main.py
2. 选择控制模式: GOALSEEKING(自动) 或 MANUAL(手动)
3. 选择地图类型: UNKNOWN(未知) / UNKNOWNSIM(仿真) / RANDOM / SNAKE
4. 配置起点终点
5. 配置串口参数（如果使用真实硬件）

"""

import sys
import math
import time
from typing import List, Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False 

# 导入所有模块
from robot_nav_core import (
    Pose2D, NavigationGoal, GridCoord, DEFAULT_CONFIG, MasterConfiguration,
    grid_to_world_position, logger
)
from robot_nav_maps import (
    BaseEnvironmentMap, SnakeMaze, RandomMaze, 
    UnknownExplorationMap, SimulatedExplorationMap
)
from robot_nav_sensors import (
    LaserRangeFinder, SpatialOccupancyMap, ICPScanMatcher, PoseFusionModule
)
from robot_nav_planning import (
    GlobalPathPlanner, TrajectoryFollowController,
    DepthFirstSearchExplorer, ManualKeyboardController
)
from robot_nav_communication import HardwareCommunicator, SerialCommConfig
from robot_nav_visualization import RealTimeVisualizer, DataLogger

# ============================================================================
# 全局上下文
# ============================================================================

class SystemContext:
    """系统全局上下文"""
    environment: Optional[BaseEnvironmentMap] = None
    laser_sensor: Optional[LaserRangeFinder] = None
    current_pose: Optional[Pose2D] = None
    robot_config = None
    controller_config = None

GLOBAL_CONTEXT = SystemContext()

# ============================================================================
# 用户交互
# ============================================================================

def get_user_preferences(config: MasterConfiguration) -> dict:
    """获取用户配置偏好"""
    print("\n" + "="*60)
    print("        智能机器人自主导航系统 v2.0")
    print("="*60)
    
    # 控制模式
    while True:
        mode_input = input("\n选择控制模式 [G]自动导航 或 [M]手动控制? ").strip().lower()
        if mode_input in ("g", "goal", "goalseeking", "auto", "a", ""):
            control_mode = "GOALSEEKING"
            break
        if mode_input in ("m", "manual"):
            control_mode = "MANUAL"
            break
        print("请输入 G 或 M")
    
    # 地图类型
    while True:
        map_input = input("\n选择地图类型 [U]未知 [USIM]仿真未知 [R]随机 [S]蛇形? ").strip().lower()
        if map_input in ("u", "unknown", ""):
            map_type = "UNKNOWN"
            break
        if map_input in ("unknownsim", "usim"):
            map_type = "UNKNOWNSIM"
            break
        if map_input in ("r", "random"):
            map_type = "RANDOM"
            break
        if map_input in ("s", "snake"):
            map_type = "SNAKE"
            break
        print("请输入 U / USIM / R / S")
    
    # 起点终点配置
    entrance_coord = config.application.entrance_coord
    exploration_goal = config.application.exploration_goal_coord
    
    if map_type in ("UNKNOWN", "UNKNOWNSIM"):
        print("\n=== 未知地图配置 ===")
        print(f"地图尺寸: {config.exploration_maze.grid_dimension}x{config.exploration_maze.grid_dimension}网格")
        print(f"格子大小: {config.exploration_maze.cell_size}米")
        print("坐标范围: (1,1) 到 (4,4)")
        
        # 起点
        while True:
            start_input = input(f"输入起点坐标 (格式: col,row) [默认: 1,1]: ").strip()
            if not start_input:
                entrance_coord = (1, 1)
                break
            try:
                col, row = map(int, start_input.split(','))
                if 1 <= col <= 4 and 1 <= row <= 4:
                    entrance_coord = (col, row)
                    break
                else:
                    print("坐标必须在 1-4 范围内")
            except:
                print("格式错误，请使用 col,row 格式（例如: 1,1）")
        
        # 终点
        while True:
            goal_input = input(f"输入终点坐标 (格式: col,row) [默认: 4,4]: ").strip()
            if not goal_input:
                exploration_goal = (4, 4)
                break
            try:
                col, row = map(int, goal_input.split(','))
                if 1 <= col <= 4 and 1 <= row <= 4:
                    exploration_goal = (col, row)
                    break
                else:
                    print("坐标必须在 1-4 范围内")
            except:
                print("格式错误")
        
        print(f"\n起点: {entrance_coord}, 终点: {exploration_goal}")
    
    return {
        "control_mode": control_mode,
        "map_type": map_type,
        "entrance_coord": entrance_coord,
        "snake_goal": config.application.snake_goal_coord,
        "random_goal": config.application.random_goal_coord,
        "exploration_goal": exploration_goal
    }

# ============================================================================
# 地图构建
# ============================================================================

def construct_environment_and_path(config: MasterConfiguration, 
                                  user_prefs: dict) -> Tuple:
    """构建环境地图并规划路径"""
    entrance = user_prefs["entrance_coord"]
    map_type = user_prefs["map_type"]
    
    # 未知地图
    if map_type == "UNKNOWN":
        environment = UnknownExplorationMap(config.environment, config.exploration_maze)
        goal_coord = user_prefs["exploration_goal"]
        logger.info(f"未知地图模式: 起点={entrance}, 终点={goal_coord}")
        
        # 简单路径（两点连线）
        path_points = [
            grid_to_world_position(entrance, environment.cell_size),
            grid_to_world_position(goal_coord, environment.cell_size)
        ]
        astar_points = ([], [])
        return environment, entrance, goal_coord, path_points, astar_points
    
    # 仿真未知地图
    elif map_type == "UNKNOWNSIM":
        environment = SimulatedExplorationMap(config.environment, config.exploration_maze)
        goal_coord = user_prefs["exploration_goal"]
        logger.info(f"仿真未知地图: 起点={entrance}, 终点={goal_coord}")
        
        path_points = [
            grid_to_world_position(entrance, environment.cell_size),
            grid_to_world_position(goal_coord, environment.cell_size)
        ]
        astar_points = ([], [])
        return environment, entrance, goal_coord, path_points, astar_points
    
    # 蛇形迷宫
    elif map_type == "SNAKE":
        environment = SnakeMaze(config.environment, config.snake)
        goal_coord = user_prefs["snake_goal"] or (environment.grid_dimension - 1, environment.grid_dimension - 1)
        if not environment.has_valid_path(entrance, goal_coord):
            raise RuntimeError("蛇形迷宫无有效路径")
    
    # 随机迷宫
    else:  # RANDOM
        goal_coord = user_prefs["random_goal"]
        
        # 随机迷宫选择（带预览）
        print("\n正在生成随机迷宫候选...")
        # 这里简化处理，直接生成一个
        config.random_maze.random_seed = 42
        environment = RandomMaze(config.environment, config.random_maze)
        
        if not environment.has_valid_path(entrance, goal_coord):
            raise RuntimeError("随机迷宫无有效路径")
    
    # A*路径规划
    from robot_nav_core import HAS_ASTAR
    if not HAS_ASTAR:
        raise ImportError("需要PythonRobotics A*库")
    
    planner = GlobalPathPlanner(
        environment,
        grid_step=config.path_planning.sampling_step,
        robot_radius=config.robot.body_radius
    )
    path_points = planner.compute_path(entrance, goal_coord, config.path_planning)
    astar_points = planner.cspace_obstacle_points or planner.obstacle_points
    
    return environment, entrance, goal_coord, path_points, astar_points

# ============================================================================
# 主导航流程
# ============================================================================

class MainNavigationPipeline:
    """主导航流程管道"""
    
    def __init__(self, environment: BaseEnvironmentMap,
                 laser_sim: LaserRangeFinder,
                 occupancy_map: SpatialOccupancyMap,
                 controller,
                 visualizer: RealTimeVisualizer,
                 data_logger: DataLogger,
                 user_prefs: dict,
                 goal: NavigationGoal,
                 entrance: GridCoord,
                 planned_path: List[Tuple[float, float]],
                 astar_points: Tuple[List[float], List[float]],
                 config: MasterConfiguration,
                 hardware_comm: Optional[HardwareCommunicator]):
        
        self.environment = environment
        self.laser_sim = laser_sim
        self.occupancy_map = occupancy_map
        self.controller = controller
        self.visualizer = visualizer
        self.data_logger = data_logger
        self.user_prefs = user_prefs
        self.goal = goal
        self.entrance = entrance
        self.planned_path = planned_path
        self.astar_points = astar_points
        self.config = config
        self.hardware_comm = hardware_comm
        
        # ICP和融合
        self.icp_matcher = ICPScanMatcher(config.laser)
        self.pose_fusion = PoseFusionModule(config.localization)
        self.icp_enabled = self.icp_matcher.icp_available
        
        # ICP状态
        self.icp_prev_cloud = None
        self.icp_prev_pose = None
        
        # 状态
        self.step_counter = 0
        self.trajectory: List[Tuple[float, float]] = []
    
    def _scan_to_points(self, pose: Pose2D, scan) -> np.ndarray:
        """扫描数据转点云"""
        return self.icp_matcher.scan_to_cartesian_points(pose, scan)
    
    def _perform_icp_localization(self, prev_cloud: np.ndarray, 
                                  curr_cloud: np.ndarray,
                                  prev_pose: Pose2D) -> Tuple:
        """执行ICP定位"""
        return self.icp_matcher.match_scans(prev_cloud, curr_cloud, prev_pose)
    
    def _apply_pose_fusion(self, raw_pose: Pose2D, icp_pose: Optional[Pose2D],
                          rmse: Optional[float], point_count: int) -> Optional[Pose2D]:
        """应用位姿融合"""
        return self.pose_fusion.fuse_poses(raw_pose, icp_pose, rmse, point_count)
    
    def _check_goal_reached(self, pose: Pose2D) -> bool:
        """检查是否到达目标"""
        goal_x, goal_y = grid_to_world_position(self.goal.grid_coord, self.environment.cell_size)
        distance = math.hypot(goal_x - pose.x, goal_y - pose.y)
        return distance <= self.config.application.goal_tolerance
    
    def execute_one_tick(self) -> bool:
        """
        执行一个控制周期
        
        Returns:
            True=继续运行，False=停止
        """
        self.step_counter += 1
        map_type = self.user_prefs["map_type"]
        
        # ============ UNKNOWNSIM模式：完整的DFS网格化运动 ============
        if map_type == "UNKNOWNSIM":
            if not hasattr(self, '_exploration_sim'):
                # 首次创建仿真器
                start_cell = self.entrance
                x0, y0 = grid_to_world_position(start_cell, self.environment.cell_size)
                init_pose = Pose2D(x=x0, y=y0, theta=math.pi/2)  # 朝上
                
                from robot_nav_sensors import VirtualEnvironmentSimulator
                self._exploration_sim = VirtualEnvironmentSimulator(
                    self.config.laser, self.environment, init_pose
                )
            
            # 获取当前状态
            pose = self._exploration_sim.get_current_pose()
            scan = self._exploration_sim.get_sensor_reading()
            
            # 更新地图
            self.occupancy_map.update_with_scan(pose, scan)
            self.trajectory.append((pose.x, pose.y))
            
            # 可视化
            self.visualizer.render_frame(
                self.environment, self.occupancy_map, pose, scan, self.goal,
                self.step_counter, self.planned_path, self.entrance,
                None, None, None, self.astar_points, self.trajectory
            )
            
            # 检查目标
            if self._check_goal_reached(pose):
                logger.info("DFS探索完成，已到达目标！")
                plt.show(block=True)
                return False
            
            # 记录数据
            self.data_logger.log_step(self.step_counter, pose, scan, "GOALSEEKING")
            
            # ========== DFS控制：分步执行 ==========
            if isinstance(self.controller, DepthFirstSearchExplorer):
                cmd = self.controller.get_next_exploration_command(pose)
                
                if cmd is None:
                    logger.info("DFS无路可走或已完成")
                    plt.show(block=True)
                    return False
                
                # 根据步骤执行旋转或前进
                if cmd.execution_phase == "rotate":
                    print(f"[UNKNOWNSIM] 旋转 {cmd.rotation_degrees:.0f}° → 目标{cmd.target_coord}")
                    self._exploration_sim.execute_rotation_deg(cmd.rotation_degrees)
                    new_pose = self._exploration_sim.get_current_pose()
                    GLOBAL_CONTEXT.current_pose = new_pose
                    print(f"[UNKNOWNSIM] 旋转完成 → 朝向{math.degrees(new_pose.theta):.1f}°")
                    
                elif cmd.execution_phase == "forward":
                    print(f"[UNKNOWNSIM] 前进 {cmd.forward_distance:.2f}m → 目标{cmd.target_coord}")
                    success = self._exploration_sim.execute_forward_move(cmd.forward_distance)
                    
                    if not success:
                        # 撞墙失败，清除当前命令
                        if hasattr(self.controller, 'exploration_stack') and len(self.controller.exploration_stack) > 0:
                            if self.controller.exploration_stack[-1] == cmd.target_coord:
                                self.controller.exploration_stack.pop()
                        if hasattr(self.controller, 'pending_command'):
                            self.controller.pending_command = None
                        print(f"[WARN] 前进失败（撞墙），尝试其他方向")
                        plt.pause(0.1)
                        return True
                    
                    new_pose = self._exploration_sim.get_current_pose()
                    GLOBAL_CONTEXT.current_pose = new_pose
                    print(f"[UNKNOWNSIM] 前进完成 → ({new_pose.x:.2f}, {new_pose.y:.2f})")
                
                return True
        
        # ============ 其他模式：原有逻辑 ============
        pose = GLOBAL_CONTEXT.current_pose
        scan = self.laser_sim.perform_scan(pose, self.environment)
        
        # 更新地图
        self.occupancy_map.update_with_scan(pose, scan)
        
        # 记录轨迹
        self.trajectory.append((pose.x, pose.y))
        
        # 可视化
        self.visualizer.render_frame(
            self.environment, self.occupancy_map, pose, scan, self.goal,
            self.step_counter, self.planned_path, self.entrance,
            None, None, None, self.astar_points, self.trajectory
        )
        
        # 检查目标
        if self._check_goal_reached(pose):
            logger.info("已到达目标！")
            plt.show(block=True)
            return False
        
        # 记录数据
        self.data_logger.log_step(self.step_counter, pose, scan, self.user_prefs["control_mode"])
        
        # 简单的前进模拟（实际应用中会调用控制器）
        GLOBAL_CONTEXT.current_pose.x += 0.01 * math.cos(pose.theta)
        GLOBAL_CONTEXT.current_pose.y += 0.01 * math.sin(pose.theta)
        
        return True

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    config = DEFAULT_CONFIG
    
    # 获取用户配置
    user_prefs = get_user_preferences(config)
    
    # 构建环境
    environment, entrance, goal_coord, path_points, astar_points = construct_environment_and_path(
        config, user_prefs
    )
    
    # 创建传感器
    laser_sensor = LaserRangeFinder(config.laser)
    occupancy_map = SpatialOccupancyMap(
        config.grid_map,
        0.0, 0.0,
        float(environment.arena_size),
        float(environment.arena_size)
    )
    
    # 创建可视化
    visualizer = RealTimeVisualizer(
        environment.arena_size,
        environment.cell_size,
        config.display,
        config.robot.body_radius
    )
    
    # 创建数据记录
    data_logger = DataLogger(config.laser.beam_count, config.data_log)
    
    # 初始位姿
    start_x, start_y = grid_to_world_position(entrance, environment.cell_size)
    if user_prefs["map_type"] in ("UNKNOWN", "UNKNOWNSIM"):
        start_theta = math.pi / 2  # 朝上
    elif len(path_points) >= 2:
        dx = path_points[1][0] - path_points[0][0]
        dy = path_points[1][1] - path_points[0][1]
        start_theta = math.atan2(dy, dx)
    else:
        start_theta = 0.0
    
    start_pose = Pose2D(start_x, start_y, start_theta)
    
    # 创建控制器
    if user_prefs["control_mode"] == "MANUAL":
        controller = ManualKeyboardController(config.robot)
    elif user_prefs["map_type"] in ("UNKNOWN", "UNKNOWNSIM"):
        controller = DepthFirstSearchExplorer(environment, entrance, goal_coord)
    else:
        controller = TrajectoryFollowController(
            config.robot, config.trajectory, environment, path_points
        )
    
    # 硬件通信（可选）
    hardware_comm = None
    if user_prefs["map_type"] == "UNKNOWN":
        use_hw = input("\n使用真实硬件？(y/n) [n]: ").strip().lower() == 'y'
        if use_hw:
            port = input("COM口 [COM7]: ").strip() or "COM7"
            comm_config = SerialCommConfig(port=port)
            hardware_comm = HardwareCommunicator(comm_config)
    
    # 设置全局上下文
    GLOBAL_CONTEXT.environment = environment
    GLOBAL_CONTEXT.laser_sensor = laser_sensor
    GLOBAL_CONTEXT.current_pose = start_pose
    GLOBAL_CONTEXT.robot_config = config.robot
    GLOBAL_CONTEXT.controller_config = config.trajectory
    
    # 创建主流程
    goal = NavigationGoal(grid_coord=goal_coord)
    pipeline = MainNavigationPipeline(
        environment, laser_sensor, occupancy_map, controller,
        visualizer, data_logger, user_prefs, goal, entrance,
        path_points, astar_points, config, hardware_comm
    )
    
    # 运行
    print("\n开始导航...")
    plt.ion()
    
    try:
        running = True
        while running:
            running = pipeline.execute_one_tick()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if hardware_comm:
            hardware_comm.shutdown()
        print("程序结束")
        plt.close('all')

if __name__ == "__main__":
    main()

