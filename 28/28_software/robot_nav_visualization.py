"""
================================================================================
智能机器人导航系统 - 可视化和数据记录模块
================================================================================
包含:
1. 实时可视化仪表盘
2. CSV数据记录器
"""

import math
import csv
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle

from robot_nav_core import (
    GridCoord, Pose2D, LaserScanData, NavigationGoal, DisplaySettings,
    DataLogSettings, logger
)
from robot_nav_maps import BaseEnvironmentMap
from robot_nav_sensors import SpatialOccupancyMap

# ============================================================================
# 可视化仪表盘
# ============================================================================

class RealTimeVisualizer:
    """实时可视化仪表盘（4面板布局）"""
    
    def __init__(self, arena_size: float, cell_size: float, config: DisplaySettings,
                 robot_radius: float):
        from matplotlib.gridspec import GridSpec
        
        self.arena_size = arena_size
        self.cell_size = cell_size
        self.config = config
        self.robot_radius = float(robot_radius)
        
        # 创建图形窗口
        self.figure = plt.figure(figsize=self.config.figure_size, constrained_layout=True)
        grid_spec = GridSpec(2, 2, figure=self.figure)
        
        # 四个子图面板
        self.ax_main_map = self.figure.add_subplot(grid_spec[0, 0])    # 左上：地图+激光
        self.ax_occupancy = self.figure.add_subplot(grid_spec[0, 1])   # 右上：占据栅格
        self.ax_planning = self.figure.add_subplot(grid_spec[1, 0])    # 左下：路径规划
        self.ax_debug = self.figure.add_subplot(grid_spec[1, 1])       # 右下：调试/ICP
        
        # 统一设置所有面板
        for ax in (self.ax_main_map, self.ax_occupancy, self.ax_planning, self.ax_debug):
            ax.set_xlim(0, self.arena_size)
            ax.set_ylim(0, self.arena_size)
            ax.set_box_aspect(1)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True)
        
        # 设置网格线
        for ax in (self.ax_main_map, self.ax_occupancy):
            ax.set_xticks(np.arange(0, self.arena_size + 1e-9, self.cell_size))
            ax.set_yticks(np.arange(0, self.arena_size + 1e-9, self.cell_size))
        
        self.ax_planning.set_title("路径规划与C空间")
        self.ax_debug.set_title("ICP点云匹配")
        
        # ICP可视化状态
        self._icp_prev_scatter = None
        self._icp_curr_scatter = None
        self._icp_transformed_scatter = None
    
    def _render_world_map(self, environment: BaseEnvironmentMap, 
                         entrance_coord: GridCoord, goal_coord: GridCoord):
        """渲染环境地图（左上和右上面板）"""
        self.ax_main_map.clear()
        self.ax_occupancy.clear()
        
        for ax in (self.ax_main_map, self.ax_occupancy):
            ax.set_xlim(0, self.arena_size)
            ax.set_ylim(0, self.arena_size)
            ax.set_box_aspect(1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.arange(0, self.arena_size + 1e-9, self.cell_size))
            ax.set_yticks(np.arange(0, self.arena_size + 1e-9, self.cell_size))
            ax.grid(True)
        
        # 绘制障碍物
        for obstacle in environment.obstacle_boxes:
            x_min, x_max, y_min, y_max = obstacle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                           facecolor="black")
            self.ax_main_map.add_patch(rect)
        
        # 绘制起点和终点
        entrance_col, entrance_row = entrance_coord
        goal_col, goal_row = goal_coord
        
        self.ax_main_map.add_patch(Rectangle(
            ((entrance_col - 1) * self.cell_size, (entrance_row - 1) * self.cell_size),
            self.cell_size, self.cell_size,
            facecolor="green", alpha=0.3, label="起点"
        ))
        
        self.ax_main_map.add_patch(Rectangle(
            ((goal_col - 1) * self.cell_size, (goal_row - 1) * self.cell_size),
            self.cell_size, self.cell_size,
            facecolor="red", alpha=0.3, label="终点"
        ))
        
        handles, labels = self.ax_main_map.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            self.ax_main_map.legend(by_label.values(), by_label.keys())
    
    def render_frame(self,
                    environment: BaseEnvironmentMap,
                    occupancy_map: SpatialOccupancyMap,
                    robot_pose: Pose2D,
                    scan_data: LaserScanData,
                    goal: NavigationGoal,
                    step_number: int,
                    planned_path: Optional[List[Tuple[float, float]]] = None,
                    entrance_coord: Optional[GridCoord] = None,
                    icp_prev_cloud: Optional[np.ndarray] = None,
                    icp_curr_cloud: Optional[np.ndarray] = None,
                    icp_transformed_cloud: Optional[np.ndarray] = None,
                    astar_obstacle_points: Optional[Tuple[List[float], List[float]]] = None,
                    robot_trajectory: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        渲染一帧
        
        Args:
            environment: 环境地图
            occupancy_map: 占据栅格地图
            robot_pose: 机器人位姿
            scan_data: 激光扫描数据
            goal: 导航目标
            step_number: 步数
            planned_path: 规划路径
            entrance_coord: 起点坐标
            icp_prev_cloud: ICP上一帧点云
            icp_curr_cloud: ICP当前帧点云
            icp_transformed_cloud: ICP变换后点云
            astar_obstacle_points: A*障碍物点
            robot_trajectory: 机器人轨迹
        """
        self._render_world_map(environment, entrance_coord or (0, 0), goal.grid_coord)
        
        # === 左上：地图+激光+路径 ===
        self.ax_main_map.plot(robot_pose.x, robot_pose.y, "bo", markersize=8)
        arrow_len = self.config.robot_arrow_length
        arrow_head = self.config.robot_arrow_head
        self.ax_main_map.arrow(
            robot_pose.x, robot_pose.y,
            arrow_len * math.cos(robot_pose.theta),
            arrow_len * math.sin(robot_pose.theta),
            head_width=arrow_head[0], head_length=arrow_head[1],
            fc="blue", ec="blue"
        )
        
        # 绘制激光束
        for angle, distance in zip(scan_data.angles, scan_data.ranges):
            end_x = robot_pose.x + distance * math.cos(robot_pose.theta + angle)
            end_y = robot_pose.y + distance * math.sin(robot_pose.theta + angle)
            self.ax_main_map.plot([robot_pose.x, end_x], [robot_pose.y, end_y],
                                 "r-", alpha=self.config.lidar_alpha,
                                 linewidth=self.config.lidar_linewidth)
        
        # 绘制规划路径
        if planned_path:
            path_x, path_y = zip(*planned_path)
            self.ax_main_map.plot(path_x, path_y, "g--", linewidth=2, alpha=0.8, label="规划路径")
        
        # 绘制实际轨迹
        if robot_trajectory and len(robot_trajectory) > 1:
            traj_x, traj_y = zip(*robot_trajectory)
            self.ax_main_map.plot(traj_x, traj_y, "m-", linewidth=2.5, alpha=0.9, label="运行轨迹")
        
        self.ax_main_map.set_title(f"环境地图 - 步数 {step_number}")
        self.ax_main_map.set_xlabel("X (米)")
        self.ax_main_map.set_ylabel("Y (米)")
        
        # === 右上：占据栅格地图 ===
        ogm_image = occupancy_map.generate_probability_image()
        self.ax_occupancy.imshow(ogm_image, 
                                extent=(0, self.arena_size, 0, self.arena_size),
                                origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
        
        self.ax_occupancy.plot(robot_pose.x, robot_pose.y, "bo", markersize=6, label="机器人")
        arrow_len_ogm = self.config.ogm_arrow_length
        arrow_head_ogm = self.config.ogm_arrow_head
        self.ax_occupancy.arrow(
            robot_pose.x, robot_pose.y,
            arrow_len_ogm * math.cos(robot_pose.theta),
            arrow_len_ogm * math.sin(robot_pose.theta),
            head_width=arrow_head_ogm[0], head_length=arrow_head_ogm[1],
            fc="blue", ec="blue"
        )
        
        # OGM上绘制轨迹
        if robot_trajectory and len(robot_trajectory) > 1:
            traj_x, traj_y = zip(*robot_trajectory)
            self.ax_occupancy.plot(traj_x, traj_y, "m-", linewidth=2, alpha=0.8, label="轨迹")
        
        self.ax_occupancy.set_title("占据栅格地图")
        self.ax_occupancy.set_xlabel("X (米)")
        self.ax_occupancy.set_ylabel("Y (米)")
        
        handles, labels = self.ax_occupancy.get_legend_handles_labels()
        if handles:
            self.ax_occupancy.legend(dict(zip(labels, handles)).values(),
                                    dict(zip(labels, handles)).keys())
        
        # === 左下：路径规划与C空间 ===
        self.ax_planning.clear()
        self.ax_planning.set_xlim(0, self.arena_size)
        self.ax_planning.set_ylim(0, self.arena_size)
        self.ax_planning.set_box_aspect(1)
        self.ax_planning.set_aspect('equal', adjustable='box')
        self.ax_planning.set_title("路径规划与C空间")
        self.ax_planning.set_xlabel("X (米)")
        self.ax_planning.set_ylabel("Y (米)")
        self.ax_planning.grid(True)
        
        if astar_obstacle_points and len(astar_obstacle_points[0]) > 0:
            self.ax_planning.plot(astar_obstacle_points[0], astar_obstacle_points[1],
                                 'x', ms=4, alpha=0.8, color='#12B2B2', label='C空间障碍')
        
        if planned_path:
            path_x, path_y = zip(*planned_path)
            self.ax_planning.plot(path_x, path_y, ".-", alpha=0.9, label='规划路径')
        
        self.ax_planning.plot(robot_pose.x, robot_pose.y, "ro", ms=4, label='机器人')
        
        try:
            radius_circle = Circle((robot_pose.x, robot_pose.y), 
                                  radius=self.robot_radius,
                                  fill=False, linestyle='--', linewidth=1.2,
                                  color='#FF7F0E', label='规划半径')
            self.ax_planning.add_patch(radius_circle)
        except:
            pass
        
        if self.ax_planning.get_legend_handles_labels()[0]:
            self.ax_planning.legend()
        
        # === 右下：ICP点云 ===
        self.ax_debug.set_xlim(0, self.arena_size)
        self.ax_debug.set_ylim(0, self.arena_size)
        self.ax_debug.set_box_aspect(1)
        self.ax_debug.set_aspect('equal', adjustable='box')
        self.ax_debug.set_title("ICP点云匹配 (灰=上一帧, 蓝=当前, 绿=变换后)")
        self.ax_debug.set_xlabel("X (米)")
        self.ax_debug.set_ylabel("Y (米)")
        self.ax_debug.grid(True)
        
        if icp_prev_cloud is not None and icp_prev_cloud.size > 0:
            if self._icp_prev_scatter is None:
                self._icp_prev_scatter = self.ax_debug.scatter(
                    icp_prev_cloud[:,0], icp_prev_cloud[:,1],
                    s=3, c="#888888", label="上一帧"
                )
            else:
                self._icp_prev_scatter.set_offsets(icp_prev_cloud)
        
        if icp_curr_cloud is not None and icp_curr_cloud.size > 0:
            if self._icp_curr_scatter is None:
                self._icp_curr_scatter = self.ax_debug.scatter(
                    icp_curr_cloud[:,0], icp_curr_cloud[:,1],
                    s=3, label="当前帧"
                )
            else:
                self._icp_curr_scatter.set_offsets(icp_curr_cloud)
        
        if icp_transformed_cloud is not None and icp_transformed_cloud.size > 0:
            if self._icp_transformed_scatter is None:
                self._icp_transformed_scatter = self.ax_debug.scatter(
                    icp_transformed_cloud[:,0], icp_transformed_cloud[:,1],
                    s=3, c="#2ca02c", label="变换后"
                )
            else:
                self._icp_transformed_scatter.set_offsets(icp_transformed_cloud)
        
        if self.ax_debug.get_legend_handles_labels()[0]:
            self.ax_debug.legend()
        
        # 刷新显示
        plt.draw()
        plt.pause(self.config.pause_duration)
        self.figure.canvas.flush_events()

# ============================================================================
# CSV数据记录器
# ============================================================================

class DataLogger:
    """CSV数据记录器"""
    
    def __init__(self, laser_ray_count: int, config: DataLogSettings):
        self.pose_file_path = Path(config.pose_csv_file)
        self.lidar_file_path = Path(config.lidar_csv_file)
        self.ray_count = laser_ray_count
        
        # 确保目录存在
        if self.pose_file_path.parent:
            self.pose_file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lidar_file_path.parent:
            self.lidar_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 删除旧文件
        for file_path in (self.pose_file_path, self.lidar_file_path):
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                with file_path.open("w", newline=""):
                    pass
        
        # 写入CSV标题
        with self.pose_file_path.open("w", newline="") as f:
            csv.writer(f).writerow(["步数", "X(米)", "Y(米)", "航向(度)", "模式"])
        
        with self.lidar_file_path.open("w", newline="") as f:
            headers = ["步数"]
            for i in range(self.ray_count):
                headers.append(f"角度{i}(度)")
                headers.append(f"距离{i}(米)")
            csv.writer(f).writerow(headers)
    
    def log_step(self, step_num: int, pose: Pose2D, scan: LaserScanData, mode: str) -> None:
        """记录一步数据"""
        # 记录位姿
        with self.pose_file_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                step_num,
                pose.x,
                pose.y,
                math.degrees(pose.theta),
                mode
            ])
        
        # 记录激光数据
        row = [step_num]
        for angle, distance in zip(scan.angles, scan.ranges):
            row.append(math.degrees(angle))
            row.append(distance)
        
        with self.lidar_file_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

if __name__ == "__main__":
    print("可视化和数据记录模块测试")
    
    from robot_nav_core import DEFAULT_CONFIG
    
    # 测试数据记录器
    logger_obj = DataLogger(360, DEFAULT_CONFIG.data_log)
    print(f"日志文件: {logger_obj.pose_file_path}, {logger_obj.lidar_file_path}")

