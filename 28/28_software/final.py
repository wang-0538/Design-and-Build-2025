import json
import math
import numpy as np
import matplotlib
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heapq import heappush, heappop
from enum import Enum
from scipy.ndimage import distance_transform_edt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体（你也可以用其他如 'Microsoft YaHei'）
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# --- 组件类 ---
class SegmentMapLoader:
    def __init__(self, json_file_path):
        self.segments, self.start_point, self.map_bounds = [], [0,0], {}
        self.load_map(json_file_path)
    def load_map(self, path):
        try:
            with open(path, 'r') as f: data = json.load(f)
            self.segments, self.start_point = data["segments"], data["start_point"]
            all_pts = [p for seg in self.segments for p in (seg["start"], seg["end"])]
            x_coords, y_coords = [p[0] for p in all_pts], [p[1] for p in all_pts]
            self.map_bounds = {"min_x": min(x_coords), "max_x": max(x_coords), "min_y": min(y_coords), "max_y": max(y_coords)}
            print(f"Map loaded from {path}. Start: {self.start_point}")
        except Exception as e: print(f"Map loading failed: {e}.")
    def get_map_info(self): return {"segments": self.segments, "start_point": self.start_point, "bounds": self.map_bounds}
    def find_exit_point(self):
        bounds = self.map_bounds
        min_x, max_x, min_y, max_y = bounds["min_x"], bounds["max_x"], bounds["min_y"], bounds["max_y"]
        start_x, start_y = self.start_point

        # 检查四条边的缺口
        candidates = []

        # 上边界 y=max_y
        covered = []
        for s in self.segments:
            if abs(s["start"][1] - max_y) < 1e-3 and abs(s["end"][1] - max_y) < 1e-3:
                covered.append(tuple(sorted((s["start"][0], s["end"][0]))))
        covered.sort()
        merged = []
        if covered:
            merged.append(list(covered[0]))
            for start, end in covered[1:]:
                if start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
        last = min_x
        for start, end in merged:
            if start > last:
                candidates.append( {'side': 'top', 'range': (last, start)} )
            last = end
        if last < max_x:
            candidates.append( {'side': 'top', 'range': (last, max_x)} )

        # 下边界 y=min_y
        covered = []
        for s in self.segments:
            if abs(s["start"][1] - min_y) < 1e-3 and abs(s["end"][1] - min_y) < 1e-3:
                covered.append(tuple(sorted((s["start"][0], s["end"][0]))))
        covered.sort()
        merged = []
        if covered:
            merged.append(list(covered[0]))
            for start, end in covered[1:]:
                if start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
        last = min_x
        for start, end in merged:
            if start > last:
                candidates.append( {'side': 'bottom', 'range': (last, start)} )
            last = end
        if last < max_x:
            candidates.append( {'side': 'bottom', 'range': (last, max_x)} )

        # 左边界 x=min_x
        covered = []
        for s in self.segments:
            if abs(s["start"][0] - min_x) < 1e-3 and abs(s["end"][0] - min_x) < 1e-3:
                covered.append(tuple(sorted((s["start"][1], s["end"][1]))))
        covered.sort()
        merged = []
        if covered:
            merged.append(list(covered[0]))
            for start, end in covered[1:]:
                if start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
        last = min_y
        for start, end in merged:
            if start > last:
                candidates.append( {'side': 'left', 'range': (last, start)} )
            last = end
        if last < max_y:
            candidates.append( {'side': 'left', 'range': (last, max_y)} )

        # 右边界 x=max_x
        covered = []
        for s in self.segments:
            if abs(s["start"][0] - max_x) < 1e-3 and abs(s["end"][0] - max_x) < 1e-3:
                covered.append(tuple(sorted((s["start"][1], s["end"][1]))))
        covered.sort()
        merged = []
        if covered:
            merged.append(list(covered[0]))
            for start, end in covered[1:]:
                if start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
        last = min_y
        for start, end in merged:
            if start > last:
                candidates.append( {'side': 'right', 'range': (last, start)} )
            last = end
        if last < max_y:
            candidates.append( {'side': 'right', 'range': (last, max_y)} )

        # 过滤掉包含起点的缺口
        def in_range(val, r):
            return r[0] - 1e-6 <= val <= r[1] + 1e-6

        filtered = []
        for c in candidates:
            if c['side'] == 'top' and abs(start_y - max_y) < 1e-3 and in_range(start_x, c['range']):
                continue
            if c['side'] == 'bottom' and abs(start_y - min_y) < 1e-3 and in_range(start_x, c['range']):
                continue
            if c['side'] == 'left' and abs(start_x - min_x) < 1e-3 and in_range(start_y, c['range']):
                continue
            if c['side'] == 'right' and abs(start_x - max_x) < 1e-3 and in_range(start_y, c['range']):
                continue
            filtered.append(c)

        if not filtered:
            # 退化：没有合法出口，默认右上角
            print("Warning: 无合法出口缺口，默认右上角")
            return [bounds['max_x']-1, bounds['max_y']-1]

        # 选择最大宽度的出口
        best = max(filtered, key=lambda c: c['range'][1] - c['range'][0])
        # 按边选出口点
        if best['side'] == 'top':
            x = (best['range'][0] + best['range'][1]) / 2
            y = max_y
        elif best['side'] == 'bottom':
            x = (best['range'][0] + best['range'][1]) / 2
            y = min_y
        elif best['side'] == 'left':
            x = min_x
            y = (best['range'][0] + best['range'][1]) / 2
        elif best['side'] == 'right':
            x = max_x
            y = (best['range'][0] + best['range'][1]) / 2
        exit_point = [x, y]
        print(f"Exit identified at: {exit_point}")
        return exit_point



class CustomLiDAR:
    def __init__(self, max_range=20.0, num_rays=500, noise_std=0.002):
        self.max_range, self.num_rays, self.noise_std = max_range, num_rays, noise_std
        self.angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    def scan(self, robot_pos, robot_angle, all_walls):
        distances, angles = [], []
        for ray_angle in self.angles:
            global_angle, min_distance = robot_angle + ray_angle, self.max_range
            for wall in all_walls:
                distance = self._ray_wall_intersection(robot_pos, global_angle, wall)
                if distance is not None and distance < min_distance: min_distance = distance
            min_distance += np.random.normal(0, self.noise_std)
            distances.append(max(0.01, min(self.max_range, min_distance)))
            angles.append(global_angle)
        return distances, angles
    def _ray_wall_intersection(self, robot_pos, angle, wall):
        x1, y1 = robot_pos; x2, y2 = x1 + np.cos(angle) * self.max_range, y1 + np.sin(angle) * self.max_range
        x3, y3 = wall['start']; x4, y4 = wall['end']
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t <= 1 and 0 <= u <= 1: return t * self.max_range
        return None

class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.1):
        self.resolution, self.origin = resolution, (x_min, y_min)
        self.grid_width, self.grid_height = int((x_max - x_min) / resolution), int((y_max - y_min) / resolution)
        self.log_odds = np.zeros((self.grid_height, self.grid_width), dtype=float)
        self.log_odds_free, self.log_odds_occ = -0.2, 1.5
        self.log_odds_max, self.log_odds_min = 6.0, -6.0
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy
    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin[0] + self.resolution/2
        wy = gy * self.resolution + self.origin[1] + self.resolution/2
        return wx, wy
    def is_in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_width and 0 <= gy < self.grid_height
    def update_from_lidar(self, robot_pos, distances, angles):
        rx, ry = self.world_to_grid(robot_pos[0], robot_pos[1])
        for dist, angle in zip(distances, angles):
            ex, ey = robot_pos[0] + dist * np.cos(angle), robot_pos[1] + dist * np.sin(angle)
            end_gx, end_gy = self.world_to_grid(ex, ey)
            for gx, gy in self._bresenham_line(rx, ry, end_gx, end_gy)[:-1]:
                if self.is_in_bounds(gx, gy):
                    self.log_odds[gy, gx] = max(self.log_odds_min, self.log_odds[gy, gx] + self.log_odds_free)
            if dist < 0.98 * self.max_range and self.is_in_bounds(end_gx, end_gy):
                self.log_odds[end_gy, end_gx] = min(self.log_odds_max, self.log_odds[end_gy, end_gx] + self.log_odds_occ)
    @property
    def max_range(self):
        return 8.0
    def get_grid_for_planning(self):
        grid = np.zeros_like(self.log_odds, dtype=np.int8)
        grid[self.log_odds > 0.1] = -1  # 障碍
        grid[self.log_odds < -0.1] = 1  # 自由
        return grid
    def _bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy
        return points

class AStar:
    def __init__(self, occupancy_grid): 
        self.grid_map = occupancy_grid

    def plan(self, start, goal, planning_grid):
        grid_copy = planning_grid.copy()
        obstacle_mask = grid_copy != 1
        
        # --- 修正：使用局部变量，避免覆盖 OccupancyGrid 的属性 ---
        distance_cost_grid = distance_transform_edt(np.logical_not(obstacle_mask))
        
        # 将距离转换为成本，距离越远成本越低
        max_dist = np.max(distance_cost_grid)
        distance_cost_grid[distance_cost_grid < 1] = 1 
        
        # 离墙越近，成本越高。这里的100是惩罚系数，可以调整
        cost_map = 100.0 / distance_cost_grid
        cost_map[obstacle_mask] = float('inf') # 障碍物成本无穷大


        start_grid = self.grid_map.world_to_grid(*start)
        goal_grid = self.grid_map.world_to_grid(*goal)
        
        if not self.grid_map.is_in_bounds(*start_grid) or not planning_grid[start_grid[1], start_grid[0]] == 1: 
            return []
        if not self.grid_map.is_in_bounds(*goal_grid) or not planning_grid[goal_grid[1], goal_grid[0]] == 1: 
            return []

        q, came_from, cost_so_far = [(0, start_grid)], {start_grid: None}, {start_grid: 0}
        
        while q:
            _, current = heappop(q)

            if current == goal_grid:
                path = []
                while current:
                    path.append(self.grid_map.grid_to_world(*current))
                    current = came_from[current]
                return path[::-1]

            (cx, cy) = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (cx + dx, cy + dy)
                if self.grid_map.is_in_bounds(*neighbor) and planning_grid[neighbor[1], neighbor[0]] == 1:
                    move_cost = math.sqrt(dx**2 + dy**2)
                    
                    # 修正: 确保使用整数索引访问成本图
                    nx, ny = int(neighbor[0]), int(neighbor[1])
                    new_cost = cost_so_far[current] + move_cost + cost_map[ny, nx]

                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        came_from[neighbor] = current
                        priority = new_cost + np.linalg.norm(np.array(neighbor) - np.array(goal_grid))
                        heappush(q, (priority, neighbor))
        return []

class FrontierExplorer:
    def __init__(self, occupancy_grid): self.grid_map = occupancy_grid
    def find_frontiers(self, planning_grid):
        frontiers = []
        for y in range(1, self.grid_map.grid_height - 1):
            for x in range(1, self.grid_map.grid_width - 1):
                if planning_grid[y, x] == 1:
                    is_frontier = False
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        if planning_grid[y + dy, x + dx] == 0:
                            is_frontier = True; break
                    if is_frontier:
                        frontiers.append(self.grid_map.grid_to_world(x, y))
        return self._cluster_frontiers(frontiers)
    def _cluster_frontiers(self, frontiers, dist_thresh=2.0):
        if not frontiers: return []
        clusters, visited = [], set()
        for i, frontier in enumerate(frontiers):
            if i in visited: continue
            visited.add(i); new_cluster, q = [frontier], [i]
            while q:
                curr_idx = q.pop(0)
                for j, other in enumerate(frontiers):
                    if j not in visited and np.linalg.norm(np.array(frontiers[curr_idx]) - np.array(other)) < dist_thresh:
                        visited.add(j); new_cluster.append(other); q.append(j)
            if len(new_cluster) > 3: clusters.append(np.mean(new_cluster, axis=0))
        return [tuple(c) for c in clusters]

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-1.5, 1.5)):
        self.Kp, self.Ki, self.Kd, self.setpoint, self.output_limits = Kp, Ki, Kd, setpoint, output_limits
        self._last_error, self._integral = 0.0, 0.0
    def __call__(self, measurement, dt):
        error = self.setpoint - measurement
        self._integral += error * dt; self._integral = np.clip(self._integral, -1, 1)
        derivative = (error - self._last_error) / dt if dt > 0 else 0
        self._last_error = error
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return np.clip(output, self.output_limits[0], self.output_limits[1])

class MissionState(Enum):
    EXPLORING = 1
    GOING_TO_START = 2
    FINAL_RUN = 3
    FINISHED = 4
    STUCK = 5
    WANDERING=6

class MazeExplorer:
    def __init__(self, maze_data):
        self.paused = False  # 控制是否暂停
        self.manual_mode = False  # 是否为手动模式
        self.map_info, self.start_pos = maze_data, tuple(maze_data['start_point'])
        self.exit_point_coords = self.map_info['find_exit_point']()
        self.pos, self.angle = list(self.start_pos), np.pi / 2
        bounds = self.get_maze_bounds()
        self.all_walls = maze_data['segments'] + self.create_boundary_walls(bounds)
        self.occupancy_grid = OccupancyGrid(bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y'])
        self.lidar, self.astar = CustomLiDAR(), AStar(self.occupancy_grid)
        self.frontier_explorer = FrontierExplorer(self.occupancy_grid)
        self.angular_pid = PIDController(Kp=4.0, Ki=0.02, Kd=0.5, output_limits=(-6.0, 6.0))
        self.PHYSICAL_RADIUS, self.PLANNING_RADIUS = 0.22, 0.05
        self.MAX_SPEED = 3.5
        self.MIN_SPEED = 1.0
        self.LOOKAHEAD_DISTANCE = 1.0
        self.path, self.path_index = [], 0
        self.mode = MissionState.EXPLORING
        self.mode_before_stuck = MissionState.EXPLORING
        self.recovery_counter = 0
        self.discovered_exit_pos = None
        self.trajectory = [self.pos[:]]
        self.DT = 0.05
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.setup_plots()
        # 一键返航按钮设置
        self.return_button_ax = self.fig.add_axes([0.85, 0.90, 0.10, 0.06])  # [left, bottom, width, height]
        self.return_button = Button(self.return_button_ax, "一键返航", color='lightgray', hovercolor='red')
        self.return_button.on_clicked(self.manual_return_home)
        self.manual_return_triggered = False  # 防止重复切换
        # 启动按钮
        self.start_button_ax = self.fig.add_axes([0.63, 0.90, 0.10, 0.06])
        self.start_button = Button(self.start_button_ax, "启动", color='lightgreen', hovercolor='lime')
        self.start_button.on_clicked(self.start_motion)
        # 暂停按钮
        self.pause_button_ax = self.fig.add_axes([0.52, 0.90, 0.10, 0.06])
        self.pause_button = Button(self.pause_button_ax, "暂停", color='lightyellow', hovercolor='orange')
        self.pause_button.on_clicked(self.pause_motion)
        # 手动模式按钮
        self.manual_button_ax = self.fig.add_axes([0.74, 0.90, 0.10, 0.06])
        self.manual_button = Button(self.manual_button_ax, "手动模式", color='lightblue', hovercolor='deepskyblue')
        self.manual_button.on_clicked(self.toggle_manual_mode)


        self.dynamic_artists = []
        self.frame_skip_counter = 0
        self.slam_map_artist = None
        self.robot_circle = None
        self.robot_arrow = None
        self.robot_slam_dot = None

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if not self.manual_mode or self.paused:
            return
        v = 0
        omega = 0
        step_size = 0.2     # 移动距离
        angle_step = np.pi / 16  # 旋转步进角
        if event.key.lower() == 'w':
            # 向前
            self.pos[0] += step_size * np.cos(self.angle)
            self.pos[1] += step_size * np.sin(self.angle)
        elif event.key.lower() == 's':
            # 向后
            self.pos[0] -= step_size * np.cos(self.angle)
            self.pos[1] -= step_size * np.sin(self.angle)
        elif event.key.lower() == 'a':
            # 左转
            self.angle += angle_step
            self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))
        elif event.key.lower() == 'd':
            # 右转
            self.angle -= angle_step
            self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))
        else:
            return
        self.trajectory.append(self.pos[:])
        self.visualize()


    def get_inflated_grid(self, grid):
        inflated_grid = np.copy(grid)
        inflation_radius_cells = math.ceil(self.PLANNING_RADIUS / self.occupancy_grid.resolution)
        obstacle_indices = np.where(grid == -1)
        obstacle_coords = list(zip(obstacle_indices[1], obstacle_indices[0]))
        if not obstacle_coords: return inflated_grid
        for ox, oy in obstacle_coords:
            for dx in range(-inflation_radius_cells, inflation_radius_cells + 1):
                for dy in range(-inflation_radius_cells, inflation_radius_cells + 1):
                    if dx**2 + dy**2 <= inflation_radius_cells**2:
                        nx, ny = ox + dx, oy + dy
                        if self.occupancy_grid.is_in_bounds(nx, ny): inflated_grid[ny, nx] = -1
        return inflated_grid

    def create_boundary_walls(self, bounds):
        # 自动为所有出口缺口留洞，其余全部封闭
        min_x, max_x, min_y, max_y = bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']

        def get_openings(side):
            result = []
            if side == "top":
                covered = []
                for s in self.map_info["segments"]:
                    if abs(s["start"][1] - max_y) < 1e-3 and abs(s["end"][1] - max_y) < 1e-3:
                        covered.append(tuple(sorted((s["start"][0], s["end"][0]))))
                covered.sort()
                merged = []
                if covered:
                    merged.append(list(covered[0]))
                    for start, end in covered[1:]:
                        if start > merged[-1][1]:
                            merged.append([start, end])
                        else:
                            merged[-1][1] = max(merged[-1][1], end)
                last = min_x
                for start, end in merged:
                    if start > last:
                        result.append((last, start))
                    last = end
                if last < max_x:
                    result.append((last, max_x))
                return result
            elif side == "bottom":
                covered = []
                for s in self.map_info["segments"]:
                    if abs(s["start"][1] - min_y) < 1e-3 and abs(s["end"][1] - min_y) < 1e-3:
                        covered.append(tuple(sorted((s["start"][0], s["end"][0]))))
                covered.sort()
                merged = []
                if covered:
                    merged.append(list(covered[0]))
                    for start, end in covered[1:]:
                        if start > merged[-1][1]:
                            merged.append([start, end])
                        else:
                            merged[-1][1] = max(merged[-1][1], end)
                last = min_x
                for start, end in merged:
                    if start > last:
                        result.append((last, start))
                    last = end
                if last < max_x:
                    result.append((last, max_x))
                return result
            elif side == "left":
                covered = []
                for s in self.map_info["segments"]:
                    if abs(s["start"][0] - min_x) < 1e-3 and abs(s["end"][0] - min_x) < 1e-3:
                        covered.append(tuple(sorted((s["start"][1], s["end"][1]))))
                covered.sort()
                merged = []
                if covered:
                    merged.append(list(covered[0]))
                    for start, end in covered[1:]:
                        if start > merged[-1][1]:
                            merged.append([start, end])
                        else:
                            merged[-1][1] = max(merged[-1][1], end)
                last = min_y
                for start, end in merged:
                    if start > last:
                        result.append((last, start))
                    last = end
                if last < max_y:
                    result.append((last, max_y))
                return result
            elif side == "right":
                covered = []
                for s in self.map_info["segments"]:
                    if abs(s["start"][0] - max_x) < 1e-3 and abs(s["end"][0] - max_x) < 1e-3:
                        covered.append(tuple(sorted((s["start"][1], s["end"][1]))))
                covered.sort()
                merged = []
                if covered:
                    merged.append(list(covered[0]))
                    for start, end in covered[1:]:
                        if start > merged[-1][1]:
                            merged.append([start, end])
                        else:
                            merged[-1][1] = max(merged[-1][1], end)
                last = min_y
                for start, end in merged:
                    if start > last:
                        result.append((last, start))
                    last = end
                if last < max_y:
                    result.append((last, max_y))
                return result

        walls = []
        # 下边界
        bottom_openings = get_openings("bottom")
        if bottom_openings:
            for i, (l, r) in enumerate(bottom_openings):
                walls.append({'start': [l, min_y], 'end': [r, min_y]})
        else:
            walls.append({'start': [min_x, min_y], 'end': [max_x, min_y]})

        # 上边界
        top_openings = get_openings("top")
        if top_openings:
            for i, (l, r) in enumerate(top_openings):
                walls.append({'start': [l, max_y], 'end': [r, max_y]})
        else:
            walls.append({'start': [min_x, max_y], 'end': [max_x, max_y]})

        # 左边界
        left_openings = get_openings("left")
        if left_openings:
            for i, (l, r) in enumerate(left_openings):
                walls.append({'start': [min_x, l], 'end': [min_x, r]})
        else:
            walls.append({'start': [min_x, min_y], 'end': [min_x, max_y]})

        # 右边界
        right_openings = get_openings("right")
        if right_openings:
            for i, (l, r) in enumerate(right_openings):
                walls.append({'start': [max_x, l], 'end': [max_x, r]})
        else:
            walls.append({'start': [max_x, min_y], 'end': [max_x, max_y]})

        return walls



    def get_maze_bounds(self): return {
        "min_x": min(p[0] for w in self.map_info["segments"] for p in (w["start"], w["end"])),
        "max_x": max(p[0] for w in self.map_info["segments"] for p in (w["start"], w["end"])),
        "min_y": min(p[1] for w in self.map_info["segments"] for p in (w["start"], w["end"])),
        "max_y": max(p[1] for w in self.map_info["segments"] for p in (w["start"], w["end"]))}

    def setup_plots(self):
        self.fig.suptitle('迷宫探索大师 v2.0（By Group 28）', fontsize=20, fontweight='bold', y=1.0)
        self.ax1.set_title('小车视角实景导航'); self.ax2.set_title('SLAM实时反馈')
        for ax in [self.ax1, self.ax2]: ax.set_aspect('equal'); ax.grid(True)
        for wall in self.all_walls: self.ax1.plot([wall['start'][0], wall['end'][0]], [wall['start'][1], wall['end'][1]], 'k-')
        self.ax1.plot(self.start_pos[0], self.start_pos[1], 'go', ms=10, label='起点'); self.ax1.legend()
        self.ax1.set_facecolor('#E3F2FD')  # 左侧淡蓝
        self.ax2.set_facecolor('#FFFDE7')  # 右侧米黄
        self.fig.patch.set_facecolor('#ECE9F7')  # 整个窗口紫白

    def step(self, frame):
        if self.discovered_exit_pos is None:
            print("[DEBUG] detected NoneType! Auto-reset.")
            self.discovered_exit_pos = self.exit_point_coords
        if self.paused:
            self.visualize()   # 保持可视化更新
            return             # 跳过运动和决策部分
        if self.manual_mode:
            # 手动模式下自动跳过自动运动
            self.visualize()
            return

        if self.mode == MissionState.STUCK:
            if self.recovery_counter > 0:
                self.pos[0] += -self.MIN_SPEED * np.cos(self.angle) * self.DT
                self.pos[1] += -self.MIN_SPEED * np.sin(self.angle) * self.DT
                self.recovery_counter -= 1
                self.trajectory.append(self.pos[:])
                self.visualize()
                return 
            else:
                print("Recovery maneuver complete. Attempting to replan.")
                self.angle += np.random.uniform(-np.pi / 4, np.pi / 4) # 增加随机转向
                self.mode = self.mode_before_stuck
                self.path = []

        if self.mode == MissionState.FINISHED:
            self.anim.event_source.stop()
            return
        distances, angles = self.lidar.scan(self.pos, self.angle, self.all_walls)
        self.occupancy_grid.update_from_lidar(self.pos, distances, angles)
        is_current_path_done = not self.path or self.path_index >= len(self.path)
        if self.mode == MissionState.EXPLORING and self.discovered_exit_pos is None:
            inflated_grid = self.get_inflated_grid(self.occupancy_grid.get_grid_for_planning())
            path_to_exit = self.astar.plan(self.pos, self.exit_point_coords, inflated_grid)
            if path_to_exit:
                print(">>> 出口已发现! 正在返回起点...")
                self.mode = MissionState.GOING_TO_START
                self.discovered_exit_pos = self.exit_point_coords
                is_current_path_done = True
        if self.mode == MissionState.GOING_TO_START:
            if np.linalg.norm(np.array(self.pos) - np.array(self.start_pos)) < 0.5:
                print(">>> 已返回起点! 开始最终冲刺...")
                self.mode = MissionState.FINAL_RUN
                self.manual_return_triggered = False   # 允许再次返航
                is_current_path_done = True
        if self.mode == MissionState.FINAL_RUN:
            if np.linalg.norm(np.array(self.pos) - np.array(self.discovered_exit_pos)) < 0.5:
                print(">>> 成功抵达出口！任务完成！")
                self.mode = MissionState.FINISHED
                return
        if is_current_path_done:
            slam_grid = self.occupancy_grid.get_grid_for_planning()
            inflated_grid = self.get_inflated_grid(slam_grid)
            new_path = []
            if self.mode == MissionState.EXPLORING:
                frontiers = self.frontier_explorer.find_frontiers(slam_grid)
                frontier_found = False
                if frontiers:
                    costs = []
                    for f in frontiers:
                        dist = np.linalg.norm(np.array(f) - np.array(self.pos))
                        angle_to_f = np.arctan2(f[1] - self.pos[1], f[0] - self.pos[0])
                        angle_diff = abs(np.arctan2(np.sin(angle_to_f - self.angle), np.cos(angle_to_f - self.angle)))
                        turn_penalty = 3.0 * (1 - np.cos(angle_diff))
                        costs.append({'frontier': f, 'cost': dist + turn_penalty})
                    costs.sort(key=lambda x: x['cost'])
                    for item in costs:
                        path_to_frontier = self.astar.plan(self.pos, item['frontier'], inflated_grid)
                        if path_to_frontier:
                            new_path = path_to_frontier
                            frontier_found = True
                            break
                if not frontier_found and frontiers:
                    print("所有前沿点在膨胀地图不可达，尝试降级用未膨胀地图...")
                    non_inflated_grid = self.occupancy_grid.get_grid_for_planning()
                    for item in frontiers:
                        path_to_frontier = self.astar.plan(self.pos, item, non_inflated_grid)
                        if path_to_frontier:
                            new_path = path_to_frontier
                            frontier_found = True
                            print("降级地图成功规划到前沿点！")
                            break
                
                if not frontier_found:
                    print("### 找不到可达的前沿点，进入WANDERING盲走模式！")
                    self.mode = MissionState.WANDERING
                    self.path = []
                    self.path_index = 0
                    return  # 本帧不再向下执行，直接进入WANDERING逻辑

            elif self.mode == MissionState.GOING_TO_START:
                new_path = self.astar.plan(self.pos, self.start_pos, inflated_grid)
                if not new_path:
                    print("返航膨胀A*失败，降级用未膨胀地图再试...")
                    non_inflated_grid = self.occupancy_grid.get_grid_for_planning()
                    new_path = self.astar.plan(self.pos, self.start_pos, non_inflated_grid)
                if not new_path:
                    print("返航路径规划彻底失败！尝试寻找离起点最近的可达点...")
                    non_inflated_grid = self.occupancy_grid.get_grid_for_planning()
                    nearest_goal = self.find_nearest_reachable_goal(self.start_pos, non_inflated_grid)
                    if nearest_goal:
                        new_path = self.astar.plan(self.pos, nearest_goal, non_inflated_grid)
                        print("实际返航目标点：", nearest_goal)

            elif self.mode == MissionState.FINAL_RUN:
                new_path = self.astar.plan(self.pos, self.discovered_exit_pos, inflated_grid)
                if not new_path:
                    print("冲刺膨胀A*失败，降级用未膨胀地图再试...")
                    non_inflated_grid = self.occupancy_grid.get_grid_for_planning()
                    new_path = self.astar.plan(self.pos, self.discovered_exit_pos, non_inflated_grid)
                    print("冲刺用未膨胀地图路径长度：", len(new_path))
                    # 如果还失败，尝试寻找离出口最近的自由格
                    if not new_path:
                        print("出口不可达，寻找最近可达自由格作为终点...")
                        nearest_goal = self.find_nearest_reachable_goal(self.discovered_exit_pos, non_inflated_grid)
                        if nearest_goal is not None:
                            new_path = self.astar.plan(self.pos, nearest_goal, non_inflated_grid)
                            print("实际冲刺目标点：", nearest_goal)
                            self.discovered_exit_pos = nearest_goal
                        else:
                            print("找不到可达自由格，继续以出口为目标")
                            self.discovered_exit_pos = self.exit_point_coords
            self.path = new_path
            self.path_index = 0
        
        v, omega = 0, 0
        if self.path and self.path_index < len(self.path):
            # --- 纯追踪（Pure Pursuit）路径跟踪逻辑 ---
            
            # 1. 找到路径上离机器人最近的点，更新我们的路径跟随进度
            min_dist_sq = float('inf')
            closest_idx = self.path_index
            for i in range(self.path_index, len(self.path)):
                dist_sq = (self.path[i][0] - self.pos[0])**2 + (self.path[i][1] - self.pos[1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_idx = i
            self.path_index = closest_idx

            # 2. 从当前位置向前搜索，找到"前瞻点"
            lookahead_point = None
            search_idx = self.path_index
            while search_idx < len(self.path):
                dist_from_robot_sq = (self.path[search_idx][0] - self.pos[0])**2 + (self.path[search_idx][1] - self.pos[1])**2
                if dist_from_robot_sq > self.LOOKAHEAD_DISTANCE**2:
                    lookahead_point = self.path[search_idx]
                    break
                search_idx += 1
            
            # 如果没找到（快到终点了），就以前方最后一个点为目标
            if not lookahead_point:
                lookahead_point = self.path[-1]

            # 3. 检查是否到达当前路径的终点
            if np.linalg.norm(np.array(self.pos) - np.array(self.path[-1])) < self.PHYSICAL_RADIUS:
                self.path = []      # 到达终点，清空路径
                self.path_index = 0

            # 4. 如果路径仍然有效，则计算速度和转向
            if self.path:
                target = lookahead_point
                angle_to_target = np.arctan2(target[1] - self.pos[1], target[0] - self.pos[0])
                angle_diff = np.arctan2(np.sin(angle_to_target - self.angle), np.cos(angle_to_target - self.angle))
                
                omega = self.angular_pid(-angle_diff, self.DT)

                # 动态速度控制：朝向目标越直，速度越快
                speed_factor = max(0, np.cos(angle_diff))
                v = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * (speed_factor ** 2)

        # --- 新：障碍物探测安全检查 ---
        self.angle += omega * self.DT
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        if v > 1e-6:
            next_pos_x = self.pos[0] + v * np.cos(self.angle) * self.DT
            next_pos_y = self.pos[1] + v * np.sin(self.angle) * self.DT
            
            is_safe = True
            grid = self.occupancy_grid.get_grid_for_planning()
            
            # 从机器人中心到下一个位置画一条线，检查沿途是否有障碍
            start_gx, start_gy = self.occupancy_grid.world_to_grid(self.pos[0], self.pos[1])
            end_gx, end_gy = self.occupancy_grid.world_to_grid(next_pos_x, next_pos_y)
            line_points = self.occupancy_grid._bresenham_line(start_gx, start_gy, end_gx, end_gy)
            
            for gx, gy in line_points:
                 if not self.occupancy_grid.is_in_bounds(gx, gy) or grid[gy, gx] == -1:
                    is_safe = False
                    break
            
            if is_safe:
                self.pos[0] = next_pos_x
                self.pos[1] = next_pos_y
            else:
                print("!!! Collision predicted. Initiating recovery. !!!")
                self.mode_before_stuck = self.mode
                self.mode = MissionState.STUCK
                self.recovery_counter = 10
        
        self.trajectory.append(self.pos[:])
        self.frame_skip_counter += 1
        self.visualize()
        # ----- 新增：WANDERING游荡逻辑 -----
        if self.mode == MissionState.WANDERING:
            # 每隔若干帧检测前沿点是否有可达的
            if self.frame_skip_counter % 5 == 0:
                slam_grid = self.occupancy_grid.get_grid_for_planning()
                inflated_grid = self.get_inflated_grid(slam_grid)
                frontiers = self.frontier_explorer.find_frontiers(slam_grid)
                found = False
                for f in frontiers:
                    if self.astar.plan(self.pos, f, inflated_grid):
                        print("发现新的可达前沿点，立即切回EXPLORING！")
                        self.path = self.astar.plan(self.pos, f, inflated_grid)
                        self.path_index = 0
                        self.mode = MissionState.EXPLORING
                        found = True
                        break
                if found:
                    # 有前沿点可达，直接return等待下一帧用正常流程走
                    self.visualize()
                    return
                if not frontiers:
                    print("### 所有区域已遍历，无前沿点，开始返航！")
                    self.mode = MissionState.GOING_TO_START
                    self.path = []
                    self.path_index = 0
                    self.visualize()
                    return
            # 没有可达前沿点，继续盲走
            v = self.MAX_SPEED
            next_pos_x = self.pos[0] + v * np.cos(self.angle) * self.DT
            next_pos_y = self.pos[1] + v * np.sin(self.angle) * self.DT

            grid = self.occupancy_grid.get_grid_for_planning()
            start_gx, start_gy = self.occupancy_grid.world_to_grid(self.pos[0], self.pos[1])
            end_gx, end_gy = self.occupancy_grid.world_to_grid(next_pos_x, next_pos_y)
            is_safe = True
            for gx, gy in self.occupancy_grid._bresenham_line(start_gx, start_gy, end_gx, end_gy):
                if not self.occupancy_grid.is_in_bounds(gx, gy) or grid[gy, gx] == -1:
                    is_safe = False
                    break
            if is_safe:
                self.pos[0] = next_pos_x
                self.pos[1] = next_pos_y
            else:
                print("!!! WANDERING碰撞，进入STUCK恢复 !!!")
                self.mode_before_stuck = MissionState.WANDERING
                self.mode = MissionState.STUCK
                self.recovery_counter = 10
            self.trajectory.append(self.pos[:])
            self.frame_skip_counter += 1
            self.visualize()
            return
        # ----- WANDERING游荡逻辑结束 -----


    def visualize(self):
        for artist in self.dynamic_artists:
            try: artist.remove()
            except: pass
        self.dynamic_artists = []
        if len(self.trajectory) > 10:
            traj_to_draw = self.trajectory[::3]
            self.dynamic_artists.extend(self.ax1.plot(*zip(*traj_to_draw), color='purple', lw=2, alpha=0.7))
        if self.robot_circle:
            try: self.robot_circle.remove()
            except: pass
        if self.robot_arrow:
            try: self.robot_arrow.remove()
            except: pass
        self.robot_circle = plt.Circle(self.pos, self.PHYSICAL_RADIUS, color='r', alpha=0.8)
        self.ax1.add_patch(self.robot_circle)
        self.robot_arrow = self.ax1.arrow(self.pos[0], self.pos[1],
                                         0.4*np.cos(self.angle), 0.4*np.sin(self.angle),
                                         head_width=0.2, fc='r', ec='r')
        if self.discovered_exit_pos and self.mode == MissionState.FINAL_RUN:
            self.dynamic_artists.append(self.ax1.plot(self.discovered_exit_pos[0], self.discovered_exit_pos[1], 'r*', ms=15)[0])
        if self.path and self.path_index < len(self.path):
            path_to_draw = self.path[self.path_index:]
            path_to_draw.insert(0, self.pos)
            self.dynamic_artists.extend(self.ax1.plot(*zip(*path_to_draw), color='dodgerblue', linestyle='--', lw=2, alpha=0.9))
        grid = self.occupancy_grid.get_grid_for_planning()
        color_grid = np.full((*grid.shape, 3), [0.5, 0.5, 0.5])
        color_grid[grid == 1], color_grid[grid == -1] = [1, 1, 1], [0, 0, 0]
        og = self.occupancy_grid
        extent = [og.origin[0], og.origin[0] + og.grid_width * og.resolution,
                 og.origin[1], og.origin[1] + og.grid_height * og.resolution]
        if hasattr(self, 'slam_map_artist') and self.slam_map_artist:
            try: self.slam_map_artist.remove()
            except: pass
        self.slam_map_artist = self.ax2.imshow(color_grid,
                                              origin='lower', extent=extent)
        if hasattr(self, 'robot_slam_dot'):
            try: self.robot_slam_dot[0].remove()
            except: pass
        self.robot_slam_dot = self.ax2.plot(self.pos[0], self.pos[1], 'ro', ms=8)
        if self.mode == MissionState.EXPLORING and self.frame_skip_counter % 3 == 0:
            frontiers = self.frontier_explorer.find_frontiers(self.occupancy_grid.get_grid_for_planning())
            if frontiers:
                self.dynamic_artists.extend(self.ax2.plot(*zip(*frontiers), 'y*', ms=6))

    def run(self):
        self.anim = animation.FuncAnimation(self.fig, self.step, frames=4000,
                                          interval=20, repeat=False)
        plt.show()
    def find_nearest_reachable_goal(self, goal_pos, planning_grid):
        gx, gy = self.occupancy_grid.world_to_grid(goal_pos[0], goal_pos[1])
        min_dist = float('inf')
        nearest = None
        for y in range(self.occupancy_grid.grid_height):
            for x in range(self.occupancy_grid.grid_width):
                if planning_grid[y, x] == 1:  # 自由格
                    dist = (x - gx)**2 + (y - gy)**2
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)
        if nearest is not None:
            return self.occupancy_grid.grid_to_world(*nearest)
        else:
            return goal_pos  # 退化情况，直接返回原目标
    def manual_return_home(self, event):
        if self.mode in [MissionState.EXPLORING, MissionState.WANDERING] and not self.manual_return_triggered:
            print(">>> [一键返航] 终止探索，立即返航！")
            self.mode = MissionState.GOING_TO_START
            self.path = []
            self.path_index = 0
            self.manual_return_triggered = True
    def pause_motion(self, event):
        self.paused = True
        print(">>> [暂停] 运动已暂停。")

    def start_motion(self, event):
        self.paused = False
        print(">>> [启动] 运动恢复。")
    def toggle_manual_mode(self, event):
        self.manual_mode = not self.manual_mode
        print(">>> 手动模式" + ("开启" if self.manual_mode else "关闭"))


def main():
    try:
        map_file = "maze1.json"
        map_loader = SegmentMapLoader(map_file)
        maze_data = map_loader.get_map_info()
        maze_data['find_exit_point'] = map_loader.find_exit_point
        explorer = MazeExplorer(maze_data)
        explorer.run()
    except FileNotFoundError:
        print(f"错误: 地图文件 '{map_file}' 未找到！请确保该文件与脚本在同一目录下。")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
