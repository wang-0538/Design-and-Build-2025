"""
================================================================================
智能机器人导航系统 - 地图模块
================================================================================
包含多种地图类型：
1. 基础地图类
2. 蛇形迷宫
3. 随机迷宫
4. 未知探索地图
5. 仿真探索地图
"""

import random
from typing import List, Iterable, Tuple, Set
from robot_nav_core import (
    GridCoord, Obstacle, EnvironmentSettings,
    SnakeMazeSettings, RandomMazeSettings, ExplorationMazeSettings,
    logger
)

# ============================================================================
# 基础地图类
# ============================================================================

class BaseEnvironmentMap:
    """环境地图基类"""
    
    def __init__(self, env_config: EnvironmentSettings, 
                 arena_size: float, cell_size: float):
        self.arena_size = float(arena_size)
        self.cell_size = float(cell_size)
        self.grid_dimension = int(round(self.arena_size / self.cell_size))
        
        self.internal_walls: List[dict] = []  # 内部墙体描述
        self.obstacle_boxes: List[Obstacle] = []  # 障碍物矩形
        
        self.wall_half_width = env_config.wall_thickness
        self.border_width = env_config.border_thickness
    
    def _construct_obstacle_boxes(self) -> None:
        """从墙体描述构建障碍物矩形"""
        boxes: List[Obstacle] = []
        hw = self.wall_half_width
        
        for wall in self.internal_walls:
            if wall.get("orientation") == "H":  # 水平墙
                y_min = wall["y"] - hw
                y_max = wall["y"] + hw
                boxes.append((wall["x_min"], wall["x_max"], y_min, y_max))
            else:  # 垂直墙
                x_min = wall["x"] - hw
                x_max = wall["x"] + hw
                boxes.append((x_min, x_max, wall["y_min"], wall["y_max"]))
        
        # 添加边界
        size = self.arena_size
        border = self.border_width
        boxes.extend([
            (0, border, 0, size),              # 左边界
            (size - border, size, 0, size),    # 右边界
            (0, size, 0, border),              # 下边界
            (0, size, size - border, size)     # 上边界
        ])
        
        self.obstacle_boxes = boxes
    
    def _check_vertical_passage(self, row_from: int, row_to: int, col: int) -> bool:
        """检查垂直方向是否畅通"""
        if abs(row_to - row_from) != 1:
            return True
        
        boundary_y = (min(row_from, row_to) + 1) * self.cell_size
        col_start = col * self.cell_size
        col_end = col_start + self.cell_size
        
        for wall in self.internal_walls:
            if wall.get("orientation") == "H" and abs(wall["y"] - boundary_y) < 1e-9:
                if wall["x_min"] < col_end and wall["x_max"] > col_start:
                    return False
        return True
    
    def _check_horizontal_passage(self, col_from: int, col_to: int, row: int) -> bool:
        """检查水平方向是否畅通"""
        if abs(col_to - col_from) != 1:
            return True
        
        boundary_x = (min(col_from, col_to) + 1) * self.cell_size
        row_start = row * self.cell_size
        row_end = row_start + self.cell_size
        
        for wall in self.internal_walls:
            if wall.get("orientation") == "V" and abs(wall["x"] - boundary_x) < 1e-9:
                if wall["y_min"] < row_end and wall["y_max"] > row_start:
                    return False
        return True
    
    def get_adjacent_cells(self, row: int, col: int) -> Iterable[Tuple[int, int]]:
        """获取相邻可达格子（4连通）"""
        n = self.grid_dimension
        
        # 右
        if col + 1 < n and self._check_horizontal_passage(col, col + 1, row):
            yield row, col + 1
        # 左
        if col - 1 >= 0 and self._check_horizontal_passage(col - 1, col, row):
            yield row, col - 1
        # 上
        if row + 1 < n and self._check_vertical_passage(row, row + 1, col):
            yield row + 1, col
        # 下
        if row - 1 >= 0 and self._check_vertical_passage(row - 1, row, col):
            yield row - 1, col
    
    def has_valid_path(self, start: GridCoord, goal: GridCoord) -> bool:
        """检查从起点到终点是否有有效路径（DFS）"""
        start_row, start_col = start[1], start[0]
        goal_row, goal_col = goal[1], goal[0]
        
        stack = [(start_row, start_col)]
        visited = set(stack)
        
        while stack:
            row, col = stack.pop()
            if (row, col) == (goal_row, goal_col):
                return True
            
            for next_row, next_col in self.get_adjacent_cells(row, col):
                if (next_row, next_col) not in visited:
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))
        
        return False

# ============================================================================
# 蛇形迷宫
# ============================================================================

class SnakeMaze(BaseEnvironmentMap):
    """蛇形走廊迷宫"""
    
    def __init__(self, env_config: EnvironmentSettings, maze_config: SnakeMazeSettings):
        super().__init__(env_config, maze_config.arena_size, maze_config.cell_size)
        
        gap_width = max(1, int(maze_config.gap_size)) * maze_config.cell_size
        walls: List[dict] = []
        
        for i in range(1, int(maze_config.wall_count) + 1):
            y_pos = i * maze_config.cell_size
            if y_pos >= maze_config.arena_size:
                break
            
            if i % 2 == 1:  # 奇数行：左侧墙
                x_min, x_max = 0.0, max(0.0, maze_config.arena_size - gap_width)
            else:  # 偶数行：右侧墙
                x_min, x_max = min(maze_config.arena_size, gap_width), maze_config.arena_size
            
            walls.append({
                "orientation": "H",
                "y": float(y_pos),
                "x_min": float(x_min),
                "x_max": float(x_max)
            })
        
        self.internal_walls = walls
        self._construct_obstacle_boxes()

# ============================================================================
# 随机迷宫
# ============================================================================

class RandomMaze(BaseEnvironmentMap):
    """随机墙体迷宫"""
    
    def __init__(self, env_config: EnvironmentSettings, maze_config: RandomMazeSettings):
        super().__init__(env_config, maze_config.arena_size, maze_config.cell_size)
        
        if maze_config.random_seed is not None:
            random.seed(maze_config.random_seed)
        
        walls: List[dict] = []
        min_length = max(1, int(maze_config.segment_min_cells))
        max_length = max(min_length, int(maze_config.segment_max_cells))
        
        for _ in range(maze_config.wall_count):
            orientation = "H" if random.random() < maze_config.horizontal_bias else "V"
            
            if orientation == "H":
                y_pos = random.randint(1, self.grid_dimension - 1) * maze_config.cell_size
                span_cells = random.randint(min_length, max_length)
                x_min = random.randint(0, self.grid_dimension - 1 - span_cells) * maze_config.cell_size
                x_max = x_min + span_cells * maze_config.cell_size
                walls.append({
                    "orientation": "H",
                    "y": float(y_pos),
                    "x_min": float(x_min),
                    "x_max": float(x_max)
                })
            else:
                x_pos = random.randint(1, self.grid_dimension - 1) * maze_config.cell_size
                span_cells = random.randint(min_length, max_length)
                y_min = random.randint(0, self.grid_dimension - 1 - span_cells) * maze_config.cell_size
                y_max = y_min + span_cells * maze_config.cell_size
                walls.append({
                    "orientation": "V",
                    "x": float(x_pos),
                    "y_min": float(y_min),
                    "y_max": float(y_max)
                })
        
        self.internal_walls = walls
        self._construct_obstacle_boxes()

# ============================================================================
# 未知探索地图
# ============================================================================

class UnknownExplorationMap(BaseEnvironmentMap):
    """未知探索地图（仅边界）"""
    
    def __init__(self, env_config: EnvironmentSettings, maze_config: ExplorationMazeSettings):
        super().__init__(env_config, maze_config.arena_size, maze_config.cell_size)
        
        # 不添加内部墙体，仅边界
        self.internal_walls: List[dict] = []
        self._construct_obstacle_boxes()
        
        logger.info(f"未知探索地图: {maze_config.grid_dimension}x{maze_config.grid_dimension}网格, "
                   f"格子大小={maze_config.cell_size}m")

# ============================================================================
# 仿真探索地图
# ============================================================================

class SimulatedExplorationMap(BaseEnvironmentMap):
    """仿真探索地图（内部有预设墙体但机器人不知道）"""
    
    def __init__(self, env_config: EnvironmentSettings, maze_config: ExplorationMazeSettings):
        super().__init__(env_config, maze_config.arena_size, maze_config.cell_size)
        
        walls: List[dict] = []
        cs = maze_config.cell_size
        
        # 构建简化的4x4全连通迷宫
        # 第2行墙体
        walls.append({"orientation": "V", "x": float(1*cs), "y_min": float(1*cs), "y_max": float(2*cs)})
        walls.append({"orientation": "H", "y": float(1*cs), "x_min": float(1*cs), "x_max": float(2*cs)})
        walls.append({"orientation": "H", "y": float(1*cs), "x_min": float(2*cs), "x_max": float(3*cs)})
        walls.append({"orientation": "V", "x": float(3*cs), "y_min": float(1*cs), "y_max": float(2*cs)})
        
        # 第3行墙体
        walls.append({"orientation": "V", "x": float(1*cs), "y_min": float(2*cs), "y_max": float(3*cs)})
        walls.append({"orientation": "H", "y": float(2*cs), "x_min": float(1*cs), "x_max": float(2*cs)})
        walls.append({"orientation": "H", "y": float(2*cs), "x_min": float(2*cs), "x_max": float(3*cs)})
        walls.append({"orientation": "V", "x": float(3*cs), "y_min": float(2*cs), "y_max": float(3*cs)})
        
        self.internal_walls = walls
        self._construct_obstacle_boxes()
        
        logger.info(f"仿真探索地图: {maze_config.grid_dimension}x{maze_config.grid_dimension}网格(简化全连通), "
                   f"内部墙体数={len(walls)}")

if __name__ == "__main__":
    print("地图模块测试")
    
    from robot_nav_core import DEFAULT_CONFIG
    
    # 测试蛇形迷宫
    snake = SnakeMaze(DEFAULT_CONFIG.environment, DEFAULT_CONFIG.snake_maze)
    print(f"蛇形迷宫: {len(snake.obstacle_boxes)}个障碍物")
    
    # 测试随机迷宫
    DEFAULT_CONFIG.random_maze.random_seed = 42
    rand = RandomMaze(DEFAULT_CONFIG.environment, DEFAULT_CONFIG.random_maze)
    print(f"随机迷宫: {len(rand.obstacle_boxes)}个障碍物")
    
    # 测试未知地图
    unknown = UnknownExplorationMap(DEFAULT_CONFIG.environment, DEFAULT_CONFIG.exploration_maze)
    print(f"未知地图: {len(unknown.obstacle_boxes)}个障碍物（仅边界）")

