# 小车 SLAM 迷宫探索 Group28

> 支持 **真实硬件** 与 **纯模拟** 两种运行方式：扫描 → 建图（占据栅格）→ 寻找前沿 → A* 路径规划 → 执行动作 → 再扫描，直至到达终点或无可探索区域。

## 运行说明

### 主要运行入口
- **运行入口**：`28\28_software\slam_update.py`  
- **切换模拟/真实模式**：打开文件并修改 **第 17 行**  
  - `SIMULATION_MODE = True`  → 启用 **模拟模式**  
  - `SIMULATION_MODE = False` → 启用 **真实小车模式**

### 纯算法模拟版本
- **文件位置**：`28\28_software\final.py`
- **迷宫配置**：提供了 `maze1.json` 和 `maze2.json` 两个复杂的模拟迷宫版本
- **使用方法**：运行前在代码第 935 行指定要加载的迷宫 JSON 文件

---

## 环境依赖

- Python 3.8+
- 建议使用虚拟环境

```bash
pip install numpy matplotlib bleak
