# 微电网调度与电池调度研究项目

## 项目概述
本项目研究微电网系统中的调度优化问题，重点关注电池储能系统的调度策略。

## 方法说明
### 优化模型
使用pulp库进行线性规划和混合整数线性规划（MILP）建模：
- LpProblem: 定义优化问题
- LpVariable: 定义决策变量
- lpSum: 对线性表达式求和
- LpMinimize: 指定最小化目标

### 主要功能
1. 负荷曲线生成
   - 根据时间段生成基础负荷
   - 添加随机波动
2. 光伏发电曲线生成
   - 根据日照时间生成基础发电量
   - 添加随机波动
3. 电价时段划分
   - 低谷时段：0.36元/kWh
   - 平段时段：0.69元/kWh
   - 高峰时段：1.23元/kWh
4. 储能系统建模
   - 充放电功率约束
   - SOC更新约束
   - 净负荷非负约束
5. 结果可视化
   - 负荷和光伏曲线
   - 电池功率和SOC曲线
   - 净负荷曲线

## 文件说明
- `microgrid_scheduling.py`: 微电网调度主程序
- `battery-scheduling-debug.py`: 电池调度调试程序
- `scheduling_results.png`: 调度结果可视化
- `scheduling_results-debug.png`: 调试结果可视化

## 运行环境
- Python 3.x
- 依赖库：见requirements.txt

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python microgrid_scheduling.py`
3. 查看结果：结果将保存为png图片
