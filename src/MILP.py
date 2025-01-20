"""
混合整数线性规划（MILP）算法实现

算法说明：
1. 将调度问题建模为混合整数线性规划问题
2. 使用线性规划求解器进行求解
3. 考虑设备启停约束、功率平衡约束等
4. 目标是最小化总运行成本

算法原理：
1. 使用PuLP库建立线性规划模型
2. 定义决策变量：
   - SOC：电池荷电状态
   - Charge：充电功率
   - Discharge：放电功率
   - IsCharging：充电状态指示变量（0/1）
3. 目标函数：最小化总用电成本
4. 约束条件：
   - 电池功率限制
   - SOC状态转移
   - 系统功率平衡
   - 电池容量限制

关键参数：
- time_points：时间点数
- battery['capacity']：电池容量（kWh）
- battery['max_power']：最大充放电功率（kW）
- battery['initial_soc']：初始SOC

依赖库：
- PuLP：用于建立和求解线性规划问题
- numpy：数值计算
- matplotlib：结果可视化

使用方法：
1. 配置config/parameters.json文件
2. 运行：python src/MILP.py
3. 结果将保存为results/scheduling_results.png

输入：
- 负荷曲线：可通过generate_load_curve生成或从CSV读取
- 光伏曲线：可通过generate_pv_curve生成或从CSV读取
- 电价：通过get_electricity_price获取

输出：
- 最优调度方案
- 总运行成本
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value
from utils import generate_load_curve, generate_pv_curve, get_electricity_price, load_curve_from_csv, pv_curve_from_csv

# 读取配置文件
with open('config/parameters.json') as f:
    params = json.load(f)

# 参数设置
time_points = params['time_points']
time = np.linspace(0, 24, time_points)

# 生成或读取负荷和光伏曲线
if params.get('load_csv'):
    load = load_curve_from_csv(r'data\load.csv')
else:
    load = generate_load_curve(time)

if params.get('pv_csv'):  
    pv = pv_curve_from_csv(r'data\pv.csv')
else:
    pv = generate_pv_curve(time)

# 创建MILP问题
prob = LpProblem("Battery_Scheduling", LpMinimize)

# 定义决策变量
battery = params['battery']
soc = LpVariable.dicts("SOC", range(time_points), 0.1 * battery['capacity'], battery['capacity'])
soc[0] = battery['initial_soc'] * battery['capacity']

charge = LpVariable.dicts("Charge", range(time_points), 0, 0.5 * battery['max_power'])
discharge = LpVariable.dicts("Discharge", range(time_points), 0, battery['max_power'])
is_charging = LpVariable.dicts("IsCharging", range(time_points), cat="Binary")

# 目标函数
prob += lpSum([(load[i] - pv[i] - discharge[i] + charge[i]) * 
              get_electricity_price(time[i]) *0.25 for i in range(time_points)])

# 约束条件
for i in range(time_points):
    prob += charge[i] <= battery['max_power'] * is_charging[i]
    prob += discharge[i] <= battery['max_power'] * (1 - is_charging[i])
    
    if i > 0:
        prob += soc[i] == soc[i-1] + charge[i]*0.25*0.95 - discharge[i]*0.25/0.95
    
    prob += load[i] - pv[i] - discharge[i] + charge[i] >= 30
    prob += load[i] - pv[i] - discharge[i] + charge[i] <= 700
    
    if i > 0:
        prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 >= 0.1 * battery['capacity']
        prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 <= battery['capacity']
    
    if i == time_points - 1:
        prob += soc[i] == battery['initial_soc'] * battery['capacity']

# 求解问题
prob.solve()

# 获取优化结果
charge_opt = np.array([value(charge[i]) for i in range(time_points)])
discharge_opt = np.array([value(discharge[i]) for i in range(time_points)])
battery_power = charge_opt - discharge_opt
net_load = load - pv - discharge_opt + charge_opt
soc_values = np.array([value(soc[i]) for i in range(time_points)]) / battery['capacity'] * 100

# 可视化结果
from utils import plot_scheduling_results
plot_scheduling_results(time, load, pv, battery_power, soc_values, net_load, save_path='results/scheduling_results-MILP.png')
pd.DataFrame({'load':net_load,'power':battery_power,'SOC':soc_values}).to_csv('results/net_load.csv')
