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
import numpy as np  # 导入numpy库用于数值计算
import pandas as pd  # 导入pandas库用于数据处理
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import json  # 导入json库用于读取配置文件
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value  # 从公共模块导入线性规划类
from utils import generate_load_curve, generate_pv_curve, get_electricity_price, load_curve_from_csv, pv_curve_from_csv  # 导入自定义工具函数

# 读取配置文件
with open(r'config\parameters.json') as f:
    params = json.load(f)  # 加载JSON格式的配置文件

# 参数设置
time_points = params['time_points']  # 获取时间点数
time = np.linspace(0, 24, time_points)  # 生成时间序列，从0到24小时

# 生成或读取负荷和光伏曲线
if params.get('load_csv'):  # 如果配置中指定了从CSV读取负荷曲线
    load = load_curve_from_csv(r'data\load.csv')  # 从CSV文件读取负荷数据
else:  # 否则生成模拟负荷曲线
    load = generate_load_curve(time)  # 调用工具函数生成负荷曲线

if params.get('pv_csv'):  # 如果配置中指定了从CSV读取光伏曲线
    pv = pv_curve_from_csv(r'data\pv.csv')  # 从CSV文件读取光伏数据
else:  # 否则生成模拟光伏曲线
    pv = generate_pv_curve(time)  # 调用工具函数生成光伏曲线

def optimize_schedule(time_points, battery_params, load, pv, time):
    """执行MILP优化调度"""
    prob = LpProblem("Battery_Scheduling", LpMinimize)
    
    # 定义决策变量
    soc = LpVariable.dicts("SOC", range(time_points), 
                         0.1 * battery_params['capacity'], 
                         battery_params['capacity'])
    soc[0] = battery_params['initial_soc'] * battery_params['capacity']
    
    charge = LpVariable.dicts("Charge", range(time_points), 
                            0, 0.5 * battery_params['max_power'])
    discharge = LpVariable.dicts("Discharge", range(time_points),
                               0, battery_params['max_power'])
    is_charging = LpVariable.dicts("IsCharging", 
                                 range(time_points), cat="Binary")

    # 目标函数
    prob += lpSum([(load[i] - pv[i] - discharge[i] + charge[i]) * 
                  get_electricity_price(time[i]) *0.25 for i in range(time_points)])

    # 约束条件
    for i in range(time_points):
        prob += charge[i] <= battery_params['max_power'] * is_charging[i]
        prob += discharge[i] <= battery_params['max_power'] * (1 - is_charging[i])
        
        if i > 0:
            prob += soc[i] == soc[i-1] + charge[i]*0.25*0.95 - discharge[i]*0.25/0.95
        
        prob += load[i] - pv[i] - discharge[i] + charge[i] >= 30
        prob += load[i] - pv[i] - discharge[i] + charge[i] <= 700
        
        if i > 0:
            prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 >= 0.1 * battery_params['capacity']
            prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 <= battery_params['capacity']
        
        if i == time_points - 1:
            prob += soc[i] == battery_params['initial_soc'] * battery_params['capacity']

    prob.solve()
    
    # 提取结果
    charge_opt = np.array([value(charge[i]) for i in range(time_points)])
    discharge_opt = np.array([value(discharge[i]) for i in range(time_points)])
    return {
        'charge': charge_opt,
        'discharge': discharge_opt,
        'soc': np.array([value(soc[i]) for i in range(time_points)]) / battery_params['capacity'] * 100,
        'net_load': load - pv - discharge_opt + charge_opt
    }

# 调用优化函数
battery = params['battery']
results = optimize_schedule(time_points, battery, load, pv, time)
charge_opt = results['charge']
discharge_opt = results['discharge']
battery_power = charge_opt - discharge_opt
net_load = results['net_load']
soc_values = results['soc']

# 可视化结果
from utils import plot_scheduling_results  # 导入绘图函数


def run_single_scenario():
    """运行单场景优化"""
    # 绘制并保存调度结果
    plot_scheduling_results(
        time=time,
        load=load,
        pv=pv,
        battery_power=battery_power,
        soc_values=soc_values,
        net_load=net_load,
        save_dir='results',
        filename_prefix='MILP'
    )
    # 保存结果到CSV
    pd.DataFrame({
        'load': net_load,
        'power': battery_power,
        'SOC': soc_values
    }).to_csv(r'results\net_load.csv')

if __name__ == "__main__":
    run_single_scenario()  # 运行单场景优化
