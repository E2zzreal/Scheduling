import json
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value

# 读取配置文件
with open('config/parameters.json') as f:
    params = json.load(f)

# 参数设置
time_points = params['time_points']
time = np.linspace(0, 24, time_points)

# 负荷曲线生成
def generate_load_curve(time):
    load = np.zeros_like(time)
    np.random.seed(42)
    for i, t in enumerate(time):
        if t < 8 or t >= 22:
            base_load = params['load']['base_night'] + 50 * np.sin(2 * np.pi * t / 24)
        else:
            base_load = params['load']['base_day'] + 300 * np.sin(2 * np.pi * (t - 8) / 14)
        load[i] = base_load + np.random.uniform(-params['load']['random_range'], params['load']['random_range'])
    return load

# 光伏发电曲线生成
def generate_pv_curve(time):
    pv = np.zeros_like(time)
    np.random.seed(42)
    for i, t in enumerate(time):
        if 7.5 <= t <= 17.5:
            base_pv = params['pv']['base_power'] + 150 * np.sin(np.pi * (t - 7.5) / 10)
            pv[i] = base_pv + np.random.uniform(-params['pv']['random_range'], params['pv']['random_range'])
        else:
            pv[i] = 0
    return pv

# 电价时段
def get_electricity_price(t):
    if 0 <= t < 6 or 22 <= t <= 24:
        return params['price']['off_peak']
    elif (6 <= t < 8) or (13 <= t < 16) or (19 <= t < 22):
        return params['price']['mid_peak']
    else:
        return params['price']['on_peak']

# 生成负荷和光伏曲线
load = generate_load_curve(time)
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
              get_electricity_price(time[i]) for i in range(time_points)])

# 约束条件
for i in range(time_points):
    prob += charge[i] <= battery['max_power'] * is_charging[i]
    prob += discharge[i] <= battery['max_power'] * (1 - is_charging[i])
    
    if i > 0:
        prob += soc[i] == soc[i-1] + charge[i]*0.25 - discharge[i]*0.25
    
    prob += load[i] - pv[i] - discharge[i] + charge[i] >= 30
    
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
fig, axs = plt.subplots(3, 1, figsize=(14, 15))

# 负荷和光伏曲线
axs[0].plot(time, load, label='Load (kW)')
axs[0].plot(time, pv, label='PV Generation (kW)')
axs[0].set_title('Load and PV Generation')
axs[0].legend()
axs[0].grid()

# 电池功率和SOC曲线
ax1 = axs[1]
ax2 = ax1.twinx()
ax1.plot(time, battery_power, label='Battery Power (kW)', color='blue')
ax1.fill_between(time, 0, battery_power, where=(battery_power > 0), color='green', alpha=0.3, label='Charging')
ax1.fill_between(time, 0, battery_power, where=(battery_power < 0), color='red', alpha=0.3, label='Discharging')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Power (kW)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.plot(time, soc_values, label='SOC (%)', color='orange', linestyle='--')
ax2.set_ylabel('SOC (%)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

axs[1].set_title('Battery Power and SOC Curve')
fig.legend(loc='upper right')
axs[1].grid()

# 净负荷曲线
axs[2].plot(time, net_load, label='Net Load (kW)')
axs[2].plot(time, load, label='Load (kW)', linestyle='--', alpha=0.5)
axs[2].set_title('Net Load and Load Comparison')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.savefig('results/scheduling_results.png')
plt.show()
