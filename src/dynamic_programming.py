import numpy as np
import pandas as pd
import json
from .utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)

"""
动态规划调度算法

算法说明：
1. 将调度问题建模为多阶段决策问题
2. 使用动态规划方法求解
3. 考虑设备状态转移和成本函数
4. 目标是最小化总运行成本

输入：
- 负荷曲线：可通过generate_load_curve生成或从CSV读取
- 光伏曲线：可通过generate_pv_curve生成或从CSV读取
- 电价：通过get_electricity_price获取

输出：
- 最优调度方案
- 总运行成本
"""
import json
from typing import List, Tuple

def dp_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
    # 参数提取
    time_points = params['time_points']
    time = np.linspace(0, 24, time_points)
    battery = params['battery']
    
    # 生成或读取负荷和光伏曲线
    if params.get('load_csv'):
        load = load_curve_from_csv(params['load_csv'])
    else:
        load = generate_load_curve(time)
        
    if params.get('pv_csv'):
        pv = pv_curve_from_csv(params['pv_csv'])
    else:
        pv = generate_pv_curve(time)
    
    # DP表初始化
    dp = np.zeros((time_points, 101))  # SOC从0%到100%
    action = np.zeros((time_points, 101))
    
    # 反向DP
    for t in range(time_points-1, -1, -1):
        for soc in range(101):
            min_cost = float('inf')
            best_action = 0
            
            # 遍历可能的充放电功率
            for power in np.linspace(-battery['max_power'], 0.5*battery['max_power'], 100):
                next_soc = soc + power*0.25/battery['capacity']*100
                
                if 10 <= next_soc <= 100:  # SOC约束
                    net_load = load[t] - pv[t] - power
                    if net_load >= 30:  # 负荷平衡约束
                        cost = net_load * get_electricity_price(time[t])
                        
                        if t < time_points-1:
                            cost += dp[t+1, int(next_soc)]
                            
                        if cost < min_cost:
                            min_cost = cost
                            best_action = power
                            
            dp[t, soc] = min_cost
            action[t, soc] = best_action
            
    # 前向计算最优路径
    soc_values = [battery['initial_soc']*100]
    charge = []
    discharge = []
    
    for t in range(time_points):
        power = action[t, int(soc_values[-1])]
        if power > 0:
            charge.append(power)
            discharge.append(0)
        else:
            charge.append(0)
            discharge.append(-power)
            
        next_soc = soc_values[-1] + power*0.25/battery['capacity']*100
        soc_values.append(next_soc)
        
    # 计算净负荷
    net_load = load - pv - np.array(discharge) + np.array(charge)
    
    # 可视化结果
    plot_scheduling_results(time, load, pv, 
                          np.array(charge) - np.array(discharge),
                          soc_values, net_load)
    
    return charge, discharge, soc_values[:-1]
