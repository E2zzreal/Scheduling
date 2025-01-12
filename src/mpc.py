import numpy as np
import pandas as pd
from .utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)

"""
模型预测控制(MPC)调度算法

算法说明：
1. 使用滚动时域优化方法
2. 在每个时间步求解有限时域优化问题
3. 考虑系统动态和未来预测
4. 目标是优化有限时域内的运行成本

输入：
- 负荷曲线：可通过generate_load_curve生成或从CSV读取
- 光伏曲线：可通过generate_pv_curve生成或从CSV读取
- 电价：通过get_electricity_price获取

输出：
- 最优调度方案
- 总运行成本
"""
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value

def mpc_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
    # 参数提取
    time_points = params['time_points']
    time = np.linspace(0, 24, time_points)
    battery = params['battery']
    horizon = 6  # 预测时域
    
    # 生成或读取负荷和光伏曲线
    if params.get('load_csv'):
        load = load_curve_from_csv(params['load_csv'])
    else:
        load = generate_load_curve(time)
        
    if params.get('pv_csv'):
        pv = pv_curve_from_csv(params['pv_csv'])
    else:
        pv = generate_pv_curve(time)
    
    # 初始化变量
    soc = [battery['initial_soc'] * battery['capacity']]
    charge = []
    discharge = []
    
    for t in range(time_points):
        # 创建MPC问题
        prob = LpProblem("MPC_Scheduling", LpMinimize)
        
        # 定义决策变量
        mpc_charge = LpVariable.dicts("Charge", range(horizon), 0, 0.5 * battery['max_power'])
        mpc_discharge = LpVariable.dicts("Discharge", range(horizon), 0, battery['max_power'])
        mpc_soc = LpVariable.dicts("SOC", range(horizon+1), 0.1 * battery['capacity'], battery['capacity'])
        mpc_soc[0] = soc[-1]
        
        # 目标函数
        prob += lpSum([(load[min(t+i, time_points-1)] - pv[min(t+i, time_points-1)] - 
                       mpc_discharge[i] + mpc_charge[i]) * 
                      get_electricity_price(time[min(t+i, time_points-1)]) 
                      for i in range(horizon)])
        
        # 约束条件
        for i in range(horizon):
            prob += mpc_charge[i] <= battery['max_power'] * (1 - (mpc_discharge[i] > 0))
            prob += mpc_discharge[i] <= battery['max_power'] * (1 - (mpc_charge[i] > 0))
            
            if i > 0:
                prob += mpc_soc[i] == mpc_soc[i-1] + mpc_charge[i]*0.25 - mpc_discharge[i]*0.25
            
            prob += load[min(t+i, time_points-1)] - pv[min(t+i, time_points-1)] - \
                    mpc_discharge[i] + mpc_charge[i] >= 30
            
            prob += mpc_soc[i] >= 0.1 * battery['capacity']
            prob += mpc_soc[i] <= battery['capacity']
            
        # 求解问题
        prob.solve()
        
        # 执行第一个时间步的控制
        charge.append(value(mpc_charge[0]))
        discharge.append(value(mpc_discharge[0]))
        soc.append(value(mpc_soc[1]))
        
    # 计算净负荷
    net_load = load - pv - np.array(discharge) + np.array(charge)
    
    # 可视化结果
    plot_scheduling_results(time, load, pv,
                          np.array(charge) - np.array(discharge),
                          soc, net_load)
            
    return charge, discharge, soc[:-1]
