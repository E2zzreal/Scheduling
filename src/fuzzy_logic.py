import numpy as np
import pandas as pd
import json
from .utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)

"""
模糊逻辑调度算法

算法说明：
1. 使用模糊逻辑控制器进行调度决策
2. 考虑SOC、电价和净负荷的模糊规则
3. 使用Mamdani推理方法
4. 目标是实现平滑的充放电控制

输入：
- 负荷曲线：可通过generate_load_curve生成或从CSV读取
- 光伏曲线：可通过generate_pv_curve生成或从CSV读取
- 电价：通过get_electricity_price获取

输出：
- 最优调度方案
- 总运行成本
- 可视化结果
"""
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List, Tuple

def fuzzy_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
    # 定义输入变量
    soc = ctrl.Antecedent(np.arange(0, 101, 1), 'soc')
    load = ctrl.Antecedent(np.arange(0, 100, 1), 'load')
    pv = ctrl.Antecedent(np.arange(0, 100, 1), 'pv')
    
    # 定义输出变量
    charge = ctrl.Consequent(np.arange(-50, 51, 1), 'charge')
    discharge = ctrl.Consequent(np.arange(-50, 51, 1), 'discharge')
    
    # 定义隶属度函数
    soc['low'] = fuzz.trimf(soc.universe, [0, 0, 50])
    soc['medium'] = fuzz.trimf(soc.universe, [30, 50, 70])
    soc['high'] = fuzz.trimf(soc.universe, [50, 100, 100])
    
    load['low'] = fuzz.trimf(load.universe, [0, 0, 50])
    load['medium'] = fuzz.trimf(load.universe, [30, 50, 70])
    load['high'] = fuzz.trimf(load.universe, [50, 100, 100])
    
    pv['low'] = fuzz.trimf(pv.universe, [0, 0, 50])
    pv['medium'] = fuzz.trimf(pv.universe, [30, 50, 70])
    pv['high'] = fuzz.trimf(pv.universe, [50, 100, 100])
    
    charge['negative'] = fuzz.trimf(charge.universe, [-50, -50, 0])
    charge['zero'] = fuzz.trimf(charge.universe, [-25, 0, 25])
    charge['positive'] = fuzz.trimf(charge.universe, [0, 50, 50])
    
    discharge['negative'] = fuzz.trimf(discharge.universe, [-50, -50, 0])
    discharge['zero'] = fuzz.trimf(discharge.universe, [-25, 0, 25])
    discharge['positive'] = fuzz.trimf(discharge.universe, [0, 50, 50])
    
    # 定义规则
    rule1 = ctrl.Rule(soc['low'] & load['high'], charge['positive'])
    rule2 = ctrl.Rule(soc['high'] & load['low'], discharge['positive'])
    rule3 = ctrl.Rule(soc['medium'] & pv['high'], charge['zero'])
    rule4 = ctrl.Rule(soc['medium'] & pv['low'], discharge['zero'])
    
    # 创建控制系统
    charge_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    charge_sim = ctrl.ControlSystemSimulation(charge_ctrl)
    
    # 初始化变量
    time_points = params['time_points']
    time = np.linspace(0, 24, time_points)
    battery = params['battery']
    
    soc_list = [battery['initial_soc'] * 100]
    charge_list = []
    discharge_list = []
    
    # 生成或读取负荷和光伏曲线
    if params.get('load_csv'):
        load_curve = load_curve_from_csv(params['load_csv'])
    else:
        load_curve = generate_load_curve(time)
        
    if params.get('pv_csv'):
        pv_curve = pv_curve_from_csv(params['pv_csv'])
    else:
        pv_curve = generate_pv_curve(time)
    
    for t in range(time_points):
        # 设置输入
        charge_sim.input['soc'] = soc_list[-1]
        charge_sim.input['load'] = load_curve[t]
        charge_sim.input['pv'] = pv_curve[t]
        
        # 计算
        charge_sim.compute()
        
        # 获取输出
        charge_power = charge_sim.output['charge']
        discharge_power = charge_sim.output['discharge']
        
        # 更新SOC
        new_soc = soc_list[-1] + (charge_power - discharge_power) * 0.25
        new_soc = max(0, min(100, new_soc))
        
        # 保存结果
        charge_list.append(charge_power)
        discharge_list.append(discharge_power)
        soc_list.append(new_soc)
    
    # 计算净负荷
    net_load = load_curve - pv_curve - np.array(discharge_list) + np.array(charge_list)
    
    # 可视化结果
    plot_scheduling_results(time, load_curve, pv_curve,
                          np.array(charge_list) - np.array(discharge_list),
                          soc_list, net_load)
    
    return charge_list, discharge_list, soc_list[:-1]
