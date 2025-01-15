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

import numpy as np
import pandas as pd
import json
from utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List, Tuple


def fuzzy_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
    # 定义输入变量
    soc = ctrl.Antecedent(np.arange(0, 101, 1), 'soc')
    load = ctrl.Antecedent(np.arange(200, 1100, 1), 'load')
    pv = ctrl.Antecedent(np.arange(0, 450, 1), 'pv')
    
    # 定义输出变量
    power = ctrl.Consequent(np.arange(-800, 501, 1), 'power')

    # 定义隶属度函数
    soc['very_low'] = fuzz.trimf(soc.universe, [5, 5, 10])
    soc['low'] = fuzz.trimf(soc.universe, [0, 0, 50])
    soc['medium'] = fuzz.trimf(soc.universe, [30, 50, 70])
    soc['high'] = fuzz.trimf(soc.universe, [50, 95, 95])
    soc['very_high'] = fuzz.trimf(soc.universe, [95, 100, 100])
    
    load['low'] = fuzz.trimf(load.universe, [0, 0, 400])
    load['medium'] = fuzz.trimf(load.universe, [300, 600, 700])
    load['high'] = fuzz.trimf(load.universe, [500, 1100, 1100])
    
    pv['low'] = fuzz.trimf(pv.universe, [0, 0, 200])
    pv['medium'] = fuzz.trimf(pv.universe, [100, 200, 300])
    pv['high'] = fuzz.trimf(pv.universe, [300, 450, 450])
    
    power['negative'] = fuzz.trimf(power.universe, [-800, -800, 0])  # 放电
    power['zero'] = fuzz.trimf(power.universe, [-100, 0, 100])       # 不充不放
    power['positive'] = fuzz.trimf(power.universe, [0, 500, 500])    # 充电
    
    # 定义规则
    rule1 = ctrl.Rule(soc['low'] & load['high'], power['positive'])
    rule2 = ctrl.Rule(soc['high'] & load['low'], power['zero'])
    rule3 = ctrl.Rule(soc['medium'] & load['medium'] & pv['high'], power['zero'])
    rule4 = ctrl.Rule(soc['medium'] & pv['low'], power['zero'])
    rule5 = ctrl.Rule(soc['low'] & load['medium'], power['positive'])
    rule6 = ctrl.Rule(soc['medium'] & load['high'], power['positive'])
    rule7 = ctrl.Rule(soc['high'] & pv['high'], power['zero'])
    rule8 = ctrl.Rule(soc['high'] & load['high'], power['zero'])
    rule9 = ctrl.Rule(soc['very_low'], power['positive'])
    rule10 = ctrl.Rule(soc['very_high'], power['zero'])

    # 新增规则以覆盖所有情况
    rule11 = ctrl.Rule(soc['low'] & load['low'], power['positive'])
    rule12 = ctrl.Rule(soc['medium'] & load['low'], power['positive'])
    rule13 = ctrl.Rule(soc['high'] & load['medium'], power['zero'])
    rule14 = ctrl.Rule(soc['very_low'] & load['low'], power['positive'])
    rule15 = ctrl.Rule(soc['very_low'] & load['medium'], power['positive'])
    rule16 = ctrl.Rule(soc['very_low'] & load['high'], power['positive'])
    rule17 = ctrl.Rule(soc['low'] & pv['high'], power['positive'])
    rule18 = ctrl.Rule(soc['medium'] & pv['medium'], power['zero'])
    rule19 = ctrl.Rule(soc['high'] & pv['medium'], power['zero'])
    rule20 = ctrl.Rule(soc['very_high'] & pv['high'] & load['high'], power['negative'])
    rule21 = ctrl.Rule(soc['very_high'] & pv['medium'] & load['high'], power['negative'])
    rule22 = ctrl.Rule(soc['very_high'] & pv['low'] & load['high'], power['negative'])

    # 创建控制系统
    power_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22])
    power_sim = ctrl.ControlSystemSimulation(power_ctrl)
    
    # 初始化变量
    time_points = params['time_points']
    time = np.linspace(0, 24, time_points)
    battery = params['battery']
    
    soc_list = [battery['initial_soc'] * 100]
    power_list = []
    
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
        power_sim.input['soc'] = soc_list[-1]
        power_sim.input['load'] = load_curve[t]
        power_sim.input['pv'] = pv_curve[t]
        
        # 计算
        power_sim.compute()
        
        # 获取输出
        if 'power' not in power_sim.output:
            print(f"Warning: No output for power at time {t}. Inputs: soc={soc_list[-1]}, load={load_curve[t]}, pv={pv_curve[t]}")
            power_value = 0  # 默认值
        else:
            power_value = power_sim.output['power']
        
        # 更新SOC
        new_soc = soc_list[-1] + power_value * 0.25/battery['capacity']*100
        new_soc = max(10, min(100, new_soc))
        
        # 保存结果
        power_list.append(power_value)
        soc_list.append(new_soc)
    
    # 计算净负荷
    net_load = load_curve - pv_curve + np.array(power_list)
    
    # 可视化结果
    plot_scheduling_results(time, load_curve, pv_curve,
                          np.array(power_list),
                          soc_list[:-1], net_load,
                          save_path='results/scheduling_results-fuzzy.png')
    
    return power_list, soc_list[:-1]


if __name__ == "__main__":
    # 读取配置文件
    with open('config/parameters.json') as f:
        params = json.load(f)
    
    # 运行模糊逻辑调度
    power, soc = fuzzy_scheduling(params)
    
    # 打印结果
    print("充放电功率:", power)
    print("SOC变化:", soc)
