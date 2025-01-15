import numpy as np
import pandas as pd
import json
from .utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)

"""
启发式规则调度算法

算法说明：
1. 基于预设的启发式规则进行调度
2. 考虑电价时段和电池状态
3. 使用简单的if-else规则进行决策
4. 目标是降低运行成本

输入：
- 负荷曲线：可通过generate_load_curve生成或从CSV读取
- 光伏曲线：可通过generate_pv_curve生成或从CSV读取
- 电价：通过get_electricity_price获取

输出：
- 调度方案
- 总运行成本
- 可视化结果
"""

def heuristic_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
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
    
    # 初始化变量
    soc = [battery['initial_soc'] * battery['capacity']]
    charge = []
    discharge = []
    
    for t in range(time_points):
        price = get_electricity_price(time[t])
        
        # 启发式规则
        if price == params['price']['off_peak'] and soc[-1] < 0.9 * battery['capacity']:
            # 低谷电价时充电
            charge_power = min(0.5 * battery['max_power'], 
                             (0.9 * battery['capacity'] - soc[-1]) / 0.25)
            charge.append(charge_power)
            discharge.append(0)
            soc.append(soc[-1] + charge_power * 0.25)
            
        elif price == params['price']['on_peak'] and soc[-1] > 0.2 * battery['capacity']:
            # 高峰电价时放电
            discharge_power = min(battery['max_power'],
                                (soc[-1] - 0.2 * battery['capacity']) / 0.25)
            charge.append(0)
            discharge.append(discharge_power)
            soc.append(soc[-1] - discharge_power * 0.25)
            
        else:
            # 其他情况不充放电
            charge.append(0)
            discharge.append(0)
            soc.append(soc[-1])
    
    # 计算净负荷
    net_load = load - pv - np.array(discharge) + np.array(charge)
    
    # 可视化结果
    plot_scheduling_results(time, load, pv,
                          np.array(charge) - np.array(discharge),
                          soc, net_load,
                          save_path='results/scheduling_results-heuristic.png')
            
    return charge, discharge, soc[:-1]

if __name__ == "__main__":
    # 读取配置文件
    with open('config/parameters.json') as f:
        params = json.load(f)
    
    # 运行模糊逻辑调度
    charge, discharge, soc = heuristic_scheduling(params)
    
    # 打印结果
    print("充电功率:", charge)
    print("放电功率:", discharge)
    print("SOC变化:", soc)
