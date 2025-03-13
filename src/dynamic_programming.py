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

示例：
>>> from src.dynamic_programming import dp_scheduling
>>> charge, discharge, soc = dp_scheduling(params)
"""
import numpy as np
import pandas as pd
import json
import logging
from .utils import (generate_load_curve, generate_pv_curve, get_electricity_price,
                   load_curve_from_csv, pv_curve_from_csv, plot_scheduling_results)
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def validate_params(params: dict) -> None:
    """验证参数是否完整"""
    required_keys = ['time_points', 'battery', 'load_csv', 'pv_csv']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"缺少必要参数: {key}")

# 读取配置文件
try:
    with open('config/parameters.json') as f:
        params = json.load(f)
    validate_params(params)
    logging.info("成功读取并验证配置文件")
except Exception as e:
    logging.error(f"配置文件读取失败: {str(e)}")
    raise

def check_soc_range(soc: float) -> float:
    """检查并修正SOC值在合理范围内"""
    if soc < 10:
        return 10
    elif soc > 100:
        return 100
    return soc

def dp_scheduling(params: dict) -> Tuple[List[float], List[float], List[float]]:
    """动态规划调度算法主函数
    
    Args:
        params (dict): 包含所有必要参数的字典
        
    Returns:
        Tuple[List[float], List[float], List[float]]: 
            返回充电功率、放电功率和SOC变化列表
            
    Raises:
        ValueError: 如果参数不完整或无效
        FileNotFoundError: 如果无法找到CSV文件
    """
    # 参数提取和验证
    try:
        time_points = params['time_points']
        time = np.linspace(0, 24, time_points)
        battery = params['battery']
        
        if not isinstance(time_points, int) or time_points <= 0:
            raise ValueError("time_points必须是正整数")
            
        logging.info(f"成功提取参数，时间点数量: {time_points}")
    except KeyError as e:
        logging.error(f"参数提取失败: {str(e)}")
        raise
    
    # 加载负荷和光伏曲线
    try:
        if params.get('load_csv'):
            logging.info(f"从CSV文件加载负荷曲线: {params['load_csv']}")
            load = load_curve_from_csv(params['load_csv'])
        else:
            logging.info("生成新的负荷曲线")
            load = generate_load_curve(time)
            
        if params.get('pv_csv'):
            logging.info(f"从CSV文件加载光伏曲线: {params['pv_csv']}")
            pv = pv_curve_from_csv(params['pv_csv'])
        else:
            logging.info("生成新的光伏曲线")
            pv = generate_pv_curve(time)
    except FileNotFoundError as e:
        logging.warning(f"CSV文件未找到: {str(e)}，正在生成新的曲线...")
        load = generate_load_curve(time)
        pv = generate_pv_curve(time)
    except Exception as e:
        logging.error(f"曲线加载失败: {str(e)}")
        raise

    # DP表初始化
    dp = np.zeros((time_points, 101))  # SOC从0%到100%
    action = np.zeros((time_points, 101))
    logging.info("成功初始化DP表和动作表")
    
    # 反向DP，优化状态转移
    logging.info("开始反向DP计算")
    for t in range(time_points-1, -1, -1):
        for soc in range(101):
            min_cost = float('inf')
            best_action = 0
            
            # 动作空间剪枝：计算有效power范围
            min_power = max(-battery['max_power'],
                          (soc/100 * battery['capacity'] - 10) / 
                          (0.25 * battery['capacity']) * 100)
            max_power = min(0.5 * battery['max_power'],
                           (100 - soc)/100 * battery['capacity'] /
                           (0.25 * battery['capacity'])) *100
            
            # 使用numpy.linspace更高效地生成power值
            power_space = np.linspace(min_power, max_power, 50)
            
            for power in power_space:
                next_soc = soc + power*0.25/battery['capacity']*100
                next_soc = check_soc_range(next_soc)  # 使用函数检查SOC范围
                
                net_load = load[t] - pv[t] + power  # 更正负荷计算
                
                cost = abs(net_load) * get_electricity_price(time[t])
                
                if t < time_points-1:
                    cost += dp[t+1, int(next_soc)]
                    
                if cost < min_cost:
                    min_cost = cost
                    best_action = power
                    
            dp[t, soc] = min_cost
            action[t, soc] = best_action
            
    # 前向计算最优路径
    logging.info("开始前向计算最优路径")
    initial_soc = battery['initial_soc'] * 100
    soc_values = [initial_soc]
    charge = []
    discharge = []
    
    for t in range(time_points):
        current_soc = int(soc_values[-1])
        power = action[t, current_soc]
        
        if power > 0:
            charge.append(power)
            discharge.append(0)
        else:
            charge.append(0)
            discharge.append(-power)
            
        next_soc = soc_values[-1] + power*0.25/battery['capacity']*100
        next_soc = check_soc_range(next_soc)  # 使用函数检查SOC范围
        soc_values.append(next_soc)
        
    net_load = load - pv + np.array(charge) - np.array(discharge)
    logging.info("成功计算最优路径")
    
    # 可视化结果并保存
    try:
        plot_scheduling_results(time, load, pv,
                              np.array(charge), np.array(discharge),
                              soc_values[:-1], net_load,
                              save_path='results/scheduling_results-dp_optimized.png')
        logging.info("成功保存可视化结果")
    except Exception as e:
        logging.error(f"结果可视化失败: {str(e)}")
        raise
    
    return charge, discharge, soc_values[:-1]

if __name__ == "__main__":
    try:
        # 读取配置文件
        with open('config/parameters.json') as f:
            params = json.load(f)
        validate_params(params)
        
        # 运行优化后的调度算法
        charge, discharge, soc = dp_scheduling(params)
        
        # 打印结果
        print("充电功率:", charge[:5], "...")  # 显示部分结果
        print("放电功率:", discharge[:5], "...")
        print("SOC变化:", [round(s,2) for s in soc[:5]], "...")
        
    except Exception as e:
        logging.error(f"程序运行失败: {str(e)}")
        raise
