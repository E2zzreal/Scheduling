"""
累计成本对比分析工具

功能：
1. 加载场景分析结果 (error_based HDF5)。
2. 筛选实际成本低于最优成本的场景。
3. 计算并绘制选定场景与最优场景的累计成本曲线对比图。
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import random

# 尝试导入工具函数，如果失败则提供备选方案
try:
    from utils import get_electricity_price
except ImportError:
    print("Warning: Could not import get_electricity_price from utils.")
    # 定义一个临时的本地版本作为后备 (需要确保电价逻辑一致)
    def get_electricity_price(hour):
        # 示例电价逻辑 (需要与 utils.py 中的实际逻辑匹配!)
        if 7 <= hour < 11 or 18 <= hour < 21:
            return 1.538 # 峰时
        elif 11 <= hour < 18 or 21 <= hour < 23:
            return 1.246 # 平时
        else:
            return 0.726 # 谷时

def load_analysis_data(filepath):
    """加载 error_based 场景分析数据，包含最优调度信息"""
    data = {}
    required_keys = ['true_load', 'true_pv', 'true_net_load', 'true_charge_power', 'true_discharge_power']
    optional_keys = ['true_soc_values'] # 可选，本分析不直接使用

    try:
        with h5py.File(filepath, 'r') as f:
            # 加载全局真实/最优数据
            data['true_cost'] = f.attrs.get('true_cost', None)
            if data['true_cost'] is None:
                raise ValueError("Attribute 'true_cost' not found in HDF5 file.")

            for key in required_keys:
                if key in f:
                    data[key] = f[key][:]
                else:
                    raise ValueError(f"Required dataset '{key}' not found in HDF5 file.")

            for key in optional_keys:
                 if key in f:
                    data[key] = f[key][:]
                 else:
                    print(f"Warning: Optional dataset '{key}' not found in HDF5 file.")

            # 加载各误差组数据
            data['groups'] = {}
            group_keys_needed = ['real_costs', 'charge_power', 'discharge_power'] # 本次分析需要的
            for group_name in f:
                if group_name.startswith('error_'):
                    try:
                        error = int(group_name.split('_')[1].replace('kW', ''))
                        grp = f[group_name]
                        group_data = {}
                        all_keys_present = True
                        for key in group_keys_needed:
                            if key in grp:
                                group_data[key] = grp[key][:]
                            else:
                                print(f"Warning: Dataset '{key}' not found in group '{group_name}'. Skipping group.")
                                all_keys_present = False
                                break
                        if all_keys_present:
                             # 获取场景数量，用于验证索引
                             num_scenarios_in_group = len(group_data['real_costs'])
                             group_data['num_scenarios'] = num_scenarios_in_group
                             data['groups'][error] = group_data
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse error level from group name '{group_name}'. Skipping.")
                    except Exception as e:
                        print(f"Error loading group '{group_name}': {e}. Skipping.")

            if not data['groups']:
                 raise ValueError("No valid 'error_XkW' groups found or loaded from HDF5 file.")

            print(f"Data loaded successfully from {filepath}")
            print(f"Optimal cost (true_cost): {data['true_cost']:.2f}")
            print(f"Error levels loaded: {list(data['groups'].keys())}")

    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data from HDF5 file {filepath}: {e}")
        return None

    return data

def calculate_cumulative_cost(schedule_charge, schedule_discharge, true_load, true_pv, time_points):
    """计算给定调度计划的累计成本曲线"""
    schedule_charge = np.array(schedule_charge)
    schedule_discharge = np.array(schedule_discharge)
    true_load = np.array(true_load)
    true_pv = np.array(true_pv)

    # 1. 计算原始净负荷
    raw_net_load = true_load - true_pv - schedule_discharge + schedule_charge

    # 2. 应用 >= 30kW 约束得到最终净负荷
    final_net_load = np.where(raw_net_load < 30, 30, raw_net_load)

    # 3. 计算瞬时成本
    time_steps = np.linspace(0, 24, time_points, endpoint=False) # endpoint=False 匹配 get_electricity_price
    instant_costs = np.zeros(time_points)
    for i in range(time_points):
        price = get_electricity_price(time_steps[i])
        instant_costs[i] = final_net_load[i] * price * 0.25 # 假设时间间隔为 0.25 小时

    # 4. 计算累计成本
    cumulative_costs = np.cumsum(instant_costs)

    return cumulative_costs, final_net_load # 返回累计成本和最终净负荷

def find_low_cost_scenarios(data, max_scenarios=6):
    """查找实际成本低于最优成本的场景"""
    low_cost_scenarios = []
    true_cost = data['true_cost']

    for error, group_data in data['groups'].items():
        real_costs = group_data['real_costs']
        indices = np.where(real_costs < true_cost)[0]
        for idx in indices:
             # 确保索引有效
             if idx < group_data['num_scenarios']:
                 low_cost_scenarios.append({
                     'error': error,
                     'scenario_id': idx,
                     'real_cost': real_costs[idx]
                 })
             else:
                 print(f"Warning: Invalid scenario index {idx} found for error {error}kW. Skipping.")


    # 按成本排序，选择最低的几个
    low_cost_scenarios.sort(key=lambda x: x['real_cost'])

    # 如果数量超过 max_scenarios，则随机选择或选择最低的
    if len(low_cost_scenarios) > max_scenarios:
        # 选择成本最低的 max_scenarios 个
        selected_scenarios = low_cost_scenarios[:max_scenarios]
        # 或者随机选择： random.sample(low_cost_scenarios, max_scenarios)
        print(f"Found {len(low_cost_scenarios)} low-cost scenarios. Selecting the {max_scenarios} lowest cost ones.")
    else:
        selected_scenarios = low_cost_scenarios
        print(f"Found {len(selected_scenarios)} low-cost scenarios.")

    return selected_scenarios

def plot_cumulative_cost_comparison(optimal_cumulative_cost, scenario_data, data, save_path):
    """绘制单个低成本场景与最优场景的累计成本对比图"""
    error = scenario_data['error']
    scenario_id = scenario_data['scenario_id']
    scenario_real_cost = scenario_data['real_cost']

    # 获取该场景的调度计划
    charge_power = data['groups'][error]['charge_power'][scenario_id]
    discharge_power = data['groups'][error]['discharge_power'][scenario_id]

    # 计算该场景的累计成本
    time_points = len(data['true_load'])
    scenario_cumulative_cost, _ = calculate_cumulative_cost(
        charge_power, discharge_power, data['true_load'], data['true_pv'], time_points
    )

    # 绘图
    time = np.linspace(0, 24, time_points) # 用于 x 轴
    plt.figure(figsize=(12, 7))
    plt.plot(time, optimal_cumulative_cost, label=f'Optimal Scenario (Final Cost: {data["true_cost"]:.2f})', color='red', linestyle='--')
    plt.plot(time, scenario_cumulative_cost, label=f'Error {error}kW - Scen {scenario_id} (Final Cost: {scenario_real_cost:.2f})', color='blue')

    plt.title(f'Cumulative Cost Comparison: Optimal vs Error {error}kW Scenario {scenario_id}')
    plt.xlabel('Time (h)')
    plt.ylabel('Cumulative Cost (€)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

if __name__ == '__main__':
    # --- 配置 ---
    hdf5_path = Path('results/scenario_analysis/error_based_scenario_analysis.h5')
    output_dir = Path('results/cumulative_cost_comparison')
    num_plots = 6 # 最多绘制多少个场景的对比图

    # --- 执行 ---
    output_dir.mkdir(parents=True, exist_ok=True) # 创建输出目录

    # 1. 加载数据
    analysis_data = load_analysis_data(hdf5_path)

    if analysis_data:
        # 2. 计算最优场景的累计成本
        time_points = len(analysis_data['true_load'])
        optimal_cumulative_cost, optimal_final_net_load = calculate_cumulative_cost(
            analysis_data['true_charge_power'],
            analysis_data['true_discharge_power'],
            analysis_data['true_load'],
            analysis_data['true_pv'],
            time_points
        )
        # 验证计算出的最优成本是否与 HDF5 中的一致
        calculated_optimal_cost = optimal_cumulative_cost[-1]
        print(f"Calculated optimal cost from schedule: {calculated_optimal_cost:.2f} (vs HDF5 true_cost: {analysis_data['true_cost']:.2f})")
        # 如果不一致，可能需要检查 calculate_cumulative_cost 或数据源

        # 3. 查找低成本场景
        low_cost_scenarios_to_plot = find_low_cost_scenarios(analysis_data, max_scenarios=num_plots)

        # 4. 为每个选定的低成本场景绘图
        if not low_cost_scenarios_to_plot:
            print("No scenarios found with real_cost < true_cost.")
        else:
            for i, scenario_info in enumerate(low_cost_scenarios_to_plot):
                plot_filename = f"comparison_err{scenario_info['error']}_scen{scenario_info['scenario_id']}.png"
                plot_save_path = output_dir / plot_filename
                try:
                    plot_cumulative_cost_comparison(
                        optimal_cumulative_cost,
                        scenario_info,
                        analysis_data,
                        plot_save_path
                    )
                except Exception as plot_e:
                    print(f"Error plotting for scenario err{scenario_info['error']}_scen{scenario_info['scenario_id']}: {plot_e}")

            print(f"\nCumulative cost comparison plots generated in: {output_dir}")

    else:
        print("Failed to load data. Analysis aborted.")
