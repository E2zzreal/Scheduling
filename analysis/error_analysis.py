"""
error模式场景分析工具
功能：
1. 成本分布对比
2. 负荷偏差分析 
3. SOC统计特征
4. 充放电模式可视化
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
# 新增导入，用于获取电价和加载参数
from utils import get_electricity_price
import json

# --- 新增辅助函数 ---
def calculate_raw_net_load(true_load, true_pv, charge_power, discharge_power):
    """计算应用约束前的原始净负荷"""
    # 确保输入是 numpy 数组
    true_load = np.array(true_load)
    true_pv = np.array(true_pv)
    charge_power = np.array(charge_power)
    discharge_power = np.array(discharge_power)
    # 检查维度匹配
    if not (true_load.shape == true_pv.shape == charge_power.shape == discharge_power.shape):
        raise ValueError(f"Dimension mismatch in calculate_raw_net_load: "
                         f"{true_load.shape}, {true_pv.shape}, {charge_power.shape}, {discharge_power.shape}")
    return true_load - true_pv - discharge_power + charge_power
# --- 辅助函数结束 ---

def load_error_data(filepath):
    """加载error模式数据"""
    with h5py.File(filepath, 'r') as f:
        data = {
            'true_load': f['true_load'][:],
            'true_pv': f['true_pv'][:],
            'true_cost': f.attrs['true_cost'],
            'true_net_load': f['true_net_load'][:],
            'groups': {}
        }

        # 尝试加载最优功率数据 (新增)
        if 'true_charge_power' in f and 'true_discharge_power' in f:
            data['true_charge_power'] = f['true_charge_power'][:]
            data['true_discharge_power'] = f['true_discharge_power'][:]
            print("Optimal charge/discharge power loaded from HDF5.")
        else:
            print("Warning: Optimal charge/discharge power not found in HDF5.")

        # --- 新增：尝试加载最优 SOC 数据 ---
        if 'true_soc_values' in f:
            data['true_soc_values'] = f['true_soc_values'][:]
            print("Optimal SOC values loaded from HDF5.")
        else:
            print("Warning: Optimal SOC values ('true_soc_values') not found in HDF5.")
        # --- 加载 SOC 结束 ---

        for group_name in f:
            if group_name.startswith('error_'):
                error = int(group_name.split('_')[1].replace('kW', ''))
                grp = f[group_name]
                data['groups'][error] = {
                    'real_costs': grp['real_costs'][:],
                    'predicted_load': grp['predicted_load'][:],
                    'real_net_load': grp['real_net_load'][:], # Corrected key from 'true_net_load'
                    'soc_values': grp['soc_values'][:],
                    'charge_power': grp['charge_power'][:],
                    'discharge_power': grp['discharge_power'][:]
                }
        return data

def plot_cost_distribution(data, save_dir):
    """绘制成本分布对比图"""
    plt.figure(figsize=(12, 6))
    errors = sorted(data['groups'].keys())
    
    for error in errors:
        costs = data['groups'][error]['real_costs']
        sns.kdeplot(costs, label=f'{error}kW Error')
        
    plt.axvline(data['true_cost'], color='black', linestyle='--', label='Optimal Cost')
    plt.title('Cost Distribution Comparison')
    plt.xlabel('Cost')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(Path(save_dir)/'cost_distribution.png')
    plt.close()

def plot_load_deviation(data, save_dir, time_points=96):
    """负荷偏差分析"""
    time = np.linspace(0, 24, time_points)
    plt.figure(figsize=(15, 8))
    
    # 绘制真实负荷
    plt.plot(time, data['true_load'], label='True Load', linewidth=2, color='black')
    
    # 绘制各误差级别的预测负荷
    for error in data['groups']:
        avg_load = np.mean(data['groups'][error]['predicted_load'], axis=0)
        plt.plot(time, avg_load, label=f'{error}kW Prediction')
    
    plt.title('Load Prediction Deviation Analysis')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.savefig(Path(save_dir)/'load_deviation.png')
    plt.close()

def plot_soc_statistics(data, save_dir):
    """SOC统计特征分析"""
    plt.figure(figsize=(12, 6))
    errors = sorted(data['groups'].keys())
    soc_stats = []
    
    for error in errors:
        soc_values = np.array(data['groups'][error]['soc_values'])
        soc_stats.append({
            'error': error,
            'mean': np.mean(soc_values),
            'std': np.std(soc_values)
        })
    
    soc_stats = pd.DataFrame(soc_stats)
    sns.barplot(data=soc_stats, x='error', y='mean', yerr=soc_stats['std'])
    plt.title('SOC Statistics by Error Level')
    plt.xlabel('Error (kW)')
    plt.ylabel('Average SOC (%)')
    plt.savefig(Path(save_dir)/'soc_statistics.png')
    plt.close()

def plot_power_patterns(data, save_dir, time_points=96):
    """充放电模式可视化"""
    time = np.linspace(0, 24, time_points)
    plt.figure(figsize=(15, 8))
    
    for error in data['groups']:
        charge_power = np.mean(data['groups'][error]['charge_power'], axis=0)
        discharge_power = np.mean(data['groups'][error]['discharge_power'], axis=0)
        battery_power = charge_power - discharge_power
        plt.plot(time, battery_power, label=f'{error}kW Error')
    
    plt.title('Battery Power Patterns')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.savefig(Path(save_dir)/'power_patterns.png')
    plt.close()

def select_low_cost_scenarios(data, num_samples=3):
    """筛选低成本场景"""
    selected = {}
    try:
        for error in data['groups']:
            # 验证必要字段存在
            required_keys = ['real_costs', 'predicted_load', 'real_net_load'] # Corrected key
            if not all(k in data['groups'][error] for k in required_keys):
                raise KeyError(f"Missing required keys in error group {error}")

            costs = data['groups'][error]['real_costs']
            true_cost = data['true_cost']
            
            # 筛选条件
            mask = costs < true_cost
            valid_indices = np.where(mask)[0]
            
            # 处理无有效场景情况
            if len(valid_indices) == 0:
                print(f"Warning: No valid scenarios for {error}kW error")
                continue
                
            # 处理样本数量
            n_samples = min(num_samples, len(valid_indices))
            if n_samples < num_samples:
                print(f"Warning: Only {len(valid_indices)} scenarios available for {error}kW, using all")
                
            selected_indices = np.random.choice(valid_indices, n_samples, replace=False)
            
            # 存储数据并验证维度
            selected[error] = {
                'predicted_load': data['groups'][error]['predicted_load'][selected_indices],
                'real_net_load': data['groups'][error]['real_net_load'][selected_indices], # Corrected key
                'scenario_ids': selected_indices
            }
            # 验证数据形状 (验证 'predicted_load' 即可，因为它们应具有相同的维度)
            if selected[error]['predicted_load'].shape[1] != len(data['true_load']):
                raise ValueError("Predicted load dimension mismatch")
                
    except Exception as e:
        print(f"Error selecting scenarios: {str(e)}")
        raise
    
    return selected

def plot_selected_scenarios(data, selected, save_dir, time_points=96):
    """绘制选定场景对比图"""
    try:
        plt.figure(figsize=(20, 15))
        time = np.linspace(0, 24, time_points)
        errors = sorted(selected.keys())
        
        # 动态计算行数（每error占一行）
        n_rows = len(errors)
        n_cols = 3
        
        for row_idx, error in enumerate(errors):
            # 验证error数据存在
            if error not in data['groups']:
                print(f"Warning: Missing data for {error}kW, skipping")
                continue
                
            scenarios = selected[error]
            for col_idx in range(n_cols):
                ax = plt.subplot(n_rows, n_cols, row_idx*n_cols + col_idx + 1)
                
                # 检查场景索引有效性
                if col_idx >= len(scenarios['scenario_ids']):
                    ax.axis('off')  # 关闭空子图
                    continue
                    
                scenario_id = scenarios['scenario_ids'][col_idx]
                try:
                    # 检查数据有效性
                    pred_load = scenarios['predicted_load'][col_idx]
                    net_load = scenarios['real_net_load'][col_idx] # Corrected key

                    if np.isnan(pred_load).any() or np.isnan(net_load).any():
                        raise ValueError("NaN values detected in load data")
                        
                    # 绘制四条曲线
                    ax.plot(time, data['true_load'], label='True Load', color='blue', linewidth=2)
                    ax.plot(time, pred_load, label='Predicted', color='orange', linestyle='--', linewidth=2)
                    ax.plot(time, net_load, label='Actual Net', color='green', linewidth=2)
                    ax.plot(time, data['true_net_load'], label='Optimal Net', color='red', linewidth=1.5, alpha=0.6, linestyle='-.')
                    
                    # 添加网格线
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # 设置子图标题
                    ax.set_title(f'Error {error}kW - Scenario {scenario_id}')
                    ax.set_xlabel('Time (h)')
                    ax.set_ylabel('Power (kW)')
                    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fontsize=6)
                    
                except Exception as e:
                    print(f"Error plotting scenario {scenario_id}: {str(e)}")
                    ax.text(0.5, 0.5, 'Data Error', ha='center')
                    ax.axis('off')
        
        # 保存处理
        plt.tight_layout()
        plt.savefig(Path(save_dir)/'selected_scenarios.png', dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"Plotting failed: {str(e)}")
        raise
    finally:
        plt.close()

# --- 新增分析函数 ---
def analyze_constraint_interaction(data, selected_scenarios, save_dir, time_points=96):
    """分析30kW约束的影响"""
    analysis_results = {}
    time = np.linspace(0, 24, time_points)
    constraint_threshold = 30
    output_path = Path(save_dir)

    print("\n--- Starting Constraint Interaction Analysis ---")

    # --- 处理最优场景 ---
    optimal_analysis = {}
    if 'true_charge_power' in data and 'true_discharge_power' in data:
        try:
            optimal_raw_net_load = calculate_raw_net_load(
                data['true_load'], data['true_pv'],
                data['true_charge_power'], data['true_discharge_power']
            )
            optimal_violations = optimal_raw_net_load < constraint_threshold
            optimal_violation_count = np.sum(optimal_violations)
            # 计算违规幅度时避免负值索引
            violation_indices = np.where(optimal_violations)[0]
            optimal_violation_magnitude = np.sum(np.maximum(0, constraint_threshold - optimal_raw_net_load[violation_indices]))

            optimal_analysis = {
                'raw_net_load': optimal_raw_net_load,
                'violation_count': optimal_violation_count,
                'violation_magnitude': optimal_violation_magnitude
            }
            print(f"Optimal Scenario: Violations={optimal_violation_count}, Total Magnitude={optimal_violation_magnitude:.2f}")
            # --- 新增：打印违规点的详细信息 ---
            if optimal_violation_count > 0:
                violation_indices = np.where(optimal_violations)[0]
                print(f"Optimal Scenario: Violation indices: {violation_indices}")
                print(f"Optimal Scenario: Raw net load at violations: {np.round(optimal_raw_net_load[violation_indices], 2)}")

                # --- 获取并打印电价和 SOC ---
                try:
                    # 加载参数以获取 time_points 和 battery capacity
                    params = {}
                    try:
                        with open('config/parameters.json') as f:
                            params = json.load(f)
                    except FileNotFoundError:
                         print("Warning: config/parameters.json not found. Using default time_points=96 and capacity=1000.")
                    time_points = params.get('time_points', 96)
                    battery_capacity = params.get('battery', {}).get('capacity', 1000) # 默认容量

                    time_steps = np.linspace(0, 24, time_points, endpoint=False) # endpoint=False 匹配 get_electricity_price

                    prices_at_violations = [get_electricity_price(time_steps[idx]) for idx in violation_indices]
                    print(f"Optimal Scenario: Prices at violations (€/kWh): {np.round(prices_at_violations, 3)}")

                    # 加载并打印最优 SOC (使用已加载的 data['true_soc_values'])
                    if 'true_soc_values' in data:
                         optimal_soc_abs = np.array(data['true_soc_values'])
                         optimal_soc_percent = (optimal_soc_abs / battery_capacity) * 100
                         soc_at_violations = optimal_soc_percent[violation_indices]
                         print(f"Optimal Scenario: SOC (%) at violations: {np.round(soc_at_violations, 1)}")
                    else:
                         print("Warning: 'true_soc_values' not loaded. Cannot print SOC at violations.")

                except Exception as detail_e:
                    print(f"Error getting details for violation points: {detail_e}")
                # --- 详细信息结束 ---

            # --- 详细信息结束 ---
        except Exception as e:
             print(f"Error calculating optimal raw net load: {e}")
             optimal_analysis = {'raw_net_load': None, 'error': str(e)} # 标记错误
    else:
        print("Warning: Optimal charge/discharge power not found in HDF5. Cannot analyze optimal raw_net_load.")
        optimal_analysis = {'raw_net_load': None} # 标记为不可用
    analysis_results['optimal'] = optimal_analysis

    # --- 处理选定的低成本场景 ---
    summary_stats = []
    for error, scenarios in selected_scenarios.items():
        analysis_results[error] = []
        if error not in data['groups']:
             print(f"Warning: Data for error {error}kW not found in data['groups']. Skipping.")
             continue
        charge_all = data['groups'][error]['charge_power']
        discharge_all = data['groups'][error]['discharge_power']

        for i, scenario_id in enumerate(scenarios['scenario_ids']):
            try:
                # 确保索引有效
                if scenario_id >= len(charge_all) or scenario_id >= len(discharge_all):
                     print(f"Warning: Invalid scenario_id {scenario_id} for error {error}kW. Skipping.")
                     continue

                charge_power = charge_all[scenario_id]
                discharge_power = discharge_all[scenario_id]

                raw_net_load = calculate_raw_net_load(
                    data['true_load'], data['true_pv'],
                    charge_power, discharge_power
                )
                violations = raw_net_load < constraint_threshold
                violation_count = np.sum(violations)
                violation_indices = np.where(violations)[0]
                violation_magnitude = np.sum(np.maximum(0, constraint_threshold - raw_net_load[violation_indices]))

                scenario_result = {
                    'scenario_id': scenario_id,
                    'raw_net_load': raw_net_load,
                    'violation_count': violation_count,
                    'violation_magnitude': violation_magnitude
                }
                analysis_results[error].append(scenario_result)
                summary_stats.append({'error': error, 'scenario_id': scenario_id, 'violations': violation_count, 'magnitude': violation_magnitude})
                # print(f"Error {error}kW - Scenario {scenario_id}: Violations={violation_count}, Total Magnitude={violation_magnitude:.2f}")
            except Exception as e:
                 print(f"Error processing Error {error}kW - Scenario {scenario_id}: {e}")
                 analysis_results[error].append({'scenario_id': scenario_id, 'error': str(e)})


    # --- 打印总结统计 ---
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        print("\nLow Cost Scenarios Constraint Violation Summary:")
        print(summary_df.groupby('error').agg(
            count=('scenario_id', 'size'),
            avg_violations=('violations', 'mean'),
            avg_magnitude=('magnitude', 'mean')
        ))

    # --- 可视化对比 (绘制第一个低成本场景与最优场景的对比) ---
    try:
        plt.figure(figsize=(15, 7))
        plot_optimal = False
        if analysis_results['optimal'].get('raw_net_load') is not None:
             plt.plot(time, analysis_results['optimal']['raw_net_load'], label='Optimal Raw Net Load', color='grey', linestyle=':', linewidth=1.5)
             # 绘制最优最终净负荷 (来自H5)
             plt.plot(time, data['true_net_load'], label='Optimal Final Net Load', color='red', linestyle='-.', linewidth=1.5, alpha=0.7)
             plot_optimal = True

        # 选择一个低成本场景绘制
        example_error = next(iter(selected_scenarios.keys()), None)
        plot_example = False
        if example_error and analysis_results.get(example_error):
            example_scenario_data = next((item for item in analysis_results[example_error] if 'error' not in item), None) # 找第一个成功的
            if example_scenario_data:
                example_id = example_scenario_data['scenario_id']
                # 绘制原始净负荷
                plt.plot(time, example_scenario_data['raw_net_load'], label=f'Low Cost (Err {example_error}, Scen {example_id}) Raw Net Load', color='purple', linewidth=1.5)
                # 从H5加载其最终净负荷
                final_net_load_low_cost = data['groups'][example_error]['true_net_load'][example_id]
                plt.plot(time, final_net_load_low_cost, label=f'Low Cost (Err {example_error}, Scen {example_id}) Final Net Load', color='green', linewidth=1.5)
                plot_example = True

        if plot_optimal or plot_example: # 只有画了东西才加其他元素
            plt.axhline(constraint_threshold, color='black', linestyle='--', label=f'{constraint_threshold}kW Constraint', linewidth=1)
            plt.title('Raw Net Load vs. Constraint Comparison (Optimal vs. Example Low Cost)')
            plt.xlabel('Time (h)')
            plt.ylabel('Power (kW)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(bottom=min(0, plt.ylim()[0])) # 确保y轴从0或更低开始
            plt.tight_layout()
            save_path = output_path / 'constraint_interaction_comparison.png'
            plt.savefig(save_path, dpi=300)
            print(f"Constraint comparison plot saved to {save_path}")
        else:
             print("Skipping constraint comparison plot: No valid data to plot.")

    except Exception as e:
        print(f"Error during constraint visualization: {e}")
    finally:
        plt.close() # 确保关闭图形

    print("--- Constraint Interaction Analysis Finished ---")
    return analysis_results
# --- 分析函数结束 ---


if __name__ == '__main__':
    try:
        # 配置参数
        hdf5_path = 'results/scenario_analysis/error_based_scenario_analysis.h5'
        output_dir = 'results/error_analysis'
        
        # 新增输出目录检查
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_dir}")
            
        # 加载数据增加文件存在检查
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
            
        data = load_error_data(hdf5_path)
        
        # 执行分析
        plot_cost_distribution(data, output_dir)
        plot_load_deviation(data, output_dir)
        plot_soc_statistics(data, output_dir)
        plot_power_patterns(data, output_dir)
        
        # 新增场景筛选和绘图
        selected = select_low_cost_scenarios(data)
        if selected: # 只有筛选到场景才绘图和分析
            plot_selected_scenarios(data, selected, output_dir)
            # 调用新增的约束分析函数
            constraint_analysis = analyze_constraint_interaction(data, selected, output_dir)
            # (可选) 在这里可以对 constraint_analysis 结果做进一步处理或打印
        else:
            print("No low-cost scenarios selected, skipping detailed plotting and constraint analysis.")

        print(f"\nAnalysis complete. Results saved to: {output_dir}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
