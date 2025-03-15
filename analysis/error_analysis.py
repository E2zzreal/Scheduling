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
        
        for group_name in f:
            if group_name.startswith('error_'):
                error = int(group_name.split('_')[1].replace('kW', ''))
                grp = f[group_name]
                data['groups'][error] = {
                    'real_costs': grp['real_costs'][:],
                    'predicted_load': grp['predicted_load'][:],
                    'true_net_load': grp['true_net_load'][:],
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
            required_keys = ['real_costs', 'predicted_load', 'true_net_load']
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
                'true_net_load': data['groups'][error]['true_net_load'][selected_indices],
                'scenario_ids': selected_indices
            }
            # 验证数据形状
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
                    net_load = scenarios['true_net_load'][col_idx]
                    
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
        plot_selected_scenarios(data, selected, output_dir)
        
        print(f"分析结果已保存至：{output_dir}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
