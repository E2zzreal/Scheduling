import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# 读取配置文件
with open('config/parameters.json') as f:
    params = json.load(f)

def generate_load_curve(time):
    """生成负荷曲线
    
    参数:
    time (np.array): 时间序列
    
    返回:
    np.array: 负荷曲线
    """
    load = np.zeros_like(time)
    np.random.seed(42)
    for i, t in enumerate(time):
        if t < 8 or t >= 22:
            base_load = params['load']['base_night'] + 50 * np.sin(2 * np.pi * t / 24)
        else:
            base_load = params['load']['base_day'] + 300 * np.sin(2 * np.pi * (t - 8) / 14)
        load[i] = base_load + np.random.uniform(-params['load']['random_range'], params['load']['random_range'])
    return load

def generate_pv_curve(time):
    """生成光伏发电曲线
    
    参数:
    time (np.array): 时间序列
    
    返回:
    np.array: 光伏发电曲线
    """
    pv = np.zeros_like(time)
    np.random.seed(42)
    for i, t in enumerate(time):
        if 7.5 <= t <= 17.5:
            base_pv = params['pv']['base_power'] + 150 * np.sin(np.pi * (t - 7.5) / 10)
            pv[i] = base_pv + np.random.uniform(-params['pv']['random_range'], params['pv']['random_range'])
        else:
            pv[i] = 0
    return pv

def get_electricity_price(t):
    """获取电价
    
    参数:
    t (float): 当前时间
    
    返回:
    float: 当前电价
    """
    if 0 <= t < 6 or 22 <= t <= 24:
        return params['price']['off_peak']
    elif (6 <= t < 8) or (11 <= t < 18) or (21 <= t < 22):
        return params['price']['mid_peak']
    elif (8 <= t < 11) or (18 <= t < 19):
        return params['price']['on_peak']
    else:
        return params['price']['rush_hour']

def plot_scheduling_results(time, load, pv, battery_power, soc_values, net_load, 
                          save_dir='results', filename_prefix='output'):
    """可视化调度结果
    
    参数:
    time (np.array): 时间序列
    load (np.array): 负荷曲线
    pv (np.array): 光伏发电曲线
    battery_power (np.array): 电池功率曲线
    soc_values (np.array): SOC曲线
    net_load (np.array): 净负荷曲线
    save_dir (str): 保存目录路径
    filename_prefix (str): 文件前缀
    
    返回:
    None
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成标准化文件名
    plot_path = f"{save_dir}/{filename_prefix}_scheduling_plot.png"
    data_path = f"{save_dir}/{filename_prefix}_scheduling_data.csv"
    
    # 保存数据到CSV
    pd.DataFrame({
        'time': time,
        'load': load,
        'pv': pv,
        'battery_power': battery_power,
        'soc': soc_values,
        'net_load': net_load
    }).to_csv(data_path, index=False)
    
    # 绘制图表
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
    plt.savefig(plot_path, dpi=300)
    plt.close()

def load_curve_from_csv(file_path):
    """从CSV文件读取负荷曲线
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    np.array: 负荷曲线数据（96个点）
    """
    df = pd.read_csv(file_path)
    if len(df) != 96:
        raise ValueError("负荷曲线数据必须包含96个时间点")
    return df['load'].values

def pv_curve_from_csv(file_path):
    """从CSV文件读取光伏曲线
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    np.array: 光伏曲线数据（96个点）
    """
    df = pd.read_csv(file_path)
    if len(df) != 96:
        raise ValueError("光伏曲线数据必须包含96个时间点")
    return df['pv'].values
