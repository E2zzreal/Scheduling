"""
微电网调度场景分析模块

功能：
1. 使用蒙特卡洛模拟进行预测不确定性分析
2. 并行处理实现高效场景评估
3. 调度结果的统计分析
4. 场景结果的可视化

依赖库：
- numpy：数值计算
- pandas：数据处理
- joblib：并行计算
- h5py：数据存储
- matplotlib：数据可视化
"""
import numpy as np  # 导入numpy库用于数值计算
import pandas as pd  # 导入pandas库用于数据处理
import h5py  # 导入h5py库用于HDF5文件存储
import joblib  # 导入joblib库用于并行计算
import json  # 导入json库用于读取配置文件
from tqdm import tqdm  # 导入tqdm库用于进度条显示
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value, PULP_CBC_CMD  # 导入线性规划类及求解器
from utils import generate_load_curve, generate_pv_curve, get_electricity_price  # 导入自定义工具函数

import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import seaborn as sns  # 导入seaborn用于美化图表

class ScenarioAnalyzer:
    def __init__(self, params, num_scenarios=1000, true_load=None, true_pv=None):
        """
        初始化场景分析器

        参数：
        params：配置参数
        num_scenarios：场景数量，默认为1000
        true_load：预设的真实负荷曲线，若未提供则自动生成
        true_pv：预设的真实光伏曲线，若未提供则自动生成
        """
        self.params = params  # 保存配置参数
        self.num_scenarios = num_scenarios  # 设置场景数量
        self.results = None  # 初始化结果存储
        self.true_load = true_load if true_load is not None else generate_load_curve(
            np.linspace(0, 24, self.params['time_points'])
        )  # 设置真实负荷
        self.true_pv = true_pv if true_pv is not None else generate_pv_curve(
            np.linspace(0, 24, self.params['time_points'])
        )  # 设置真实光伏
        # 初始化 true_cost 属性，以便验证函数可以访问
        self.true_cost = None
        self.true_net_load = None
        self.true_schedule = None

    def generate_scenarios_accurancy(self, accuracy_level):
        """
        生成指定准确度水平的负荷和光伏场景

        参数：
        accuracy_level：预测准确度（0到1之间）

        返回：
        包含负荷和光伏曲线的场景列表
        """
        # 使用固定的真实负荷和光伏作为基准
        base_load = self.true_load
        base_pv = self.true_pv

        scenarios = []
        for _ in range(self.num_scenarios):
            # 根据准确度水平生成噪声
            load_noise = np.random.normal(0,
                (1 - accuracy_level) * self.params['load']['random_range'],
                size=self.params['time_points'])


            # 添加噪声到基准曲线
            scenarios.append({
                'load': base_load + load_noise,
                'pv': base_pv
            })
        return scenarios

    def generate_scenarios_error(self, error):
        """
        生成指定准确度水平的负荷和光伏场景

        参数：
        accuracy_level：预测准确度（0到1之间）

        返回：
        包含负荷和光伏曲线的场景列表
        """
        # 使用固定的真实负荷和光伏作为基准
        base_load = self.true_load
        base_pv = self.true_pv

        scenarios = []
        for _ in range(self.num_scenarios):
            # 根据准确度水平生成噪声
            load_noise = np.random.normal(error,
                0.5*error,
                size=self.params['time_points'])


            # 添加噪声到基准曲线
            scenarios.append({
                'load': base_load + np.random.choice([-1, 1])*load_noise,
                'pv': base_pv
            })
        return scenarios

    def _calculate_real_cost(self, schedule, true_load, true_pv):
        """
        计算实际成本

        参数：
        schedule：调度方案 (包含 'charge_power', 'discharge_power' 的字典)
        true_load：真实负荷
        true_pv：真实光伏

        返回：
        real_net_load (应用约束后的净负荷), total_cost (总成本)
        """
        # --- 成本计算核心逻辑 ---
        # 确保输入是 numpy 数组以进行向量化操作
        true_load = np.array(true_load)
        true_pv = np.array(true_pv)
        # 从 schedule 字典中获取功率列表
        schedule_charge = np.array(schedule['charge_power'])
        schedule_discharge = np.array(schedule['discharge_power'])

        if not (true_load.shape == true_pv.shape == schedule_charge.shape == schedule_discharge.shape):
             raise ValueError(f"Dimension mismatch in _calculate_real_cost: "
                              f"{true_load.shape}, {true_pv.shape}, {schedule_charge.shape}, {schedule_discharge.shape}")

        # 步骤 1: 计算原始净负荷 (Raw Net Load)
        # 这是基于真实负荷、真实光伏以及优化器给出的充放电计划计算的，未应用任何运行约束。
        raw_net_load = true_load - true_pv - schedule_discharge + schedule_charge

        # 步骤 2: 应用运行约束 (>= 30kW) 得到实际净负荷 (Real Net Load)
        # 使用 np.where 将所有低于 30kW 的原始净负荷强制拉高到 30kW。
        # 这是模拟实际运行时电网对最低负荷的要求。
        real_net_load = np.where(raw_net_load < 30, 30, raw_net_load)

        # 步骤 3: 计算总成本 (Total Cost)
        # 基于应用了约束后的实际净负荷 (real_net_load) 和对应时段的电价计算总费用。
        time_points = len(true_load)
        time_steps = np.linspace(0, 24, time_points, endpoint=False) # 确保时间步长与电价函数匹配
        total_cost = sum(real_net_load[i] * get_electricity_price(time_steps[i]) * 0.25 # 乘以时间间隔 (0.25小时)
                         for i in range(time_points))

        return real_net_load, total_cost
        # --- 成本计算核心逻辑结束 ---

    def run_analysis(self, type = 'accuracy', accuracy_levels=[0.7, 0.8, 0.9, 0.95], error_levels=[30,50,100,150]):
        """
        运行多准确度水平的场景分析

        参数：
        accuracy_levels：要分析的准确度水平列表

        返回：
        包含所有场景分析结果的字典
        """
        results = {}
        # 计算真实最优成本
        print("Calculating optimal schedule and cost (constraint >= 30)...")
        true_schedule_result = self._run_scenario({
            'load': self.true_load,
            'pv': self.true_pv
        })
        # self.true_schedule 现在包含 'schedule', 'total_cost_objective', 'soc_values' 等
        self.true_schedule = true_schedule_result
        print("Optimal schedule calculated.")

        # 使用 _calculate_real_cost 计算最优成本 (self.true_cost) 和最优最终净负荷 (self.true_net_load)
        # 注意：这里使用的是最优调度计划 (self.true_schedule['schedule']) 和 真实 负荷/光伏数据，
        # 并且 _calculate_real_cost 内部会应用 >= 30kW 的约束。
        # 因此，self.true_cost 代表的是完美预测下的 *实际* 运行成本。
        self.true_net_load, self.true_cost = self._calculate_real_cost(
            self.true_schedule['schedule'], self.true_load, self.true_pv
        )
        print(f"Optimal cost calculated via _calculate_real_cost: {self.true_cost:.2f}")

        # (可选) 提取最优电池功率用于绘图或其他分析
        self.true_battery_power = np.array(self.true_schedule['schedule']['charge_power']) - np.array(self.true_schedule['schedule']['discharge_power'])


        if type == 'accuracy':
            for accuracy in accuracy_levels:
                print(f"正在运行准确度水平为 {accuracy*100}% 的分析")
                scenarios = self.generate_scenarios_accurancy(accuracy)  # 生成场景
                        # 使用并行处理运行场景
                scenario_results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(self._run_scenario)(scenario)
                    for scenario in tqdm(scenarios)
                )

                # 计算每个场景的真实成本
                for result in scenario_results:
                    # result 包含该误差场景优化得到的 'schedule', 'total_cost_objective' 等
                    # 使用 _calculate_real_cost 计算该误差场景下的 *实际* 运行成本
                    # 输入：该场景的调度计划 (result['schedule'])，但使用 *真实* 负荷/光伏 (self.true_load, self.true_pv)
                    # 输出：该场景下的实际净负荷 (real_net_load_scenario) 和实际成本 (real_cost_scenario)
                    real_net_load_scenario, real_cost_scenario = self._calculate_real_cost(
                        result['schedule'],
                        self.true_load,
                        self.true_pv
                    )
                    # 将计算得到的实际结果存回该场景的 result 字典
                    result['real_net_load'] = real_net_load_scenario # 保存最终净负荷 (应用了 >=30 约束)
                    result['real_cost'] = real_cost_scenario # 保存实际成本

                results[accuracy] = scenario_results  # 保存该准确度下的所有场景结果

        elif type == 'error':
            for error in error_levels:
                print(f"正在运行误差水平为 {error} 的分析")
                scenarios = self.generate_scenarios_error(error) # 生成场景

                # 使用并行处理运行场景
                scenario_results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(self._run_scenario)(scenario)
                    for scenario in tqdm(scenarios)
                )

                # 计算每个场景的真实成本
                for result in scenario_results:
                    # 同上，使用该误差场景的调度计划和 *真实* 负荷/光伏，通过 _calculate_real_cost 计算实际成本
                    real_net_load_scenario, real_cost_scenario = self._calculate_real_cost(
                        result['schedule'],
                        self.true_load,
                        self.true_pv
                    )
                    # 存回结果
                    result['real_net_load'] = real_net_load_scenario
                    result['real_cost'] = real_cost_scenario

                results[error] = scenario_results  # 保存该误差水平下的所有场景结果

        self.results = results
        return results

    def _run_scenario(self, scenario, single_scenario=False):
        """
        运行单个场景的MILP优化

        参数：
        scenario：包含负荷和光伏曲线的场景
        single_scenario：是否为单场景优化模式

        返回：
        包含优化结果的字典
        """
        # 创建MILP问题实例
        prob = LpProblem("Scenario_Analysis", LpMinimize)

        # 如果是单场景模式，使用更详细的求解器输出
        solver = PULP_CBC_CMD(msg=1 if single_scenario else 0)

        # 定义变量
        battery = self.params['battery']
        time_points = self.params['time_points']

        soc = LpVariable.dicts("SOC", range(time_points), 0.1 * battery['capacity'], battery['capacity'])
        soc[0] = battery['initial_soc'] * battery['capacity']

        charge = LpVariable.dicts("Charge", range(time_points), 0, 0.5 * battery['max_power'])
        discharge = LpVariable.dicts("Discharge", range(time_points), 0, battery['max_power'])
        is_charging = LpVariable.dicts("IsCharging", range(time_points), cat="Binary")

        # 生成时间序列
        time = np.linspace(0, 24, time_points, endpoint=False) # 匹配 get_electricity_price

        # 目标函数
        prob += lpSum([(scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i]) *
                      get_electricity_price(time[i]) * 0.25 for i in range(time_points)])

        # 约束条件
        for i in range(time_points):
            # 充电功率约束
            prob += charge[i] <= battery['max_power'] * is_charging[i]
            # 放电功率约束
            prob += discharge[i] <= battery['max_power'] * (1 - is_charging[i])

            if i > 0:
                # SOC状态转移方程
                prob += soc[i] == soc[i-1] + charge[i]*0.25*0.95 - discharge[i]*0.25/0.95

            # 功率平衡约束
            prob += scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i] >= 30
            prob += scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i] <= 1000

            if i > 0:
                # SOC范围约束 (修正：确保 SOC 约束与转移方程一致)
                prob += soc[i-1] + charge[i]*0.25*0.95 - discharge[i]*0.25/0.95 >= 0.1 * battery['capacity']
                prob += soc[i-1] + charge[i]*0.25*0.95 - discharge[i]*0.25/0.95 <= battery['capacity']

            if i == time_points - 1:
                # 最终SOC约束
                prob += soc[i] == battery['initial_soc'] * battery['capacity']

        # 求解并返回结果
        prob.solve(solver)

        # 获取优化结果
        charge_opt = [value(charge[i]) for i in range(time_points)]
        discharge_opt = [value(discharge[i]) for i in range(time_points)]
        soc_opt = [value(soc[i]) for i in range(time_points)] # 保存绝对值

        # 如果是单场景模式，进行可视化和结果保存
        if single_scenario:
            from utils import plot_scheduling_results
            battery_power = np.array(charge_opt) - np.array(discharge_opt)
            soc_percent = np.array(soc_opt) / battery['capacity'] * 100
            # 注意：这里的 net_load 是优化器内部看到的，可能不完全等于后续计算的 raw_net_load
            net_load_optimizer = np.array(scenario['load']) - np.array(scenario['pv']) - np.array(discharge_opt) + np.array(charge_opt)
            plot_scheduling_results(time, scenario['load'], scenario['pv'],
                                 battery_power, soc_percent, net_load_optimizer,
                                 save_dir='results/single_scenario',
                                 filename_prefix='MILP')
            pd.DataFrame({
                'load': net_load_optimizer,
                'power': battery_power,
                'SOC_percent': soc_percent
            }).to_csv('results/single_scenario/net_load.csv')

        return {
            'schedule': {  # 保存调度方案
                'charge_power': charge_opt,
                'discharge_power': discharge_opt
            },
            'total_cost_objective': value(prob.objective),  # 优化器目标函数值
            'soc_values': soc_opt,  # SOC 绝对值
            # 'charge_power': charge_opt, # 重复，已在 schedule 中
            # 'discharge_power': discharge_opt, # 重复
            'predicted_load': scenario['load'] # 优化时使用的负荷
        }

    def save_results(self, type='accuracy'):
        """
        将分析结果保存到HDF5文件

        参数：
        filename：保存结果的文件路径
        """
        if self.results is None or self.true_cost is None:
             print("Error: Analysis results or true_cost not available. Run run_analysis first.")
             return

        if type == 'accuracy':
            filename='results/scenario_analysis/accuracy_based_scenario_analysis.h5'
            with h5py.File(filename, 'w') as f:
                # 保存全局真实数据
                f.attrs['true_cost'] = self.true_cost # 使用 run_analysis 中计算好的 self.true_cost
                f.create_dataset('true_load', data=self.true_load)
                f.create_dataset('true_pv', data=self.true_pv)
                f.create_dataset('true_net_load', data=self.true_net_load) # 使用 run_analysis 中计算好的
                # 保存最优调度计划和 SOC
                if hasattr(self, 'true_schedule') and self.true_schedule:
                     f.create_dataset('true_charge_power', data=self.true_schedule['schedule']['charge_power'])
                     f.create_dataset('true_discharge_power', data=self.true_schedule['schedule']['discharge_power'])
                     if 'soc_values' in self.true_schedule:
                         f.create_dataset('true_soc_values', data=self.true_schedule['soc_values']) # 保存绝对值
                     else:
                         print("Warning: Optimal SOC values not found in true_schedule for saving.")
                else:
                     print("Warning: True schedule data not available for saving in accuracy mode.")

                for accuracy, results_list in self.results.items():
                    # 为每个准确度水平创建组
                    grp = f.create_group(f"accuracy_{int(accuracy*100)}%")
                    # 保存各项结果数据
                    # 注意：从 run_analysis 保存的 result 字典中提取数据
                    grp.create_dataset('total_cost_objective', data=[r['total_cost_objective'] for r in results_list]) # 优化器目标值
                    grp.create_dataset('real_costs', data=[r['real_cost'] for r in results_list]) # 实际评估成本
                    grp.create_dataset('soc_values', data=[r['soc_values'] for r in results_list]) # 各场景 SOC 绝对值
                    grp.create_dataset('charge_power', data=[r['schedule']['charge_power'] for r in results_list])
                    grp.create_dataset('discharge_power', data=[r['schedule']['discharge_power'] for r in results_list])
                    grp.create_dataset('predicted_load', data=[r['predicted_load'] for r in results_list])
                    grp.create_dataset('real_net_load', data=[r['real_net_load'] for r in results_list]) # 实际评估净负荷
                    # 计算并保存成本差异
                    cost_differences = [r['real_cost'] - self.true_cost for r in results_list]
                    grp.create_dataset('cost_differences', data=cost_differences)

        elif type == 'error':
            filename='results/scenario_analysis/error_based_scenario_analysis.h5'
            with h5py.File(filename, 'w') as f:
                # 保存全局真实数据
                f.attrs['true_cost'] = self.true_cost
                f.create_dataset('true_load', data=self.true_load)
                f.create_dataset('true_pv', data=self.true_pv)
                f.create_dataset('true_net_load', data=self.true_net_load)
                # 保存最优调度计划和 SOC
                if hasattr(self, 'true_schedule') and self.true_schedule:
                     f.create_dataset('true_charge_power', data=self.true_schedule['schedule']['charge_power'])
                     f.create_dataset('true_discharge_power', data=self.true_schedule['schedule']['discharge_power'])
                     if 'soc_values' in self.true_schedule:
                         f.create_dataset('true_soc_values', data=self.true_schedule['soc_values'])
                     else:
                         print("Warning: Optimal SOC values not found in true_schedule for saving.")
                else:
                     print("Warning: True schedule data not available for saving in error mode.")

                for error, results_list in self.results.items():
                    # 为每个准确度水平创建组
                    grp = f.create_group(f"error_{error}kW")
                    # 保存各项结果数据
                    grp.create_dataset('total_cost_objective', data=[r['total_cost_objective'] for r in results_list])
                    grp.create_dataset('real_costs', data=[r['real_cost'] for r in results_list])
                    grp.create_dataset('soc_values', data=[r['soc_values'] for r in results_list])
                    grp.create_dataset('charge_power', data=[r['schedule']['charge_power'] for r in results_list])
                    grp.create_dataset('discharge_power', data=[r['schedule']['discharge_power'] for r in results_list])
                    grp.create_dataset('predicted_load', data=[r['predicted_load'] for r in results_list])
                    grp.create_dataset('real_net_load', data=[r['real_net_load'] for r in results_list])
                    # 计算并保存成本差异
                    cost_differences = [r['real_cost'] - self.true_cost for r in results_list]
                    grp.create_dataset('cost_differences', data=cost_differences)

        print(f"Results saved to {filename}")


    def plot_results(self, type = 'accuracy'):
        """
        生成场景分析结果的可视化图表
        """
        if self.results is None or self.true_cost is None:
             print("Error: Analysis results or true_cost not available. Run run_analysis first.")
             return

        time_points = self.params['time_points']  # 获取时间点数
        time = np.linspace(0, 24, time_points)  # 生成时间序列，从0到24小时

        if type == 'accuracy':
            # --- Accuracy Analysis Plotting (2x2 Layout) ---
            plt.figure(figsize=(18, 12)) # 保持 2x2 布局
            sns.set_style('whitegrid')

            # --- 子图 1: 成本分布 ---
            plt.subplot(2, 2, 1)
            for accuracy, results_list in self.results.items():
                costs = [r['real_cost'] for r in results_list]
                sns.kdeplot(costs, label=f'{int(accuracy*100)}% Accuracy')
            plt.axvline(self.true_cost, color='red', linestyle='--', label=f'True Cost ({self.true_cost:.2f})')
            plt.title('Real Cost Distribution')
            plt.xlabel('Real Cost')
            plt.ylabel('Density')
            plt.legend()

            # --- 子图 2: 实际净负荷 ---
            plt.subplot(2, 2, 2)
            for accuracy, results_list in self.results.items():
                avg_real_net_load = np.mean([r['real_net_load'] for r in results_list], axis=0)
                plt.plot(time, avg_real_net_load, label=f'{accuracy*100}% Accuracy Load')
            plt.plot(time, self.true_net_load, label='Optimal Load', linestyle='--')
            plt.title('Average Real Net Load Pattern (Constraint >= 30 Applied)')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()

            # --- 子图 3: SOC 统计 ---
            plt.subplot(2, 2, 3)
            soc_stats = []
            battery_capacity = self.params.get('battery', {}).get('capacity', 1000)
            for accuracy, results_list in self.results.items():
                soc_values_percent = np.array([r['soc_values'] for r in results_list]) / battery_capacity * 100
                soc_stats.append({
                    'accuracy': accuracy,
                    'mean': np.mean(soc_values_percent),
                    'std': np.std(soc_values_percent)
                })
            soc_stats_df = pd.DataFrame(soc_stats)
            sns.barplot(data=soc_stats_df, x='accuracy', y='mean', yerr=soc_stats_df['std'])
            plt.title('Average SOC (%) at Different Accuracy Levels')
            plt.xlabel('Accuracy Level')
            plt.ylabel('Average SOC (%)')

            # --- 子图 4: 充放电模式 ---
            plt.subplot(2, 2, 4)
            for accuracy, results_list in self.results.items():
                charge_power = np.mean([r['schedule']['charge_power'] for r in results_list], axis=0)
                discharge_power = np.mean([r['schedule']['discharge_power'] for r in results_list], axis=0)
                battery_power = charge_power - discharge_power
                plt.plot(time, battery_power, label=f'{accuracy*100}% Power')
            if hasattr(self, 'true_battery_power'):
                 plt.plot(time, self.true_battery_power, label = 'Optimal Power', linestyle='--')
            plt.title('Average Battery Charging/Discharging Pattern')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()

            plt.tight_layout()
            plt.savefig('results/scenario_analysis/accuracy_based_scenario_analysis.png', dpi=300)
            plt.close()
            print("Accuracy analysis plots saved.")

        elif type == 'error':
            # --- Error Analysis Plotting (2x3 Layout) ---
            fig, axes = plt.subplots(2, 3, figsize=(24, 12)) # 创建 2x3 子图网格
            sns.set_style('whitegrid')
            axes_flat = axes.flatten() # 展平以便索引

            # --- 子图 1 (0,0): 成本分布 ---
            ax1 = axes_flat[0]
            for error, results_list in self.results.items():
                costs = [r['real_cost'] for r in results_list]
                sns.kdeplot(costs, ax=ax1, label=f'{error}kW Error')
            ax1.axvline(self.true_cost, color='red', linestyle='--', label=f'True Cost ({self.true_cost:.2f})')
            ax1.set_title('Real Cost Distribution')
            ax1.set_xlabel('Real Cost')
            ax1.set_ylabel('Density')
            ax1.legend()

            # --- 子图 2 (0,1): 实际净负荷 ---
            ax2 = axes_flat[1]
            for error, results_list in self.results.items():
                avg_real_net_load = np.mean([r['real_net_load'] for r in results_list], axis=0)
                ax2.plot(time, avg_real_net_load, label=f'{error}kW Error Load')
            ax2.plot(time, self.true_net_load, label='Optimal Load', linestyle='--')
            ax2.set_title('Average Real Net Load Pattern (Constraint >= 30 Applied)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Power (kW)')
            ax2.legend()

            # --- 子图 3 (0,2): SOC 统计 ---
            ax3 = axes_flat[2]
            soc_stats = []
            battery_capacity = self.params.get('battery', {}).get('capacity', 1000)
            error_levels_sorted = sorted(self.results.keys()) # 确保箱线图 x 轴有序
            for error in error_levels_sorted:
                results_list = self.results[error]
                soc_values_percent = np.array([r['soc_values'] for r in results_list]) / battery_capacity * 100
                soc_stats.append({
                    'error': error,
                    'mean': np.mean(soc_values_percent),
                    'std': np.std(soc_values_percent)
                })
            soc_stats_df = pd.DataFrame(soc_stats)
            sns.barplot(data=soc_stats_df, x='error', y='mean', yerr=soc_stats_df['std'], ax=ax3)
            ax3.set_title('Average SOC (%) at Different Error Levels')
            ax3.set_xlabel('Error Level (kW)')
            ax3.set_ylabel('Average SOC (%)')

            # --- 子图 4 (1,0): 充放电模式 ---
            ax4 = axes_flat[3]
            for error in error_levels_sorted:
                results_list = self.results[error]
                charge_power = np.mean([r['schedule']['charge_power'] for r in results_list], axis=0)
                discharge_power = np.mean([r['schedule']['discharge_power'] for r in results_list], axis=0)
                battery_power = charge_power - discharge_power
                ax4.plot(time, battery_power, label=f'{error}kW Error Power')
            if hasattr(self, 'true_battery_power'):
                 ax4.plot(time, self.true_battery_power, label = 'Optimal Power', linestyle='--')
            ax4.set_title('Average Battery Charging/Discharging Pattern')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Power (kW)')
            ax4.legend()

            # --- 子图 5 (1,1): 成本差值分布 (新增) ---
            ax5 = axes_flat[4]
            cost_diff_data = []
            for error in error_levels_sorted:
                results_list = self.results[error]
                for r in results_list:
                    cost_diff = r['real_cost'] - self.true_cost
                    cost_diff_data.append({'Error Level': error, 'Cost Difference': cost_diff})
            cost_diff_df = pd.DataFrame(cost_diff_data)

            # 使用箱线图展示分布和均值
            sns.boxplot(data=cost_diff_df, x='Error Level', y='Cost Difference', ax=ax5, showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})

            # 计算并标注均值
            means = cost_diff_df.groupby('Error Level')['Cost Difference'].mean()
            # 动态调整标签垂直偏移量，基于 y 轴范围
            y_min, y_max = ax5.get_ylim()
            vertical_offset = (y_max - y_min) * 0.03 # 偏移量为 y 轴范围的 3%

            for i, error_level in enumerate(error_levels_sorted):
                 # 确保均值存在再添加文本
                 if error_level in means:
                     mean_val = means[error_level]
                     # 调整文本位置，避免重叠
                     ax5.text(i, mean_val + vertical_offset, f'Mean:\n{mean_val:.2f}',
                              horizontalalignment='center', size='small', color='black', weight='semibold',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)) # 添加背景框

            ax5.axhline(0, color='red', linestyle='--', label='Optimal Cost Level (Diff=0)') # 添加 0 参考线
            ax5.set_title('Cost Difference Distribution vs Optimal')
            ax5.set_xlabel('Error Level (kW)')
            ax5.set_ylabel('Cost Difference (€)')
            ax5.legend()

            # --- 子图 6 (1,2): 空白或备用 ---
            axes_flat[5].axis('off') # 保持空白

            plt.tight_layout()
            plt.savefig('results/scenario_analysis/error_based_scenario_analysis.png', dpi=300)
            plt.close()
            print("Error analysis plots saved.")

    # --- 新增：验证最优成本计算的函数 ---
    def verify_optimal_cost_calculation(self):
        """运行最优场景并详细打印成本计算过程"""
        print("\n--- Verifying Optimal Cost Calculation (Constraint >= 30) ---")
        # 1. 运行最优场景优化
        true_scenario = {'load': self.true_load, 'pv': self.true_pv}
        # 使用 single_scenario=False 避免求解器过多输出
        true_schedule_result = self._run_scenario(true_scenario, single_scenario=False)
        true_schedule_dict = true_schedule_result['schedule'] # 获取包含 charge/discharge 的字典
        print("Optimal schedule obtained.")

        # 2. 使用 _calculate_real_cost 计算成本
        #   a. 计算原始净负荷 (在 _calculate_real_cost 内部完成)
        #   b. 应用约束得到最终净负荷 (在 _calculate_real_cost 内部完成)
        #   c. 计算总成本
        real_net_load, calculated_cost = self._calculate_real_cost(
            true_schedule_dict, self.true_load, self.true_pv
        )
        print(f"Calculated Optimal Cost via _calculate_real_cost: {calculated_cost:.2f}")

        # 3. 打印一些详细信息以供比较
        # 重新计算 raw_net_load 以便检查
        raw_net_load = self.true_load - self.true_pv - np.array(true_schedule_dict['discharge_power']) + np.array(true_schedule_dict['charge_power'])
        violation_indices = np.where(raw_net_load < 30)[0]
        print(f"Points where raw_net_load < 30: {len(violation_indices)}")
        if len(violation_indices) > 0:
             print(f"Indices: {violation_indices}")
             print(f"Raw net load at these points: {np.round(raw_net_load[violation_indices], 2)}")
             print(f"Final net load (after np.where) at these points: {np.round(real_net_load[violation_indices], 2)}")

        print("--- Verification Finished ---")
        return calculated_cost
    # --- 验证函数结束 ---

if __name__ == "__main__":
    import argparse

    # 从JSON文件读取配置参数
    params_file = 'config/parameters.json'
    try:
        with open(params_file) as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"Error: Parameters file not found at {params_file}")
        exit() # 参数文件是必需的

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Microgrid Scenario Analysis Tool")
    parser.add_argument('--scenario', action='store_true', help='Run multi-scenario analysis (default if no flags)')
    parser.add_argument('--error', action='store_true', help='Use error-based scenario generation (requires --scenario)')
    parser.add_argument('--verify_cost', action='store_true', help='Run only the optimal cost verification')
    # 可以添加更多参数，例如指定 accuracy levels 或 error levels

    args = parser.parse_args()

    # 创建分析器实例
    analyzer = ScenarioAnalyzer(parameters)

    if args.verify_cost:
        # 只运行成本验证
        analyzer.verify_optimal_cost_calculation()
    elif args.scenario:
        # 运行多场景分析
        analysis_type = 'error' if args.error else 'accuracy' # 默认 accuracy
        print(f"Starting multi-scenario analysis (type: {analysis_type})...")
        analyzer.run_analysis(type=analysis_type) # 可以传递 accuracy/error levels 参数
        analyzer.save_results(type=analysis_type)
        analyzer.plot_results(type=analysis_type)
        print("Multi-scenario analysis complete.")
    else:
        # 默认行为：如果未指定 --scenario 或 --verify_cost，可以运行单场景优化或提示用户
        print("Running single scenario optimization (default action)...")
        # 使用 analyzer 实例中已有的 true_load 和 true_pv
        scenario = {'load': analyzer.true_load, 'pv': analyzer.true_pv}
        analyzer._run_scenario(scenario, single_scenario=True) # single_scenario=True 显示详细输出和绘图
