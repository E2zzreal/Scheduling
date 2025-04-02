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
        schedule：调度方案
        true_load：真实负荷
        true_pv：真实光伏
        
        返回：
        实际成本
        """
        # 计算原始净负荷
        raw_net_load = true_load - true_pv - schedule['discharge_power'] + schedule['charge_power']
        # 应用下限约束
        real_net_load = np.where(raw_net_load < 30, 30, raw_net_load)
        return real_net_load, sum(real_net_load[i] * get_electricity_price(i) * 0.25 
                  for i in range(len(true_load)))

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
        true_schedule = self._run_scenario({
            'load': self.true_load,
            'pv': self.true_pv
        })
        self.true_schedule = true_schedule # <--- 新增: 保存最优调度结果
        self.true_battery_power = np.array(true_schedule['charge_power']) - np.array(true_schedule['discharge_power'])
        self.true_net_load, self.true_cost = self._calculate_real_cost(true_schedule, self.true_load, self.true_pv)
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
                    result['true_load'], result['real_cost'] = self._calculate_real_cost(
                        result['schedule'], 
                        self.true_load,
                        self.true_pv
                    )
                
                results[accuracy] = scenario_results  # 保存结果

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
                    result['true_load'], result['real_cost'] = self._calculate_real_cost(
                        result['schedule'], 
                        self.true_load,
                        self.true_pv
                    )
                
                results[error] = scenario_results  # 保存结果
            
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
        time = np.linspace(0, 24, time_points)
        
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
            
            # 功率平衡约束 (临时修改：将下限改为 0)
            prob += scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i] >= 0
            prob += scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i] <= 1000
            
            if i > 0:
                # SOC范围约束
                prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 >= 0.1 * battery['capacity']
                prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 <= battery['capacity']
            
            if i == time_points - 1:
                # 最终SOC约束
                prob += soc[i] == battery['initial_soc'] * battery['capacity']
        
        # 求解并返回结果
        prob.solve(solver)
        
        # 获取优化结果
        charge_opt = np.array([value(charge[i]) for i in range(time_points)])
        discharge_opt = np.array([value(discharge[i]) for i in range(time_points)])
        battery_power = charge_opt - discharge_opt
        soc_values = np.array([value(soc[i]) for i in range(time_points)]) / battery['capacity'] * 100
        
        # 如果是单场景模式，进行可视化和结果保存
        if single_scenario:
            from utils import plot_scheduling_results
            time = np.linspace(0, 24, time_points)
            net_load = scenario['load'] - scenario['pv'] - discharge_opt + charge_opt
            plot_scheduling_results(time, scenario['load'], scenario['pv'], 
                                 battery_power, soc_values, net_load,
                                 save_dir='results/single_scenario',
                                 filename_prefix='MILP')
            pd.DataFrame({
                'load': net_load,
                'power': battery_power,
                'SOC': soc_values
            }).to_csv('results/single_scenario/net_load.csv')
        
        return {
            'schedule': {  # 保存调度方案
                'charge_power': [value(charge[i]) for i in range(time_points)],
                'discharge_power': [value(discharge[i]) for i in range(time_points)]
            },
            'total_cost': value(prob.objective),  # 预测成本
            'soc_values': [value(soc[i]) for i in range(time_points)],  # SOC值
            'charge_power': [value(charge[i]) for i in range(time_points)],  # 充电功率
            'discharge_power': [value(discharge[i]) for i in range(time_points)],  # 放电功率
            'predicted_load': scenario['load']  # 预测负荷
        }

    def save_results(self, type='accuracy'):
        """
        将分析结果保存到HDF5文件
        
        参数：
        filename：保存结果的文件路径
        """
        if type == 'accuracy':
            filename='results/scenario_analysis/accuracy_based_scenario_analysis.h5'
            with h5py.File(filename, 'w') as f:
                # 保存全局真实数据
                f.attrs['true_cost'] = self.true_cost
                f.create_dataset('true_load', data=self.true_load)
                f.create_dataset('true_pv', data=self.true_pv)
                f.create_dataset('true_net_load', data=self.true_net_load)
                # 新增：保存最优充放电功率和 SOC
                if hasattr(self, 'true_schedule'):
                     f.create_dataset('true_charge_power', data=self.true_schedule['charge_power'])
                     f.create_dataset('true_discharge_power', data=self.true_schedule['discharge_power'])
                     # --- 同时保存最优 SOC ---
                     if 'soc_values' in self.true_schedule:
                         f.create_dataset('true_soc_values', data=self.true_schedule['soc_values'])
                     else:
                         print("Warning: Optimal SOC values not found in true_schedule.")
                     # --- 保存 SOC 结束 ---
                else:
                     print("Warning: True schedule data not available for saving in accuracy mode.")

                for accuracy, results in self.results.items():
                    # 为每个准确度水平创建组
                    grp = f.create_group(f"accuracy_{int(accuracy*100)}%")
                    # 保存各项结果数据
                    grp.create_dataset('total_costs', data=[r['total_cost'] for r in results])
                    grp.create_dataset('real_costs', data=[r['real_cost'] for r in results])
                    grp.create_dataset('soc_values', data=[r['soc_values'] for r in results])
                    grp.create_dataset('charge_power', data=[r['charge_power'] for r in results])
                    grp.create_dataset('discharge_power', data=[r['discharge_power'] for r in results])
                    grp.create_dataset('predicted_load', data=[r['predicted_load'] for r in results])
                    grp.create_dataset('true_net_load', data=[r['true_load'] for r in results])
                    # 计算并保存成本差异
                    cost_differences = [r['real_cost'] - self.true_cost for r in results]
                    grp.create_dataset('cost_differences', data=cost_differences)
        elif type == 'error':
            filename='results/scenario_analysis/error_based_scenario_analysis.h5'
            with h5py.File(filename, 'w') as f:
                # 保存全局真实数据
                f.attrs['true_cost'] = self.true_cost
                f.create_dataset('true_load', data=self.true_load)
                f.create_dataset('true_pv', data=self.true_pv)
                f.create_dataset('true_net_load', data=self.true_net_load)
                # 新增：保存最优充放电功率和 SOC
                if hasattr(self, 'true_schedule'):
                     f.create_dataset('true_charge_power', data=self.true_schedule['charge_power'])
                     f.create_dataset('true_discharge_power', data=self.true_schedule['discharge_power'])
                     # --- 同时保存最优 SOC ---
                     if 'soc_values' in self.true_schedule:
                         f.create_dataset('true_soc_values', data=self.true_schedule['soc_values'])
                     else:
                         print("Warning: Optimal SOC values not found in true_schedule.")
                     # --- 保存 SOC 结束 ---
                else:
                     print("Warning: True schedule data not available for saving in error mode.")

                for error, results in self.results.items():
                    # 为每个准确度水平创建组
                    grp = f.create_group(f"error_{error}kW")
                    # 保存各项结果数据
                    grp.create_dataset('total_costs', data=[r['total_cost'] for r in results])
                    grp.create_dataset('real_costs', data=[r['real_cost'] for r in results])
                    grp.create_dataset('soc_values', data=[r['soc_values'] for r in results])
                    grp.create_dataset('charge_power', data=[r['charge_power'] for r in results])
                    grp.create_dataset('discharge_power', data=[r['discharge_power'] for r in results])
                    grp.create_dataset('predicted_load', data=[r['predicted_load'] for r in results])
                    grp.create_dataset('true_net_load', data=[r['true_load'] for r in results])
                    # 计算并保存成本差异
                    cost_differences = [r['real_cost'] - self.true_cost for r in results]
                    grp.create_dataset('cost_differences', data=cost_differences)

    def plot_results(self, type = 'accuracy'):
        """
        生成场景分析结果的可视化图表
        """
        time_points = self.params['time_points']  # 获取时间点数
        time = np.linspace(0, 24, time_points)  # 生成时间序列，从0到24小时
        # 设置图表布局
        plt.figure(figsize=(18, 12))
        sns.set_style('whitegrid')
        
        if type == 'accuracy':
            # 绘制成本分布图
            plt.subplot(2, 2, 1)
            for accuracy, results in self.results.items():
                costs = [r['real_cost'] for r in results]
                sns.kdeplot(costs, label=f'{int(accuracy*100)}% Accuracy')
            # 添加真实成本参考线
            plt.axvline(self.true_cost, color='red', linestyle='--', label='True Cost')
            plt.title('Real Cost Distribution')
            plt.xlabel('Real Cost')
            plt.ylabel('Density')
            plt.legend()
            
            #绘制实际净负荷统计图
            plt.subplot(2, 2, 2)
            for accuracy, results in self.results.items():
                True_net_load = np.mean([r['true_load'] for r in results], axis=0)
                plt.plot(time, True_net_load, label=f'{accuracy}% Accuracy Load')
            plt.plot(time, self.true_net_load, label='Optimal Load', linestyle='--')
            plt.title('Average Ture Load Pattern')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()
            
            # 绘制SOC统计图
            plt.subplot(2, 2, 3)
            soc_stats = []
            for accuracy, results in self.results.items():
                soc_values = np.array([r['soc_values'] for r in results])
                soc_stats.append({
                    'accuracy': accuracy,
                    'mean': np.mean(soc_values),
                    'std': np.std(soc_values)
                })
            soc_stats = pd.DataFrame(soc_stats)
            sns.barplot(data=soc_stats, x='accuracy', y='mean', yerr=soc_stats['std'])
            plt.title('Average SOC at Different Accuracy Levels')
            plt.xlabel('Accuracy Level')
            plt.ylabel('Average SOC (%)')
            
            # 绘制充放电模式图
            plt.subplot(2, 2, 4)
            for accuracy, results in self.results.items():
                charge_power = np.mean([r['charge_power'] for r in results], axis=0)
                discharge_power = np.mean([r['discharge_power'] for r in results], axis=0)
                battery_power = charge_power - discharge_power 
                plt.plot(time, battery_power, label=f'{accuracy}% Power')
            plt.plot(time, self.true_battery_power, label = 'Optimal Power', linestyle='--')
            plt.title('Average Battery Charging/Discharging Pattern')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()
            
            plt.tight_layout()  # 调整布局
            plt.savefig('results/scenario_analysis/accurancy_scenario_analysis.png', dpi=300)  # 保存图表
            plt.close()  # 关闭图表

        elif type == 'error':
            # 绘制成本分布图
            plt.subplot(2, 2, 1)
            for error, results in self.results.items():
                costs = [r['real_cost'] for r in results]
                sns.kdeplot(costs, label=f'{error}KW Error')
            # 添加真实成本参考线
            plt.axvline(self.true_cost, color='red', linestyle='--', label='True Cost')
            plt.title('Real Cost Distribution')
            plt.xlabel('Real Cost')
            plt.ylabel('Density')
            plt.legend()

            #绘制实际净负荷统计图
            plt.subplot(2, 2, 2)
            for error, results in self.results.items():
                True_net_load = np.mean([r['true_load'] for r in results], axis=0)
                plt.plot(time, True_net_load, label=f'{error}kW Error Load')
            plt.plot(time, self.true_net_load, label='Optimal Load', linestyle='--')
            plt.title('Average Ture Load Pattern')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()
            
            # 绘制SOC统计图
            plt.subplot(2, 2, 3)
            soc_stats = []
            for error, results in self.results.items():
                soc_values = np.array([r['soc_values'] for r in results])
                soc_stats.append({
                    'error': error,
                    'mean': np.mean(soc_values),
                    'std': np.std(soc_values)
                })
            soc_stats = pd.DataFrame(soc_stats)
            sns.barplot(data=soc_stats, x='error', y='mean', yerr=soc_stats['std'])
            plt.title('Average SOC at Different Accuracy Levels')
            plt.xlabel('Accuracy Level')
            plt.ylabel('Average SOC (%)')
            
            # 绘制充放电模式图
            plt.subplot(2, 2, 4)
            for error, results in self.results.items():
                charge_power = np.mean([r['charge_power'] for r in results], axis=0)
                discharge_power = np.mean([r['discharge_power'] for r in results], axis=0)
                battery_power = charge_power - discharge_power 
                plt.plot(time, battery_power, label=f'{error}KW Error Power')
            plt.plot(time, self.true_battery_power, label = 'Optimal Power', linestyle='--')
            plt.title('Average Battery Charging/Discharging Pattern')
            plt.xlabel('Time')
            plt.ylabel('Power (kW)')
            plt.legend()
            
            plt.tight_layout()  # 调整布局
            plt.savefig('results/scenario_analysis/error_based_scenario_analysis.png', dpi=300)  # 保存图表
            plt.close()  # 关闭图表

if __name__ == "__main__":
    import argparse
    
    # 从JSON文件读取配置参数
    with open('config/parameters.json') as f:
        parameters = json.load(f)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', action='store_true', help='Run scenario analysis')
    # parser.add_argument('--accuracy', type=float, help='Run accuracy analysis')
    parser.add_argument('--error', action='store_true', help='Run error based analysis')
    args = parser.parse_args()
    
    analyzer = ScenarioAnalyzer(parameters)  # 创建场景分析器
    
    if args.scenario:  # 如果指定了场景分析模式
        if args.error:
            analyzer.run_analysis(type='error')
            analyzer.save_results(type='error')  # 保存分析结果
            analyzer.plot_results(type='error')  # 生成可视化图表
        else:
            analyzer.run_analysis()  # 运行场景分析
            analyzer.save_results()  # 保存分析结果
            analyzer.plot_results()  # 生成可视化图表
    else:  # 否则运行单场景优化 (修改为运行真实负荷场景，约束 >= 0)
        print("Running single scenario optimization with true load and pv (relaxed constraint >= 0)...")
        # 使用 analyzer 实例中已有的 true_load 和 true_pv
        scenario = {'load': analyzer.true_load, 'pv': analyzer.true_pv}
        # 调用 _run_scenario 并获取结果，设置 single_scenario=True 获取详细输出
        relaxed_schedule_result_0 = analyzer._run_scenario(scenario, single_scenario=True)

        # --- 在这里添加分析新结果的代码 ---
        print("\n--- Analyzing Relaxed Constraint (>=0) Results ---")
        # 获取放松约束后的调度计划
        schedule_relaxed_0 = relaxed_schedule_result_0['schedule']

        # 计算新计划下的 raw_net_load_relaxed
        try:
            # 尝试从 analysis 模块导入，如果失败则打印错误
            try:
                from analysis.error_analysis import calculate_raw_net_load
            except ImportError:
                 # 如果直接运行此脚本可能找不到 analysis 模块，尝试相对导入或调整路径
                 # 作为备选，可以直接复制 calculate_raw_net_load 函数到这里，但不推荐
                 print("Warning: Could not import calculate_raw_net_load from analysis.error_analysis.")
                 # 定义一个临时的本地版本作为后备
                 def calculate_raw_net_load(true_load, true_pv, charge_power, discharge_power):
                     return np.array(true_load) - np.array(true_pv) - np.array(discharge_power) + np.array(charge_power)

            raw_net_load_relaxed_0 = calculate_raw_net_load(
                analyzer.true_load, analyzer.true_pv,
                np.array(schedule_relaxed_0['charge_power']), np.array(schedule_relaxed_0['discharge_power'])
            )
            # 之前定位的违规点索引 (原始 >= 30 约束下的)
            original_violation_indices = [31, 34, 35, 36, 39, 41, 42, 72, 73, 75, 77, 79, 80, 81, 82, 83]
            print(f"Relaxed (>=0) Raw Net Load at original violation points: {np.round(raw_net_load_relaxed_0[original_violation_indices], 2)}")
            # 检查是否有新的低于 30 的点
            new_violations = raw_net_load_relaxed_0 < 30
            print(f"Relaxed (>=0) Total points below 30: {np.sum(new_violations)}")
            if np.sum(new_violations) > 0:
                 print(f"Relaxed (>=0) Min raw net load: {np.min(raw_net_load_relaxed_0):.2f}")


            # 使用原始的 _calculate_real_cost 计算最终成本 (它仍然强制 >= 30)
            _, cost_relaxed_0 = analyzer._calculate_real_cost(schedule_relaxed_0, analyzer.true_load, analyzer.true_pv)
            print(f"Relaxed (>=0) Schedule Cost (evaluated with >=30 constraint): {cost_relaxed_0:.2f}")

        except Exception as e:
            print(f"Error during relaxed (>=0) analysis: {e}")

        print("--- Relaxed Constraint (>=0) Analysis Finished ---")
