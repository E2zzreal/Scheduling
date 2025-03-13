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
from tqdm import tqdm  # 导入tqdm库用于进度条显示
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, value, PULP_CBC_CMD  # 从公共模块导入线性规划类及求解器
from utils import generate_load_curve, generate_pv_curve, get_electricity_price  # 导入自定义工具函数

class ScenarioAnalyzer:
    def __init__(self, params, num_scenarios=1000):
        """
        初始化场景分析器
        
        参数：
        params：配置参数
        num_scenarios：场景数量，默认为1000
        """
        self.params = params  # 保存配置参数
        self.num_scenarios = num_scenarios  # 设置场景数量
        self.results = None  # 初始化结果存储
        
    def generate_scenarios(self, accuracy_level):
        """
        生成指定准确度水平的负荷和光伏场景
        
        参数：
        accuracy_level：预测准确度（0到1之间）
        
        返回：
        包含负荷和光伏曲线的场景列表
        """
        # 生成基准负荷和光伏曲线
        base_load = generate_load_curve(np.linspace(0, 24, self.params['time_points']))
        base_pv = generate_pv_curve(np.linspace(0, 24, self.params['time_points']))
        
        scenarios = []
        for _ in range(self.num_scenarios):
            # 根据准确度水平生成噪声
            load_noise = np.random.normal(0, 
                (1 - accuracy_level) * self.params['load']['random_range'],
                size=self.params['time_points'])
                
            pv_noise = np.random.normal(0,
                (1 - accuracy_level) * self.params['pv']['random_range'],
                size=self.params['time_points'])
                
            # 添加噪声到基准曲线
            scenarios.append({
                'load': base_load + load_noise,
                'pv': base_pv + pv_noise
            })
        return scenarios

    def run_analysis(self, accuracy_levels=[0.7, 0.8, 0.9, 0.95]):
        """
        运行多准确度水平的场景分析
        
        参数：
        accuracy_levels：要分析的准确度水平列表
        
        返回：
        包含所有场景分析结果的字典
        """
        results = {}
        
        for accuracy in accuracy_levels:
            print(f"正在运行准确度水平为 {accuracy*100}% 的分析")
            scenarios = self.generate_scenarios(accuracy)  # 生成场景
            
            # 使用并行处理运行场景
            scenario_results = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self._run_scenario)(scenario)
                for scenario in tqdm(scenarios)
            )
            
            results[accuracy] = scenario_results  # 保存结果
            
        self.results = results
        return results

    def _run_scenario(self, scenario):
        """
        运行单个场景的MILP优化
        
        参数：
        scenario：包含负荷和光伏曲线的场景
        
        返回：
        包含优化结果的字典
        """
        # 创建MILP问题实例
        prob = LpProblem("Scenario_Analysis", LpMinimize)
        
        # 定义变量
        battery = self.params['battery']
        time_points = self.params['time_points']
        
        soc = LpVariable.dicts("SOC", range(time_points), 0.1 * battery['capacity'], battery['capacity'])
        soc[0] = battery['initial_soc'] * battery['capacity']
        
        charge = LpVariable.dicts("Charge", range(time_points), 0, 0.5 * battery['max_power'])
        discharge = LpVariable.dicts("Discharge", range(time_points), 0, battery['max_power'])
        is_charging = LpVariable.dicts("IsCharging", range(time_points), cat="Binary")
        
        # 目标函数
        prob += lpSum([(scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i]) * 
                      get_electricity_price(i) * 0.25 for i in range(time_points)])
        
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
            prob += scenario['load'][i] - scenario['pv'][i] - discharge[i] + charge[i] <= 700
            
            if i > 0:
                # SOC范围约束
                prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 >= 0.1 * battery['capacity']
                prob += soc[i-1] + charge[i]*0.25 - discharge[i]*0.25 <= battery['capacity']
            
            if i == time_points - 1:
                # 最终SOC约束
                prob += soc[i] == battery['initial_soc'] * battery['capacity']
        
        # 求解并返回结果
        prob.solve(PULP_CBC_CMD(msg=0))  # 调用求解器并禁用详细输出
        return {
            'total_cost': value(prob.objective),  # 总成本
            'soc_values': [value(soc[i]) for i in range(time_points)],  # SOC值
            'charge_power': [value(charge[i]) for i in range(time_points)],  # 充电功率
            'discharge_power': [value(discharge[i]) for i in range(time_points)]  # 放电功率
        }

    def save_results(self, filename='results/scenario_analysis.h5'):
        """
        将分析结果保存到HDF5文件
        
        参数：
        filename：保存结果的文件路径
        """
        with h5py.File(filename, 'w') as f:
            for accuracy, results in self.results.items():
                # 为每个准确度水平创建组
                grp = f.create_group(f"accuracy_{int(accuracy*100)}")
                # 保存各项结果数据
                grp.create_dataset('total_costs', data=[r['total_cost'] for r in results])
                grp.create_dataset('soc_values', data=[r['soc_values'] for r in results])
                grp.create_dataset('charge_power', data=[r['charge_power'] for r in results])
                grp.create_dataset('discharge_power', data=[r['discharge_power'] for r in results])

    def plot_results(self):
        """
        生成场景分析结果的可视化图表
        """
        import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
        import seaborn as sns  # 导入seaborn用于美化图表
        
        # 设置图表布局
        plt.figure(figsize=(15, 10))
        sns.set_style('whitegrid')
        
        # 绘制成本分布图
        plt.subplot(2, 2, 1)
        for accuracy, results in self.results.items():
            costs = [r['total_cost'] for r in results]
            sns.kdeplot(costs, label=f'{int(accuracy*100)}% Accuracy')
        plt.title('Total Cost Distribution')
        plt.xlabel('Total Cost')
        plt.ylabel('Density')
        plt.legend()
        
        # 绘制SOC统计图
        plt.subplot(2, 2, 2)
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
        plt.subplot(2, 1, 2)
        for accuracy, results in self.results.items():
            charge_power = np.mean([r['charge_power'] for r in results], axis=0)
            discharge_power = np.mean([r['discharge_power'] for r in results], axis=0)
            plt.plot(charge_power, label=f'{int(accuracy*100)}% Charge')
            plt.plot(discharge_power, '--', label=f'{int(accuracy*100)}% Discharge')
        plt.title('Average Charging/Discharging Pattern')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.legend()
        
        plt.tight_layout()  # 调整布局
        plt.savefig('results/scenario_analysis.png')  # 保存图表
        plt.close()  # 关闭图表

if __name__ == "__main__":
    from config import parameters  # 导入配置文件
    analyzer = ScenarioAnalyzer(parameters)  # 创建场景分析器
    analyzer.run_analysis()  # 运行分析
    analyzer.save_results()  # 保存分析结果
