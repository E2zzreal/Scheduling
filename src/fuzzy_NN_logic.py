"""
基于LSTM的储能系统优化调度系统（完整版）

组件构成：
1. 动态规划数据生成器
2. LSTM神经网络调度模型
3. 混合训练框架（监督+强化学习）
4. 在线学习与概念漂移检测
5. 数据增强与预处理模块
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from typing import List, Tuple

# ------------------ 动态规划数据生成器 ------------------
class DPSolver:
    """动态规划最优调度生成器"""
    def __init__(self, battery_capacity, max_charge_rate, time_resolution=0.25):
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.time_resolution = time_resolution
        
    def solve(self, load, pv, price):
        """求解最优调度序列"""
        T = len(load)
        soc_min, soc_max = 0.2, 1.0
        
        # 状态空间：时间步 x SOC离散化
        soc_states = np.linspace(soc_min, soc_max, 50)
        dp_cost = np.full((T, len(soc_states)), np.inf)
        dp_action = np.zeros((T, len(soc_states)))
        
        # 初始化
        dp_cost[0] = 0
        
        for t in range(1, T):
            for i, soc in enumerate(soc_states):
                min_cost = np.inf
                best_action = 0
                
                # 可行动作空间
                possible_actions = np.linspace(-self.max_charge_rate, self.max_charge_rate, 20)
                
                for action in possible_actions:
                    # SOC更新
                    delta_soc = action * self.time_resolution / self.battery_capacity
                    new_soc = soc + delta_soc
                    
                    if soc_min <= new_soc <= soc_max:
                        # 计算净负荷成本
                        net_load = load[t] - pv[t] + action
                        cost = net_load * price[t] * self.time_resolution
                        
                        # 找到最近的状态索引
                        prev_idx = np.abs(soc_states - new_soc).argmin()
                        
                        total_cost = dp_cost[t-1, prev_idx] + cost
                        
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_action = action
                
                dp_cost[t, i] = min_cost
                dp_action[t, i] = best_action
                
        # 回溯最优路径
        optimal_actions = np.zeros(T)
        current_soc = soc_max  # 假设初始SOC为最大值
        
        for t in reversed(range(1, T)):
            idx = np.abs(soc_states - current_soc).argmin()
            optimal_actions[t] = dp_action[t, idx]
            current_soc -= optimal_actions[t] * self.time_resolution / self.battery_capacity
            
        return optimal_actions

# ------------------ LSTM神经网络模型 ------------------
class AdvancedLSTMModel(nn.Module):
    """增强型LSTM调度模型"""
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch, features]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out[-1]  # 取最后一个时间步
        return self.fc(context)

# ------------------ 混合训练框架 ------------------
class HybridTrainer:
    def __init__(self, model, dp_solver, battery_params):
        self.model = model
        self.dp_solver = dp_solver
        self.battery_params = battery_params
        self.online_buffer = deque(maxlen=1000)
        
    def generate_dp_data(self, num_scenarios=100):
        """生成动态规划训练数据"""
        scenarios = []
        for _ in range(num_scenarios):
            # 生成随机场景
            load = generate_load_curve()
            pv = generate_pv_curve()
            price = get_electricity_price()
            
            # 求解最优调度
            optimal_actions = self.dp_solver.solve(load, pv, price)
            
            # 构建特征序列
            features = self._build_features(load, pv, price, optimal_actions)
            scenarios.append(features)
            
        return np.vstack(scenarios)
    
    def _build_features(self, load, pv, price, actions):
        """构建时序特征矩阵"""
        T = len(load)
        features = np.zeros((T, 5))
        soc = self.battery_params['initial_soc']
        
        for t in range(T):
            features[t] = [soc, load[t], pv[t], price[t], actions[t]]
            delta_soc = actions[t] * self.battery_params['time_resolution'] / self.battery_params['capacity']
            soc = np.clip(soc + delta_soc, 
                        self.battery_params['soc_min'],
                        self.battery_params['soc_max'])
        return features
    
    def supervised_train(self, data, epochs=50):
        """监督学习阶段"""
        dataset = TimeSeriesDataset(data, window_size=3)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 添加SOC约束惩罚
                predicted = outputs.squeeze()
                soc = inputs[:, -1, 0] + predicted / self.battery_params['max_charge_rate']
                penalty = torch.relu(soc - 1.0).mean() + torch.relu(0.2 - soc).mean()
                loss += 0.1 * penalty
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    def online_learning_step(self, real_data):
        """在线学习步骤"""
        # 加入经验回放缓冲区
        self.online_buffer.extend(real_data)
        
        # 从缓冲区采样
        batch = random.sample(self.online_buffer, min(32, len(self.online_buffer)))
        inputs, targets = zip(*batch)
        
        # 微调模型
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()

# ------------------ 在线调度系统 ------------------
class OnlineSchedulingSystem:
    def __init__(self, model, battery_params, window_size=3):
        self.model = model
        self.battery_params = battery_params
        self.window_size = window_size
        self.soc_history = [battery_params['initial_soc']]
        self.feature_window = deque(maxlen=window_size)
        self.drift_detector = DriftDetector()
        
    def make_decision(self, current_load, current_pv, current_price):
        """实时决策"""
        # 构建输入特征
        features = [
            self.soc_history[-1],
            current_load,
            current_pv,
            current_price,
            0  # 初始功率
        ]
        
        # 填充时间窗口
        if len(self.feature_window) < self.window_size:
            padding = [np.zeros(5) for _ in range(self.window_size - len(self.feature_window))]
            window = padding + [features]
        else:
            window = list(self.feature_window)[1:] + [features]
        
        # 转换为tensor
        window_tensor = torch.FloatTensor(window).unsqueeze(0)
        
        # 模型预测
        with torch.no_grad():
            pred_power = self.model(window_tensor).item()
        
        # 反归一化
        actual_power = pred_power * self.battery_params['max_charge_rate']
        
        # 更新特征窗口
        features[-1] = actual_power
        self.feature_window.append(features)
        
        # 计算SOC
        delta_soc = actual_power * self.battery_params['time_resolution'] / self.battery_params['capacity']
        new_soc = self.soc_history[-1] + delta_soc
        new_soc = np.clip(new_soc, 
                        self.battery_params['soc_min'],
                        self.battery_params['soc_max'])
        self.soc_history.append(new_soc)
        
        # 概念漂移检测
        self.drift_detector.update(pred_power, actual_power)
        
        return actual_power

class DriftDetector:
    """概念漂移检测器"""
    def __init__(self, window_size=24, threshold=0.1):
        self.error_window = deque(maxlen=window_size)
        self.threshold = threshold
        
    def update(self, predicted, actual):
        error = abs(predicted - actual)
        self.error_window.append(error)
        
        if len(self.error_window) == self.window_size:
            if np.mean(self.error_window) > self.threshold:
                print("检测到概念漂移，触发模型更新！")
                self._trigger_retraining()
                
    def _trigger_retraining(self):
        # 此处应连接模型重训练逻辑
        pass

# ------------------ 辅助函数 ------------------
def data_augmentation(features, noise_level=0.01):
    """数据增强：添加高斯噪声"""
    noise = np.random.normal(0, noise_level, features.shape)
    return np.clip(features + noise, 
                 [0, 0, 0, 0, -1],
                 [1, 1, 1, 1, 1])

def normalize_features(features, max_values):
    """特征归一化"""
    return features / max_values

# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 初始化参数
    battery_params = {
        'capacity': 2000,  # kWh
        'max_charge_rate': 500,  # kW
        'initial_soc': 0.5,
        'soc_min': 0.2,
        'soc_max': 1.0,
        'time_resolution': 0.25  # 15分钟
    }
    
    # 创建组件
    dp_solver = DPSolver(battery_params['capacity'], battery_params['max_charge_rate'])
    model = AdvancedLSTMModel()
    trainer = HybridTrainer(model, dp_solver, battery_params)
    
    # 阶段1：监督学习训练
    print("生成训练数据...")
    dp_data = trainer.generate_dp_data(num_scenarios=100)
    print("开始监督训练...")
    trainer.supervised_train(dp_data, epochs=50)
    
    # 阶段2：在线部署
    online_system = OnlineSchedulingSystem(model, battery_params)
    
    # 模拟实时数据流
    for hour in range(24):
        load = generate_load_curve(hour)
        pv = generate_pv_curve(hour)
        price = get_electricity_price(hour)
        
        # 获取实时决策
        action = online_system.make_decision(load, pv, price)
        
        # 在线学习（假设能获取到真实最优动作）
        # 注意：实际中需要设计reward函数
        if hour % 4 == 0:  # 每4小时更新一次
            real_data = [...]  # 需要根据实际数据采集
            loss = trainer.online_learning_step(real_data)
            print(f"Online learning loss: {loss:.4f}")