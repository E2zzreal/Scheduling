"""
基于LSTM的储能系统优化调度（完整实现版）
包含动态规划加速、安全约束、模型版本控制、在线学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from numba import jit
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import random

# ------------------ 工具函数 ------------------
def generate_load_curve(time_points):
    """生成模拟负荷曲线"""
    base_load = 300 + 200 * np.sin(np.pi * time_points / 12)
    noise = 50 * np.random.randn(len(time_points))
    return np.clip(base_load + noise, 200, 1000).astype(np.float32)

def generate_pv_curve(time_points):
    """生成模拟光伏曲线"""
    pv = 400 * np.sin(np.pi * (time_points - 6)/12)
    pv[pv < 0] = 0
    noise = 20 * np.random.randn(len(time_points))
    return np.clip(pv + noise, 0, 450).astype(np.float32)

def get_electricity_price(time_points):
    """获取分时电价"""
    price = np.ones_like(time_points) * 0.5
    peak_hours = (time_points >= 18) | (time_points <= 9)
    price[peak_hours] = 0.8
    return price.astype(np.float32)

# ------------------ 加速动态规划 ------------------
@jit(nopython=True, parallel=True)
def dp_core_loop(dp_cost, dp_action, soc_states, possible_actions,
                 load, pv, price, battery_capacity, time_resolution):
    T, num_states = dp_cost.shape
    soc_min, soc_max = 0.2, 1.0

    for t in range(1, T):
        for i in range(num_states):
            soc = soc_states[i]
            min_cost = np.inf
            best_action = 0.0
            
            for action in possible_actions:
                delta_soc = action * time_resolution / battery_capacity
                new_soc = soc + delta_soc
                
                if soc_min <= new_soc <= soc_max:
                    prev_idx = np.argmin(np.abs(soc_states - new_soc))
                    net_load = load[t] - pv[t] + action
                    cost = net_load * price[t] * time_resolution
                    total_cost = dp_cost[t-1, prev_idx] + cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_action = action
            
            dp_cost[t, i] = min_cost
            dp_action[t, i] = best_action
    return dp_cost, dp_action

class AcceleratedDPSolver:
    """Numba加速的动态规划求解器"""
    def __init__(self, battery_params):
        self.capacity = battery_params['capacity']
        self.max_rate = battery_params['max_charge_rate']
        self.time_res = battery_params['time_resolution']
        
    def solve(self, load, pv, price):
        T = len(load)
        soc_states = np.linspace(0.2, 1.0, 50)
        possible_actions = np.linspace(-self.max_rate, self.max_rate, 20)
        
        dp_cost = np.full((T, len(soc_states)), np.inf)
        dp_action = np.zeros((T, len(soc_states)))
        dp_cost[0] = 0
        
        dp_cost, dp_action = dp_core_loop(
            dp_cost, dp_action, soc_states.astype(np.float64),
            possible_actions.astype(np.float64),
            load.astype(np.float64), pv.astype(np.float64),
            price.astype(np.float64),
            np.float64(self.capacity),
            np.float64(self.time_res)
        )
        
        return self._trace_back(dp_action, soc_states)

    def _trace_back(self, dp_action, soc_states):
        T = len(dp_action)
        optimal_actions = np.zeros(T)
        current_soc = 1.0
        
        for t in reversed(range(1, T)):
            idx = np.argmin(np.abs(soc_states - current_soc))
            optimal_actions[t] = dp_action[t, idx]
            current_soc -= optimal_actions[t] * self.time_res / self.capacity
            
        return optimal_actions

# ------------------ 神经网络模型 ------------------
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
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out[-1]
        return self.fc(context)

# ------------------ 数据管道 ------------------
class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, data, window_size=3):
        self.data = data
        self.window_size = window_size
        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.data)-self.window_size):
            X.append(self.data[i:i+self.window_size])
            y.append(self.data[i+self.window_size, -1])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------ 训练框架 ------------------
class HybridTrainer:
    def __init__(self, model, battery_params):
        self.model = model
        self.battery_params = battery_params
        self.online_buffer = deque(maxlen=1000)
        self.checkpointer = ModelCheckpoint()
        self.dp_solver = AcceleratedDPSolver(battery_params)

    def generate_dp_data(self, num_scenarios=100):
        """生成动态规划训练数据"""
        scenarios = []
        for _ in range(num_scenarios):
            time = np.linspace(0, 24, 96)
            load = generate_load_curve(time)
            pv = generate_pv_curve(time)
            price = get_electricity_price(time)
            
            actions = self.dp_solver.solve(load, pv, price)
            features = self._build_features(load, pv, price, actions)
            scenarios.append(features)
        return np.vstack(scenarios)

    def _build_features(self, load, pv, price, actions):
        """构建特征矩阵"""
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

    def supervised_train(self, data, epochs=100):
        """监督学习训练"""
        dataset = TimeSeriesDataset(data)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 加载已有检查点
        start_epoch, _ = self.checkpointer.load_latest(self.model, optimizer)
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 添加SOC约束
                predicted = outputs.squeeze()
                soc = inputs[:, -1, 0] + predicted / self.battery_params['max_charge_rate']
                penalty = torch.relu(soc - 1.0).mean() + torch.relu(0.2 - soc).mean()
                loss += 0.1 * penalty
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            self.checkpointer.save(self.model, optimizer, epoch, avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    def online_learning_step(self, batch_data):
        """在线学习步骤"""
        if not batch_data:
            return
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        inputs, targets = zip(*batch_data)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

# ------------------ 在线系统 ------------------
class OnlineSchedulingSystem:
    def __init__(self, model, battery_params):
        self.model = model
        self.battery_params = battery_params
        self.soc_history = [battery_params['initial_soc']]
        self.feature_window = deque(maxlen=3)
        self.safety_layer = SafetyLayer(battery_params)
        self.drift_detector = DriftDetector()

    def make_decision(self, current_load, current_pv, current_price):
        """实时决策"""
        # 构建输入特征
        features = [
            self.soc_history[-1],
            current_load,
            current_pv,
            current_price,
            0.0  # 初始功率
        ]
        
        # 填充时间窗口
        if len(self.feature_window) < 3:
            window = [np.zeros(5) for _ in range(3 - len(self.feature_window))] + [features]
        else:
            window = list(self.feature_window)[1:] + [features]
        
        # 模型预测
        window_tensor = torch.FloatTensor(np.array(window)).unsqueeze(0)
        with torch.no_grad():
            pred_power = self.model(window_tensor).item()
        
        # 应用安全约束
        safe_power = self.safety_layer.clamp_power(
            pred_power * self.battery_params['max_charge_rate'],
            self.soc_history[-1]
        )
        
        # 更新状态
        delta_soc = safe_power * self.battery_params['time_resolution'] / self.battery_params['capacity']
        new_soc = np.clip(self.soc_history[-1] + delta_soc,
                        self.battery_params['soc_min'],
                        self.battery_params['soc_max'])
        
        self.soc_history.append(new_soc)
        self.feature_window.append([
            new_soc,
            current_load,
            current_pv,
            current_price,
            safe_power
        ])
        
        # 漂移检测
        self.drift_detector.update(pred_power, safe_power)
        
        return safe_power

# ------------------ 辅助类 ------------------
class ModelCheckpoint:
    """模型版本管理"""
    def __init__(self, save_dir="checkpoints", max_keep=5):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.checkpoints = []
    
    def save(self, model, optimizer, epoch, loss):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"checkpoint_ep{epoch}_{loss:.4f}_{timestamp}.pt"
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_keep:
            os.remove(self.checkpoints.pop(0))
    
    def load_latest(self, model, optimizer):
        if not self.checkpoints:
            return 0, float('inf')
        
        checkpoint = torch.load(self.checkpoints[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

class SafetyLayer:
    """安全约束层"""
    def __init__(self, battery_params):
        self.params = battery_params
    
    def clamp_power(self, power, current_soc):
        max_charge = min(
            self.params['max_charge_rate'],
            (self.params['soc_max'] - current_soc) * self.params['capacity'] / self.params['time_resolution']
        )
        max_discharge = max(
            -self.params['max_charge_rate'],
            (self.params['soc_min'] - current_soc) * self.params['capacity'] / self.params['time_resolution']
        )
        return np.clip(power, max_discharge, max_charge)

class DriftDetector:
    """概念漂移检测"""
    def __init__(self, window_size=24, threshold=0.1):
        self.error_window = deque(maxlen=window_size)
        self.threshold = threshold
    
    def update(self, predicted, actual):
        error = abs(predicted - actual)
        self.error_window.append(error)
        
        if len(self.error_window) == self.error_window.maxlen:
            if np.mean(self.error_window) > self.threshold:
                print("Alert: Concept drift detected!")
                self._trigger_retraining()
    
    def _trigger_retraining(self):
        # 实现模型重训练逻辑
        pass

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 初始化配置
    battery_config = {
        'capacity': 2000,      # kWh
        'max_charge_rate': 500, # kW
        'initial_soc': 0.5,
        'soc_min': 0.2,
        'soc_max': 1.0,
        'time_resolution': 0.25 # 15分钟
    }
    
    # 创建组件
    model = AdvancedLSTMModel()
    trainer = HybridTrainer(model, battery_config)
    online_system = OnlineSchedulingSystem(model, battery_config)
    
    # 训练阶段
    print("Generating training data...")
    train_data = trainer.generate_dp_data(num_scenarios=100)
    
    print("Starting supervised training...")
    trainer.supervised_train(train_data, epochs=100)
    
    # 模拟在线运行
    print("Starting online operation...")
    time_points = np.linspace(0, 24, 96)
    load = generate_load_curve(time_points)
    pv = generate_pv_curve(time_points)
    price = get_electricity_price(time_points)
    
    power_output = []
    soc_history = []
    
    for t in range(len(time_points)):
        action = online_system.make_decision(load[t], pv[t], price[t])
        power_output.append(action)
        soc_history.append(online_system.soc_history[-1])
        
        # 每6小时在线学习
        if t % 24 == 0 and t > 0:
            real_data = [(torch.randn(3,5), torch.randn(1)) for _ in range(10)]  # 模拟真实数据
            trainer.online_learning_step(real_data)
    
    # 可视化结果
    net_load = load - pv + np.array(power_output)
    plot_scheduling_results(time_points, load, pv, power_output, soc_history, net_load,
                          'results/full_scheduling.png')
    
    print("Operation completed. Results saved.")