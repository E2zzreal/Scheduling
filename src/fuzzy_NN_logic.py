"""
基于LSTM的储能系统优化调度算法

算法说明：
1. 使用LSTM网络捕捉时间序列特征
2. 考虑历史状态对当前决策的影响
3. 采用滑动窗口时间序列数据处理
4. 集成注意力机制增强重要时间点识别

输入：
- 时序特征：[SOC, 负荷, 光伏, 电价, 历史功率] 
- 时间窗口长度：3小时历史数据
- 电池参数约束

输出：
- 最优充放电功率序列
- 总运行成本分析
- 时间注意力权重可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class LSTMScheduler(nn.Module):
    """
    LSTM神经网络调度模型
    输入维度：(batch_size, seq_len, input_size)
    输出维度：(batch_size, 1)
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMScheduler, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        
        # 时间注意力机制
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        
        # 最终输出
        output = self.fc(context)
        return output, attn_weights

class TimeSeriesDataset(Dataset):
    """时间序列数据集生成器"""
    def __init__(self, data, window_size=3):
        """
        data: 完整时序数据 (seq_len, features)
        window_size: 输入时间窗口长度
        """
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # 输入为滑动窗口数据
        x = self.data[idx:idx+self.window_size]
        # 输出为下一时刻目标值
        y = self.data[idx+self.window_size, -1]  # 假设最后一列为目标功率
        return torch.FloatTensor(x), torch.FloatTensor([y])

def create_sequences(data, window_size):
    """创建时间序列样本"""
    sequences = []
    for i in range(len(data)-window_size):
        seq = data[i:i+window_size]
        target = data[i+window_size, -1]
        sequences.append((seq, target))
    return sequences

def train_lstm(model, train_loader, epochs=100, lr=0.001):
    """LSTM模型训练"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # 添加SOC约束惩罚
            predicted_power = outputs.squeeze()
            soc = inputs[:, -1, 0] + predicted_power  # 简化的SOC更新
            penalty = torch.relu(soc - 1.0).mean() + torch.relu(0.2 - soc).mean()
            loss += 0.1 * penalty
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

def lstm_scheduling(params: dict) -> Tuple[List[float], List[float], float]:
    """
    LSTM优化调度主函数
    
    参数：
    params - 包含：
        - battery: 电池参数
        - time_points: 时间点数
        - window_size: 时间窗口长度
        - model_path: 预训练模型路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_points = params['time_points']
    battery = params['battery']
    window_size = params.get('window_size', 3)

    # 初始化模型
    model = LSTMScheduler(input_size=5, hidden_size=64).to(device)
    if params.get('model_path'):
        model.load_state_dict(torch.load(params['model_path']))
    
    # 生成时序数据
    time = np.linspace(0, 24, time_points)
    load = load_curve_from_csv(params['load_csv']) if params.get('load_csv') else generate_load_curve(time)
    pv = pv_curve_from_csv(params['pv_csv']) if params.get('pv_csv') else generate_pv_curve(time)
    price = get_electricity_price(time)
    
    # 创建特征矩阵 [SOC, Load, PV, Price, Power]
    features = np.zeros((time_points, 5))
    soc = battery['initial_soc']
    features[0, :] = [soc, load[0], pv[0], price[0], 0]
    
    # 实时调度循环
    power = []
    attention_weights = []
    model.eval()
    
    with torch.no_grad():
        for t in range(1, time_points):
            # 构建时间窗口
            if t < window_size:
                # 填充初始窗口
                window = np.vstack([features[:t], np.zeros((window_size-t, 5))])
            else:
                window = features[t-window_size:t]
            
            # 转换为tensor
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            
            # 模型预测
            pred_power, attn = model(window_tensor)
            norm_power = pred_power.item()
            
            # 反归一化功率
            max_power = battery['max_charge_rate']
            actual_power = norm_power * max_power
            
            # 计算SOC变化
            delta_soc = actual_power * params['time_resolution'] / battery['capacity']
            new_soc = soc + delta_soc
            new_soc = max(battery['soc_min'], min(battery['soc_max'], new_soc))
            
            # 更新特征矩阵
            features[t, 0] = new_soc
            features[t, 1:] = [load[t], pv[t], price[t], actual_power]
            
            # 记录结果
            power.append(actual_power)
            attention_weights.append(attn.cpu().numpy())
            soc = new_soc

    # 后处理分析
    net_load = load - pv + np.array(power)
    cost = np.sum(net_load * price) * params['time_resolution']
    
    # 可视化注意力权重
    plot_attention(np.array(attention_weights), 
                 time[window_size:],
                 save_path='results/attention_weights.png')

    return power, features[:, 0], cost

def plot_attention(weights, time, save_path):
    """可视化时间注意力权重"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.imshow(weights.squeeze().T, aspect='auto', 
             cmap='viridis', interpolation='nearest')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.title('Temporal Attention Distribution')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

# 其他辅助函数需与原始代码保持兼容...