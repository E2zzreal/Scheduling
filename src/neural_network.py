import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 加载数据
net_load = pd.read_csv('results/net_load.csv')
pv = pd.read_csv('data/pv.csv')
with open('config/parameters.json') as f:
    params = json.load(f)

# 数据预处理
def preprocess_data(net_load, pv, params):
    # 合并数据
    data = net_load.copy()
    data['pv'] = pv['pv'].values
    
    # 添加电价信息
    time_points = params['time_points']
    prices = np.zeros(time_points)
    # 假设前24个点为off_peak, 25-48为mid_peak, 49-72为on_peak, 73-96为rush_hour
    prices[:24] = params['price']['off_peak']
    prices[24:48] = params['price']['mid_peak'] 
    prices[48:72] = params['price']['on_peak']
    prices[72:] = params['price']['rush_hour']
    data['price'] = prices
    
    # 添加负荷预测误差特征
    data['load_error'] = np.random.normal(0, 0.1, len(data))  # 假设预测误差服从正态分布
    
    # 特征和目标
    X = data[['SOC', 'load', 'pv', 'load_error']].values
    y = data['power'].values
    
    # 如果y是1D数组，reshape为2D
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 添加模糊特征
    fuzzy_scaler = StandardScaler()
    fuzzy_features = fuzzy_scaler.fit_transform(data[['load', 'load_error']])
    
    # 确保所有数据维度一致
    assert X.shape[0] == y.shape[0], f"X和y样本数不一致: {X.shape[0]} vs {y.shape[0]}"
    assert data['price'].values.shape[0] == y.shape[0], "price和y样本数不一致"
    
    return X, y, scaler, data

# 模糊逻辑层
class FuzzyLayer(nn.Module):
    def __init__(self):
        super(FuzzyLayer, self).__init__()
        # 定义模糊规则参数
        self.low = nn.Parameter(torch.tensor(0.2))
        self.medium = nn.Parameter(torch.tensor(0.5))
        self.high = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, x):
        # 计算隶属度
        low_membership = torch.sigmoid((self.low - x) * 10)
        medium_membership = torch.exp(-((x - self.medium) ** 2) / 0.1)
        high_membership = torch.sigmoid((x - self.high) * 10)
        
        # 模糊化处理
        return torch.cat([low_membership, medium_membership, high_membership], dim=1)

# 定义神经网络模型
class PowerOptimizer(nn.Module):
    def __init__(self, input_size):
        super(PowerOptimizer, self).__init__()
        self.fuzzy = FuzzyLayer()
        self.fc1 = nn.Linear(input_size + 3, 64)  # 增加3个模糊特征
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 96)  # 输出96个时间点的功率
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 分离负荷和误差特征
        load = x[:, 1:2]  # 负荷是第二个特征
        load_error = x[:, 3:4]  # 误差是第四个特征
        
        # 模糊化处理负荷和误差
        fuzzy_load = self.fuzzy(load)
        fuzzy_error = self.fuzzy(load_error)
        
        # 合并所有特征
        x = torch.cat([x, fuzzy_load, fuzzy_error], dim=1)
        
        # 通过神经网络
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 添加误差补偿
        x = x * (1 + 0.1 * load_error)  # 根据误差调整输出
        
        return x

# 自定义损失函数：最小化总电价成本并添加约束
def cost_loss(pred, price, soc, initial_soc, load_error):
    # 计算96个点的总成本
    cost = torch.sum(pred * price)
    
    # 考虑预测误差影响
    error_impact = torch.mean(torch.abs(load_error))  # 计算平均预测误差
    cost = cost * (1 + error_impact)  # 误差越大，成本权重越高
    
    # 约束1：净负荷在30-700之间
    load_min = 30
    load_max = 700
    load_violation = torch.sum(torch.relu(load_min - pred) + torch.relu(pred - load_max))
    
    # 约束2：SOC在10-100之间
    soc_min = 10
    soc_max = 100
    soc_violation = torch.sum(torch.relu(soc_min - soc) + torch.relu(soc - soc_max))
    
    # 约束3：初始和结束SOC差值在5以内
    soc_diff = torch.abs(soc[:, -1] - initial_soc)
    soc_diff_violation = torch.sum(torch.relu(soc_diff - 5))
    
    # 约束4：避免频繁充放电
    power_change = torch.diff(pred, dim=1)
    power_change_violation = torch.sum(torch.abs(power_change) > 50)  # 单次变化不超过50
    
    # 总损失 = 成本 + 约束惩罚
    penalty_weight = 1000  # 惩罚权重
    total_loss = cost + penalty_weight * (
        load_violation + soc_violation + soc_diff_violation + power_change_violation
    )
    
    return total_loss

# 主程序
if __name__ == '__main__':
    # 数据预处理
    X, y, scaler, data = preprocess_data(net_load, pv, params)
    
    # 转换为PyTorch张量
    # 确保所有张量的batch size一致
    batch_size = len(data)
    X_tensor = torch.FloatTensor(X)
    # 检查y的维度
    if len(y) == 96:  # 如果y是单条数据
        y_tensor = torch.FloatTensor(y.values.reshape(1, 96))  # reshape为(1, 96)
        price_tensor = torch.FloatTensor(data['price'].values.reshape(1, 96))  # reshape为(1, 96)
    else:  # 如果y是多条数据
        y_tensor = torch.FloatTensor(y.values.reshape(-1, 96))  # 自动计算batch size
        price_tensor = torch.FloatTensor(data['price'].values.reshape(-1, 96))  # 自动计算batch size
    initial_soc_tensor = torch.FloatTensor(data['SOC'].values.reshape(-1, 1))  # 初始SOC
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
        X_tensor, y_tensor, price_tensor, test_size=0.2, random_state=42)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train, price_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型、优化器和损失函数
    model = PowerOptimizer(input_size=X_train.shape[1])
    optimizer = optim.Adam(model.parameters())
    
    # 训练模型
    num_epochs = 200  # 增加训练轮数
    best_loss = float('inf')
    early_stop_patience = 10
    patience_counter = 0
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y, batch_price in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # 计算SOC变化
            soc = initial_soc_tensor + torch.cumsum(outputs, dim=1) * 0.25  # 假设0.25是时间步长系数
            batch_load_error = batch_X[:, 3:4]  # 获取当前batch的load_error
            loss = cost_loss(outputs, batch_price, soc, initial_soc_tensor, batch_load_error)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 计算平均epoch loss
        epoch_loss /= len(train_loader)
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            # 计算测试集的SOC变化
            test_soc = initial_soc_tensor + torch.cumsum(test_outputs, dim=1) * 0.25
            test_load_error = X_test[:, 3:4]  # 获取测试集的load_error
            test_loss = cost_loss(test_outputs, price_test, test_soc, initial_soc_tensor, test_load_error)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 早停机制
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'models/neural_network_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {test_loss.item():.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'models/neural_network.pth')
    np.save('models/scaler.npy', scaler)
    
    # 模型评估
    print('\n模型评估:')
    model.load_state_dict(torch.load('models/neural_network_best.pth'))
    model.eval()
    
    # 计算测试集成本
    with torch.no_grad():
        test_outputs = model(X_test)
        test_soc = initial_soc_tensor + torch.cumsum(test_outputs, dim=1) * 0.25
        test_cost = torch.sum(test_outputs * price_test, dim=1).mean().item()
        
        # 计算不同电价时段的成本
        off_peak_cost = torch.sum(test_outputs[:, :24] * price_test[:, :24]).item()
        mid_peak_cost = torch.sum(test_outputs[:, 24:48] * price_test[:, 24:48]).item()
        on_peak_cost = torch.sum(test_outputs[:, 48:72] * price_test[:, 48:72]).item()
        rush_hour_cost = torch.sum(test_outputs[:, 72:] * price_test[:, 72:]).item()
        
        # 计算约束违反情况
        load_violation = torch.sum(torch.relu(30 - test_outputs) + torch.relu(test_outputs - 700)).item()
        soc_violation = torch.sum(torch.relu(10 - test_soc) + torch.relu(test_soc - 100)).item()
        soc_diff = torch.abs(test_soc[:, -1] - initial_soc_tensor)
        soc_diff_violation = torch.sum(torch.relu(soc_diff - 5)).item()
        
    print(f'测试集平均成本: {test_cost:.2f}')
    print(f'各时段成本分布:')
    print(f'  - 低谷时段: {off_peak_cost:.2f}')
    print(f'  - 平峰时段: {mid_peak_cost:.2f}')
    print(f'  - 高峰时段: {on_peak_cost:.2f}')
    print(f'  - 尖峰时段: {rush_hour_cost:.2f}')
    print('\n约束违反情况:')
    print(f'  - 负荷约束违反: {load_violation:.2f}')
    print(f'  - SOC约束违反: {soc_violation:.2f}')
    print(f'  - SOC变化约束违反: {soc_diff_violation:.2f}')
    
    # 可视化结果
    import matplotlib.pyplot as plt
    
    # 绘制成本曲线
    plt.figure(figsize=(12, 6))
    plt.plot(test_outputs[0].numpy() * price_test[0].numpy(), label='每小时成本')
    plt.axvline(x=24, color='r', linestyle='--', label='低谷结束')
    plt.axvline(x=48, color='g', linestyle='--', label='平峰结束') 
    plt.axvline(x=72, color='b', linestyle='--', label='高峰结束')
    plt.title('每小时成本曲线')
    plt.xlabel('时间 (小时)')
    plt.ylabel('成本 (元)')
    plt.legend()
    plt.grid()
    plt.savefig('results/cost_curve.png')
    
    # 绘制SOC变化曲线
    plt.figure(figsize=(12, 6))
    plt.plot(test_soc[0].numpy(), label='SOC')
    plt.axhline(y=10, color='r', linestyle='--', label='SOC下限')
    plt.axhline(y=100, color='r', linestyle='--', label='SOC上限')
    plt.title('SOC变化曲线')
    plt.xlabel('时间 (小时)')
    plt.ylabel('SOC (%)')
    plt.legend()
    plt.grid()
    plt.savefig('results/soc_curve.png')
