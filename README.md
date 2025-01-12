# 能源管理系统调度算法

本项目实现了多种能源管理系统的调度算法，包括：
- 混合整数线性规划（MILP）
- 动态规划（DP）
- 模型预测控制（MPC）
- 启发式规则
- 模糊逻辑控制

## 主要功能

1. 负荷曲线生成
   - 支持随机生成负荷曲线
   - 支持从CSV文件读取负荷曲线
   - 曲线格式：96个时间点（15分钟间隔）

2. 光伏发电曲线生成
   - 支持随机生成光伏曲线
   - 支持从CSV文件读取光伏曲线
   - 曲线格式：96个时间点（15分钟间隔）

3. 电价获取
   - 支持分时电价获取
   - 支持自定义电价时段

4. 调度算法
   - 提供多种调度算法实现
   - 支持算法性能比较

## 文件说明

- src/MILP.py: 混合整数线性规划算法
- src/dynamic_programming.py: 动态规划算法
- src/mpc.py: 模型预测控制算法
- src/heuristic_rules.py: 启发式规则算法
- src/fuzzy_logic.py: 模糊逻辑控制算法
- src/utils.py: 工具函数（曲线生成、CSV读取等）
- config/parameters.json: 系统参数配置
- results/: 结果可视化图片

## 使用方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置参数：
   - 编辑config/parameters.json文件
   - 可选配置：
     - load_csv: 负荷曲线CSV文件路径（如"data/load.csv"）
     - pv_csv: 光伏曲线CSV文件路径（如"data/pv.csv"）

3. 运行算法：
   ```bash
   # 运行MILP算法
   python src/MILP.py

   # 运行动态规划算法
   python src/dynamic_programming.py

   # 运行MPC算法
   python src/mpc.py

   # 运行启发式规则算法
   python src/heuristic_rules.py

   # 运行模糊逻辑算法
   python src/fuzzy_logic.py
   ```

4. CSV文件格式：
   - 负荷曲线CSV：
     - 列名：load
     - 数据：96个时间点的负荷值
   - 光伏曲线CSV：
     - 列名：pv
     - 数据：96个时间点的光伏功率值
   - 示例：
     ```csv
     load
     300.5
     310.2
     ...
     ```

5. 查看结果：
   - 结果将保存为png图片
   - 图片保存在results/目录下

## 依赖库

- numpy
- pandas
- pulp
- scikit-fuzzy
- matplotlib

## 算法说明

每个算法文件开头都包含详细的算法说明，包括：
- 算法原理
- 输入输出
- 使用示例
- 注意事项
