# 项目结构说明

## 目录结构
```
.
├── README.md                # 项目说明文档
├── requirements.txt         # 项目依赖
├── config/                  # 配置文件目录
│   └── parameters.json      # 系统参数配置
├── src/                     # 源代码目录
│   └── scheduling.py        # 主调度程序
├── results/                 # 结果文件目录
│   └── scheduling_results.png  # 调度结果可视化
└── projectstructure.md      # 项目结构说明
```

## 模块说明
1. 核心模块
   - `src/scheduling.py`: 实现微电网调度优化算法
   - 包含负荷曲线生成、光伏发电曲线生成、电池调度优化等功能

2. 配置文件
   - `config/parameters.json`: 包含系统运行参数
   - 可配置负荷、光伏、电池、电价等参数

3. 结果可视化
   - 使用matplotlib生成调度结果图表
   - 结果保存至results/目录
   - 包含负荷曲线、光伏曲线、电池SOC曲线等

4. 依赖管理
   - 使用requirements.txt管理Python依赖
   - 建议使用virtualenv创建虚拟环境
   - 主要依赖：pulp, numpy, matplotlib

## 运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 修改配置文件：`config/parameters.json`
3. 运行主程序：`python src/scheduling.py`
4. 查看结果：`results/scheduling_results.png`
