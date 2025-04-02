# 能源管理系统调度算法与场景分析

本项目实现了多种能源管理系统的调度算法，并提供了场景分析功能以评估预测不确定性对调度效果的影响。

## 主要功能

1.  **调度算法实现**:
    *   混合整数线性规划（MILP）
    *   动态规划（DP）
    *   模型预测控制（MPC）
    *   启发式规则
    *   模糊逻辑控制
2.  **基础数据处理**:
    *   负荷/光伏曲线生成（随机或从CSV读取）
    *   分时电价获取
3.  **场景分析**:
    *   基于蒙特卡洛模拟，分析不同预测准确度或误差水平对调度成本的影响。
    *   使用并行处理高效评估大量场景。
    *   生成详细的分析结果（HDF5数据）和可视化图表。

## 文件说明

-   **`src/`**: 包含各种调度算法的实现。
    -   `MILP.py`: 混合整数线性规划算法。
    -   `dynamic_programming.py`: 动态规划算法。
    -   `mpc.py`: 模型预测控制算法。
    -   `heuristic_rules.py`: 启发式规则算法。
    -   `fuzzy_logic.py`: 模糊逻辑控制算法。
    -   `fuzzy_NN_logic.py`: 模糊神经网络逻辑算法 (新增)。
-   **`scenario_analysis.py`**: 场景分析主模块，用于生成和评估不同预测水平下的调度场景。
-   **`analysis/`**: 包含用于后处理和可视化场景分析结果的脚本。
    -   `error_analysis.py`: 分析 `error_based` 场景结果的脚本。
    -   `cumulative_cost_analysis.py`: 分析累计成本对比的脚本。
-   **`utils.py`**: 工具函数（曲线生成、CSV读取、电价获取等）。
-   **`config/parameters.json`**: 系统参数配置（电池、负荷、时间点等）。
-   **`results/`**: 存放运行结果。
    -   `results/scenario_analysis/`: 场景分析的 HDF5 数据文件和可视化图片。
    -   `results/cumulative_cost_comparison/`: 累计成本对比图。
    -   `results/single_scenario/`: 单场景优化结果（如果运行）。
    -   (可能包含其他算法直接输出的图片)
-   **`readme_error.md`**: 记录成本计算验证和相关分析结论的文档。
-   **`requirements.txt`**: 项目依赖库列表。

## 使用方法

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置参数**:
    *   编辑 `config/parameters.json` 文件，设置电池容量、功率、初始SOC、时间点数等。
    *   (可选) 在 `parameters.json` 中配置 `load_csv` 和 `pv_csv` 路径以使用文件数据，否则将随机生成。

3.  **运行单个调度算法**:
    ```bash
    # 运行MILP算法 (示例)
    python src/MILP.py
    # ... (类似地运行其他 src/ 目录下的算法脚本) ...
    ```

4.  **运行场景分析**:
    ```bash
    # 运行基于误差水平的场景分析 (生成 HDF5 数据和 PNG 图)
    python scenario_analysis.py --scenario --error

    # 运行基于准确度水平的场景分析 (生成 HDF5 数据和 PNG 图)
    # python scenario_analysis.py --scenario --accuracy # 注意：accuracy 模式可能未完全测试

    # (可选) 仅验证最优成本计算逻辑
    # python scenario_analysis.py --verify_cost
    ```

5.  **运行后处理分析脚本**: (在运行场景分析生成 HDF5 文件后)
    ```bash
    # 运行 error_analysis 脚本生成详细图表
    python -m analysis.error_analysis

    # 运行 cumulative_cost_analysis 脚本生成累计成本对比图
    python -m analysis.cumulative_cost_analysis
    ```

6.  **查看结果**:
    *   调度算法和场景分析的图表保存在 `results/` 下的相应子目录中。
    *   场景分析的详细数据保存在 `results/scenario_analysis/` 下的 `.h5` 文件中。
    *   成本计算验证和结论见 `readme_error.md`。

## 依赖库

-   numpy
-   pandas
-   pulp (用于 MILP)
-   joblib (用于并行计算)
-   h5py (用于数据存储)
-   matplotlib (用于绘图)
-   seaborn (用于绘图美化)
-   scikit-fuzzy (如果使用模糊逻辑)
-   tqdm (用于进度条)

## 注意事项

-   确保 `utils.py` 中的电价逻辑 (`get_electricity_price`) 与实际需求一致。
-   运行分析脚本（如 `error_analysis.py`）前，需要先运行 `scenario_analysis.py --scenario --error` (或 `--accuracy`) 来生成对应的 HDF5 数据文件。
-   分析脚本使用 `-m` 标志运行，以确保正确处理相对导入。
