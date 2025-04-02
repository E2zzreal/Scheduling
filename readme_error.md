# 微电网调度场景分析：成本计算验证与最终结论

本文档记录了对 `scenario_analysis.py` 中成本计算逻辑的验证过程，并基于验证结果得出了关于预测误差对成本影响的最终结论。

## 初始观察与问题

在早期分析中，HDF5 文件 (`results/scenario_analysis/error_based_scenario_analysis.h5`) 中存储的最优成本 (`true_cost`) 属性值约为 12192.04。基于此基准，运行 `analysis/cumulative_cost_analysis.py` 发现大量（1312个）预测误差场景的实际计算成本低于此值。这引发了对“预测不准成本反而更低”这一反常现象的深入探究。

## 成本计算验证与错误溯源

在 `analysis/cumulative_cost_analysis.py` 脚本的初步运行中，发现根据 HDF5 文件加载的最优调度计划重新计算出的最优成本 (`Calculated optimal cost from schedule: 5163.96`) 与 HDF5 文件属性中存储的 `true_cost` (`12192.04`) 存在巨大差异。这表明初始 HDF5 文件中的 `true_cost` 值是错误的，导致了之前错误的“低成本场景”判断。

**错误原因分析 (推测)**:

由于无法回溯生成错误 HDF5 文件时的代码版本，以下是基于当前代码逻辑对旧版本可能错误的推测：

1.  **`run_analysis` 中计算 `self.true_cost` 时出错**:
    *   可能错误地将优化器的目标函数值 (`true_schedule_result['total_cost_objective']`) 或其他不相关的值赋给了 `self.true_cost`。
    *   调用 `_calculate_real_cost` 时传递了错误的参数（例如，使用了预测负荷而非真实负荷）。
2.  **`_calculate_real_cost` 函数本身存在 Bug (旧版本)**:
    *   **最可能的错误**: 在计算总成本的循环中，**错误地使用了未应用 `>= 30kW` 约束的 `raw_net_load`**，而不是应用了约束后的 `real_net_load`。
        *   **旧错误逻辑 (推测)**: `total_cost = sum(raw_net_load[i] * price * 0.25)`
        *   **当前正确逻辑**: 先 `real_net_load = np.where(raw_net_load < 30, 30, raw_net_load)`，然后 `total_cost = sum(real_net_load[i] * price * 0.25)`。
    *   如果 `raw_net_load` 在某些高电价时段远低于 30（例如为负值），使用 `raw_net_load` 计算会导致成本显著偏离正确值。
    *   其他可能性：旧版本中电价获取逻辑错误、时间步长 (`* 0.25`) 计算错误等。
3.  **`save_results` 中写入 HDF5 时出错**:
    *   可能错误地将 `self.true_cost` 之外的变量值写入了 HDF5 的 `true_cost` 属性。

**当前代码验证**:

为了验证当前代码的成本计算逻辑：
1.  在 `scenario_analysis.py` 中添加了 `verify_optimal_cost_calculation` 函数，该函数：
    *   调用 `_run_scenario` 获取最优调度计划 (`true_schedule_dict`)。
    *   调用 `_calculate_real_cost(true_schedule_dict, self.true_load, self.true_pv)` 计算成本。
    *   打印计算过程中的关键值（`raw_net_load` 低于 30 的点、最终 `real_net_load`、计算出的总成本）。
2.  修改 `if __name__ == "__main__":` 块以调用此验证函数。
3.  运行 `python scenario_analysis.py`。

**验证结果 (基于当前代码)**:
```
--- Verifying Optimal Cost Calculation (Constraint >= 30) ---
Optimal schedule obtained.
Calculated Optimal Cost via _calculate_real_cost: 5163.90
Points where raw_net_load < 30: 16
Indices: [31 34 35 36 39 41 42 72 73 75 77 79 80 81 82 83]
Raw net load at these points: [30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30.]
Final net load (after np.where) at these points: [30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30.]
--- Verification Finished ---
```
这确认了 **当前版本** 的 `_calculate_real_cost` 函数，在给定最优调度计划和真实负荷/光伏时，计算出的最优成本约为 **5163.90**。这表明当前的成本计算逻辑是正确的。因此，初始 HDF5 文件中的错误值 (12192.04) 必定源于 **过去** 代码执行中的错误。

## 基于正确基准的再分析

1.  恢复 `scenario_analysis.py` 的主执行块（确保 `run_analysis` 和 `save_results` 使用正确的逻辑）。
2.  重新运行 `python scenario_analysis.py --scenario --error` 以生成包含正确 `true_cost` (5163.90) 的 HDF5 文件。
3.  再次运行 `python -m analysis.cumulative_cost_analysis`。

**再分析结果**:
```
Data loaded successfully from results\scenario_analysis\error_based_scenario_analysis.h5
Optimal cost (true_cost): 5163.90
Error levels loaded: [100 150 30 50]
Calculated optimal cost from schedule: 5163.90 (vs HDF5 true_cost: 5163.90)
Found 0 low-cost scenarios.
No scenarios found with real_cost < true_cost.
```
这次，重新计算的成本与 HDF5 中的 `true_cost` 一致，并且 **没有发现任何成本低于最优成本的误差场景**。

## 最终（修正后）结论

1.  **成本计算准确性**: 之前的分析受到了 HDF5 文件中存储的错误 `true_cost` 的误导。经过验证，正确的基准最优成本约为 **5163.90**。
2.  **完美预测成本最低**: 当使用正确的基准成本时，所有预测不准确的场景的实际成本都 **高于或等于** 最优成本。这符合理论预期：**完美预测能够带来最低的运行成本**。
3.  **约束边界与数值精度**: 完美预测场景下，优化器为了最小化成本，会将净负荷精确推至 `>= 30kW` 的约束边界。后续成本计算中，由于潜在的数值精度差异，这些边界点可能被计算为略低于 30，然后在 `_calculate_real_cost` 中被强制拉回 30。这解释了为什么最优成本本身会受到 30kW 约束的影响（即无约束时的理论最低成本会更低）。
4.  **预测误差的影响**: 预测误差导致调度计划偏离最优，从而在应用 `>= 30kW` 约束进行实际成本评估时，通常会导致更高的成本。误差越大，偏离越远，成本增加可能越多。
5.  **可视化更新**: `scenario_analysis.py` 的 `plot_results` 函数已更新（针对 `type='error'`），现在生成的 `error_based_scenario_analysis.png` 包含一个额外的子图（通常是第 5 个，位于 2x3 布局的中间底部）。该子图使用箱线图展示了不同误差水平下，各场景实际成本与最优成本的差值分布，并标注了每个误差水平的平均成本差值，有助于更直观地量化预测误差带来的额外成本。

**总结**: 最初观察到的“预测不准成本更低”现象是由于数据记录错误。在修正基准成本后，结果表明完美预测确实能获得最低成本，而预测误差会增加成本。分析过程中的约束边界效应和数值精度问题仍然是理解模型行为的重要方面。
