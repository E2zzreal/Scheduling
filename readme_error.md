# 微电网调度场景分析：成本异常现象探究

本文档记录了对 `scenario_analysis.py` 中观察到的异常现象——即预测不准确的场景有时比完美预测场景成本更低——的分析过程和结论。

## 初始问题

在使用 `scenario_analysis.py` 分析不同预测误差水平对调度成本的影响时，发现某些误差较大的场景（例如 100kW Error）在最终的实际成本评估中，其平均成本或部分场景成本低于完美预测（0 Error）下的最优成本 (`True Cost`)。这与直觉相悖，因为理论上完美预测应该导致最低成本。

代码核心逻辑：
1.  **优化 (`_run_scenario`)**: 基于 **预测** 负荷 (`scenario['load']`) 进行 MILP 优化，目标是最小化预测成本，约束包括净负荷 `>= 30kW`。
2.  **成本评估 (`_calculate_real_cost`)**: 使用优化得到的 **调度计划** (`schedule`) 和 **真实** 负荷/光伏 (`true_load`, `true_pv`) 计算 **原始净负荷** (`raw_net_load`)，然后 **强制应用** `>= 30kW` 约束得到最终用于计算成本的 `real_net_load` (`np.where(raw_net_load < 30, 30, raw_net_load)`)。

## 分析过程与发现

### 假设 1：30kW 约束与预测误差的相互作用

*   **分析**: 完美预测下的最优调度 (`true_schedule`)，虽然在优化时满足了基于 `true_load` 的 `>= 30` 约束，但在后续使用 `true_load` 计算 `raw_net_load` 时，结果可能因数值精度等原因略低于 30。这些点在 `_calculate_real_cost` 中会被强制拉回 30，产生额外成本。而预测不准的场景，其调度计划可能“碰巧”使得计算出的 `raw_net_load` 更少地低于 30，从而避免了这部分额外成本。
*   **验证**: 修改 `analysis/error_analysis.py` 以计算并分析 `raw_net_load`。
    *   **数据**: 初始运行时（包含效率因子），分析脚本输出 `Optimal Scenario: Violations=16 Total Magnitude=0.00`。同时，低成本场景数据显示（例如）：
        ```
        Low Cost Scenarios Constraint Violation Summary:
               count  avg_violations  avg_magnitude
        error
        30         3        0.666667       3.056074
        50         3        0.666667      10.449104
        100        3        0.333333       8.137562
        150        3        0.000000       0.000000
        ```
        这表明最优场景确实触发了约束，而部分低成本误差场景触发次数更少。

### 假设 2：效率因素的影响

*   **分析**: SOC 状态转移方程中的充放电效率因子 (`*0.95`, `/0.95`) 可能间接影响优化决策，导致 `raw_net_load` 计算偏差。
*   **实验**: 临时移除 `_run_scenario` 中的效率因子，重新运行完整分析 (`scenario_analysis.py --scenario --error` 后接 `python -m analysis.error_analysis`)。
*   **结果**: 分析脚本输出 `Optimal Scenario: Violations=22 Total Magnitude=0.00`。违规次数从 16 次 **增加** 到 22 次。
*   **结论**: **效率因素不是主要原因**，甚至其存在反而略微减少了边界违规现象。

### 假设 3：优化目标与约束边界的相互作用及数值精度

*   **分析**: 优化器为了最小化成本，可能将净负荷精确推至 `>= 30` 的约束边界。后续 NumPy 计算时，微小的数值精度差异可能导致结果略低于 30。
*   **验证 1 (检查边界值)**: 修改 `analysis/error_analysis.py` 打印违规点的详细信息（电价、SOC、`raw_net_load` 值）。运行 `python -m analysis.error_analysis`。
*   **结果 1 (数据)**:
    ```
    Optimal Scenario: Violations=16 Total Magnitude=0.00
    Optimal Scenario: Violation indices: [31 34 35 36 39 41 42 72 73 75 77 79 80 81 82 83]
    Optimal Scenario: Raw net load at violations: [30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30. 30.]
    Optimal Scenario: Prices at violations (€/kWh): [0.726 1.246 1.246 1.246 1.246 1.246 1.246 1.246 1.246 1.246 1.538 1.538 1.538 1.538 1.538 1.538]
    Optimal Scenario: SOC (%) at violations: [100.   90.5  86.5  83.1  73.1  65.8  62.8  56.1  53.1  47.3  42.1  37.   33.4  29.9  26.6  23.5]
    ```
    发现所有违规点的 `raw_net_load` 值（四舍五入后）**精确等于 30.00**。这些点主要发生在电价中低、SOC 范围较宽的时段。这强烈表明优化器正是在利用约束边界。
*   **验证 2 (放松约束实验 >= 29)**: 修改 `_run_scenario` 将约束改为 `>= 29`，修改 `if __name__ == "__main__":` 块以运行单场景真实负荷，执行 `python scenario_analysis.py`。
*   **结果 2 (数据)**:
    ```
    Relaxed Raw Net Load at original violation points: [29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29. 29.]
    Relaxed Schedule Cost (evaluated with >=30 constraint): 12202.61
    ```
    在原始违规点，`raw_net_load` 现在精确地降到了 **29.0**。使用 `>= 29` 约束得到的调度计划，在原始 `>= 30` 成本评估逻辑下，成本为 **12202.61**。
*   **验证 3 (放松约束实验 >= 0)**: 修改 `_run_scenario` 将约束改为 `>= 0`，修改 `if __name__ == "__main__":` 块以运行单场景真实负荷，执行 `python scenario_analysis.py`。
*   **结果 3 (数据)**:
    ```
    Relaxed (>=0) Raw Net Load at original violation points: [-0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0. -0.]
    Relaxed (>=0) Total points below 30: 26
    Relaxed (>=0) Min raw net load: -0.00
    Relaxed (>=0) Schedule Cost (evaluated with >=30 constraint): 12460.38
    ```
    在原始违规点，`raw_net_load` 现在精确地降到了 **0.0**。总共有 26 个点的 `raw_net_load` 低于 30。使用 `>= 0` 约束得到的调度计划，在原始 `>= 30` 成本评估逻辑下，成本为 **12460.38**。
*   **结论**: **优化目标与约束边界的相互作用是主要原因**。优化器确实在成本驱动下将净负荷推至约束允许的最低边界。数值精度差异导致后续计算时出现略低于边界的值。比较不同约束下的评估成本（`>=30` 评估：原始约 12500，`>=29` 评估：12202.61，`>=0` 评估：12460.38）表明，稍微放宽约束可能在特定评估逻辑下有利，但过度放宽则不然。

## 最终结论

1.  **根本原因**: 预测不准确场景成本有时低于完美预测场景，是由于 **30kW 净负荷下限约束** 在 **成本评估阶段 (`_calculate_real_cost`) 的强制应用** 与 **优化阶段 (`_run_scenario`) 基于不同负荷（预测 vs 真实）寻找边界解** 之间的相互作用，以及潜在的 **数值精度差异** 造成的。
2.  **优化器行为**: 优化器在追求成本最低时，会精确地将净负荷推至当前约束（例如 `>= 30`, `>= 29`, `>= 0`）的下限。
3.  **评估逻辑**: 最终成本取决于使用哪个调度计划，并用 **固定的 `>= 30` 评估逻辑** 来计算。这导致稍微放宽约束（如 `>= 29`）得到的计划在最终评估时可能成本更低，但完全放开约束（`>= 0`）则可能因评估时的强制调整而导致成本增加。
4.  **预测误差的“伪优势”**: 预测误差有时会“幸运地”产生一个调度计划，使得其在真实负荷下计算出的 `raw_net_load` 能更好地“规避”评估阶段 `>= 30` 约束的强制调整，从而显得成本更低。

此分析揭示了模型约束、优化目标、预测不确定性以及评估逻辑之间复杂的相互影响。
