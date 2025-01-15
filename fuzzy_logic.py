
# 定义规则
rule1 = ctrl.Rule(soc['low'] & load['high'], power['positive'])
rule2 = ctrl.Rule(soc['high'] & load['low'], power['zero'])
rule3 = ctrl.Rule(soc['medium'] & load['medium'] & pv['high'], power['zero'])
rule4 = ctrl.Rule(soc['medium'] & pv['low'], power['zero'])
rule5 = ctrl.Rule(soc['low'] & load['medium'], power['positive'])
rule6 = ctrl.Rule(soc['medium'] & load['high'], power['positive'])
rule7 = ctrl.Rule(soc['high'] & pv['high'], power['zero'])
rule8 = ctrl.Rule(soc['high'] & load['high'], power['zero'])
rule9 = ctrl.Rule(soc['very_low'], power['positive'])
rule10 = ctrl.Rule(soc['very_high'], power['zero'])

# 新增规则以覆盖所有情况
rule11 = ctrl.Rule(soc['low'] & load['low'], power['positive'])
rule12 = ctrl.Rule(soc['medium'] & load['low'], power['positive'])
rule13 = ctrl.Rule(soc['high'] & load['medium'], power['zero'])
rule14 = ctrl.Rule(soc['very_low'] & load['low'], power['positive'])
rule15 = ctrl.Rule(soc['very_low'] & load['medium'], power['positive'])
rule16 = ctrl.Rule(soc['very_low'] & load['high'], power['positive'])
rule17 = ctrl.Rule(soc['low'] & pv['high'], power['positive'])
rule18 = ctrl.Rule(soc['medium'] & pv['medium'], power['zero'])
rule19 = ctrl.Rule(soc['high'] & pv['medium'], power['zero'])
rule20 = ctrl.Rule(soc['very_high'] & pv['high'], power['negative'])
rule21 = ctrl.Rule(soc['very_high'] & pv['medium'], power['negative'])
rule22 = ctrl.Rule(soc['very_high'] & pv['low'], power['negative'])

# 创建控制系统
power_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                 rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
                                 rule21, rule22])
