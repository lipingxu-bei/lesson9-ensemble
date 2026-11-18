# PBL 课程任务与材料索引

本项目的仓库名称是 `lesson9-ensemble`，但使用的教学材料名称为 **第 10 讲：集成学习与 Bootstrapping**。仓库编号不改变课程内容。本仓库采用课程方案 A：WDBC。

## 原始课程材料

- [`student-worksheet.md`](student-worksheet.md)：教师原始 A–H 工作单（原样保留）。
- [`workplan.md`](workplan.md)：课程工作计划（原样保留，含备选 CTG 方案；本仓库仅实践 WDBC）。
- [`ensemble-pbl.pptx`](ensemble-pbl.pptx)：原课程集成学习教学课件（原样保留）。
- 本项目的 Python 实验代码位于 [`../src/`](../src/)，不是课程原始示范代码。

## 仓库要解决的研究问题

在相同数据划分和评价规则下，Bagging、随机森林和 Boosting 是否优于单棵决策树？它们相对于逻辑回归、RBF-SVM 是否仍有优势？换一组训练记录后，预测是否更稳定？

重点比较 **分类效果、重抽样稳定性、误判类型和模型解释成本**，而非仅凭一次 AUC 选择“最好模型”。

## 已开展的任务与证据位置

| 工作单 | 任务 | 对应代码或结果 |
|---|---|---|
| A | 决策树深度、Bootstrap 后的根节点 | `results/A_tree_depth.csv`；`results/A_bootstrap_roots.csv` |
| B | 硬投票、软投票及方差公式 | `student-worksheet.md`；`reports/` 中的课程报告 |
| C | 1,000 次 Bootstrap 和 OOB 原理 | `results/C_bootstrap_1000.csv`；Fig1 |
| D | 单树、Bagging、随机森林及特征重要性 | `results/D_*.csv`；Fig2、Fig4 |
| E | AdaBoost 权重、梯度提升训练过程 | `results/E_*.csv`；Fig5 |
| F | 七模型的固定验证和开发集 CV | `results/D_F_validation_models.csv`；`results/F_development_cv.csv`；Fig3 |
| G | 固定验证记录的重复训练稳定性 | `results/D_G_prediction_stability.csv`；Fig2、Fig3 |
| H | Bagging 与 Boosting 机制对照、研究备忘录 | `reports/` 中的课程报告 |

上表中 `results/`、`reports/` 等路径均相对于仓库根目录。工作单给出的示例树与手算题，和代码实际拟合的模型实验应区分开来。

## 结论能够说明什么，不能说明什么

- 可以：说明在本次 WDBC 划分和开发阶段评价下，集成树与单树相比的表现及预测波动。
- 不可以：由固定验证 AUC 推断对新医院或临床患者同样有效；把特征重要性解释为致病因果；将锁定测试集当作调参数据。
- 当前项目只有锁定测试特征文件，**不包含锁定测试标签，也未进行最终独立测试**。

## 与课程 Harness PPT 的对应

本目录提供「问题、工作单和课程材料」，具体复现命令和证据链见根目录 `README.md`，数据来源见 `data/README.md`，学习反思见 `reflection.md`，同伴复现记录见 `peer_review/reproduction.md`。
