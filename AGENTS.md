# AGENTS.md — lesson9-ensemble

## 项目目标
本仓库是 WDBC（乳腺肿块数据）集成学习与 Bootstrap 课程作业，包含实验、结果、绘图和学生报告。

## 强制规则
1. **不得修改 `reports/` 中用户的任何文字、标点、公式内容、段落顺序或结论**；修改报告必须先得到用户明确授权。
2. **绝不读取或使用教师答案、锁定测试标签**；开发和调参仅使用 `data/student/train.csv`、`validation.csv`。
3. `sample_id` 不得作为模型输入。只对 LR / SVM 在训练折内进行 StandardScaler 标准化，防止数据泄漏。
4. 重要实验的默认随机种子是 `1010`。修改模型参数、数据划分、标签定义或评价规则时必须说明变化及对数值的影响。
5. 不可编造 AUC、误判个数或图表，不可把验证集评估声称为独立临床验证。
6. 实验输出保存在 `results/`，绘图保存在 `figures/`，不把教师数据加入 Git。
7. 优先添加可重复运行的函数、单元测试和 README 说明，不要只替换报告里的数字。

## 运行命令
```bash
python -m pip install -r requirements.txt
python -m src.run_experiments
python -m src.plot_results
python -m unittest discover -s tests -v
```

## 文件分工
- `src/run_experiments.py`: 训练、Bootstrap、模型对比、稳定性、特征重要性、Boosting
- `src/plot_results.py`: 仅根据数据/实验 CSV 绘图，避免重复训练全部模型（ROC 子图例外）
- `data/student/`: 教师提供的学生版本数据，不允许覆盖
- `reports/`: 用户本人编辑的最终 PDF，不允许改写
