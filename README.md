# lesson9-ensemble｜集成学习与 Bootstrap（WDBC）

基于第 10 讲课程的 **方案 A（WDBC）**，研究单棵决策树、Bagging、随机森林、AdaBoost、梯度提升、逻辑回归和 RBF-SVM。GitHub 仓库名称采用 `lesson9-ensemble`，不代表变更了课程第 10 讲的实验内容。

## 项目目录

```text
lesson9-ensemble/
├── README.md
├── AGENTS.md
├── .gitignore
├── requirements.txt
├── run_all.bat             # Windows一键执行测试、实验和绘图
├── data/student/            # 原始学生数据：训练 / 验证 / 锁定测试特征
├── src/run_experiments.py   # 主实验；重生成 CSV
├── src/plot_results.py     # 画 Fig1–Fig5 多面板论文图
├── results/                 # 已有实验结果 CSV
├── figures/                 # 已有实验图
├── reports/                 # 用户最终上传 PDF（原文件直接复制）
├── docs/student-worksheet.md
├── docs/OPTIMIZATION_NOTES.md
└── tests/                   # 自动数据检查
```

## 快速运行（Windows PowerShell）

```powershell
cd D:\lesson9-ensemble
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_experiments
python -m src.plot_results
```

也可以在安装依赖后双击 `run_all.bat`，它将按顺序执行测试、实验和画图。

实验默认随机种子 `1010`。运行完成后，CSV 更新到 `results/`，5组多面板图保存到 `figures/`。切勿直接运行 `src` 下的文件后期望它依赖当前工作目录：推荐始终使用 `python -m src.xxx`。

### 自定义路径和抽样次数

```powershell
python -m src.run_experiments --help
python -m src.plot_results --help
```

`--stability-repeats`、`--bootstrap-repeats` 等参数可以用于调试，但**改变后生成的结果不能与报告中原有数值直接一一比较**。不加参数运行即复现原来的实验设计。

## 数据与防泄漏原则

- `train.csv`（341条）用于训练；`validation.csv`（114条）用于模型比较；`locked-test-features.csv`（114条）只留作最后评价、**没有对应标签**。
- 30个细胞核形态特征是输入，`malignant=1` 为恶性；`sample_id` 只是索引，永远不进入特征矩阵。
- 逻辑回归与 SVM 均使用 `Pipeline(StandardScaler(), estimator)`，五折交叉验证时折内拟合标准化器。
- 报告中的五折CV是把训练与验证集合并（455条）后的**开发阶段辅助检查**，不是独立测试；锁定测试标签未使用。
- 所有实验和图表为课程研究用途，不是临床诊断或外部推广的证据。

## 主要输出及其含义

| 文件 | 内容 |
|---|---|
| `A_tree_depth.csv` | 不同树深下训练与验证集表现 |
| `A_bootstrap_roots.csv` | 重抽样后根节点使用的特征 |
| `C_bootstrap_1000.csv` | 每次抽样不同记录占比及OOB占比 |
| `D_F_validation_models.csv` | 七模型的AUC、BA、FP、FN等 |
| `F_development_cv.csv` | 开发集五折交叉验证 |
| `D_tree_count.csv` | Bagging/随机森林的树数实验 |
| `D_G_prediction_stability.csv` | 训练重抽样后的预测波动和类别翻转率 |
| `D_feature_importance.csv` | Gini 与置换重要性 |
| `D_high_correlation_pairs.csv` | 训练数据中高度相关的特征组合 |
| `E_gradient_boosting_stages.csv` | 学习率和Boosting轮次分析 |
| `E_adaboost_weight_trace.csv` | AdaBoost权重更新的教学示例 |
| `metadata.json` | 随机种子、输入校验和、运行参数等 |

## 关于报告与代码的关系

`reports/集成学习_WDBC_用户最终原稿.pdf` 是直接复制用户最后上传的PDF，未对其内容做任何文字改写。代码优化只影响实验项目；如果未来升级scikit-learn导致某些结果有差异，应记录版本并重新核对，**不能静默替换报告里的数字**。

## GitHub 上传

```powershell
git init
git add .
git commit -m "Initialize lesson9 ensemble coursework"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/lesson9-ensemble.git
git push -u origin main
```

建议创建 **Private** 仓库，不要上传 `data/teacher/` 或其他评分答案。
