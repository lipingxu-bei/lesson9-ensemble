# WDBC 数据来源与使用说明

## 1. 来源与用途

本项目使用课程方案 A：Breast Cancer Wisconsin (Diagnostic)，简称 WDBC。课堂离线数据由 `sklearn.datasets.load_breast_cancer` 生成，UCI 原始数据页面：<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>，DOI：`10.24432/C5DW2B`。

原课程数据说明见 `pbl/` 中的课程材料。若要将数据用于非课程项目或公开再分发，请再核验来源署名和适用许可。

## 2. 目录和划分

| 文件 | 记录数 | 标签 | 作用 |
|---|---:|---|---|
| `student/train.csv` | 341 | 有 | 拟合模型、训练折内预处理 |
| `student/validation.csv` | 114 | 有 | 开发阶段比较模型与结果分析 |
| `student/locked-test-features.csv` | 114 | 无 | 预留的锁定测试特征，不参与选择模型 |

三份 CSV 均包含 `sample_id` 和 30 个连续形态特征；训练/验证文件另包含 `malignant` 标签。已检查学生版三个 CSV 没有缺失单元格，且各文件内部 `sample_id` 不重复。

## 3. 观察单位与字段

- **一行是什么？** 一条乳腺肿块细针穿刺图像导出的细胞核形态汇总记录，不能直接当作未来患病风险或独立患者级结论。
- **`sample_id`**：记录标识，仅用于跟踪与核对，**不参与模型输入**。
- **`malignant`**：课程重编码标签，`1=恶性`、`0=良性`。`locked-test-features.csv` 不包含此列。
- **30 个特征**：10 类形态量各有 `mean`、`error`、`worst` 三组汇总，包括半径（radius）、纹理（texture）、周长（perimeter）、面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹度（concavity）、凹点（concave points）、对称性（symmetry）与分形维数（fractal dimension）。实际完整拼写以 CSV 表头为准。
- **标签分布**：训练集恶性 127、良性 214；验证集恶性 43、良性 71。

## 4. 数据处理与防泄漏规则

1. 训练只使用 `train.csv`；不得打开教师版锁定测试标签来选模型。
2. 所有模型必须排除 `sample_id`、`malignant` 等非输入列；特征列顺序以训练表为准。
3. 逻辑回归和 SVM 的 StandardScaler 应包含在 Pipeline 中，并只在当前训练数据或 CV 训练折上拟合。
4. 训练集与验证集不交叉使用来拟合预处理。当前报告中的开发集五折 CV 利用训练与验证合并的 455 条记录，仅作开发阶段检查。
5. 改变数据划分、随机种子或处理规则，需要新建实验记录，不能悄悄替换旧报告数字。

## 5. 隐私与发布边界

当前仓库只包含课程公开/教学用途的学生数据版本，没有教师参考标签、密钥或身份信息字段。尽管如此，公开仓库前仍应确认课程允许公开这些 CSV。

不得上传 `data/teacher/`、`locked-test-labels.csv`、`all-measurements.csv` 或其他评分答案；`.gitignore` 中已有相应排除规则。项目实验结果不是临床诊断建议，也不支持跨中心效果结论。

## 6. 运行与结果核对

从仓库根目录执行：

```powershell
python -m src.run_experiments
python -m src.plot_results
```

结果写入 `results/`，图写入 `figures/`，`results/metadata.json` 记录数据 SHA256 和参数、包版本等信息。
