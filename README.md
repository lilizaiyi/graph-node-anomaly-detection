# 图卷积自编码器的节点异常检测

深度学习课程设计使用 Books 和 Disney 两个公开属性图，比较邻居聚合及结构重构对无监督异常排序的影响。主方法采用两层 GCN，重构设计参考了天津大学武南南老师团队的 [DECLARE](https://ijcai.org/proceedings/2025/387) 和 [DOMINANT](https://doi.org/10.1137/1.9781611975673.67)。[BOND 基准](https://proceedings.neurips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract.html) 提供数据版本，原始发布文件和许可证位于 `data/source/`。

## 复现报告实验

在项目根目录使用 D 盘项目的 Python 环境。现有环境已安装 Python 3.13.6、PyTorch 2.15.0 CPU 版和 NumPy 2.5.4。

```powershell
.\.venv\Scripts\python.exe prepare_benchmarks.py
.\.venv\Scripts\python.exe run_course_experiment.py --output results/reproduce
```

解压课程附件到另一目录时，可以用已经安装了 NumPy 和 PyTorch 的 Python 运行这两个脚本。缺少依赖时，在项目虚拟环境安装 `requirements-deep.txt`，Windows CPU 实测版本记录于 `requirements-lock.txt`。若手头缺少原始压缩包，运行 `prepare_benchmarks.py --download` 可从公开源下载并验证 SHA-256。

报告使用 `results/course_final/` 中的正式记录。三个训练版本在两张图上分别跑 5 个种子，共 30 次训练，每次 200 轮。运行结果保存如下。

| 文件 | 内容 |
|---|---|
| `data/benchmarks.json` | 数据源、实际规模和完整 SHA-256 |
| `results/course_final/metrics.csv` | 每个种子或确定性基线的 AUROC、AP、P@10 |
| `results/course_final/summary.csv` | 各方法的均值与样本标准差 |
| `results/course_final/metadata.json` | 数据版本、参数、环境及评价协议 |
| `results/course_final/{books,disney}/` | 每轮损失、只含参数的权重文件和逐节点分数 |

每次重复实验请指定新的输出目录。节点编号为原数据行号，从 0 开始。双重构模型的三个评分文件共享同一套训练权重。异常标注仅用于评分结束后的评价。

报告 PDF 位于 `output/pdf/`。可用 PDF 阅读器添加姓名、学号，或用装有 NumPy 和 ReportLab 的 Python 运行下列命令，输出符合课程命名规则的副本。脚本使用 Windows 自带的微软雅黑字体。

```powershell
python scripts/make_course_report.py --student-id '你的学号' --student-name '你的姓名' --output 'output/pdf/学号-姓名.pdf'
```

课程实验涉及的来源和 AI 辅助说明见报告第 6 页。下面保留此前的合成图示例，便于单独验证基础流程。

## 合成图流程检查

在项目根目录执行（Python 3.10+）：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
```

默认生成固定随机种子的合成图，运行 PCA 属性重构、邻域属性残差和两者的分数融合。
它们是基础对照方法，不是三个深度学习模型；邻域残差依赖结构和属性，不应称为“仅结构”。
输出在 results/smoke/：汇总、逐节点分数、指标、实验配置和环境。
合成结果仅用于流程检查，不支持真实数据或论文性能结论。

## 深度学习实验

建议使用项目虚拟环境安装依赖，避免修改系统环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deep.txt
.\.venv\Scripts\python.exe run_experiment.py --deep --output results/deep_smoke
```

实现了两层 GCN 编码器、属性解码器和内积结构解码器。分别训练 dual、attribute、structure 三种重构目标，每组 3 个种子。
这些是“重构目标消融”，三个模型的输入都包含属性和结构，不能宣称是“输入信息消融”。
这是教学用 GCN 双重构基线，不是 DECLARE 或 DOMINANT 的严格复现。
默认 CPU、固定 100 轮；不使用标签选择训练轮次或参数。原始两项误差的尺度可能不同，后续应分析损失平衡。
结构损失包含全部非对角点对，稀疏图中非边占比较高，这是该起步基线的局限。

## 接入真实数据

将数据保存为 NumPy NPZ，字段如下：

| 字段 | 形状 | 含义 |
|---|---|---|
| x | N×F | 数值节点属性 |
| a | N×N | 无向非负邻接矩阵，载入后转为二值并移除自环 |
| y | N | 0 正常、1 异常，仅用于评价 |

```python
import numpy as np
np.savez_compressed('data/my_graph.npz', x=x, a=a, y=y)
```

```powershell
python run_experiment.py --data data/my_graph.npz --output results/my_graph
# PyTorch 环境下增加 --deep 运行深度学习基线与消融
```

载入器拒绝非对称图、非法标签和非有限数值，避免悄悄改变任务。
同一节点的属性、邻接矩阵行列、标签必须严格对齐。
目前是稠密小图实现，节点上限 5000；不是可扩展的大图训练器。大型数据须先改用稀疏表示和采样，不能直接转稠密。
不同运行请使用不同输出目录，避免覆盖已有实验。

## 评价协议

- 传导式无监督：全图属性和结构可见，全图标准化，标签仅用于计算 ROC-AUC 和 AP。
- 分数越大越异常；指标按同分阈值分组，避免同分排序导致偏差。
- 同一个数据文件、相同预处理用于所有模型。固定合成数据种子，训练随机种子另行控制。
- 确定性基础方法只运行一次；GCN 汇报训练随机种子的均值和样本标准差。
- 默认融合权重 0.5，提前固定。不可在测试标签上挑选最佳权重再声称无监督结果。
- 没有阈值化分类结果；不报告依赖真实异常数量确定阈值的 F1。

此前的方案、论文笔记和提纲保存在 `docs/`，正式实验以本页开头的复现命令为准。
