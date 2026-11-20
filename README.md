# 图节点异常检测

在 Books 和 Disney 两个共购网络上比较几种异常检测方法，看看加入邻居信息和结构重构后，异常节点能不能排得更靠前。对照方法有 PCA、邻域残差、MLP 自编码器和两种 GCN 自编码器。

## 运行

在项目目录打开 Windows CMD。已有 `.venv` 的话，从第三条开始。

```cmd
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-lock.txt
.\.venv\Scripts\python.exe prepare_benchmarks.py --download
.\.venv\Scripts\python.exe run_course_experiment.py --output results/reproduce
```

数据包缺失时才会下载。训练用 CPU，每个模型跑 5 个随机种子，每次 200 轮。跑完先看 `results/reproduce/summary.csv`，单次结果在 `metrics.csv`，各数据集的子目录里有训练损失、模型权重和节点排名。再跑一次时换一个输出目录名即可。

## 结果

下面是 GCN 双重构在 5 个种子上的均值。

| 数据集 | AUROC | AP |
|---|---:|---:|
| Books | 0.5190 | 0.0231 |
| Disney | 0.4153 | 0.0510 |

两张图上，只用节点属性的 MLP 得到的 AP 都高于双重构。Disney 上的邻域残差则拿到了最高的 AUROC（0.6638）。

## 代码入口

`prepare_benchmarks.py` 准备数据，`run_course_experiment.py` 负责训练和汇总，模型在 `src/graph_autoencoder.py`。训练参数可以在 `configs/course.json` 里改。[逐行讲解](docs/正式实验代码逐行讲解.md)

数据取自 [PyGOD](https://github.com/pygod-team/data)。模型参考 [GCN](https://arxiv.org/abs/1609.02907)、[DOMINANT](https://doi.org/10.1137/1.9781611975673.67) 和 [DECLARE](https://ijcai.org/proceedings/2025/387)。
