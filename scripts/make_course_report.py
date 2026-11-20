"""Build the six-page PDF from the preserved course experiment output."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

import numpy as np
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
TITLE = '图卷积自编码器的节点异常检测'
MARGIN = 53
WIDTH = A4[0] - 2 * MARGIN - 12
INK = colors.HexColor('#17232B')
ACCENT = colors.HexColor('#164A62')
MUTED = colors.HexColor('#465A65')
GRID = colors.HexColor('#CCDCE4')
PALE = colors.HexColor('#EDF4F7')
ALTERNATE = colors.HexColor('#F7F9FA')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def register_fonts(font_dir):
    pdfmetrics.registerFont(TTFont('Report', str(font_dir / 'msyh.ttc'), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ReportBold', str(font_dir / 'msyhbd.ttc'), subfontIndex=0))
    pdfmetrics.registerFontFamily('Report', normal='Report', bold='ReportBold')


def style(name, size, leading, *, bold=False, color=INK, after=0, before=0, align=0):
    return ParagraphStyle(
        name, fontName='ReportBold' if bold else 'Report', fontSize=size, leading=leading,
        textColor=color, spaceBefore=before, spaceAfter=after, wordWrap='CJK',
        alignment=align, allowWidows=0, allowOrphans=0,
    )


TITLE_STYLE = style('title', 25, 37, bold=True, color=colors.black, after=5)
SUBTITLE = style('subtitle', 13, 22, color=MUTED, after=12)
LABEL = style('label', 10.3, 17, color=ACCENT, after=7)
H1 = style('section', 19, 28, bold=True, color=ACCENT, after=12)
H2 = style('subsection', 12.1, 19, bold=True, before=8, after=4)
BODY = style('body', 10.8, 18.2, after=9)
SMALL = style('small', 9.65, 15.6, color=MUTED, after=6)
CAPTION = style('caption', 9.4, 15, color=MUTED, after=7, before=3)
FORMULA = style('formula', 11, 19, color=ACCENT, after=7, align=TA_CENTER)
REFERENCE = style('reference', 8.9, 13, after=4)
REFERENCE.wordWrap = None
REFERENCE.leftIndent = 18
REFERENCE.firstLineIndent = -18
CELL = style('cell', 9.5, 14.8)
CELL_CENTER = style('center cell', 9.3, 14.5, align=TA_CENTER)
HEADER = style('table header', 9.8, 14, bold=True, color=ACCENT)
HEADER_CENTER = style('center table header', 9.6, 14, bold=True, color=ACCENT, align=TA_CENTER)


def table(rows, fractions, centered=()):
    cells = []
    for index, row in enumerate(rows):
        row_cells = []
        for col, value in enumerate(row):
            if index == 0:
                cell_style = HEADER_CENTER if col in centered else HEADER
            else:
                cell_style = CELL_CENTER if col in centered else CELL
            row_cells.append(Paragraph(str(value), cell_style))
        cells.append(row_cells)
    result = Table(cells, colWidths=[WIDTH * f for f in fractions], repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALTERNATE]),
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, GRID),
        ('LINEBELOW', (0, -1), (-1, -1), 0.6, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    return result


def architecture():
    drawing = Drawing(WIDTH, 112)

    def box(x, y, width, title, detail):
        drawing.add(Rect(x, y, width, 37, rx=4, ry=4, fillColor=PALE, strokeColor=GRID))
        drawing.add(String(x + width / 2, y + 22, title, fontName='ReportBold',
                           fontSize=10, textAnchor='middle', fillColor=INK))
        drawing.add(String(x + width / 2, y + 8.5, detail, fontName='Report',
                           fontSize=8.2, textAnchor='middle', fillColor=MUTED))

    def arrow(start, end, y):
        drawing.add(Line(start, y, end - 4, y, strokeColor=ACCENT, strokeWidth=1.1))
        drawing.add(Polygon([end - 6, y - 3.3, end, y, end - 6, y + 3.3],
                            fillColor=ACCENT, strokeColor=None))

    box(0, 36, 91, '属性图', '节点属性 X、邻接 A')
    box(108, 36, 116, '两层图卷积', '对称归一化传播')
    box(241, 36, 83, '节点嵌入 Z', '8 维表示')
    box(352, 63, WIDTH - 352, '属性解码器', 'MLP 重构节点属性')
    box(352, 9, WIDTH - 352, '结构解码器', '投影内积预测连边')
    arrow(91, 108, 54.5)
    arrow(224, 241, 54.5)
    drawing.add(Line(324, 54.5, 337, 54.5, strokeColor=ACCENT, strokeWidth=1.1))
    drawing.add(Line(337, 27.5, 337, 81.5, strokeColor=ACCENT, strokeWidth=1.1))
    arrow(337, 352, 81.5)
    arrow(337, 352, 27.5)
    return drawing


def training_plot(results, config):
    figure = Drawing(WIDTH, 224)
    styles = [
        ('mlp_ae', 'MLP-AE', colors.HexColor('#1F5C8B')),
        ('gcn_attr', 'GCN 属性重构', colors.HexColor('#338166')),
        ('gcn_dual', 'GCN 双重构', colors.HexColor('#A85A2C')),
    ]
    for number, (_, label, tint) in enumerate(styles):
        x = 87 + 110 * number
        figure.add(Line(x, 11, x + 18, 11, strokeColor=tint, strokeWidth=2))
        figure.add(String(x + 22, 8, label, fontName='Report', fontSize=9, fillColor=INK))
    panel_width = WIDTH / 2 - 43
    for index, dataset in enumerate(('books', 'disney')):
        panel_x = index * WIDTH / 2 + 34
        series = []
        for method, _, _ in styles:
            runs = [
                read_csv(results / dataset / f'{method}_{seed}_loss.csv')
                for seed in config['seeds']
            ]
            series.append([
                (epoch, mean(float(run[epoch - 1]['loss']) for run in runs))
                for epoch in range(1, config['epochs'] + 1)
            ])
        plot = LinePlot()
        plot.x = panel_x
        plot.y = 48
        plot.width = panel_width
        plot.height = 139
        plot.data = series
        for i, (_, _, tint) in enumerate(styles):
            plot.lines[i].strokeColor = tint
            plot.lines[i].strokeWidth = 1.55
        plot.xValueAxis.valueMin = 1
        plot.xValueAxis.valueMax = config['epochs']
        plot.xValueAxis.valueSteps = [1, 50, 100, 150, config['epochs']]
        plot.yValueAxis.valueMin = 0
        plot.yValueAxis.valueMax = 1.08
        plot.yValueAxis.valueSteps = [0, .25, .5, .75, 1]
        plot.yValueAxis.labelTextFormat = '%.2f'
        for axis in (plot.xValueAxis, plot.yValueAxis):
            axis.labels.fontName = 'Report'
            axis.labels.fontSize = 8.1
            axis.labels.fillColor = MUTED
            axis.strokeColor = GRID
        plot.yValueAxis.visibleGrid = True
        plot.yValueAxis.gridStrokeColor = GRID
        plot.yValueAxis.gridStrokeWidth = 0.45
        figure.add(plot)
        middle = panel_x + panel_width / 2
        figure.add(String(middle, 202, dataset.title(), fontName='ReportBold',
                          fontSize=11, fillColor=ACCENT, textAnchor='middle'))
        figure.add(String(middle, 24, '训练轮次', fontName='Report', fontSize=8.8,
                          fillColor=MUTED, textAnchor='middle'))
    return figure


def footer(canvas, doc):
    canvas.setTitle(TITLE)
    canvas.setSubject('深度学习课程设计技术报告')
    canvas.setFont('Report', 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN + 6, 31, '深度学习课程设计  2026 秋季')
    canvas.drawRightString(A4[0] - MARGIN - 6, 31, f'{doc.page} / 6')


def linked(label, url):
    return f'<link href="{url}" color="#164A62">{label}</link>'


def build_report(results, data_dir, output, student_id, student_name):
    meta = json.loads((results / 'metadata.json').read_text(encoding='utf-8'))
    config = meta['config']
    by_method = {(row['dataset'], row['method']): row for row in read_csv(results / 'summary.csv')}
    datasets = meta['datasets']
    seeds = ', '.join(str(seed) for seed in config['seeds'])
    epochs = config['epochs']
    k = config['review_budget']
    total_fits = meta['fit_count']

    def value(dataset, method, metric='ap'):
        return float(by_method[dataset, method][f'{metric}_mean'])

    def metric_text(dataset, method, key):
        row = by_method[dataset, method]
        number = f"{float(row[f'{key}_mean']):.4f}"
        deviation = row[f'{key}_std']
        return f'{number} ± {float(deviation):.4f}' if deviation else number

    story = []

    def para(text, paragraph_style=BODY):
        story.append(Paragraph(text, paragraph_style))

    def heading(text):
        para(text, H2)

    def new_page(label):
        story.append(PageBreak())
        para(label, H1)

    para('深度学习课程设计  2026 秋季', LABEL)
    para(TITLE, TITLE_STYLE)
    para('Books 与 Disney 的无监督对比实验', SUBTITLE)
    name = escape(student_name) if student_name else '____________'
    code = escape(student_id) if student_id else '________________'
    para(f'姓名 {name}&nbsp;&nbsp;&nbsp;&nbsp;学号 {code}', SMALL)
    para('图神经网络方向　|　附实验代码与运行记录', SMALL)
    story.append(Spacer(1, 15))
    heading('摘要')
    para(
        f'本文考察图卷积中的邻居聚合，以及将结构误差纳入异常评分，能否改善静态属性图的节点异常排序。'
        f'在两个保留原始异常标注的公开共购图上，实现参数规模匹配的 MLP 与 GCN 属性自编码器，并为 GCN 增加结构解码器。'
        f'每个训练版本在每张图上使用 {len(config["seeds"])} 个随机种子，共完成 {total_fits} 次训练。'
        f'在 Books 和 Disney 上，双重构的平均精确率 AP 分别为 {value("books", "gcn_dual"):.4f} 和 '
        f'{value("disney", "gcn_dual"):.4f}，低于仅用属性的 MLP（{value("books", "mlp_ae"):.4f} 和 '
        f'{value("disney", "mlp_ae"):.4f}）。Disney 的邻域残差获得最高 AUROC '
        f'{value("disney", "neighbor_residual", "roc_auc"):.4f}。'
        '图关系可提供局部差异信号，但在本实验中通过 GCN 聚合和重构利用结构，未稳定提高异常排序。'
    )
    para('关键词　图卷积网络　节点异常检测　无监督学习　自编码器', SMALL)
    story.append(Spacer(1, 10))
    para('1 研究问题', H1)
    para(
        '在商品共购网络中，人工核查通常只能覆盖少量商品。节点的评价属性可以显示自身偏离，'
        '连接关系则提供局部参照。一个商品可能与全网大多数商品相似，却和经常与它一起购买的商品差异很大。'
        '反过来，如果邻居只是购物搭配关系，直接平均邻居属性可能掩盖节点自己的异常线索。'
    )
    para(
        '给定包含 N 个节点的无向图，输入为节点属性矩阵 X 和邻接矩阵 A，输出为每个节点的异常分数。'
        '分数越高，核查优先级越高。异常标注只用于事后检验排序。本文用两个对照回答图信息在何处提供了增益。'
    )
    story.append(table([
        ('研究问题', '对照设计'),
        ('RQ1 邻居聚合是否有帮助', '比较参数规模和属性目标相同的 MLP-AE 与 GCN 属性重构。'),
        ('RQ2 结构目标与结构评分是否有帮助', '加入结构解码器重新训练，再固定双重构模型，对比单项评分和联合评分。'),
    ], (.39, .61)))

    new_page('2 数据与实验环境')
    heading('公开属性图与异常标注')
    para(
        '数据取自 BOND 基准 [4] 的 PyGOD 官方数据仓库 [6]。两张图的边表示商品共购，属性包括价格、评分和评论数量等统计量。'
        'Books 的节点是图书，异常标注来自 amazonfail 标签。Disney 的节点是影片，异常标注来自人工评定 [5]。'
        '实验保留数据包中的标签及原始节点行序。'
    )
    para('表 1　预处理后的数据规模', CAPTION)
    dataset_table = [('数据集', '节点数 N', '无向边数', '属性维数', '异常数', '异常比例', '边密度')]
    for dataset in ('books', 'disney'):
        row = datasets[dataset]
        n, edges, positives = row['nodes'], row['edges_undirected'], row['anomalies']
        dataset_table.append((
            dataset.title(), f'{n:,}', f'{edges:,}', row['features'], positives,
            f'{100 * positives / n:.2f}%', f'{200 * edges / (n * (n - 1)):.2f}%',
        ))
    story.append(table(dataset_table, (.15, .14, .15, .13, .12, .15, .16), centered=range(1, 7)))
    story.append(Spacer(1, 6))
    para(
        '边数按无向点对去重后计数，边密度为实际边数与可能边数之比。Books 的发布文件包含 21 列属性，'
        '其中 1 列为常数。本报告的维数以实际使用的文件为准。', SMALL
    )
    heading('预处理与数据校验')
    para(
        '转换脚本读取属性、边索引和标签，将每条边写入邻接矩阵的两个方向，合并重复边并移除对角自环。'
        '载入时检查节点数量、矩阵对称性、属性有限值以及 0 和 1 的二值标签。每列属性减去全图均值，再除以标准差，'
        '分母下限设为 10<super>-8</super>。常量特征中心化后保留为零列。'
    )
    para(
        'Books 原始压缩包的 SHA-256 前 12 位为 <b>' + datasets['books']['archive_sha256'][:12] +
        '</b>，Disney 为 <b>' + datasets['disney']['archive_sha256'][:12] +
        '</b>。完整哈希、下载地址和转换后文件哈希记录在 data/benchmarks.json，'
        '训练脚本在运行前重新校验输入。'
    )
    heading('软件与硬件')
    story.append(table([
        ('项目', '实验设置'),
        ('运行平台', 'Windows 11，Intel Core i5-11400H，CPU 训练，PyTorch 使用 2 个计算线程'),
        ('软件版本', f'Python {meta["python"]}，PyTorch {meta["torch"]}，NumPy {meta["numpy"]}'),
        ('代码与数据形态', 'PyTorch 实现稀疏图传播，NumPy 完成数据处理与评价。下载原始数据后生成本地 NPZ。'),
    ], (.25, .75)))
    heading('传导式无监督评价')
    para(
        '训练和评分都在同一张可观测图上完成。所有方法共享同一图和同一份标准化属性。节点异常标签在模型输出分数后才进入指标计算，'
        '所以这里只检验给定图内的异常排序。跨图迁移和新节点检测需要单独验证。'
    )

    new_page('3 方法与技术')
    para(
        '两层 GCN [1] 将节点自身与近邻属性组合，用低维表示重构输入。DOMINANT [2] 和天津大学武南南等人的 '
        'DECLARE [3] 均使用属性与结构重构。DECLARE 还结合图 Transformer、扩散视图与对比学习。'
        '本实验实现其中的双重构思路，以单个 GCN 编码器控制实验变量。'
    )
    story.append(architecture())
    para('图 1　两层 GCN 的节点表示与两条重构分支', CAPTION)
    heading('图卷积与属性重构')
    para(
        '先为编码器加自环，使聚合保留自身信息。D 是 A + I 的度矩阵，W 和 b 是可学习参数。'
        '以标准化后的 X 为输入，编码器先映射到 32 维，再压缩到 8 维。'
    )
    para(
        'P = D<super>-1/2</super>(A + I)D<super>-1/2</super><br/>'
        'H = ReLU(PXW<sub>0</sub> + b<sub>0</sub>)'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Z = PHW<sub>1</sub> + b<sub>1</sub>',
        FORMULA
    )
    para(
        '属性解码器从 Z 经 32 维隐藏层还原每个节点的原始维数。节点属性误差 e<sub>X</sub> 是重构向量和标准化输入'
        '之间的逐维均方误差。MLP 对照将 P 替换为单位作用，编码器和属性解码器的可学习参数维持相同形状。'
    )
    heading('结构重构与联合评分')
    para(
        '结构解码器将每个嵌入 z<sub>i</sub> 线性投影到 q<sub>i</sub>，以缩放内积加一个标量偏置 b 得到连边 logit。'
        'A 的非对角元素提供连边目标。稀疏图的无边点对远多于有边点对，因此分别求两类的二元交叉熵均值，再等权平均。'
    )
    para(
        'ℓ<sub>ij</sub> = q<sub>i</sub><super>T</super>q<sub>j</sub> / √8 + b<br/>'
        'e<sub>A</sub>(i) = [有边平均 BCE + 无边平均 BCE] / (2 ln 2)',
        FORMULA
    )
    para(
        'BCE 以连边 logit 为输入并采用数值稳定的 softplus 计算。若某一类为空，只使用存在类别的均值。'
        '除以 ln 2 后，零 logit 的结构误差为 1。训练时在每个可用类别内抽样，评分时对该节点的全部非对角点对精确计算。'
    )
    para(
        's<sub>i</sub> = 0.5 e<sub>X</sub>(i) + 0.5 e<sub>A</sub>(i)<br/>'
        'L = 节点抽样重构分数的均值',
        FORMULA
    )
    para('等权用于训练和联合评分。单项排序直接使用对应的节点误差，便于定位联合评分中的变化。', SMALL)

    new_page('4 设计实现与训练过程')
    heading('基线与消融安排')
    story.append(table([
        ('方法', '实现与作用'),
        ('PCA 属性重构', '用标准化属性的前 8 个主方向重构，提供线性属性基线。'),
        ('邻域属性残差', '计算节点与一阶邻居属性均值之间的均方差，检查局部属性差异。'),
        ('MLP-AE', '采用两层编码与逐节点解码，供 RQ1 隔离邻居聚合。'),
        ('GCN 属性重构', '加入两次归一化聚合，训练和评分只使用属性重构误差。'),
        ('GCN 双重构', '保留同一属性模型并训练结构分支，联合评分用于 RQ2。'),
    ], (.285, .715)))
    story.append(Spacer(1, 7))
    para(
        '每张图分别训练全部深度模型。Adam 学习率为 0.005，固定 200 轮，种子为 '
        f'{seeds}。每轮使用全体节点计算属性损失，对每个节点从有边与无边集合各均匀有放回抽样 '
        f'{config["pairs_per_class"]} 次。训练结束后统一用最后一轮权重评分，评价标签不参与参数或轮次选择。'
    )
    story.append(training_plot(results, config))
    para('图 2　五个种子的平均训练目标。横轴为轮次，纵轴为各方法自身的损失。', CAPTION)
    books_loss, disney_loss = [
        [
            mean(float(read_csv(results / dataset / f'gcn_dual_{seed}_loss.csv')[index]['loss'])
                 for seed in config['seeds'])
            for index in (0, -1)
        ]
        for dataset in ('books', 'disney')
    ]
    para(
        f'双重构平均损失在 Books 上从 {books_loss[0]:.3f} 降到 {books_loss[1]:.3f}，'
        f'在 Disney 上从 {disney_loss[0]:.3f} 降到 {disney_loss[1]:.3f}。'
        '损失下降说明模型拟合了训练目标，异常排序仍需要用独立于损失的指标验证。'
    )
    heading('从代码到实验产物')
    para(
        'prepare_benchmarks.py 负责数据转换，src/graph_autoencoder.py 实现编码器、抽样损失和精确评分。'
        'run_course_experiment.py 根据 configs/course.json 循环训练，保存每轮损失、每次模型权重及逐节点分数和排名。'
        '不同运行使用独立目录，结果可从原始分数重新计算。'
    )

    new_page('5 实验结果')
    para(
        'AUROC 衡量随机抽取的异常节点相对正常节点排在前面的能力。平均精确率 AP 按召回增量对精确率加权，'
        '在异常比例较低时关注排名前部的表现。所有节点同分时，AP 等于异常比例，Books 为 0.0197，Disney 为 0.0484。'
        '另以固定的前 10 名模拟人工核查预算。所有分数从高到低排序，异常为正类。'
    )
    para('表 2　主要对比结果。神经模型为五次均值 ± 样本标准差，确定性基线为单次结果。', CAPTION)
    model_names = [
        ('pca', 'PCA 属性重构'), ('neighbor_residual', '邻域属性残差'), ('mlp_ae', 'MLP-AE'),
        ('gcn_attr', 'GCN 属性重构'), ('gcn_dual', 'GCN 双重构'),
    ]
    main_table = [('方法', 'Books<br/>AUROC', 'Books<br/>AP', 'Disney<br/>AUROC', 'Disney<br/>AP')]
    for method, name in model_names:
        cells = []
        for dataset in ('books', 'disney'):
            for key in ('roc_auc', 'ap'):
                rendered = metric_text(dataset, method, key)
                if value(dataset, method, key) == max(value(dataset, candidate, key) for candidate, _ in model_names):
                    rendered = f'<b>{rendered}</b>'
                cells.append(rendered)
        main_table.append((name, *cells))
    story.append(table(main_table, (.215, .19625, .19625, .19625, .19625), centered=range(1, 5)))
    heading('RQ1 邻居聚合后 AP 下降')
    para(
        f'对照表 2，GCN 属性重构的 AP 比 MLP-AE 在 Books 低 '
        f'{value("books", "mlp_ae") - value("books", "gcn_attr"):.4f}，在 Disney 低 '
        f'{value("disney", "mlp_ae") - value("disney", "gcn_attr"):.4f}。'
        'Disney 的邻域属性残差取得最高 AUROC，说明直接比较节点与邻居的差异也能利用图关系。'
        '图关系的收益取决于边的用法和评价指标。'
    )
    heading('RQ2 同一模型中的结构评分未带来整体增益')
    para('表 3　固定双重构训练权重后的评分消融，数值为五次 AP 均值 ± 样本标准差。', CAPTION)
    scoring_rows = [
        ('dual_attr_score', '只用属性误差'), ('dual_struct_score', '只用结构误差'), ('gcn_dual', '等权联合评分'),
    ]
    story.append(table(
        [('评分方式', 'Books AP', 'Disney AP')] + [
            (name, metric_text('books', method, 'ap'), metric_text('disney', method, 'ap'))
            for method, name in scoring_rows
        ],
        (.40, .30, .30), centered=(1, 2)
    ))
    story.append(Spacer(1, 7))
    para(
        '在两张图上，等权联合评分的 AP 均低于同一模型只用属性误差的结果。'
        '与从头训练的 GCN 属性重构比较，完整双重构在 Books 的 AP 略高，在 Disney 较低。'
        '训练目标的影响和最终评分的影响应分别看待。'
    )
    para(
        f'以固定的前 {k} 名为核查预算，GCN 双重构在两张图的全部种子上均未命中已标异常。'
        f'MLP-AE 在 Books 平均每次命中 {k * value("books", "mlp_ae", "precision_at_k"):.1f} 个，'
        f'在 Disney 平均命中 {k * value("disney", "mlp_ae", "precision_at_k"):.1f} 个。'
        '在这个预算下的低命中率限制了上述方法直接用于核查的价值。'
    )

    new_page('6 案例分析与改进')
    heading('Books 节点的误报与漏报')
    para(
        '以种子 42 为例，选取双重构最高分的标注正常节点、被 MLP 排入前十的异常节点，'
        '以及结构分支提高排名但仍未进入核查范围的异常节点。节点编号为数据文件的零起始行号。'
    )
    with np.load(data_dir / 'books.npz', allow_pickle=False) as data:
        degree = data['a'].sum(axis=1)
    folder = results / 'books'
    rows = {
        method: {int(row['node_id']): row for row in read_csv(folder / f'{method}_42_scores.csv')}
        for method in ('gcn_dual', 'dual_attr_score', 'dual_struct_score', 'mlp_ae')
    }
    case_rows = [('节点', '标签', '度', '属性误差', '结构误差', '联合排名', 'MLP 排名')]
    for node in (1034, 557, 1286):
        case_rows.append((
            node, rows['gcn_dual'][node]['label'], int(degree[node]),
            f'{float(rows["dual_attr_score"][node]["anomaly_score"]):.3f}',
            f'{float(rows["dual_struct_score"][node]["anomaly_score"]):.3f}',
            rows['gcn_dual'][node]['rank'], rows['mlp_ae'][node]['rank'],
        ))
    para('表 4　同一批节点的分数分解与排名。标签 1 为异常，排名 1 为最可疑。', CAPTION)
    story.append(table(case_rows, (.12, .095, .085, .17, .17, .18, .18), centered=range(7)))
    story.append(Spacer(1, 6))
    para(
        '节点 1034 的属性误差为全图最高，推高了它的联合分数。异常节点 557 从 MLP 的第 7 名跌至联合评分的第 54 名，'
        '一旦采用前十名的预算就会漏检。节点 1286 的结构误差排第 40，属性误差排第 1243，'
        '联合后升至第 496。该节点体现了结构分支的补充作用，也显示等权融合可能压低某一分支的强信号。'
    )
    heading('局限与后续验证')
    para(
        '两个数据集同属共购图，Disney 只有 6 个异常，个别节点的排名变化就会放大 AP 波动。样本标准差反映的是初始化'
        '和抽样变化，无法回答跨图泛化。下一步可加入自特征残差连接，与现有 GCN 做配对对照，检验保留节点自身特征'
        '是否能缓解漏报。结构分支可按局部度分组检查误差分布，并在其他有独立标注的图上验证分数校准。'
    )
    heading('参考文献与工具说明')
    refs = [
        '[1] Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. '
        'International Conference on Learning Representations (ICLR), 2017. '
        + linked('arXiv:1609.02907', 'https://arxiv.org/abs/1609.02907'),
        '[2] Kaize Ding, Jundong Li, Rohit Bhanushali, and Huan Liu. Deep Anomaly Detection on Attributed Networks. '
        'Proceedings of the 2019 SIAM International Conference on Data Mining (SDM), pp. 594-602. '
        + linked('doi:10.1137/1.9781611975673.67', 'https://doi.org/10.1137/1.9781611975673.67'),
        '[3] Nannan Wu, Hongdou Dong, Wenjun Wang, and Yiming Zhao. Dual Encoder Contrastive Learning with '
        'Augmented Views for Graph Anomaly Detection. Proceedings of the Thirty-Fourth International Joint '
        'Conference on Artificial Intelligence (IJCAI), 2025, pp. 3480-3488. '
        + linked('doi:10.24963/ijcai.2025/387', 'https://doi.org/10.24963/ijcai.2025/387'),
        '[4] Kay Liu, Yingtong Dou, Yue Zhao, Xueying Ding, Xiyang Hu, Ruitong Zhang, Kaize Ding, Canyu Chen, '
        'Hao Peng, Kai Shu, Lichao Sun, Jundong Li, George H. Chen, Zhihao Jia, and Philip S. Yu. '
        'BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs. '
        'Advances in Neural Information Processing Systems 35 (NeurIPS), 2022, pp. 27021-27035. '
        + linked('doi:10.52202/068431-1959', 'https://doi.org/10.52202/068431-1959'),
        '[5] Jundong Li, Harsh Dani, Xia Hu, and Huan Liu. Radar: Residual Analysis for Anomaly Detection in '
        'Attributed Networks. Proceedings of the Twenty-Sixth International Joint Conference on Artificial '
        'Intelligence (IJCAI), 2017, pp. 2152-2158. '
        + linked('doi:10.24963/ijcai.2017/299', 'https://doi.org/10.24963/ijcai.2017/299'),
        '[6] PyGOD Team. Data repository for PyGOD. GitHub, '
        + linked('github.com/pygod-team/data', 'https://github.com/pygod-team/data')
        + '. Books 与 Disney 数据包，2026-11-20 访问。',
    ]
    for reference in refs:
        para(reference, REFERENCE)
    para('OpenAI Codex 辅助编写与检查代码、整理报告。表和图来自本项目实测记录。', REFERENCE)

    document = SimpleDocTemplate(
        str(output), pagesize=A4, title=TITLE, author=student_name,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=50, bottomMargin=52,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    if document.page != 6:
        raise RuntimeError(f'Expected six report pages, generated {document.page}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/course_final')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/pdf' / f'{TITLE}.pdf')
    parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
    parser.add_argument('--student-id', default='')
    parser.add_argument('--student-name', default='')
    args = parser.parse_args()
    register_fonts(args.font_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_report(args.results, args.data_dir, args.output, args.student_id, args.student_name)
    print(args.output)


if __name__ == '__main__':
    main()
