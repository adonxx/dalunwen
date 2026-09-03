"""Build the updated 2026-08-28 Drone-YOLO progress report PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output" / "pdf" / "2026.8.28阶段性实验汇报_更新版.pdf"
FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")

INK = colors.HexColor("#303236")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#D9DDE3")
PANEL = colors.HexColor("#F5F6F8")
BLUE = colors.HexColor("#4263EB")
GREEN = colors.HexColor("#0CA678")
ORANGE = colors.HexColor("#F08C00")
RED = colors.HexColor("#C92A2A")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(FONT_REGULAR), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", str(FONT_BOLD), subfontIndex=0))


class MetricCards(Flowable):
    """Draw four compact metric cards across one page."""

    def __init__(self, items: list[tuple[str, str, str]], width: float, height: float = 64):
        super().__init__()
        self.items = items
        self.width = width
        self.height = height

    def draw(self) -> None:
        card_width = (self.width - 12 * (len(self.items) - 1)) / len(self.items)
        for index, (value, label, accent) in enumerate(self.items):
            x = index * (card_width + 12)
            self.canv.setFillColor(PANEL)
            self.canv.setStrokeColor(LINE)
            self.canv.roundRect(x, 0, card_width, self.height, 5, fill=1, stroke=1)
            self.canv.setFillColor(colors.HexColor(accent))
            self.canv.rect(x, self.height - 4, card_width, 4, fill=1, stroke=0)
            self.canv.setFillColor(INK)
            self.canv.setFont("MicrosoftYaHei-Bold", 15)
            self.canv.drawCentredString(x + card_width / 2, self.height - 27, value)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("MicrosoftYaHei", 8.5)
            self.canv.drawCentredString(x + card_width / 2, 13, label)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=23,
            leading=31,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName="MicrosoftYaHei",
            fontSize=10,
            leading=16,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=17,
            leading=23,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=7,
            borderColor=LINE,
            borderWidth=0,
            borderPadding=(0, 0, 4, 0),
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=12,
            leading=18,
            textColor=INK,
            spaceBefore=6,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.5,
            leading=16,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.2,
            leading=15,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "note": ParagraphStyle(
            "NoteCN",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=8.3,
            leading=13,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "CalloutCN",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.3,
            leading=15,
            textColor=INK,
            backColor=colors.HexColor("#EEF3FF"),
            borderColor=colors.HexColor("#B9C7FF"),
            borderWidth=0.7,
            borderPadding=8,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "small_center": ParagraphStyle(
            "SmallCenter",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def heading(text: str, number: int, style: ParagraphStyle) -> KeepTogether:
    line = Table([[Paragraph(f"{number}. {text}", style)]], colWidths=[100 * mm])
    line.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([line, Spacer(1, 3)])


def make_table(
    data: list[list[str]],
    widths: list[float],
    font_size: float = 8.5,
    highlight_rows: tuple[int, ...] = (),
) -> Table:
    converted = []
    for row_index, row in enumerate(data):
        converted.append(
            [
                Paragraph(
                    str(value),
                    ParagraphStyle(
                        f"cell-{row_index}-{col_index}",
                        fontName="MicrosoftYaHei-Bold" if row_index == 0 else "MicrosoftYaHei",
                        fontSize=font_size,
                        leading=font_size + 4,
                        textColor=INK,
                        alignment=TA_LEFT if col_index == 0 else TA_CENTER,
                    ),
                )
                for col_index, value in enumerate(row)
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF3")),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFB")]),
    ]
    for row in highlight_rows:
        commands.extend(
            [
                ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#EAF7F2")),
                ("TEXTCOLOR", (0, row), (-1, row), GREEN),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def bullet(text: str, s: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"•&nbsp;&nbsp;{text}", s["bullet"])


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(doc.leftMargin, 17 * mm, width - doc.rightMargin, 17 * mm)
    canvas.setFont("MicrosoftYaHei", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 11 * mm, "无人机小目标检测与轻量化 · 阶段性实验汇报")
    canvas.drawRightString(width - doc.rightMargin, 11 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def build() -> None:
    register_fonts()
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="2026.8.28阶段性实验汇报",
        author="AllenX",
        subject="Drone-YOLO复现、小目标切片与尺度感知融合实验",
    )
    story = []

    # Page 1: baseline and completed screening.
    story.extend(
        [
            Paragraph("无人机小目标检测与轻量化阶段性汇报", s["title"]),
            Paragraph("更新日期：2026年8月28日　研究对象：VisDrone2019　基线：YOLOv11s", s["subtitle"]),
            heading("阶段完成内容", 1, s["h1"]),
            Paragraph(
                "已完成Drone-YOLO四阶段复现、P2/P5检测尺度消融、自适应加权MF-FPN多随机种子复验、NWD回归损失筛选、SAHI切片推理、Scale-48尺度感知融合及真实联合推理验证。当前实验链条已从结构复现推进到小目标精度与推理效率的联合优化。",
                s["body"],
            ),
            heading("基础复现结果", 2, s["h1"]),
            make_table(
                [
                    ["模型", "mAP50", "mAP50-95", "Small AP", "参数量", "GFLOPs"],
                    ["YOLOv11s", "39.64%", "23.23%", "13.52%", "9.432M", "21.56"],
                    ["MF-FPN", "41.08%", "24.27%", "15.63%", "3.169M", "24.52"],
                    ["MF-FPN+LSCD", "40.24%", "23.66%", "15.27%", "2.772M", "19.19"],
                    ["完整模型", "41.40%", "24.34%", "15.62%", "2.772M", "19.19"],
                ],
                [37 * mm, 23 * mm, 28 * mm, 25 * mm, 23 * mm, 23 * mm],
                highlight_rows=(4,),
            ),
            Spacer(1, 6),
            Paragraph(
                "完整模型相对YOLOv11s的mAP50提高1.76个百分点、Small AP提高2.09个百分点，参数量减少约70.6%。结果支持MF-FPN与Inner-WIoU的有效性；LSCD显著降低参数量和计算量，但其独立精度增益尚未复现。",
                s["callout"],
            ),
            heading("已完成的候选方法筛选", 3, s["h1"]),
            make_table(
                [
                    ["候选方向", "关键结果", "判定"],
                    ["加权MF-FPN", "3个种子平均ΔSmall AP=-0.15 pp；仅1/3为正", "停止晋级"],
                    ["NWD回归损失", "ΔSmall AP=-0.99 pp；ΔmAP50-95=-0.47 pp", "停止晋级"],
                    ["Inner-WIoU+NWD", "ΔSmall AP=-0.31 pp；ΔmAP50-95=-0.21 pp", "停止晋级"],
                ],
                [39 * mm, 84 * mm, 36 * mm],
            ),
            Spacer(1, 5),
            bullet("P2检测层是小目标收益的主要来源，但需要持续关注尺度间能力分配。", s),
            bullet("加权融合的小幅单次收益未能跨随机种子稳定复现，不能写成有效贡献。", s),
            bullet("NWD作为框回归损失未改善小目标，后续不再增加训练轮数。", s),
        ]
    )

    # Page 2: SAHI and Scale-48.
    story.extend(
        [
            PageBreak(),
            Paragraph("新增实验：切片推理与尺度感知融合", s["title"]),
            Paragraph("目的：验证整图缩放造成的小目标像素损失，并恢复切片对中、大目标上下文的破坏。", s["subtitle"]),
            heading("SAHI切片推理", 1, s["h1"]),
            make_table(
                [
                    ["方法", "AP50-95", "Small AP", "Medium AP", "Large AP", "延迟"],
                    ["Whole-640", "23.51%", "14.75%", "34.10%", "45.29%", "26.3 ms"],
                    ["SAHI-640", "23.32%", "18.20%", "29.72%", "25.35%", "57.5 ms"],
                    ["SAHI-960", "23.76%", "18.66%", "30.14%", "27.02%", "33.0 ms"],
                ],
                [34 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
                highlight_rows=(3,),
            ),
            Spacer(1, 5),
            Paragraph(
                "SAHI-960使Small AP提高3.91个百分点，证明整图缩放确实是当前小目标瓶颈之一；但Large AP下降18.28个百分点，说明纯切片缺失完整上下文，不能直接作为最终方案。",
                s["callout"],
            ),
            heading("Scale-48尺度感知融合", 2, s["h1"]),
            Paragraph(
                "融合规则：保留全部Whole-640预测，仅补充SAHI-960中预测面积小于48²像素的框，再执行类别感知NMS。48像素是预测尺度容差，并非新的网络层或输入尺度。",
                s["body"],
            ),
            make_table(
                [
                    ["方法", "AP50-95", "Small AP", "Small AR", "Medium AP", "Large AP"],
                    ["Whole-640", "23.51%", "14.75%", "24.46%", "34.10%", "45.29%"],
                    ["SAHI-960", "23.76%", "18.66%", "31.64%", "30.14%", "27.02%"],
                    ["Union-All", "26.53%", "19.53%", "33.28%", "34.70%", "42.18%"],
                    ["Scale-32", "26.34%", "19.24%", "33.25%", "34.28%", "44.09%"],
                    ["Scale-48", "26.82%", "19.55%", "33.35%", "34.81%", "44.09%"],
                ],
                [34 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
                highlight_rows=(5,),
            ),
            Spacer(1, 7),
            Image(
                str(ROOT / "reports" / "sahi_scale_fusion" / "figures" / "fusion_size_metrics.png"),
                width=159 * mm,
                height=79 * mm,
            ),
            Paragraph("图1　整图、纯切片与尺度感知融合的分尺寸AP对比。", s["small_center"]),
        ]
    )

    # Page 3: true runtime and next experiment.
    joint_record = json.loads(
        (ROOT / "reports" / "scale48_joint_inference" / "metrics" / "joint_scale48.json").read_text(
            encoding="utf-8"
        )
    )
    efficiency = joint_record["efficiency"]
    story.extend(
        [
            PageBreak(),
            Paragraph("真实联合推理与下一阶段计划", s["title"]),
            Paragraph("548张验证图像、同一模型进程、单次图像解码、FP32、RTX 4080 SUPER。", s["subtitle"]),
            heading("真实Scale-48联合推理", 1, s["h1"]),
            MetricCards(
                [
                    ("26.82%", "mAP50-95", "#4263EB"),
                    ("19.55%", "Small AP", "#0CA678"),
                    (f"{efficiency['mean_end_to_end_ms']:.2f} ms", "平均端到端延迟", "#F08C00"),
                    (f"{efficiency['fps_from_mean_end_to_end']:.2f}", "FPS", "#9775FA"),
                ],
                width=159 * mm,
            ),
            Spacer(1, 8),
            make_table(
                [
                    ["复验项目", "结果"],
                    ["离线融合与真实联合推理AP差异", "0.000个百分点（完全复现）"],
                    ["Whole-640 → Scale-48延迟", "26.35 → 35.13 ms/图（1.33×）"],
                    ["P95延迟 / 峰值显存", "40.59 ms / 1088 MB"],
                    ["完整性", "548张、156799框、零越界、检查点一致"],
                ],
                [75 * mm, 84 * mm],
                highlight_rows=(1,),
            ),
            Spacer(1, 7),
            Image(
                str(ROOT / "reports" / "scale48_joint_inference" / "figures" / "joint_timing_breakdown.png"),
                width=115 * mm,
                height=68.5 * mm,
            ),
            Paragraph("图2　真实Scale-48联合推理平均耗时分解。", s["small_center"]),
            heading("当前可以汇报的核心结论", 2, s["h1"]),
            bullet("Drone-YOLO复现实现了小目标精度提升和模型参数量显著下降。", s),
            bullet("加权MF-FPN和NWD回归未通过预注册筛选，形成了可解释的失败证据。", s),
            bullet("SAHI证实输入像素损失是小目标瓶颈；Scale-48通过整图与切片互补获得Small AP +4.80 pp、总体AP +3.32 pp。", s),
            bullet("真实联合程序完整复现离线精度，平均28.47 FPS；当前新增规则不增加模型参数，但仍需降低切片计算。", s),
            heading("本轮新增实验", 3, s["h1"]),
            Paragraph(
                "密度自适应切片已按预注册方案完成：512张训练图像无标签校准，三组门控在548张验证图像上真实计时；结果与下一步见第4页。",
                s["callout"],
            ),
        ]
    )

    # Page 4: density-gated slicing results completed after the 08-28 update began.
    story.extend(
        [
            PageBreak(),
            Paragraph("新增实验：密度自适应切片", s["title"]),
            Paragraph("训练集512张无标签校准；验证集548张；阈值冻结后不在验证集继续搜索。", s["subtitle"]),
            heading("Gate-Q30关键结果", 1, s["h1"]),
            MetricCards(
                [
                    ("26.49%", "mAP50-95", "#4263EB"),
                    ("19.19%", "Small AP", "#0CA678"),
                    ("24.36 ms", "平均端到端延迟", "#F08C00"),
                    ("41.05", "FPS", "#9775FA"),
                ],
                width=159 * mm,
            ),
            Spacer(1, 6),
            make_table(
                [
                    ["方法", "AP50-95", "Small AP", "延迟", "较Full降时"],
                    ["Whole-640", "23.51%", "14.75%", "26.35 ms", "—"],
                    ["Full Scale-48", "26.82%", "19.55%", "35.13 ms", "—"],
                    ["Top-1", "26.55%", "19.31%", "31.72 ms", "9.7%"],
                    ["Gate-Q20", "26.53%", "19.26%", "25.63 ms", "27.0%"],
                    ["Gate-Q30", "26.49%", "19.19%", "24.36 ms", "30.7%"],
                ],
                [39 * mm, 27 * mm, 29 * mm, 32 * mm, 32 * mm],
                highlight_rows=(5,),
            ),
            Spacer(1, 6),
            Image(
                str(ROOT / "reports" / "density_gated_sahi" / "figures" / "gate_accuracy_latency.png"),
                width=107 * mm,
                height=70 * mm,
            ),
            Paragraph("图3　密度门控的Small AP—端到端延迟权衡。", s["small_center"]),
            heading("预注册判定与解释", 2, s["h1"]),
            bullet("Gate-Q30将平均选择切片数降至0.84，切片计算减少56.0%，延迟降低30.7%。", s),
            bullet("Gate-Q30的Small AP下降0.36 pp，达到保持条件；总体AP下降0.34 pp，超过0.20 pp上限。", s),
            bullet("三组门控均未同时满足全部条件，因此停止在验证集上继续调阈值；Full Scale-48保留为精度方案。", s),
            heading("下一步：转向训练阶段优化", 3, s["h1"]),
            Paragraph(
                "优先验证RFLA式高斯感受野标签分配或小目标区域采样：先做独立单变量消融，再与Full Scale-48组合，判断能否提升Small AP且不增加推理开销。",
                s["callout"],
            ),
        ]
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
