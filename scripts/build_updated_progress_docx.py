"""Build an editable Word version of the 2026-08-28 progress report."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output" / "docx" / "2026.8.28阶段性实验汇报_更新版.docx"

INK = "303236"
MUTED = "6B7280"
LINE = "D9DDE3"
HEADER_FILL = "E9EDF2"
PANEL_FILL = "F5F6F8"
HIGHLIGHT_FILL = "E8F5F0"
CALLOUT_FILL = "EEF3FF"
CALLOUT_BORDER = "B9C7FF"
BLUE = "4263EB"
GREEN = "0CA678"
ORANGE = "F08C00"
PURPLE = "9775FA"

FONT = "Microsoft YaHei"
FONT_BOLD = "Microsoft YaHei"

# standard_business_brief preset with named overrides for the supplied A4 PDF:
# A4, 20 mm side margins, 16/20 mm top/bottom, Microsoft YaHei, gray-indigo palette.
PAGE_WIDTH_DXA = 11906
PAGE_HEIGHT_DXA = 16838
CONTENT_WIDTH_DXA = 9638
TABLE_INDENT_DXA = 120


def set_run_font(run, *, size: float | None = None, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 65, start: int = 120, bottom: int = 65, end: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int], *, indent: int = TABLE_INDENT_DXA) -> None:
    if sum(widths) != CONTENT_WIDTH_DXA:
        raise ValueError(f"Table widths must sum to {CONTENT_WIDTH_DXA}, got {sum(widths)}")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def set_table_borders(table, color: str = LINE, size: int = 5) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), str(size))
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def paragraph_border(paragraph, *, color: str, size: int = 6, space: int = 4, edge: str = "bottom") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    border = OxmlElement(f"w:{edge}")
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), str(size))
    border.set(qn("w:space"), str(space))
    border.set(qn("w:color"), color)
    p_bdr.append(border)


def paragraph_shading(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, before, after in (
        ("Heading 1", 16, 6, 4),
        ("Heading 2", 13.5, 6, 3),
        ("Heading 3", 11.5, 5, 3),
    ):
        style = styles[name]
        style.font.name = FONT_BOLD
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_BOLD)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_BOLD)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_BOLD)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(INK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = FONT
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    caption.font.size = Pt(8)
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(4)


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Mm(20)
    section.right_margin = Mm(20)
    section.top_margin = Mm(16)
    section.bottom_margin = Mm(20)
    section.header_distance = Mm(8)
    section.footer_distance = Mm(9)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    set_run_font(run, size=8, color=MUTED)


def configure_footer(doc: Document) -> None:
    section = doc.sections[0]
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(0)
    tabs = paragraph.paragraph_format.tab_stops
    tabs.add_tab_stop(Mm(166))
    left = paragraph.add_run("无人机小目标检测与轻量化 · 阶段性实验汇报")
    set_run_font(left, size=8, color=MUTED)
    tab = paragraph.add_run("\t第 ")
    set_run_font(tab, size=8, color=MUTED)
    add_page_field(paragraph)
    tail = paragraph.add_run(" 页")
    set_run_font(tail, size=8, color=MUTED)


def create_bullet_numbering(doc: Document) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    level.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet")
    level.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "•")
    level.append(lvl_text)
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    level.append(lvl_jc)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "720")
    ind.set(qn("w:hanging"), "360")
    p_pr.append(ind)
    level.append(p_pr)
    r_pr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT)
    fonts.set(qn("w:hAnsi"), FONT)
    fonts.set(qn("w:eastAsia"), FONT)
    r_pr.append(fonts)
    level.append(r_pr)
    abstract.append(level)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def add_title(doc: Document, text: str, subtitle: str, *, page_break_before: bool = False) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.page_break_before = page_break_before
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, size=22, bold=True, color=INK)
    meta = doc.add_paragraph()
    meta.paragraph_format.space_before = Pt(0)
    meta.paragraph_format.space_after = Pt(9)
    meta.paragraph_format.keep_with_next = True
    run = meta.add_run(subtitle)
    set_run_font(run, size=9.5, color=MUTED)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_paragraph(text, style=f"Heading {level}")
    paragraph_border(paragraph, color=LINE, size=5, space=3)


def add_body(doc: Document, text: str, *, after: float = 4) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    run = paragraph.add_run(text)
    set_run_font(run, size=9.5, color=INK)


def add_bullet(doc: Document, text: str, num_id: int) -> None:
    paragraph = doc.add_paragraph(style="Normal")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.2
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, num])
    p_pr.insert(0, num_pr)
    run = paragraph.add_run(text)
    set_run_font(run, size=9.1, color=INK)


def add_callout(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Mm(2)
    paragraph.paragraph_format.right_indent = Mm(2)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph_shading(paragraph, CALLOUT_FILL)
    paragraph_border(paragraph, color=CALLOUT_BORDER, size=7, space=7, edge="top")
    paragraph_border(paragraph, color=CALLOUT_BORDER, size=7, space=7, edge="bottom")
    run = paragraph.add_run(text)
    set_run_font(run, size=9.1, color=INK)


def add_table(
    doc: Document,
    rows: list[list[str]],
    widths: list[int],
    *,
    highlight_rows: tuple[int, ...] = (),
    font_size: float = 8.5,
) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    set_table_geometry(table, widths)
    set_table_borders(table)
    repeat_header(table.rows[0])
    for row_index, values in enumerate(rows):
        for col_index, value in enumerate(values):
            cell = table.cell(row_index, col_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index == 0:
                set_cell_shading(cell, HEADER_FILL)
            elif row_index in highlight_rows:
                set_cell_shading(cell, HIGHLIGHT_FILL)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.1
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(value))
            set_run_font(run, size=font_size, bold=row_index == 0, color=INK)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.line_spacing = 0.2


def add_metric_strip(doc: Document, items: list[tuple[str, str, str]]) -> None:
    widths = [2409, 2409, 2410, 2410]
    table = doc.add_table(rows=1, cols=4)
    set_table_geometry(table, widths, indent=0)
    set_table_borders(table, color=LINE, size=5)
    for index, (value, label, accent) in enumerate(items):
        cell = table.cell(0, index)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, PANEL_FILL)
        set_cell_margins(cell, top=70, start=80, bottom=65, end=80)
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        top = OxmlElement("w:top")
        top.set(qn("w:val"), "single")
        top.set(qn("w:sz"), "22")
        top.set(qn("w:color"), accent)
        borders.append(top)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(1)
        value_run = paragraph.add_run(value)
        set_run_font(value_run, size=13.5, bold=True, color=INK)
        label_p = cell.add_paragraph()
        label_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label_p.paragraph_format.space_after = Pt(0)
        label_run = label_p.add_run(label)
        set_run_font(label_run, size=7.5, color=MUTED)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_figure(doc: Document, path: Path, width_mm: float, caption: str, alt_text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run()
    shape = run.add_picture(str(path), width=Mm(width_mm))
    doc_pr = shape._inline.docPr
    doc_pr.set("descr", alt_text)
    caption_p = doc.add_paragraph(caption, style="Caption")
    caption_p.paragraph_format.keep_with_next = False


def add_page_break(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.add_run().add_break()
    paragraph.runs[0]._r.br_lst[0].set(qn("w:type"), "page")


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    configure_footer(doc)
    bullet_num_id = create_bullet_numbering(doc)

    core = doc.core_properties
    core.title = "2026.8.28阶段性实验汇报"
    core.subject = "Drone-YOLO复现、小目标切片与尺度感知融合实验"
    core.author = "AllenX"
    core.keywords = "无人机小目标检测; 轻量化; Drone-YOLO; SAHI; Scale-48"

    # Page 1
    add_title(doc, "无人机小目标检测与轻量化阶段性汇报", "更新日期：2026年8月28日  研究对象：VisDrone2019  基线：YOLOv11s")
    add_heading(doc, "1. 阶段完成内容")
    add_body(doc, "已完成Drone-YOLO四阶段复现、P2/P5检测尺度消融、自适应加权MF-FPN多随机种子复验、NWD回归损失筛选、SAHI切片推理、Scale-48尺度感知融合及真实联合推理验证。当前实验链条已从结构复现推进到小目标精度与推理效率的联合优化。")
    add_heading(doc, "2. 基础复现结果")
    add_table(
        doc,
        [
            ["模型", "mAP50", "mAP50-95", "Small AP", "参数量", "GFLOPs"],
            ["YOLOv11s", "39.64%", "23.23%", "13.52%", "9.432M", "21.56"],
            ["MF-FPN", "41.08%", "24.27%", "15.63%", "3.169M", "24.52"],
            ["MF-FPN+LSCD", "40.24%", "23.66%", "15.27%", "2.772M", "19.19"],
            ["完整模型", "41.40%", "24.34%", "15.62%", "2.772M", "19.19"],
        ],
        [2200, 1380, 1620, 1500, 1500, 1438],
        highlight_rows=(4,),
    )
    add_callout(doc, "完整模型相对YOLOv11s的mAP50提高1.76个百分点、Small AP提高2.09个百分点，参数量减少约70.6%。结果支持MF-FPN与Inner-WIoU的有效性；LSCD显著降低参数量和计算量，但其独立精度增益尚未复现。")
    add_heading(doc, "3. 已完成的候选方法筛选")
    add_table(
        doc,
        [
            ["候选方向", "关键结果", "判定"],
            ["加权MF-FPN", "3个种子平均ΔSmall AP=-0.15 pp；仅1/3为正", "停止晋级"],
            ["NWD回归损失", "ΔSmall AP=-0.99 pp；ΔmAP50-95=-0.47 pp", "停止晋级"],
            ["Inner-WIoU+NWD", "ΔSmall AP=-0.31 pp；ΔmAP50-95=-0.21 pp", "停止晋级"],
        ],
        [2350, 5100, 2188],
    )
    add_bullet(doc, "P2检测层是小目标收益的主要来源，但需要持续关注尺度间能力分配。", bullet_num_id)
    add_bullet(doc, "加权融合的小幅单次收益未能跨随机种子稳定复现，不能写成有效贡献。", bullet_num_id)
    add_bullet(doc, "NWD作为框回归损失未改善小目标，后续不再增加训练轮数。", bullet_num_id)

    # Page 2
    add_title(doc, "新增实验：切片推理与尺度感知融合", "目的：验证整图缩放造成的小目标像素损失，并恢复切片对中、大目标上下文的破坏。", page_break_before=True)
    add_heading(doc, "1. SAHI切片推理")
    add_table(
        doc,
        [
            ["方法", "AP50-95", "Small AP", "Medium AP", "Large AP", "延迟"],
            ["Whole-640", "23.51%", "14.75%", "34.10%", "45.29%", "26.3 ms"],
            ["SAHI-640", "23.32%", "18.20%", "29.72%", "25.35%", "57.5 ms"],
            ["SAHI-960", "23.76%", "18.66%", "30.14%", "27.02%", "33.0 ms"],
        ],
        [2100, 1450, 1500, 1650, 1550, 1388],
        highlight_rows=(3,),
    )
    add_callout(doc, "SAHI-960使Small AP提高3.91个百分点，证明整图缩放确实是当前小目标瓶颈之一；但Large AP下降18.28个百分点，说明纯切片缺失完整上下文，不能直接作为最终方案。")
    add_heading(doc, "2. Scale-48尺度感知融合")
    add_body(doc, "融合规则：保留全部Whole-640预测，仅补充SAHI-960中预测面积小于48²像素的框，再执行类别感知NMS。48像素是预测尺度容差，并非新的网络层或输入尺度。", after=3)
    add_table(
        doc,
        [
            ["方法", "AP50-95", "Small AP", "Small AR", "Medium AP", "Large AP"],
            ["Whole-640", "23.51%", "14.75%", "24.46%", "34.10%", "45.29%"],
            ["SAHI-960", "23.76%", "18.66%", "31.64%", "30.14%", "27.02%"],
            ["Union-All", "26.53%", "19.53%", "33.28%", "34.70%", "42.18%"],
            ["Scale-32", "26.34%", "19.24%", "33.25%", "34.28%", "44.09%"],
            ["Scale-48", "26.82%", "19.55%", "33.35%", "34.81%", "44.09%"],
        ],
        [2100, 1450, 1500, 1500, 1550, 1538],
        highlight_rows=(5,),
    )
    add_figure(
        doc,
        ROOT / "reports" / "sahi_scale_fusion" / "figures" / "fusion_size_metrics.png",
        100,
        "图1  整图、纯切片与尺度感知融合的分尺寸AP对比。",
        "整图、SAHI切片和Scale-48融合在Small、Medium、Large AP上的柱状图对比",
    )

    # Page 3
    add_title(doc, "真实联合推理与下一阶段计划", "548张验证图像、同一模型进程、单次图像解码、FP32、RTX 4080 SUPER。", page_break_before=True)
    add_heading(doc, "1. 真实Scale-48联合推理")
    add_metric_strip(
        doc,
        [("26.82%", "mAP50-95", BLUE), ("19.55%", "Small AP", GREEN), ("35.13 ms", "平均端到端延迟", ORANGE), ("28.47", "FPS", PURPLE)],
    )
    add_table(
        doc,
        [
            ["复验项目", "结果"],
            ["离线融合与真实联合推理AP差异", "0.000个百分点（完全复现）"],
            ["Whole-640 → Scale-48延迟", "26.35 → 35.13 ms/图（1.33×）"],
            ["P95延迟 / 峰值显存", "40.59 ms / 1088 MB"],
            ["完整性", "548张、156799框、零越界、检查点一致"],
        ],
        [4500, 5138],
        highlight_rows=(1,),
    )
    add_figure(
        doc,
        ROOT / "reports" / "scale48_joint_inference" / "figures" / "joint_timing_breakdown.png",
        86,
        "图2  真实Scale-48联合推理平均耗时分解。",
        "Scale-48联合推理的图像解码、Whole-640、SAHI-960和融合步骤耗时柱状图",
    )
    add_heading(doc, "2. 当前可以汇报的核心结论")
    add_bullet(doc, "Drone-YOLO复现实现了小目标精度提升和模型参数量显著下降。", bullet_num_id)
    add_bullet(doc, "加权MF-FPN和NWD回归未通过预注册筛选，形成了可解释的失败证据。", bullet_num_id)
    add_bullet(doc, "SAHI证实输入像素损失是小目标瓶颈；Scale-48通过整图与切片互补获得Small AP +4.80 pp、总体AP +3.32 pp。", bullet_num_id)
    add_bullet(doc, "真实联合程序完整复现离线精度，平均28.47 FPS；当前新增规则不增加模型参数，但仍需降低切片计算。", bullet_num_id)
    add_heading(doc, "3. 本轮新增实验")
    add_callout(doc, "密度自适应切片已按预注册方案完成：512张训练图像无标签校准，三组门控在548张验证图像上真实计时；结果与下一步见第4页。")

    # Page 4
    add_title(doc, "新增实验：密度自适应切片", "训练集512张无标签校准；验证集548张；阈值冻结后不在验证集继续搜索。", page_break_before=True)
    add_heading(doc, "1. Gate-Q30关键结果")
    add_metric_strip(
        doc,
        [("26.49%", "mAP50-95", BLUE), ("19.19%", "Small AP", GREEN), ("24.36 ms", "平均端到端延迟", ORANGE), ("41.05", "FPS", PURPLE)],
    )
    add_table(
        doc,
        [
            ["方法", "AP50-95", "Small AP", "延迟", "较Full降时"],
            ["Whole-640", "23.51%", "14.75%", "26.35 ms", "—"],
            ["Full Scale-48", "26.82%", "19.55%", "35.13 ms", "—"],
            ["Top-1", "26.55%", "19.31%", "31.72 ms", "9.7%"],
            ["Gate-Q20", "26.53%", "19.26%", "25.63 ms", "27.0%"],
            ["Gate-Q30", "26.49%", "19.19%", "24.36 ms", "30.7%"],
        ],
        [2300, 1650, 1750, 1900, 2038],
        highlight_rows=(5,),
    )
    add_figure(
        doc,
        ROOT / "reports" / "density_gated_sahi" / "figures" / "gate_accuracy_latency.png",
        78,
        "图3  密度门控的Small AP—端到端延迟权衡。",
        "Whole-640、Full Scale-48和三种密度门控策略的小目标精度与延迟散点图",
    )
    add_heading(doc, "2. 预注册判定与解释")
    add_bullet(doc, "Gate-Q30将平均选择切片数降至0.84，切片计算减少56.0%，延迟降低30.7%。", bullet_num_id)
    add_bullet(doc, "Gate-Q30的Small AP下降0.36 pp，达到保持条件；总体AP下降0.34 pp，超过0.20 pp上限。", bullet_num_id)
    add_bullet(doc, "三组门控均未同时满足全部条件，因此停止在验证集上继续调阈值；Full Scale-48保留为精度方案。", bullet_num_id)
    add_heading(doc, "3. 下一步：转向训练阶段优化")
    add_callout(doc, "优先验证RFLA式高斯感受野标签分配或小目标区域采样：先做独立单变量消融，再与Full Scale-48组合，判断能否提升Small AP且不增加推理开销。")

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
