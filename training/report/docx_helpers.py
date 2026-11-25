"""Small helpers to build the report with python-docx (our own code)."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def new_doc():
    d = Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)
    st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(4); st.paragraph_format.line_spacing = 1.1
    for name, size in [("Heading 1", 14), ("Heading 2", 11.5)]:
        h = d.styles[name]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        h.paragraph_format.space_before = Pt(8); h.paragraph_format.space_after = Pt(3)
    return d


def para(d, text, bold_lead=None, style=None, size=None, italic=False, align=None):
    p = d.add_paragraph(style=style)
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
    for i, chunk in enumerate(text.split("**")):   # **bold** inline
        r = p.add_run(chunk); r.bold = (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)
    if align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def bullet(d, text):
    return para(d, text, style="List Bullet")


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hex_fill); tcPr.append(s)


def table(d, header, rows, widths=None, size=8.5, highlight_row=None, caption=None):
    if caption:
        para(d, caption, size=9, italic=True).paragraph_format.keep_with_next = True
    t = d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(size)
        shade(c, "DCE6F1")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
            if highlight_row is not None and ri == highlight_row:
                r.bold = True; shade(cells[i], "FFF2CC")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(d, path, width_cm, caption):
    d.add_picture(path, width=Cm(width_cm)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(d, caption, size=9, italic=True, align="center")
