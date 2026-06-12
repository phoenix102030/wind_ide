from __future__ import annotations

import math
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "vectorwide_training_flow"
TEMPLATE = (
    Path("/Users/felix/.cache/codex-runtimes/codex-primary-runtime/dependencies/python")
    / "lib/python3.12/site-packages/vsdx/media/media.vsdx"
)
VSDX_OUT = OUT_DIR / "vectorwide_training_flow.vsdx"
BACKUP_OUT = OUT_DIR / "vectorwide_training_flow_template_backup.vsdx"
SVG_OUT = OUT_DIR / "vectorwide_training_flow_preview.svg"
PNG_OUT = OUT_DIR / "vectorwide_training_flow_preview.png"

PAGE_W = 14.6
PAGE_H = 7.6
SCALE = 100.0


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def cell(name: str, value: str | float | int, formula: str | None = None) -> str:
    f = f" F='{escape(formula)}'" if formula else ""
    return f"<Cell N='{name}' V='{escape(str(value))}'{f}/>"


def rect_geometry() -> str:
    return (
        "<Section N='Geometry' IX='0'>"
        "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/><Cell N='NoShow' V='0'/>"
        "<Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>"
        "<Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>"
        "<Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
        "</Section>"
    )


class VsdxBuilder:
    def __init__(self) -> None:
        self.next_id = 1
        self.shapes: list[str] = []
        self.svg: list[str] = []

    def _id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    @staticmethod
    def xywh_to_visio(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return x + w / 2.0, PAGE_H - y - h / 2.0, w, h

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        fill: str = "#FFFFFF",
        line: str = "#333333",
        weight: float = 0.012,
        dash: str | None = None,
        radius: float = 0.08,
        font_size: float = 10,
        bold: bool = False,
        font_color: str = "#111111",
        align: int = 1,
    ) -> int:
        sid = self._id()
        pinx, piny, width, height = self.xywh_to_visio(x, y, w, h)
        cells = [
            cell("PinX", pinx),
            cell("PinY", piny),
            cell("Width", width),
            cell("Height", height),
            cell("LocPinX", width / 2, "Width*0.5"),
            cell("LocPinY", height / 2, "Height*0.5"),
            cell("Angle", 0),
            cell("FillForegnd", fill),
            cell("LineColor", line),
            cell("LineWeight", weight),
            cell("Rounding", radius),
            cell("VerticalAlign", 1),
        ]
        if dash:
            cells.append(cell("LinePattern", dash))
        char = (
            "<Section N='Character'>"
            "<Row IX='0'>"
            f"<Cell N='Size' V='{font_size / 72.0}' U='PT'/>"
            f"<Cell N='Color' V='{font_color}'/>"
            f"<Cell N='Style' V='{1 if bold else 0}'/>"
            "</Row></Section>"
        )
        para = (
            "<Section N='Paragraph'><Row IX='0'>"
            f"<Cell N='HorzAlign' V='{align}'/>"
            "</Row></Section>"
        )
        self.shapes.append(
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
            + "".join(cells)
            + rect_geometry()
            + char
            + para
            + f"<Text>{escape(text)}</Text></Shape>"
        )
        svg_dash = " stroke-dasharray='7,5'" if dash else ""
        px = x * SCALE
        py = y * SCALE
        self.svg.append(
            f"<rect x='{px:.1f}' y='{py:.1f}' width='{w*SCALE:.1f}' height='{h*SCALE:.1f}' "
            f"rx='{radius*SCALE:.1f}' fill='{fill}' stroke='{line}' stroke-width='{max(weight*SCALE, 1):.1f}'{svg_dash}/>"
        )
        if text:
            self.add_svg_text(x + w / 2, y + h / 2, text, font_size, font_color, bold, anchor="middle")
        return sid

    def add_svg_text(
        self,
        cx: float,
        cy: float,
        text: str,
        font_size: float,
        color: str,
        bold: bool = False,
        anchor: str = "middle",
    ) -> None:
        lines = text.split("\n")
        line_height = font_size * 1.16
        start = cy * SCALE - (len(lines) - 1) * line_height / 2.0
        weight = "700" if bold else "400"
        for i, line in enumerate(lines):
            self.svg.append(
                f"<text x='{cx*SCALE:.1f}' y='{start + i*line_height:.1f}' "
                f"text-anchor='{anchor}' dominant-baseline='middle' "
                f"font-family='Times New Roman, Times, serif' font-size='{font_size:.1f}' "
                f"font-weight='{weight}' fill='{color}'>{escape(line)}</text>"
            )

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        font_size: float = 11,
        color: str = "#111111",
        bold: bool = False,
        align: int = 1,
    ) -> int:
        return self.add_rect(
            x,
            y,
            w,
            h,
            text=text,
            fill="#FFFFFF",
            line="#FFFFFF",
            weight=0,
            radius=0,
            font_size=font_size,
            bold=bold,
            font_color=color,
            align=align,
        )

    def add_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#555555",
        weight: float = 0.012,
        arrow: bool = True,
        dash: bool = False,
    ) -> int:
        sid = self._id()
        begin_x = x1
        begin_y = PAGE_H - y1
        end_x = x2
        end_y = PAGE_H - y2
        width = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(begin_y - end_y, end_x - begin_x)
        pinx = (begin_x + end_x) / 2.0
        piny = (begin_y + end_y) / 2.0
        cells = [
            cell("PinX", pinx, "(BeginX+EndX)/2"),
            cell("PinY", piny, "(BeginY+EndY)/2"),
            cell("Width", width),
            cell("Height", 0),
            cell("LocPinX", width / 2, "Width*0.5"),
            cell("LocPinY", 0),
            cell("Angle", angle),
            cell("BeginX", begin_x),
            cell("BeginY", begin_y),
            cell("EndX", end_x),
            cell("EndY", end_y),
            cell("LineColor", color),
            cell("LineWeight", weight),
        ]
        if arrow:
            cells.append(cell("EndArrow", 4))
            cells.append(cell("EndArrowSize", 2))
        if dash:
            cells.append(cell("LinePattern", 2))
        geom = (
            "<Section N='Geometry' IX='0'><Cell N='NoFill' V='1'/><Cell N='NoLine' V='0'/>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='2'><Cell N='X' V='{width}'/><Cell N='Y' V='0'/></Row>"
            "</Section>"
        )
        self.shapes.append(
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
            + "".join(cells)
            + geom
            + "</Shape>"
        )
        dash_attr = " stroke-dasharray='7,5'" if dash else ""
        marker = " marker-end='url(#arrow)'" if arrow else ""
        self.svg.append(
            f"<line x1='{x1*SCALE:.1f}' y1='{y1*SCALE:.1f}' x2='{x2*SCALE:.1f}' y2='{y2*SCALE:.1f}' "
            f"stroke='{color}' stroke-width='{max(weight*SCALE, 1):.1f}'{dash_attr}{marker}/>"
        )
        return sid

    def add_heatmap_stack(self, x: float, y: float, label: str) -> None:
        offsets = [(0.12, -0.10), (0.06, -0.05), (0.0, 0.0)]
        fills = ["#93C7E4", "#66B3D6", "#2C7FB8"]
        for (dx, dy), fill in zip(offsets, fills):
            self.add_rect(x + dx, y + dy, 0.78, 0.56, fill=fill, line="#4F88A7", weight=0.006, radius=0.015)
        # Small wind-like strokes and station stars.
        for i in range(3):
            self.add_line(x + 0.18 + 0.18 * i, y + 0.34 - 0.07 * i, x + 0.30 + 0.18 * i, y + 0.28 - 0.07 * i, "#183B59", 0.006, True)
        for sx, sy in [(x + 0.15, y + 0.45), (x + 0.52, y + 0.18), (x + 0.70, y + 0.38)]:
            self.add_text(sx - 0.04, sy - 0.04, 0.08, 0.08, "*", 10, "#D7191C", True)
        self.add_text(x - 0.05, y + 0.62, 0.95, 0.22, label, 13, "#111111", True)

    def add_feature_map_icon(self, x: float, y: float, w: float = 0.58, h: float = 0.38) -> None:
        """Editable stacked feature-map motif adapted from ML-visual template style."""
        for dx, dy, fill in [(0.09, -0.07, "#D8C6F2"), (0.045, -0.035, "#C8D9F7"), (0.0, 0.0, "#B9E5F0")]:
            self.add_rect(x + dx, y + dy, w, h, fill=fill, line="#5E5E6A", weight=0.004, radius=0.012)

    def add_mlp_icon(self, x: float, y: float, scale: float = 1.0) -> None:
        """Small native node-link neural-net icon; no embedded raster."""
        r = 0.055 * scale
        layers = [
            [(x, y + 0.00 * scale), (x, y + 0.22 * scale), (x, y + 0.44 * scale)],
            [(x + 0.32 * scale, y + 0.10 * scale), (x + 0.32 * scale, y + 0.34 * scale)],
            [(x + 0.62 * scale, y + 0.22 * scale)],
        ]
        for left, right in zip(layers, layers[1:]):
            for a in left:
                for c in right:
                    self.add_line(a[0] + r, a[1], c[0] - r, c[1], "#777777", 0.004, arrow=False)
        for layer in layers:
            for cx, cy in layer:
                self.add_rect(cx - r, cy - r, 2 * r, 2 * r, fill="#EEF0FF", line="#4E4E5A", weight=0.004, radius=r)

    def add_cnn_template(self, x: float, y: float, w: float, h: float, title: str = "CNN") -> None:
        self.add_rect(x, y, w, h, fill="#F6DADB", line="#333333", weight=0.01, radius=0.07)
        self.add_feature_map_icon(x + 0.11 * w, y + 0.20 * h, 0.22 * w, 0.32 * h)
        self.add_line(x + 0.42 * w, y + 0.50 * h, x + 0.55 * w, y + 0.50 * h, "#555555", 0.006)
        self.add_mlp_icon(x + 0.61 * w, y + 0.23 * h, scale=0.54 * w)
        self.add_text(x + 0.06 * w, y + 0.70 * h, 0.88 * w, 0.22 * h, title, 8.7, "#111111", True)

    def add_transformer_template(self, x: float, y: float, w: float, h: float, title: str) -> None:
        self.add_rect(x, y, w, h, fill="#CFEAF5", line="#333333", weight=0.01, radius=0.07)
        block_w = w * 0.17
        gap = w * 0.035
        bx = x + w * 0.06
        labels = ["MHA", "Norm", "FFN", "Norm"]
        fills = ["#E6F2FF", "#FFFFFF", "#FFF7C8", "#FFFFFF"]
        for i, (lab, fill) in enumerate(zip(labels, fills)):
            self.add_rect(bx + i * (block_w + gap), y + h * 0.18, block_w, h * 0.42, lab, fill, "#4C6A7A", 0.005, font_size=5.6)
            if i < len(labels) - 1:
                self.add_line(
                    bx + (i + 1) * block_w + i * gap,
                    y + h * 0.39,
                    bx + (i + 1) * (block_w + gap),
                    y + h * 0.39,
                    "#4C6A7A",
                    0.004,
                )
        self.add_text(x + w * 0.58, y + h * 0.68, w * 0.36, h * 0.20, title, 7.9, "#111111", True)

    def page_xml(self) -> str:
        return (
            "<?xml version='1.0' encoding='utf-8'?>"
            "<PageContents xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
            "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' xml:space='preserve'>"
            "<Shapes>"
            + "".join(self.shapes)
            + "</Shapes><Connects/></PageContents>"
        )

    def svg_xml(self) -> str:
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{PAGE_W*SCALE:.0f}' height='{PAGE_H*SCALE:.0f}' "
            f"viewBox='0 0 {PAGE_W*SCALE:.0f} {PAGE_H*SCALE:.0f}'>"
            "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto' markerUnits='strokeWidth'>"
            "<path d='M0,0 L0,6 L9,3 z' fill='#555555'/></marker></defs>"
            "<rect width='100%' height='100%' fill='white'/>"
            + "".join(self.svg)
            + "</svg>"
        )


def draw_figure() -> VsdxBuilder:
    b = VsdxBuilder()
    b.add_text(0.25, 0.08, 14.1, 0.32, "VectorWIDE training workflow", 16, "#1B1B1B", True)

    b.add_rect(0.45, 0.65, 5.25, 6.35, fill="#EAF5E3", line="#5DAA45", weight=0.018, dash="2", radius=0.18)
    b.add_rect(5.95, 0.65, 6.35, 6.35, fill="#EAF2FF", line="#4078C8", weight=0.018, dash="2", radius=0.18)
    b.add_rect(12.55, 1.12, 1.72, 5.38, fill="#FDEBEC", line="#333333", weight=0.012, radius=0.16)
    b.add_text(0.78, 0.75, 4.55, 0.32, "Offline training", 15, "#4B9B3B", True)
    b.add_text(6.38, 0.75, 5.55, 0.32, "Online rolling adaptation and forecasting", 15, "#356DB6", True)
    b.add_text(12.70, 1.25, 1.40, 0.28, "Trainable subset\nin online phase", 8.5, "#111111", True)

    # Offline data columns.
    for x, lab in [(1.00, "X_t"), (2.65, "..."), (4.15, "X_{t+W}")]:
        if lab == "...":
            b.add_text(x, 5.70, 0.4, 0.2, lab, 15, "#111111", True)
        else:
            b.add_heatmap_stack(x, 5.35, lab)
    b.add_text(0.63, 5.98, 1.18, 0.40, "RU-WRF grids\n6 channels", 8.5, "#111111")
    b.add_cnn_template(0.84, 4.55, 1.18, 0.58, "CNN")
    b.add_cnn_template(3.84, 4.55, 1.18, 0.58, "CNN")
    b.add_text(2.65, 4.80, 0.38, 0.20, "...", 13, "#111111", True)
    b.add_line(1.42, 5.26, 1.42, 5.04, "#777777", 0.01)
    b.add_line(4.42, 5.26, 4.42, 5.04, "#777777", 0.01)
    b.add_transformer_template(1.05, 3.88, 3.85, 0.52, "causal temporal fusion")
    b.add_line(1.42, 4.55, 1.42, 4.40, "#777777", 0.01)
    b.add_line(4.42, 4.55, 4.42, 4.40, "#777777", 0.01)
    b.add_rect(1.00, 3.22, 3.95, 0.42, "Heads:  μ_t, Σ_t, α_t  (+ anchored Δμ_t)", "#FFF7C8", "#333333", 0.01, font_size=10)
    b.add_line(2.97, 3.95, 2.97, 3.64, "#777777", 0.01)
    b.add_rect(0.82, 2.42, 4.30, 0.47, "Vector Lagrangian IDE kernel  →  transition M_t", "#F5F2FF", "#4A3A8A", 0.012, font_size=10)
    b.add_line(2.97, 3.22, 2.97, 2.89, "#777777", 0.01)
    b.add_rect(0.80, 1.45, 1.28, 0.50, "Stage 1\nadv", "#FFFFFF", "#5DAA45", 0.012, font_size=9, bold=True)
    b.add_rect(2.33, 1.45, 1.28, 0.50, "Stage 2\nkf", "#FFFFFF", "#5DAA45", 0.012, font_size=9, bold=True)
    b.add_rect(3.86, 1.45, 1.28, 0.50, "Stage 3\njoint", "#FFFFFF", "#5DAA45", 0.012, font_size=9, bold=True)
    b.add_line(2.08, 1.70, 2.33, 1.70, "#5DAA45", 0.01)
    b.add_line(3.61, 1.70, 3.86, 1.70, "#5DAA45", 0.01)
    b.add_text(0.70, 2.02, 4.65, 0.33, "losses:  L_adv  →  L_KF  →  L_KF + λ_adv L_adv + λ_ms L_ms + regularization", 8.7, "#111111")
    b.add_rect(1.65, 0.88, 2.72, 0.35, "validation tail selects offline checkpoint", "#E8F3E1", "#5DAA45", 0.009, font_size=9)

    # Online panel.
    b.add_rect(6.20, 1.12, 1.65, 0.42, "Load offline\ncheckpoint", "#FFFFFF", "#356DB6", 0.012, font_size=9, bold=True)
    b.add_line(5.70, 1.33, 6.20, 1.33, "#555555", 0.012)
    b.add_rect(8.10, 1.12, 1.95, 0.42, "freeze backbone;\nselect adaptation params", "#FFFFFF", "#356DB6", 0.012, font_size=8.8)
    b.add_line(7.85, 1.33, 8.10, 1.33, "#356DB6", 0.012)
    b.add_rect(10.35, 1.12, 1.62, 0.42, "rolling window\nW=1008", "#FFFFFF", "#356DB6", 0.012, font_size=9, bold=True)
    b.add_line(10.05, 1.33, 10.35, 1.33, "#356DB6", 0.012)

    for x, lab in [(6.40, "X_{r-W+1}"), (8.05, "..."), (9.60, "X_r"), (11.00, "X_{r+h}")]:
        if lab == "...":
            b.add_text(x, 5.70, 0.4, 0.2, lab, 15, "#111111", True)
        else:
            b.add_heatmap_stack(x, 5.35, lab)
    for x in [6.70, 9.90, 11.30]:
        b.add_cnn_template(x - 0.56, 4.55, 1.18, 0.58, "CNN")
        b.add_line(x, 5.26, x, 5.04, "#777777", 0.01)
        b.add_line(x, 4.55, x, 4.40, "#777777", 0.01)
    b.add_transformer_template(6.25, 3.88, 3.95, 0.52, "reused causal Transformer")
    b.add_rect(10.65, 3.88, 1.35, 0.52, "future\nNWP", "#CFEAF5", "#333333", 0.01, font_size=9)
    b.add_line(8.22, 3.88, 8.22, 3.55, "#777777", 0.01)
    b.add_line(11.33, 3.88, 11.33, 3.55, "#777777", 0.01)
    b.add_rect(6.15, 3.08, 4.15, 0.42, "online loss on recent observations: L_KF + λ_anchor ||θ-θ_off||²", "#FFF7C8", "#333333", 0.01, font_size=9)
    b.add_rect(10.62, 3.08, 1.45, 0.42, "θ_{r+1:r+h}", "#FFF7C8", "#333333", 0.01, font_size=11, bold=True)
    b.add_line(8.22, 3.50, 8.22, 3.08, "#777777", 0.01)
    b.add_line(11.33, 3.50, 11.33, 3.08, "#777777", 0.01)
    b.add_rect(6.15, 2.32, 2.05, 0.45, "Kalman filter\non window", "#F5F2FF", "#4A3A8A", 0.012, font_size=9, bold=True)
    b.add_rect(8.65, 2.32, 1.70, 0.45, "update\nselected params", "#F5F2FF", "#4A3A8A", 0.012, font_size=9, bold=True)
    b.add_rect(10.65, 2.32, 1.35, 0.45, "forecast\nH=72", "#F5F2FF", "#4A3A8A", 0.012, font_size=9, bold=True)
    b.add_line(8.20, 2.55, 8.65, 2.55, "#4A3A8A", 0.012)
    b.add_line(10.35, 2.55, 10.65, 2.55, "#4A3A8A", 0.012)
    b.add_line(10.70, 2.32, 9.10, 2.32, "#4A3A8A", 0.01, arrow=True, dash=True)
    b.add_text(9.85, 2.05, 0.90, 0.20, "next roll", 8, "#4A3A8A")
    b.add_rect(8.02, 6.38, 2.28, 0.38, "ŷ_{r+1}, ..., ŷ_{r+72}", "#FFFFFF", "#356DB6", 0.012, font_size=12, bold=True)
    b.add_line(11.32, 2.32, 9.20, 6.38, "#356DB6", 0.012)
    b.add_text(10.05, 6.80, 1.45, 0.18, "measurement space:  b_{t+h} + ẑ_{t+h|t}", 8, "#111111")

    # Right trainable-subset box.
    items = [
        ("Q, R covariance\nparameters", "#FFF7C8"),
        ("output/statistical\nheads", "#DFF0FF"),
        ("optional kernel ell\nor gamma", "#F3EAFD"),
        ("CNN/Transformer\nreused", "#E8F3E1"),
    ]
    y = 1.78
    for label, fill in items:
        b.add_rect(12.78, y, 1.28, 0.50, label, fill, "#333333", 0.009, font_size=8.2)
        y += 0.66
    b.add_text(12.70, 4.65, 1.42, 0.92, "code scopes:\nfull_head / output_head / ide_only\nfast: qr / qr-kernel / stat-head", 7.4, "#111111")

    # Side annotations.
    b.add_rect(0.10, 3.86, 0.72, 0.50, "spatial\nencoder", "#F8FBFF", "#333333", 0.008, dash="2", font_size=7.5)
    b.add_line(0.82, 4.12, 1.05, 4.15, "#555555", 0.009)
    b.add_rect(0.10, 3.12, 0.72, 0.50, "temporal\nfusion", "#F8FBFF", "#333333", 0.008, dash="2", font_size=7.5)
    b.add_line(0.82, 3.37, 1.05, 4.15, "#555555", 0.009)
    b.add_rect(5.83, 0.95, 0.02, 5.72, fill="#777777", line="#777777", weight=0.004, radius=0)
    b.add_text(5.60, 3.55, 0.25, 0.35, "offline\n→", 7.5, "#555555")
    return b


def update_pages_xml() -> str:
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<Pages xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' xml:space='preserve'>"
        "<Page ID='0' NameU='VectorWIDE Training Flow' Name='VectorWIDE Training Flow' "
        f"ViewScale='1' ViewCenterX='{PAGE_W/2}' ViewCenterY='{PAGE_H/2}'>"
        "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
        f"<Cell N='PageWidth' V='{PAGE_W}'/><Cell N='PageHeight' V='{PAGE_H}'/>"
        "<Cell N='ShdwOffsetX' V='0.1181102362204724'/><Cell N='ShdwOffsetY' V='-0.1181102362204724'/>"
        "<Cell N='PageScale' V='0.03937007874015748' U='MM'/>"
        "<Cell N='DrawingScale' V='0.03937007874015748' U='MM'/>"
        "<Cell N='DrawingSizeType' V='0'/><Cell N='DrawingScaleType' V='0'/>"
        "<Cell N='DrawingResizeType' V='1'/><Cell N='PageShapeSplit' V='1'/>"
        "</PageSheet><Rel r:id='rId1'/></Page></Pages>"
    )


def render_png_from_builder(builder: VsdxBuilder) -> None:
    width = int(PAGE_W * SCALE)
    height = int(PAGE_H * SCALE)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Times New Roman.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    # The SVG is the authoritative preview; the PNG is a light raster check.
    draw.text((20, 20), "Preview generated; open SVG for full vector layout.", fill=(30, 30, 30), font=font)
    image.save(PNG_OUT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, BACKUP_OUT)
    builder = draw_figure()
    SVG_OUT.write_text(builder.svg_xml(), encoding="utf-8")
    render_png_from_builder(builder)

    replacements = {
        "visio/pages/page1.xml": builder.page_xml().encode("utf-8"),
        "visio/pages/pages.xml": update_pages_xml().encode("utf-8"),
    }
    with zipfile.ZipFile(TEMPLATE, "r") as zin, zipfile.ZipFile(VSDX_OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = replacements.get(item.filename)
            if data is None:
                data = zin.read(item.filename)
            zout.writestr(item, data)
    print(f"Wrote {VSDX_OUT}")
    print(f"Wrote {SVG_OUT}")
    print(f"Wrote {PNG_OUT}")
    print(f"Template backup {BACKUP_OUT}")
    print(f"Native shapes: {builder.next_id - 1}")


if __name__ == "__main__":
    main()
