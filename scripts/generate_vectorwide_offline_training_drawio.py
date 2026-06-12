from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "vectorwide_training_flow"
DRAWIO_OUT = OUT_DIR / "vectorwide_offline_training.drawio"


class Drawio:
    def __init__(self, page_w: int = 760, page_h: int = 980) -> None:
        self.next_id = 2
        self.page_w = page_w
        self.page_h = page_h
        self.cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']

    def _id(self) -> str:
        value = f"n{self.next_id}"
        self.next_id += 1
        return value

    @staticmethod
    def label(text: str) -> str:
        return escape(text).replace("\\n", "&lt;br&gt;").replace("\n", "&lt;br&gt;")

    def vertex(self, x: float, y: float, w: float, h: float, value: str, style: str) -> str:
        cid = self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value="{self.label(value)}" style="{escape(style)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
        )
        return cid

    def edge(self, source: str, target: str, value: str = "", dashed: bool = False) -> str:
        cid = self._id()
        dash = "dashed=1;dashPattern=6 6;" if dashed else ""
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;endArrow=block;endFill=1;strokeColor=#777777;strokeWidth=2;{dash}"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{self.label(value)}" style="{escape(style)}" edge="1" parent="1" source="{source}" target="{target}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        return cid

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        fill: str,
        stroke: str,
        size: int = 14,
        bold: bool = False,
        dashed: bool = False,
        rounded: bool = True,
    ) -> str:
        dash = "dashed=1;dashPattern=8 6;" if dashed else ""
        font = "fontStyle=1;" if bold else ""
        round_style = "rounded=1;arcSize=8;" if rounded else "rounded=0;"
        style = (
            f"{round_style}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;"
            f"fontFamily=Times New Roman;fontSize={size};align=center;verticalAlign=middle;spacing=6;{font}{dash}"
        )
        return self.vertex(x, y, w, h, value, style)

    def text(self, x: float, y: float, w: float, h: float, value: str, size: int = 14, bold: bool = False) -> str:
        font = "fontStyle=1;" if bold else ""
        style = (
            "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            f"whiteSpace=wrap;fontFamily=Times New Roman;fontSize={size};{font}"
        )
        return self.vertex(x, y, w, h, value, style)

    def heatmap_stack(self, x: float, y: float, label: str) -> str:
        self.box(x + 18, y - 14, 92, 64, "", "#B8DCEF", "#508AA8", 1, rounded=False)
        self.box(x + 9, y - 7, 92, 64, "", "#8EC8E5", "#508AA8", 1, rounded=False)
        main = self.box(x, y, 92, 64, "", "#2F86BD", "#2B6D99", 1, rounded=False)
        self.vertex(
            x + 19,
            y + 21,
            58,
            24,
            "",
            "shape=mxgraph.basic.arc;html=1;strokeColor=#1F3E58;strokeWidth=2;fillColor=none;endArrow=block;endFill=1;",
        )
        for sx, sy in [(x + 13, y + 48), (x + 56, y + 28), (x + 76, y + 42)]:
            self.text(sx, sy, 14, 14, "*", 16, True)
        self.text(x - 10, y + 72, 115, 30, label, 18, True)
        return main

    def mini_network(self, x: float, y: float) -> None:
        coords = [(x, y), (x, y + 27), (x, y + 54), (x + 48, y + 14), (x + 48, y + 41), (x + 92, y + 27)]
        nodes = []
        for cx, cy in coords:
            nodes.append(
                self.vertex(
                    cx,
                    cy,
                    17,
                    17,
                    "",
                    "ellipse;whiteSpace=wrap;html=1;fillColor=#E8ECFF;strokeColor=#555555;strokeWidth=1.2;",
                )
            )
        style = "edgeStyle=straight;html=1;endArrow=none;strokeColor=#777777;strokeWidth=1;"
        for a in nodes[:3]:
            for b in nodes[3:5]:
                cid = self._id()
                self.cells.append(
                    f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>'
                )
        for a in nodes[3:5]:
            cid = self._id()
            self.cells.append(
                f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1" source="{a}" target="{nodes[5]}"><mxGeometry relative="1" as="geometry"/></mxCell>'
            )

    def cnn_icon(self, x: float, y: float) -> str:
        outer = self.box(x, y, 128, 72, "", "#F8DDDD", "#333333", 1)
        self.box(x + 14, y + 20, 32, 24, "", "#B8E4EB", "#5B8B9A", 1, rounded=False)
        self.box(x + 22, y + 13, 32, 24, "", "#C8D9F7", "#5B6F9A", 1, rounded=False)
        self.vertex(x + 58, y + 29, 22, 16, "", "shape=singleArrow;direction=east;html=1;fillColor=#555555;strokeColor=#555555;")
        self.mini_network(x + 86, y + 10)
        self.text(x + 22, y + 50, 84, 20, "CNN", 14, True)
        return outer

    def transformer_icon(self, x: float, y: float) -> str:
        outer = self.box(x, y, 360, 64, "", "#D9EEF7", "#333333", 1)
        names = ["MHA", "Norm", "FFN", "Norm"]
        fills = ["#E8F2FF", "#FFFFFF", "#FFF5C4", "#FFFFFF"]
        bx = x + 20
        for i, name in enumerate(names):
            self.box(bx + i * 82, y + 15, 66, 28, name, fills[i], "#5B7280", 10)
            if i < 3:
                self.vertex(bx + 66 + i * 82, y + 21, 18, 16, "", "shape=singleArrow;direction=east;html=1;fillColor=#5B7280;strokeColor=#5B7280;")
        self.text(x + 112, y + 44, 140, 18, "Transformer", 14, True)
        return outer

    def xml(self) -> str:
        root = "".join(self.cells)
        model = (
            f'<mxGraphModel dx="900" dy="1100" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.page_w}" pageHeight="{self.page_h}" math="0" shadow="0">'
            f"<root>{root}</root></mxGraphModel>"
        )
        return (
            '<mxfile host="app.diagrams.net" modified="2026-06-12T00:00:00.000Z" agent="Codex" version="30.0.4" type="device">'
            f'<diagram id="vectorwide-offline-training" name="Offline Training">{model}</diagram></mxfile>'
        )


def build() -> Drawio:
    d = Drawio()

    panel = d.box(150, 35, 455, 900, "", "#EEF7EB", "#5DAA45", 18, True, dashed=True)
    d.text(230, 55, 295, 32, "Offline Training", 22, True)

    # Inputs across a sequence window.
    x1 = d.heatmap_stack(210, 775, "X₁")
    xt = d.heatmap_stack(455, 775, "X_T")
    d.text(352, 806, 44, 24, "...", 20, True)

    # CNN spatial encoder.
    cnn1 = d.cnn_icon(195, 635)
    cnnt = d.cnn_icon(440, 635)
    d.text(352, 660, 44, 24, "...", 18, True)
    d.edge(x1, cnn1)
    d.edge(xt, cnnt)

    # Lambda spatial representations.
    lam1 = d.text(225, 565, 90, 30, "λ₁", 24, True)
    lamt = d.text(470, 565, 90, 30, "λ_T", 24, True)
    d.text(352, 568, 44, 24, "...", 18, True)
    d.edge(cnn1, lam1)
    d.edge(cnnt, lamt)

    # Transformer temporal fusion.
    trans = d.transformer_icon(198, 475)
    d.edge(lam1, trans)
    d.edge(lamt, trans)

    # Deep parameters.
    theta1 = d.text(225, 365, 90, 34, "θ₁", 24, True)
    thetat = d.text(470, 365, 90, 34, "θ_T", 24, True)
    d.text(352, 370, 44, 24, "...", 18, True)
    d.edge(trans, theta1)
    d.edge(trans, thetat)

    # Objective and parameter types.
    obj = d.box(
        215,
        180,
        330,
        112,
        "Z\\nargmax  L(Z | Ω, Φ)\\nΩ, Φ",
        "#EEF7EB",
        "#EEF7EB",
        20,
        True,
        rounded=False,
    )
    d.edge(theta1, obj)
    d.edge(thetat, obj)
    d.text(225, 308, 150, 26, "Ω: statistical", 15, True)
    d.text(410, 308, 150, 26, "Φ: deep", 15, True)

    # Side labels, concise.
    temporal_note = d.box(15, 485, 120, 58, "Temporal\\nfeature fusion", "#F8FBFF", "#333333", 13, dashed=True)
    spatial_note = d.box(5, 625, 135, 70, "Spatial encoder\\nCNN blocks", "#FFF2F2", "#333333", 12, dashed=True)
    d.edge(temporal_note, trans, dashed=True)
    d.edge(spatial_note, cnn1, dashed=True)
    return d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagram = build()
    DRAWIO_OUT.write_text(diagram.xml(), encoding="utf-8")
    print(f"Wrote {DRAWIO_OUT}")
    print(f"Editable draw.io cells: {diagram.next_id - 2}")


if __name__ == "__main__":
    main()
