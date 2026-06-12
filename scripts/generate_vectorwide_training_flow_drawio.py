from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "vectorwide_training_flow"
DRAWIO_OUT = OUT_DIR / "vectorwide_training_flow.drawio"


class Drawio:
    def __init__(self) -> None:
        self.next_id = 2
        self.cells: list[str] = [
            '<mxCell id="0"/>',
            '<mxCell id="1" parent="0"/>',
        ]

    def _id(self) -> str:
        value = f"n{self.next_id}"
        self.next_id += 1
        return value

    @staticmethod
    def _label(label: str) -> str:
        return escape(label).replace("\\n", "&lt;br&gt;").replace("\n", "&lt;br&gt;")

    def vertex(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str = "",
        style: str = "",
        parent: str = "1",
    ) -> str:
        cid = self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value="{self._label(label)}" style="{escape(style)}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
            "</mxCell>"
        )
        return cid

    def edge(
        self,
        source: str,
        target: str,
        label: str = "",
        style: str = "",
        parent: str = "1",
    ) -> str:
        cid = self._id()
        if not style:
            style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#555555;strokeWidth=2;"
        self.cells.append(
            f'<mxCell id="{cid}" value="{self._label(label)}" style="{escape(style)}" edge="1" parent="{parent}" source="{source}" target="{target}">'
            '<mxGeometry relative="1" as="geometry"/>'
            "</mxCell>"
        )
        return cid

    def text(self, x: float, y: float, w: float, h: float, label: str, size: int = 14, bold: bool = False) -> str:
        weight = "fontStyle=1;" if bold else ""
        return self.vertex(
            x,
            y,
            w,
            h,
            label,
            f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Times New Roman;fontSize={size};{weight}",
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        fill: str = "#FFFFFF",
        stroke: str = "#333333",
        size: int = 13,
        bold: bool = False,
        rounded: bool = True,
        dashed: bool = False,
    ) -> str:
        weight = "fontStyle=1;" if bold else ""
        dash = "dashed=1;dashPattern=8 6;" if dashed else ""
        round_flag = "rounded=1;arcSize=8;" if rounded else "rounded=0;"
        return self.vertex(
            x,
            y,
            w,
            h,
            label,
            f"{round_flag}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontFamily=Times New Roman;fontSize={size};align=center;verticalAlign=middle;spacing=6;{weight}{dash}",
        )

    def note(self, x: float, y: float, w: float, h: float, label: str) -> str:
        return self.box(x, y, w, h, label, "#F7F7F7", "#777777", 11, rounded=True, dashed=True)

    def stack_icon(self, x: float, y: float, label: str) -> str:
        self.box(x + 18, y - 12, 90, 58, "", "#B8DCEF", "#508AA8", 1, rounded=False)
        self.box(x + 9, y - 6, 90, 58, "", "#8EC8E5", "#508AA8", 1, rounded=False)
        main = self.box(x, y, 90, 58, "", "#2F86BD", "#2B6D99", 1, rounded=False)
        self.vertex(
            x + 20,
            y + 17,
            54,
            24,
            "",
            "shape=mxgraph.basic.arc;html=1;strokeColor=#1F3E58;strokeWidth=2;fillColor=none;endArrow=block;endFill=1;",
        )
        self.text(x - 5, y + 64, 110, 24, label, 13, True)
        return main

    def mini_network(self, x: float, y: float) -> None:
        coords = [(x, y), (x, y + 26), (x, y + 52), (x + 52, y + 13), (x + 52, y + 39), (x + 100, y + 26)]
        node_ids = []
        for cx, cy in coords:
            node_ids.append(
                self.vertex(
                    cx,
                    cy,
                    16,
                    16,
                    "",
                    "ellipse;whiteSpace=wrap;html=1;fillColor=#E8ECFF;strokeColor=#555555;strokeWidth=1;",
                )
            )
        edge_style = "edgeStyle=straight;html=1;endArrow=none;strokeColor=#777777;strokeWidth=1;"
        for a in node_ids[:3]:
            for b in node_ids[3:5]:
                self.edge(a, b, "", edge_style)
        for a in node_ids[3:5]:
            self.edge(a, node_ids[5], "", edge_style)

    def cnn_motif(self, x: float, y: float, label: str) -> str:
        outer = self.box(x, y, 160, 86, "", "#F8DDDD", "#333333", 1)
        self.box(x + 15, y + 16, 38, 28, "", "#B8E4EB", "#5B8B9A", 1, rounded=False)
        self.box(x + 23, y + 10, 38, 28, "", "#C8D9F7", "#5B6F9A", 1, rounded=False)
        self.vertex(
            x + 68,
            y + 31,
            22,
            18,
            "",
            "shape=singleArrow;direction=east;html=1;fillColor=#555555;strokeColor=#555555;",
        )
        self.mini_network(x + 95, y + 15)
        self.text(x + 14, y + 58, 132, 20, label, 12, True)
        return outer

    def transformer_motif(self, x: float, y: float, w: float, label: str) -> str:
        outer = self.box(x, y, w, 74, "", "#D9EEF7", "#333333", 1)
        names = ["MHA", "Norm", "FFN", "Norm"]
        fills = ["#E8F2FF", "#FFFFFF", "#FFF5C4", "#FFFFFF"]
        bx = x + 18
        for i, name in enumerate(names):
            self.box(bx + i * 92, y + 16, 74, 30, name, fills[i], "#5B7280", 10)
            if i < 3:
                self.vertex(
                    bx + 74 + i * 92,
                    y + 23,
                    20,
                    16,
                    "",
                    "shape=singleArrow;direction=east;html=1;fillColor=#5B7280;strokeColor=#5B7280;",
                )
        self.text(x + w - 208, y + 49, 190, 20, label, 11, True)
        return outer

    def graph_xml(self) -> str:
        root = "".join(self.cells)
        return (
            '<mxGraphModel dx="1500" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1500" pageHeight="900" math="0" shadow="0">'
            f"<root>{root}</root></mxGraphModel>"
        )

    def xml(self) -> str:
        return (
            '<mxfile host="app.diagrams.net" modified="2026-06-12T00:00:00.000Z" agent="Codex" version="30.0.4" type="device">'
            f'<diagram id="vectorwide-training-flow" name="VectorWIDE Training Flow">{self.graph_xml()}</diagram>'
            "</mxfile>"
        )


def build() -> Drawio:
    d = Drawio()
    d.text(25, 20, 1450, 34, "VectorWIDE model structure and training workflow", 20, True)

    # Main containers.
    d.box(35, 80, 865, 760, "", "#EEF7EB", "#5DAA45", 16, True, dashed=True)
    d.box(925, 80, 540, 760, "", "#EAF2FF", "#4078C8", 16, True, dashed=True)
    d.text(310, 95, 315, 28, "Model structure at time t", 16, True)
    d.text(1020, 95, 350, 28, "Training and forecasting procedure", 16, True)

    # Model structure.
    x_stack = d.stack_icon(80, 180, "NWP field X_t\\n6 channels, 40x40")
    station = d.note(82, 300, 140, 64, "station coordinates\\nS = {s_1,...,s_3}")
    target = d.note(82, 390, 145, 76, "target state\\nZ_t = [U_1,U_2,U_3,V_1,V_2,V_3]^T")

    cnn = d.cnn_motif(280, 160, "CNN backbone\\nConv-GN-SiLU-Pool")
    attention = d.box(280, 270, 160, 58, "channel + spatial\\nattention, avg pool", "#F7E8F2", "#8B5A7A", 12)
    trans = d.transformer_motif(500, 150, 330, "causal temporal encoder")
    heads = d.box(
        515,
        260,
        300,
        92,
        "stochastic advection heads\\nmu_t: component advection mean\\nSigma_t: 4x4 covariance,   alpha_t: 2x2 mixing",
        "#FFF5C4",
        "#8A6D1D",
        11,
    )
    anchor = d.note(525, 372, 280, 58, "anchored mode in code:\\nmu_t = A_anchor,t + delta_mu_t")
    kernel = d.box(
        310,
        480,
        505,
        86,
        "Vector Lagrangian IDE kernel\\nprojects stochastic advection moments onto spatial redistribution",
        "#F0EBFF",
        "#4A3A8A",
        13,
        True,
    )
    block = d.box(
        335,
        595,
        455,
        74,
        "time-varying block transition K_t\\nwithin-component and cross-component propagation",
        "#FFFFFF",
        "#4A3A8A",
        12,
    )
    ssm = d.box(
        295,
        705,
        540,
        70,
        "linear-Gaussian state-space layer\\nY_t = K_t Y_{t-1} + eta_t,    Z_t = H_t Y_t + epsilon_t    (H_t = I here)",
        "#FFFFFF",
        "#333333",
        12,
    )

    d.edge(x_stack, cnn)
    d.edge(cnn, attention)
    d.edge(attention, trans)
    d.edge(trans, heads)
    d.edge(heads, kernel)
    d.edge(anchor, heads, "", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#8A6D1D;strokeWidth=2;")
    d.edge(station, kernel)
    d.edge(kernel, block)
    d.edge(block, ssm)
    d.edge(target, ssm, "Kalman likelihood", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#555555;strokeWidth=2;")

    # Training workflow.
    data = d.box(
        970,
        145,
        190,
        80,
        "offline split\\nX, residual Z, NWP baseline\\nA_anchor, V_star, covariance proxies",
        "#FFFFFF",
        "#4078C8",
        11,
    )
    val = d.note(1215, 145, 175, 80, "validation tail\\nselects best offline checkpoint")
    adv = d.box(975, 270, 160, 68, "Stage 1: adv\\ntrain encoder/heads\\nfreeze kernel, Q/R", "#E8F3E1", "#5DAA45", 12, True)
    kf = d.box(1170, 270, 160, 68, "Stage 2: kf\\ntrain kernel + Q/R\\nfreeze network", "#E8F3E1", "#5DAA45", 12, True)
    joint = d.box(1072, 375, 170, 72, "Stage 3: joint\\ntrain all modules\\nforecast-aware loss", "#E8F3E1", "#5DAA45", 12, True)
    ckpt = d.box(1075, 485, 160, 58, "offline checkpoint\\nVectorWIDE theta_off", "#FFFFFF", "#5DAA45", 12, True)
    online = d.box(
        970,
        600,
        190,
        82,
        "online rolling window\\nW = 1008 ten-min steps\\nupdate selected parameters",
        "#FFFFFF",
        "#4078C8",
        11,
    )
    subset = d.note(1215, 590, 190, 105, "trainable subset in code\\nfull_head / output_head / ide_only\\nfast mode: qr, qr-kernel, stat-head")
    forecast = d.box(
        1030,
        735,
        245,
        62,
        "forecast distribution\\nH = 72 lead times in paper setting\\nadd future NWP baseline back to residual forecasts",
        "#FFF5C4",
        "#8A6D1D",
        11,
        True,
    )

    d.edge(data, adv)
    d.edge(adv, kf)
    d.edge(kf, joint)
    d.edge(joint, ckpt)
    d.edge(val, ckpt, "checkpoint metric", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#777777;strokeWidth=2;exitX=0.5;exitY=1;entryX=1;entryY=0.5;")
    d.edge(ckpt, online)
    d.edge(subset, online, "", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#4078C8;strokeWidth=2;")
    d.edge(online, forecast)
    d.edge(forecast, online, "advance by stride", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#4078C8;strokeWidth=2;")

    bridge = d.note(850, 355, 92, 78, "same model\\nreused online")
    d.edge(ssm, bridge, "", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#555555;strokeWidth=2;")
    d.edge(bridge, ckpt, "", "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;dashed=1;strokeColor=#555555;strokeWidth=2;")

    return d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = build()
    DRAWIO_OUT.write_text(d.xml(), encoding="utf-8")
    print(f"Wrote {DRAWIO_OUT}")
    print(f"Editable draw.io cells: {d.next_id - 2}")


if __name__ == "__main__":
    main()
