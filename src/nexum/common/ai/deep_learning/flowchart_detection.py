from typing import Sequence

import numpy as np
from easyocr import Reader

from nexum.common.ai.deep_learning.base_model import BaseModel
from nexum.document.models import (
    FlowchartNode,
    FlowchartEdge,
    FlowchartInput,
    FlowchartGraph,
    FlowchartConfig,
)


class FlowchartExtractor(BaseModel[FlowchartInput, FlowchartGraph | None]):
    def __init__(self, config: FlowchartConfig | None = None) -> None:
        self.config = config or FlowchartConfig()
        self.ocr = Reader(['pt', 'en'], gpu=False)

    def predict(self, input_data: FlowchartInput) -> FlowchartGraph | None:
        image = input_data.image
        boxes = self._run_easyocr(image)
        nodes = self._extract_nodes(boxes)

        if len(nodes) < 2:
            return None

        sorted_nodes = sorted(nodes, key=lambda n: n.center[1])
        edges = self._build_sequential_edges(sorted_nodes)
        mermaid = self._generate_mermaid(sorted_nodes, edges)

        return FlowchartGraph(
            nodes=sorted_nodes,
            edges=edges,
            mermaid=mermaid,
        )

    def _run_easyocr(self, image: np.ndarray):
        return self.ocr.readtext(image)

    def _extract_nodes(self, boxes) -> list[FlowchartNode]:
        nodes = []
        node_id = 1

        for bbox, text, _ in boxes:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            w = x2 - x1
            h = y2 - y1

            nodes.append(
                FlowchartNode(
                    id=f"node_{node_id}",
                    label=text.strip(),
                    shape="rectangle",
                    box=(x1, y1, w, h),
                    center=(x1 + w // 2, y1 + h // 2),
                )
            )
            node_id += 1

        return nodes

    @staticmethod
    def _build_sequential_edges(nodes: Sequence[FlowchartNode]):
        return [
            FlowchartEdge(
                source=nodes[i].id,
                target=nodes[i + 1].id,
                direction="down",
            )
            for i in range(len(nodes) - 1)
        ]

    @staticmethod
    def _generate_mermaid(nodes, edges):
        mermaid = ["graph TD"]
        for node in nodes:
            clean = node.label.replace('"', "'")
            mermaid.append(f'    {node.id}["{clean}"]')

        for edge in edges:
            mermaid.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(mermaid)
