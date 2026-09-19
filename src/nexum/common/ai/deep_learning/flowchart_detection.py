from typing import Sequence

import cv2
import numpy as np
import pytesseract

from nexum.common.ai.deep_learning.base_model import BaseModel
from nexum.document.models import FlowchartNode, FlowchartEdge, FlowchartInput, FlowchartGraph, FlowchartConfig


class FlowchartExtractor(BaseModel[FlowchartInput, FlowchartGraph | None]):
    def __init__(self, config: FlowchartConfig | None = None) -> None:
        self.config = config or FlowchartConfig()

    def predict(self, input_data: FlowchartInput) -> FlowchartGraph | None:
        gray = self._ensure_gray(input_data.image)
        binary = self._binarize(gray)
        contours = self._find_contours(binary)

        nodes = self._extract_nodes(gray, contours, input_data.lang)
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

    @staticmethod
    def _ensure_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] in (3, 4):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            gray,
            self.config.threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
        return binary

    @staticmethod
    def _find_contours(binary: np.ndarray) -> Sequence[np.ndarray]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _extract_nodes(
        self,
        gray: np.ndarray,
        contours: Sequence[np.ndarray],
        lang: str,
    ) -> list[FlowchartNode]:
        nodes: list[FlowchartNode] = []
        node_id = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(
                cnt, self.config.polygon_epsilon_ratio * peri, True
            )
            x, y, w, h = cv2.boundingRect(cnt)

            crop = self._crop_node_region(gray, x, y, w, h)
            text = self._ocr_text(crop, lang)
            shape = self._classify_shape(approx)

            nodes.append(
                FlowchartNode(
                    id=f"node_{node_id}",
                    label=text,
                    shape=shape,
                    box=(x, y, w, h),
                    center=(x + w // 2, y + h // 2),
                )
            )
            node_id += 1

        return nodes

    def _crop_node_region(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        pad = self.config.padding
        return gray[
            max(0, y + pad): y + h - pad,
            max(0, x + pad): x + w - pad,
        ]

    def _ocr_text(self, crop: np.ndarray, lang: str) -> str:
        if crop.size == 0:
            return ""
        return pytesseract.image_to_string(
            crop,
            lang=lang,
            config=self.config.tesseract_config,
        ).strip()

    @staticmethod
    def _classify_shape(approx: np.ndarray) -> str:
        vertices = len(approx)
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            return (
                "diamond"
                if aspect_ratio > 1.2 or aspect_ratio < 0.8
                else "rectangle"
            )
        if vertices > 5:
            return "circle_or_ellipse"
        return "polygon"

    @staticmethod
    def _build_sequential_edges(
        nodes: Sequence[FlowchartNode],
    ) -> list[FlowchartEdge]:
        return [
            FlowchartEdge(
                source=nodes[i].id,
                target=nodes[i + 1].id,
                direction="down",
            )
            for i in range(len(nodes) - 1)
        ]

    @staticmethod
    def _generate_mermaid(
        nodes: Sequence[FlowchartNode],
        edges: Sequence[FlowchartEdge],
    ) -> str:
        mermaid_lines = ["graph TD"]
        for node in nodes:
            clean_text = node.label.replace('"', "'") or node.shape
            if node.shape == "diamond":
                mermaid_lines.append(f'    {node.id}{{"{clean_text}"}}')
            elif node.shape == "circle_or_ellipse":
                mermaid_lines.append(f'    {node.id}(("{clean_text}"))')
            else:
                mermaid_lines.append(f'    {node.id}["{clean_text}"]')

        for edge in edges:
            mermaid_lines.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(mermaid_lines)
