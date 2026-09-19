import io
import logging
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from nexum.common.ai.deep_learning.flowchart_detection import (
    FlowchartExtractor,
    FlowchartGraph,
    FlowchartInput,
)
from nexum.common.errors import NexumRuntimeError
from nexum.common.helpers.image import run_ocr, run_ocr_data
from nexum.common.helpers.table import extract_table_dataframe
from nexum.document.models import ImageOCRConfig, Parsed, ImageContext, CaptionInput
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class ImageOCRParser(BaseParser):
    def __init__(self, config: ImageOCRConfig | None = None):
        resolved_config = config or ImageOCRConfig()
        super().__init__(resolved_config)
        self.config: ImageOCRConfig = resolved_config

    def parse(self, raw: bytes) -> Parsed:
        try:
            ctx = self._create_context(raw)
            tables, rows = self._extract_tables(ctx)
            caption = self._extract_caption(ctx)
            flowcharts = self._extract_flowcharts(ctx)
            blocks = self._extract_residual_blocks(ctx)

            metadata = {
                "source_type": "image",
                "width": ctx.width,
                "height": ctx.height,
                "lang": ctx.lang,
                "psm": ctx.psm,
                "has_tables": len(tables) > 0,
                "tables_count": len(tables),
                "has_flowcharts": len(flowcharts) > 0,
                "flowcharts_count": len(flowcharts),
            }

            return Parsed(
                metadata=metadata,
                caption=caption,
                tables=tables,
                blocks=blocks,
                rows=rows,
                flowcharts=flowcharts,
            )

        except Exception as exc:
            logger.error(f"Image parsing failed: {exc}")
            raise NexumRuntimeError(f"Image parsing failed: {exc}")

    def _create_context(self, raw: bytes) -> ImageContext:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(pil)
        width, height = pil.size
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return ImageContext(
            pil=pil,
            gray=gray,
            text_canvas=gray.copy(),
            width=width,
            height=height,
            total_pixels=width * height,
            lang=getattr(self.config, "lang", "por"),
            psm=getattr(self.config, "psm", 3),
        )

    def _extract_tables(
        self, ctx: ImageContext
    ) -> tuple[list[pd.DataFrame], list[list[Any]]]:
        if not getattr(self.config, "table_as_df", True):
            return [], []

        tables: list[pd.DataFrame] = []
        rows: list[list[Any]] = []

        from nexum.common.ai.deep_learning.table_detection import (
            TableDetectionInput,
            TableDetector,
        )

        detector = getattr(self, "table_detector", None) or TableDetector()
        detected = detector.predict(
            TableDetectionInput(image=ctx.pil, threshold=0.6)
        )

        pad_x = max(10, int(ctx.width * 0.008))
        pad_y = max(10, int(ctx.height * 0.008))

        for tbl in detected:
            xmin = int(tbl.box.xmin)
            ymin = int(tbl.box.ymin)
            xmax = int(tbl.box.xmax)
            ymax = int(tbl.box.ymax)

            c_xmin = max(0, xmin - pad_x)
            c_ymin = max(0, ymin - pad_y)
            c_xmax = min(ctx.width, xmax + pad_x)
            c_ymax = min(ctx.height, ymax + pad_y)

            table_crop = ctx.gray[c_ymin:c_ymax, c_xmin:c_xmax]
            if table_crop.size == 0:
                continue

            cleaned_crop = self._remove_table_grid_lines(table_crop)
            ocr_data = run_ocr_data(cleaned_crop, lang=ctx.lang, psm=6)
            df = extract_table_dataframe(ocr_data)

            if df is not None and not df.empty:
                df = df.map(
                    lambda v: "X"
                    if str(v).strip().upper() in {"XK", "XX", "K", "X.", "+"}
                    else v
                )
                tables.append(df)
                df_flat = df.reset_index()
                rows.extend([df_flat.columns.tolist()] + df_flat.values.tolist())

            mask_pad_right = max(35, int(ctx.width * 0.015))
            cv2.rectangle(
                ctx.text_canvas,
                (max(0, c_xmin - 20), max(0, c_ymin - 20)),
                (
                    min(ctx.width, c_xmax + mask_pad_right),
                    min(ctx.height, c_ymax + 20),
                ),
                255,
                -1,
            )

        return tables, rows

    @staticmethod
    def _remove_table_grid_lines(crop: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(crop, 210, 255, cv2.THRESH_BINARY_INV)
        line_k_w = max(30, int(crop.shape[1] * 0.08))
        line_k_h = max(20, int(crop.shape[0] * 0.08))

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_k_w, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_k_h))

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        grid_mask = cv2.bitwise_or(h_lines, v_lines)

        result = crop.copy()
        result[grid_mask > 0] = 255
        return result

    def _extract_caption(self, ctx: ImageContext) -> str | None:
        if not getattr(self.config, "enable_caption", True):
            return None
        try:
            from nexum.common.ai.deep_learning.image_caption import ImageCaptionModel
            caption_model = ImageCaptionModel()
            return caption_model.predict(CaptionInput(image=ctx.pil))
        except Exception as exc:
            logger.error(f"Failed to generate caption: {exc}")
            return None

    def _extract_flowcharts(self, ctx: ImageContext) -> list[FlowchartGraph]:
        if not getattr(self.config, "enable_flowchart", True):
            return []

        try:

            extractor = getattr(self, "flowchart_extractor", None) or FlowchartExtractor()
            graph = extractor.predict(
                FlowchartInput(image=ctx.text_canvas, lang=ctx.lang)
            )
            if not graph:
                return []

            valid_nodes = self._filter_valid_flowchart_nodes(ctx, graph.nodes)
            if len(valid_nodes) < 2:
                return []

            valid_edges = self._build_flowchart_edges(ctx, valid_nodes)
            rendered_mermaid = self._render_mermaid(valid_nodes, valid_edges)

            resolved_graph = FlowchartGraph(
                nodes=valid_nodes,
                edges=valid_edges,
                mermaid=rendered_mermaid,
            )

            for node in valid_nodes:
                nx, ny, nw, nh = node.box
                m_pad = max(6, int(min(nw, nh) * 0.1))
                cv2.rectangle(
                    ctx.text_canvas,
                    (max(0, nx - m_pad), max(0, ny - m_pad)),
                    (
                        min(ctx.width, nx + nw + m_pad),
                        min(ctx.height, ny + nh + m_pad),
                    ),
                    255,
                    -1,
                )

            return [resolved_graph]

        except Exception as exc:
            logger.error(f"Failed to extract flowchart: {exc}")
            return []

    @staticmethod
    def _filter_valid_flowchart_nodes(ctx: ImageContext, raw_nodes: list[dict[str, Any]]) -> list[
        dict[str, Any]]:
        valid_nodes = []
        for n in raw_nodes:
            bx, by, bw, bh = n["box"]
            node_area = bw * bh
            aspect_ratio = float(bw) / bh if bh > 0 else 0

            is_valid_size = (ctx.total_pixels * 0.003) <= node_area <= (ctx.total_pixels * 0.08)
            is_valid_dimension = bw < (ctx.width * 0.35) and bh < (ctx.height * 0.30)
            is_valid_aspect = 0.35 <= aspect_ratio <= 3.2

            if is_valid_size and is_valid_dimension and is_valid_aspect:
                crop_test = ctx.gray[by: by + bh, bx: bx + bw]
                if crop_test.size > 0:
                    valid_nodes.append(n)
            elif node_area > (ctx.total_pixels * 0.08):
                cv2.rectangle(
                    ctx.text_canvas,
                    (max(0, bx - 10), max(0, by - 10)),
                    (min(ctx.width, bx + bw + 10), min(ctx.height, by + bh + 10)),
                    255,
                    -1,
                )
        return valid_nodes

    @staticmethod
    def _build_flowchart_edges(ctx: ImageContext, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges = []
        decision_node = next((n for n in nodes if n.get("shape") == "diamond"), None)

        if not decision_node:
            sorted_nodes = sorted(nodes, key=lambda n: n["center"][1])
            for i in range(len(sorted_nodes) - 1):
                edges.append({"from": sorted_nodes[i]["id"], "to": sorted_nodes[i + 1]["id"], "direction": "down"})
            return edges

        dec_x, dec_y = decision_node["center"]

        top_nodes = sorted(
            [n for n in nodes if n["center"][1] < dec_y and abs(n["center"][0] - dec_x) < (ctx.width * 0.15)],
            key=lambda n: n["center"][1]
        )
        for i in range(len(top_nodes) - 1):
            edges.append({"from": top_nodes[i]["id"], "to": top_nodes[i + 1]["id"], "direction": "down"})
        if top_nodes:
            edges.append({"from": top_nodes[-1]["id"], "to": decision_node["id"], "direction": "down"})

        left_nodes = sorted([n for n in nodes if n["center"][0] < dec_x - (ctx.width * 0.08)],
                            key=lambda n: n["center"][1])
        right_nodes = sorted([n for n in nodes if n["center"][0] > dec_x + (ctx.width * 0.08)],
                             key=lambda n: n["center"][1])

        for ln in left_nodes:
            edges.append({"from": decision_node["id"], "to": ln["id"], "label": "Não"})

        if right_nodes:
            edges.append({"from": decision_node["id"], "to": right_nodes[0]["id"], "label": "Sim"})
            for i in range(len(right_nodes) - 1):
                edges.append({"from": right_nodes[i]["id"], "to": right_nodes[i + 1]["id"], "direction": "down"})

        return edges

    @staticmethod
    def _render_mermaid(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
        lines = ["graph TD"]
        for n in nodes:
            label = n.get("label", "").replace('"', "'") or n.get("shape", "node")
            shape = n.get("shape")
            if shape == "diamond":
                lines.append(f'    {n["id"]}{{"{label}"}}')
            elif shape == "circle_or_ellipse":
                lines.append(f'    {n["id"]}(("{label}"))')
            else:
                lines.append(f'    {n["id"]}["{label}"]')

        for e in edges:
            if "label" in e:
                lines.append(f'    {e["from"]} -- {e["label"]} --> {e["to"]}')
            else:
                lines.append(f'    {e["from"]} --> {e["to"]}')

        return "\n".join(lines)

    @staticmethod
    def _extract_residual_blocks(ctx: ImageContext) -> list[str]:
        _, thresh_inv = cv2.threshold(ctx.text_canvas, 220, 255, cv2.THRESH_BINARY_INV)

        k_w = max(15, int(ctx.width * 0.02))
        k_h = max(8, int(ctx.height * 0.01))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        dilated = cv2.dilate(thresh_inv, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_block_area = ctx.total_pixels * 0.002
        valid_clusters = [
            cv2.boundingRect(cnt)
            for cnt in contours
            if cv2.contourArea(cnt) >= min_block_area
        ]
        valid_clusters.sort(key=lambda b: (b[1], b[0]))

        blocks = []
        for cx, cy, cw, ch in valid_clusters:
            pad = 12
            bx1 = max(0, cx - pad)
            by1 = max(0, cy - pad)
            bx2 = min(ctx.width, cx + cw + pad)
            by2 = min(ctx.height, cy + ch + pad)

            block_crop = ctx.gray[by1:by2, bx1:bx2]
            raw_text = run_ocr(block_crop, lang=ctx.lang, psm=6)

            for line in raw_text.splitlines():
                cleaned = line.strip()
                if len(cleaned) > 1 and not all(c in ".,-|_~:;[](){}/\\ " for c in "".join(line.split())):
                    blocks.append(cleaned)

        return blocks


if __name__ == '__main__':
    from pathlib import Path
    from pprint import pprint

    image_path = Path(r"D:\docs\complete.png")
    image_bytes = image_path.read_bytes()
    config = ImageOCRConfig(
        lang="por",
        enable_binarization=False,
        enable_denoise=False,
        enable_sharpen=False,
        enable_clahe=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
        enable_osd=False,
        enable_deskew=False,
        psm=3,
        table_as_df=True,
        enable_flowchart=True,
        enable_caption=True,
    )
    parser = ImageOCRParser(config)

    try:
        resultado = parser.parse(image_bytes)
        pprint(resultado.dict())
    except NexumRuntimeError as err:
        print(f"Falha na execução do OCR: {err}")
