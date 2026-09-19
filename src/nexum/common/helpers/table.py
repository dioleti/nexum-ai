import csv
from typing import Any, Sequence

import pandas as pd


def extract_table_dataframe(ocr_data: dict[str, list[Any]]) -> pd.DataFrame:
    words: list[dict[str, Any]] = []
    n_elements = len(ocr_data.get("text", []))

    for i in range(n_elements):
        raw_text = str(ocr_data["text"][i]).strip()
        if raw_text:
            words.append(
                {
                    "text": raw_text,
                    "left": int(ocr_data["left"][i]),
                    "top": int(ocr_data["top"][i]),
                    "width": int(ocr_data["width"][i]),
                    "height": int(ocr_data["height"][i]),
                    "center_x": int(ocr_data["left"][i]) + (int(ocr_data["width"][i]) // 2),
                    "center_y": int(ocr_data["top"][i]) + (int(ocr_data["height"][i]) // 2),
                }
            )

    if not words:
        return pd.DataFrame()

    avg_height = sum(w["height"] for w in words) / len(words)
    y_tolerance = max(avg_height * 0.8, 15.0)

    words.sort(key=lambda w: w["center_y"])
    lines: list[list[dict[str, Any]]] = []

    for word in words:
        placed = False
        for line in lines:
            line_avg_y = sum(w["center_y"] for w in line) / len(line)
            if abs(word["center_y"] - line_avg_y) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    lines.sort(key=lambda line: sum(w["center_y"] for w in line) / len(line))

    for line in lines:
        line.sort(key=lambda w: w["left"])

    if len(lines) < 2:
        return pd.DataFrame()

    header_words = lines[0]
    columns: list[dict[str, Any]] = []

    curr_col = None
    for w in header_words:
        if curr_col is None:
            curr_col = {
                "text": w["text"],
                "left": w["left"],
                "right": w["left"] + w["width"],
            }
        else:
            if (w["left"] - curr_col["right"]) < 35:
                curr_col["text"] += f" {w['text']}"
                curr_col["right"] = w["left"] + w["width"]
            else:
                curr_col["center_x"] = (curr_col["left"] + curr_col["right"]) // 2
                columns.append(curr_col)
                curr_col = {
                    "text": w["text"],
                    "left": w["left"],
                    "right": w["left"] + w["width"],
                }
    if curr_col:
        curr_col["center_x"] = (curr_col["left"] + curr_col["right"]) // 2
        columns.append(curr_col)

    if len(columns) < 2:
        return pd.DataFrame()

    row_label_title = columns[0]["text"].rstrip(":")
    data_columns = columns[1:]

    records: list[dict[str, Any]] = []
    for line in lines[1:]:
        row_name_tokens = []
        row_data = {c["text"]: "" for c in data_columns}
        first_data_col_left = data_columns[0]["left"]

        for item in line:
            if item["center_x"] < (first_data_col_left - 20):
                row_name_tokens.append(item["text"])
            else:
                closest_col = min(
                    data_columns,
                    key=lambda c: abs(c["center_x"] - item["center_x"])
                )
                row_data[closest_col["text"]] = item["text"]

        name = " ".join(row_name_tokens).strip()
        if name:
            record = {row_label_title: name}
            record.update(row_data)
            records.append(record)

    df = pd.DataFrame(records)
    if not df.empty and row_label_title in df.columns:
        df.set_index(row_label_title, inplace=True)

    return df


def detect_table_boundaries(
    lines: Sequence[str],
    delimiter: str | None = None,
) -> tuple[int, bool]:
    clean_lines = [line.strip() for line in lines if line.strip()]
    if not clean_lines:
        return 0, False

    target_delimiter = delimiter
    if not target_delimiter:
        try:
            sample_blob = "\n".join(clean_lines[:10])
            target_delimiter = csv.Sniffer().sniff(sample_blob).delimiter
        except csv.Error:
            target_delimiter = ","

    skip_count = 0
    for line in clean_lines:
        if line.count(target_delimiter) < 1:
            skip_count += 1
        else:
            break

    has_header = False
    if skip_count < len(clean_lines):
        try:
            remaining_blob = "\n".join(
                clean_lines[skip_count: skip_count + 10]
            )
            has_header = csv.Sniffer().has_header(remaining_blob)
        except csv.Error:
            has_header = True

    return skip_count, has_header
