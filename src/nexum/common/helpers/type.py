import re
from datetime import datetime

from nexum.common.date import formats

INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

def infer(value: str):
    v = value.strip()
    if not v:
        return None

    lv = v.lower()

    if lv == "true":
        return True
    if lv == "false":
        return False

    if INT_RE.match(v):
        return int(v)

    if FLOAT_RE.match(v):
        try:
            return float(v)
        except ValueError:
            pass

    if "," in v:
        vb = v.replace(".", "").replace(",", ".")
        if FLOAT_RE.match(vb):
            try:
                return float(vb)
            except ValueError:
                pass

    try:
        dt = datetime.fromisoformat(v)
        return dt
    except ValueError:
        pass

    for fmt in formats:
        try:
            dt = datetime.strptime(v, fmt)
            if dt.time() == datetime.min.time():
                return dt.date()
            return dt
        except ValueError:
            pass

    return v
