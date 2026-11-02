"""Small deterministic IO/exact-arithmetic helpers. No GPU or connector access."""
from __future__ import annotations
import hashlib, json, math, re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


def canonical(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_for(master: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(canonical([master, *parts]).encode()).digest()[:8], 'big') & ((1 << 63) - 1)


def exact(x: Any) -> Fraction:
    """Same exact-number bounds/grammar as the F6 reference. No tolerance repair."""
    if x is None or isinstance(x, bool):
        raise ValueError('not an exact number')
    if isinstance(x, Fraction):
        result = x
    elif isinstance(x, (int, Decimal)):
        if isinstance(x, Decimal) and (not x.is_finite() or abs(x.adjusted()) > 24):
            raise ValueError('decimal bounds')
        result = Fraction(x)
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError('nonfinite')
        result = Fraction(str(x))
    elif isinstance(x, str):
        text = x.strip()
        if len(text) > 64 or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?|[+-]?\d+\s*/\s*[+-]?\d+', text):
            raise ValueError('number grammar')
        result = Fraction(text.replace(' ', ''))
    else:
        raise ValueError('number type')
    if abs(result.numerator) > 10**12 or result.denominator > 10**9:
        raise ValueError('number bounds')
    return result


def enc(x: Fraction | int) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'


def finite(x: Fraction) -> bool:
    d = x.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(canonical(row) + '\n' for row in rows), encoding='utf-8')


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
