"""Base schema class with deterministic IDs and dict/JSON round-tripping.

Ported from queering-nlp-bias (src/common/base_schema.py), with torch tensor
handling replaced by numpy handling since this repo's schemas never hold tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import dataclass, fields, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np

from src.common.file_io import load_json


def _qfloat(x: float, places: int = 8) -> float:
    """Stable decimal rounding: converts via str -> Decimal -> quantize."""
    if math.isnan(x):
        return 0.0
    if math.isinf(x):
        return 1e10 if x > 0 else -1e10
    q = Decimal(1) / (Decimal(10) ** places)
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    f = float(d)
    return 0.0 if f == 0.0 else f


def _canon(obj: Any, places: int = 8) -> Any:
    """Canonicalize object for deterministic hashing / JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [_canon(v, places) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        obj = float(obj)
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return _qfloat(obj, places)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {
            f.name: _canon(getattr(obj, f.name), places)
            for f in fields(obj)
            if not f.name.startswith("_")
        }
    if isinstance(obj, dict):
        return {
            k: _canon(v, places)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, (list, tuple)):
        return [_canon(v, places) for v in obj]
    return obj


def deterministic_id_from_dataclass(
    data_class_obj: Any, places: int = 8, digest_bytes: int = 16
) -> str:
    """Generate a deterministic ID from a dataclass object."""
    payload = json.dumps(
        _canon(data_class_obj, places),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.blake2b(
        payload.encode("utf-8"), digest_size=digest_bytes
    ).hexdigest()


@dataclass
class BaseSchema:
    """Base class for schema dataclasses: to_dict/from_dict/from_json + stable IDs."""

    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    def to_dict(self) -> dict:
        return _canon(self)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    @classmethod
    def _convert_value(cls, val: Any, field_type: Any) -> Any:
        """Convert a value to the expected field type."""
        origin = get_origin(field_type)
        if origin is Union or isinstance(field_type, types.UnionType):
            args = [a for a in get_args(field_type) if a is not type(None)]
            if len(args) == 1:
                field_type = args[0]

        if val is None:
            return None
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return field_type(val) if not isinstance(val, field_type) else val
        if (
            is_dataclass(field_type)
            and hasattr(field_type, "from_dict")
            and isinstance(val, dict)
        ):
            return field_type.from_dict(val)
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0] if get_args(field_type) else None
            if item_type:
                return [cls._convert_value(item, item_type) for item in val]
        if get_origin(field_type) is dict:
            args = get_args(field_type)
            if args and is_dataclass(args[1]):
                return {k: cls._convert_value(v, args[1]) for k, v in val.items()}
        return val

    @classmethod
    def from_dict(cls, d: dict):
        """Recursively construct a dataclass instance from a nested dict."""
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            val = d[f.name]
            field_type = hints.get(f.name)
            kwargs[f.name] = cls._convert_value(val, field_type) if field_type else val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path):
        """Load from JSON file."""
        return cls.from_dict(load_json(path))
