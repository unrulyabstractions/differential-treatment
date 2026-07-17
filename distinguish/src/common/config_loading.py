"""Pipeline-config loading: defaults patched by JSON files.

configs/config.json holds the complete defaults (documentation by example).
Any --config file is a partial patch of the PipelineConfig shape: top-level
keys, with nested objects patching nested section configs field-wise. Keys
starting with "_" are documentation and ignored; unknown keys fail loudly.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from src.common.base_schema import BaseSchema
from src.common.file_io import load_json
from src.common.run_config import PipelineConfig


def load_pipeline_config(paths: list[Path]) -> PipelineConfig:
    """Defaults patched by each config file in order."""
    config = PipelineConfig()
    for path in paths:
        patch = load_json(Path(path))
        if not isinstance(patch, dict):
            raise ValueError(f"Config file must hold a JSON object: {path}")
        _patch_schema(config, patch, source=Path(path))
    return config


def _patch_schema(schema: BaseSchema, patch: dict, source: Path) -> None:
    """Field-wise update; nested dicts patch nested schemas recursively."""
    valid = {f.name for f in fields(schema)}
    for key, value in patch.items():
        if key.startswith("_"):
            continue
        if key not in valid:
            raise ValueError(
                f"Unknown config key '{key}' for {type(schema).__name__} in {source}"
            )
        current = getattr(schema, key)
        if is_dataclass(current) and isinstance(value, dict):
            _patch_schema(current, value, source)
        else:
            hint = get_type_hints(type(schema))[key]
            setattr(schema, key, type(schema)._convert_value(value, hint))
