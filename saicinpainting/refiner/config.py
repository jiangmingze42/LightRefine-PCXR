from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping in config file: {path}")
    return data


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_by_path(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require_path(config: Mapping[str, Any], dotted_key: str) -> str:
    value = get_by_path(config, dotted_key)
    if value in (None, ""):
        raise ConfigError(f"Missing required config value: {dotted_key}")
    return str(value)


def resolve_path(path: str | None, base_dir: str | Path | None = None) -> str | None:
    if path in (None, ""):
        return None
    p = Path(path).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return str(p)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cur = result
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise ConfigError(f"Override conflicts with a non-mapping key: {key}")
        cur[parts[-1]] = value
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = load_yaml(path)
    return deep_update(config, parse_overrides(overrides))
