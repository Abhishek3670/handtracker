"""Configuration loader and schema for touchless media controller."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_ACTIONS: dict[str, str] = {
    "circle_cw": "volume_up",
    "circle_ccw": "volume_down",
    "peace_sign": "play_pause",
    "swipe_right": "next_track",
    "swipe_left": "prev_track",
    "fist": "mute",
}

@dataclass
class MediaConfig:
    wake_gesture: str = "open_palm"
    wake_duration_s: float = 1.0
    idle_timeout_s: float = 4.0
    volume_step: int = 2
    actions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))

    def normalize(self) -> None:
        self.wake_gesture = str(self.wake_gesture).lower().strip().replace(" ", "_").replace("-", "_")
        self.wake_duration_s = max(0.1, float(self.wake_duration_s))
        self.idle_timeout_s = max(0.5, float(self.idle_timeout_s))
        self.volume_step = max(1, min(50, int(self.volume_step)))
        normalized_actions = {}
        for k, v in self.actions.items():
            norm_k = str(k).lower().strip().replace(" ", "_").replace("-", "_")
            norm_v = str(v).lower().strip().replace(" ", "_").replace("-", "_")
            normalized_actions[norm_k] = norm_v
        self.actions = normalized_actions

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaConfig:
        config = cls(
            wake_gesture=data.get("wake_gesture", "open_palm"),
            wake_duration_s=float(data.get("wake_duration_s", 1.0)),
            idle_timeout_s=float(data.get("idle_timeout_s", 4.0)),
            volume_step=int(data.get("volume_step", 2)),
            actions=dict(data.get("actions", DEFAULT_ACTIONS)),
        )
        config.normalize()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None = None) -> MediaConfig:
        if path is None:
            path = "config.yaml"
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        
        content = file_path.read_text(encoding="utf-8")
        data: dict[str, Any] = {}
        if file_path.suffix.lower() in (".yaml", ".yml"):
            if yaml is not None:
                data = yaml.safe_load(content) or {}
            else:
                data = cls._parse_simple_yaml(content)
        elif file_path.suffix.lower() == ".json":
            try:
                data = json.loads(content)
            except Exception:
                data = {}
        else:
            try:
                data = yaml.safe_load(content) if yaml else cls._parse_simple_yaml(content)
            except Exception:
                data = {}
        return cls.from_dict(data if isinstance(data, dict) else {})

    @staticmethod
    def _parse_simple_yaml(content: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        current_dict: dict[str, Any] = result
        current_section: str | None = None
        for line in content.splitlines():
            line = line.rstrip()
            if not line or line.strip().startswith("#"):
                continue
            if line.startswith("  ") and current_section is not None:
                subline = line.strip()
                if ":" in subline:
                    k, v = subline.split(":", 1)
                    k = k.strip().strip("\"'")
                    v = v.strip().strip("\"'")
                    if v.lower() in ("true", "yes"): val: Any = True
                    elif v.lower() in ("false", "no"): val = False
                    elif v.isdigit(): val = int(v)
                    else:
                        try: val = float(v)
                        except ValueError: val = v
                    current_dict[k] = val
            elif ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().strip("\"'")
                v = v.strip()
                if not v or v.startswith("#"):
                    current_section = k
                    result[k] = {}
                    current_dict = result[k]
                else:
                    current_section = None
                    v = v.strip("\"'")
                    if v.lower() in ("true", "yes"): val = True
                    elif v.lower() in ("false", "no"): val = False
                    elif v.isdigit(): val = int(v)
                    else:
                        try: val = float(v)
                        except ValueError: val = v
                    result[k] = val
                    current_dict = result
        return result

    def save(self, path: str | Path = "config.yaml") -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if file_path.suffix.lower() in (".yaml", ".yml") and yaml is not None:
            file_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        else:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def create_default(cls, path: str | Path = "config.yaml") -> MediaConfig:
        cfg = cls()
        cfg.normalize()
        cfg.save(path)
        return cfg

    def get_action_for_gesture(self, gesture_name: str) -> str | None:
        if not gesture_name:
            return None
        norm = str(gesture_name).lower().strip().replace(" ", "_").replace("-", "_")
        if norm in self.actions:
            return self.actions[norm]
        if norm == "peace" and "peace_sign" in self.actions:
            return self.actions["peace_sign"]
        if norm == "peace_sign" and "peace" in self.actions:
            return self.actions["peace"]
        return None
