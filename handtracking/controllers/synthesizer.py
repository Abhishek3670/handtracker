"""Native OS key event synthesizer for multimedia controls."""
from __future__ import annotations
import sys
from typing import Callable

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

ACTION_KEY_MAP: dict[str, int] = {
    "volume_up": VK_VOLUME_UP,
    "volume_down": VK_VOLUME_DOWN,
    "mute": VK_VOLUME_MUTE,
    "play_pause": VK_MEDIA_PLAY_PAUSE,
    "next_track": VK_MEDIA_NEXT_TRACK,
    "prev_track": VK_MEDIA_PREV_TRACK,
    "stop": VK_MEDIA_STOP,
}


class KeySynthesizer:
    """Dispatches media key strokes to the operating system or records them in dry-run mode."""

    def __init__(self, dry_run: bool = False, custom_handler: Callable[[str], None] | None = None):
        self.dry_run = dry_run
        self.custom_handler = custom_handler
        self.history: list[str] = []
        self.key_history: list[int] = []

    def send_action(self, action: str) -> bool:
        """Execute a named action (e.g. 'volume_up', 'play_pause')."""
        norm_action = str(action).lower().strip().replace(" ", "_").replace("-", "_")
        self.history.append(norm_action)

        if self.custom_handler is not None:
            self.custom_handler(norm_action)
            return True

        vk_code = ACTION_KEY_MAP.get(norm_action)
        if vk_code is not None:
            return self.send_key(vk_code)
        return False

    def send_key(self, vk_code: int) -> bool:
        """Send a virtual key press and release event."""
        self.key_history.append(vk_code)
        if self.dry_run:
            return True

        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                KEYEVENTF_EXTENDEDKEY = 0x0001
                KEYEVENTF_KEYUP = 0x0002
                user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
                user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
                return True
            except Exception:
                return False
        return True

    def clear_history(self) -> None:
        self.history.clear()
        self.key_history.clear()
