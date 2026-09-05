import pytest
from pathlib import Path
from handtracking.config import MediaConfig, DEFAULT_ACTIONS


def test_media_config_defaults():
    cfg = MediaConfig()
    assert cfg.wake_gesture == "open_palm"
    assert cfg.wake_duration_s == 1.0
    assert cfg.idle_timeout_s == 4.0
    assert cfg.volume_step == 2
    assert cfg.actions["circle_cw"] == "volume_up"
    assert cfg.actions["circle_ccw"] == "volume_down"
    assert cfg.actions["peace_sign"] == "play_pause"
    assert cfg.actions["swipe_right"] == "next_track"
    assert cfg.actions["swipe_left"] == "prev_track"
    assert cfg.actions["fist"] == "mute"


def test_media_config_normalization():
    cfg = MediaConfig(
        wake_gesture="  Open-Palm  ",
        wake_duration_s=-5.0,
        idle_timeout_s=0.1,
        volume_step=100,
        actions={"Circle-CW": "Volume Up", "Swipe-Right": "Next-Track"},
    )
    cfg.normalize()
    assert cfg.wake_gesture == "open_palm"
    assert cfg.wake_duration_s == 0.1  # min clamped
    assert cfg.idle_timeout_s == 0.5  # min clamped
    assert cfg.volume_step == 50  # max clamped
    assert cfg.actions["circle_cw"] == "volume_up"
    assert cfg.actions["swipe_right"] == "next_track"


def test_media_config_save_and_load_yaml(tmp_path: Path):
    yaml_file = tmp_path / "custom_config.yaml"
    cfg = MediaConfig(wake_gesture="peace", volume_step=5)
    cfg.save(yaml_file)
    assert yaml_file.exists()

    loaded = MediaConfig.load(yaml_file)
    assert loaded.wake_gesture == "peace"
    assert loaded.volume_step == 5
    assert loaded.idle_timeout_s == 4.0


def test_media_config_save_and_load_json(tmp_path: Path):
    json_file = tmp_path / "custom_config.json"
    cfg = MediaConfig(wake_gesture="fist", wake_duration_s=2.0)
    cfg.save(json_file)
    assert json_file.exists()

    loaded = MediaConfig.load(json_file)
    assert loaded.wake_gesture == "fist"
    assert loaded.wake_duration_s == 2.0


def test_media_config_load_nonexistent_returns_defaults():
    cfg = MediaConfig.load("non_existent_file_path_12345.yaml")
    assert cfg.wake_gesture == "open_palm"
    assert cfg.volume_step == 2


def test_media_config_get_action_for_gesture():
    cfg = MediaConfig()
    assert cfg.get_action_for_gesture("circle_cw") == "volume_up"
    assert cfg.get_action_for_gesture("Circle CW") == "volume_up"
    assert cfg.get_action_for_gesture("peace") == "play_pause"
    assert cfg.get_action_for_gesture("Peace Sign") == "play_pause"
    assert cfg.get_action_for_gesture("fist") == "mute"
    assert cfg.get_action_for_gesture("unknown_gesture") is None
    assert cfg.get_action_for_gesture("") is None
