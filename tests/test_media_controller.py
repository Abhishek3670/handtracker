import pytest
from handtracking.config import MediaConfig
from handtracking.controllers.media import MediaController
from handtracking.controllers.synthesizer import KeySynthesizer, VK_VOLUME_UP, VK_VOLUME_DOWN, VK_MEDIA_PLAY_PAUSE


def test_media_controller_ignores_commands_when_sleeping():
    synthesizer = KeySynthesizer(dry_run=True)
    ctrl = MediaController(synthesizer=synthesizer, initial_volume=50)

    # In sleeping state, volume gestures should not trigger action
    actions = ctrl.process_gestures(static_gestures=["fist"], temporal_gestures=["Circle CW"], timestamp=10.0)
    assert actions == []
    assert ctrl.volume == 50
    assert len(synthesizer.history) == 0


def test_media_controller_executes_actions_when_active():
    synthesizer = KeySynthesizer(dry_run=True)
    cfg = MediaConfig(wake_duration_s=0.5, volume_step=5)
    ctrl = MediaController(config=cfg, synthesizer=synthesizer, initial_volume=50)

    # 1. Wake controller with continuous open_palm
    ctrl.process_gestures(static_gestures=["open_palm"], timestamp=10.0)
    ctrl.process_gestures(static_gestures=["open_palm"], timestamp=10.6)
    assert ctrl.state_machine.is_active is True

    # 2. Trigger Volume Up via Circle CW
    actions = ctrl.process_gestures(temporal_gestures=["Circle CW"], timestamp=11.0)
    assert actions == ["volume_up"]
    assert ctrl.volume == 55
    assert synthesizer.history[-1] == "volume_up"
    assert synthesizer.key_history[-1] == VK_VOLUME_UP
    assert "Volume Up 🔊 55%" in ctrl.get_active_toast(timestamp=11.0)

    # 3. Trigger Volume Down via Circle CCW (after cooldown)
    actions = ctrl.process_gestures(temporal_gestures=["Circle CCW"], timestamp=11.5)
    assert actions == ["volume_down"]
    assert ctrl.volume == 50
    assert synthesizer.history[-1] == "volume_down"
    assert synthesizer.key_history[-1] == VK_VOLUME_DOWN

    # 4. Trigger Play/Pause via Peace Sign
    actions = ctrl.process_gestures(static_gestures=["peace"], timestamp=12.0)
    assert actions == ["play_pause"]
    assert synthesizer.history[-1] == "play_pause"
    assert synthesizer.key_history[-1] == VK_MEDIA_PLAY_PAUSE


def test_media_controller_volume_clamping_and_mute():
    synthesizer = KeySynthesizer(dry_run=True)
    ctrl = MediaController(synthesizer=synthesizer, initial_volume=98)
    ctrl.state_machine.wake(timestamp=1.0)

    # Volume up clamped at 100
    ctrl.process_gestures(temporal_gestures=["Circle CW"], timestamp=1.5)
    assert ctrl.volume == 100
    ctrl.process_gestures(temporal_gestures=["Circle CW"], timestamp=2.0)
    assert ctrl.volume == 100

    # Mute toggle
    ctrl.process_gestures(static_gestures=["fist"], timestamp=2.5)
    assert ctrl.is_muted is True
    assert "Mute 🔇" in ctrl.get_active_toast(timestamp=2.5)

    ctrl.process_gestures(static_gestures=["fist"], timestamp=3.2)
    assert ctrl.is_muted is False


def test_media_controller_cooldown_rate_limits():
    synthesizer = KeySynthesizer(dry_run=True)
    ctrl = MediaController(synthesizer=synthesizer, initial_volume=50)
    ctrl.state_machine.wake(timestamp=10.0)

    # Trigger track skip
    actions1 = ctrl.process_gestures(temporal_gestures=["Swipe Right"], timestamp=10.1)
    assert actions1 == ["next_track"]

    # Immediate second swipe within 0.6s cooldown should be rejected
    actions2 = ctrl.process_gestures(temporal_gestures=["Swipe Right"], timestamp=10.2)
    assert actions2 == []

    # After cooldown elapsed
    actions3 = ctrl.process_gestures(temporal_gestures=["Swipe Right"], timestamp=10.8)
    assert actions3 == ["next_track"]
