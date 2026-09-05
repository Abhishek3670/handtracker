import pytest
from handtracking.controllers.state_machine import ControllerState, ControllerStateMachine


def test_state_machine_initial_state():
    sm = ControllerStateMachine(wake_gesture="open_palm", wake_duration_s=1.0, idle_timeout_s=4.0)
    assert sm.state == ControllerState.SLEEPING
    assert sm.is_sleeping is True
    assert sm.is_waking is False
    assert sm.is_active is False
    assert sm.hold_progress == 0.0


def test_state_machine_wake_flow():
    sm = ControllerStateMachine(wake_gesture="open_palm", wake_duration_s=1.0, idle_timeout_s=4.0)

    # Irrelevant gesture does not wake
    state = sm.update(["fist"], timestamp=10.0)
    assert state == ControllerState.SLEEPING
    assert sm.hold_progress == 0.0

    # Wake gesture starts WAKING
    state = sm.update(["open_palm"], timestamp=10.0)
    assert state == ControllerState.WAKING
    assert sm.hold_progress == 0.0

    # Halfway through hold duration
    state = sm.update(["open_palm"], timestamp=10.5)
    assert state == ControllerState.WAKING
    assert pytest.approx(sm.hold_progress, 0.01) == 0.5

    # Reached 1.0s continuous hold -> ACTIVE
    state = sm.update(["open_palm"], timestamp=11.0)
    assert state == ControllerState.ACTIVE
    assert sm.is_active is True
    assert sm.hold_progress == 1.0


def test_state_machine_interrupted_wake_reverts_to_sleeping():
    sm = ControllerStateMachine(wake_gesture="open_palm", wake_duration_s=1.0)

    # Start waking
    sm.update(["open_palm"], timestamp=10.0)
    sm.update(["open_palm"], timestamp=10.6)
    assert sm.is_waking is True

    # Released / changed gesture before 1s completes
    sm.update(["fist"], timestamp=10.7)
    assert sm.state == ControllerState.SLEEPING
    assert sm.hold_progress == 0.0


def test_state_machine_idle_timeout_auto_sleep():
    sm = ControllerStateMachine(wake_gesture="open_palm", wake_duration_s=1.0, idle_timeout_s=4.0)

    # Wake up
    sm.update(["open_palm"], timestamp=10.0)
    sm.update(["open_palm"], timestamp=11.0)
    assert sm.is_active is True

    # Check time until sleep
    assert pytest.approx(sm.time_until_sleep(12.0), 0.01) == 3.0

    # Activity resets timer
    sm.record_activity(13.0)
    assert pytest.approx(sm.time_until_sleep(14.0), 0.01) == 3.0

    # 4.1s of inactivity causes transition to SLEEPING
    state = sm.update([], timestamp=17.1)
    assert state == ControllerState.SLEEPING
    assert sm.is_sleeping is True


def test_state_machine_manual_wake_sleep_reset():
    sm = ControllerStateMachine()
    sm.wake(timestamp=5.0)
    assert sm.is_active is True
    assert sm.hold_progress == 1.0

    sm.sleep()
    assert sm.is_sleeping is True
    assert sm.hold_progress == 0.0

    sm.wake(timestamp=10.0)
    sm.reset()
    assert sm.is_sleeping is True
