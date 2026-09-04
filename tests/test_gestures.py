from handtracking.gestures import (EventState, FingerPoseAnalyzer, FingerState,
                                    FingerStates, GestureDefinition, GestureEventDispatcher,
                                    GestureRecognizer, GestureType)
from handtracking.inference.models import BoundingBox, Handedness, HandLandmarks, Landmark3D


def hand(extended):
    p = [Landmark3D(0, 0, 0)] * 21
    for n, base in enumerate((5, 9, 13, 17)):
        x = (n - 1.5) * .08
        ys = (.15, .35, .55, .75) if n in extended else (.12, .08, .04, .01)
        for i, y in enumerate(ys): p[base + i] = Landmark3D(x, y, 0)
    thumb_ys = (.1, .3, .5, .7) if extended else (.08, .04, .02, .01)
    for i, y in enumerate(thumb_ys): p[1 + i] = Landmark3D(-.2 if extended else -.03, y, 0)
    return HandLandmarks(tuple(p), Handedness("Left", .9), BoundingBox.from_landmarks(p))


def test_finger_pose_analyzer_classifies_extended_and_curled():
    states = FingerPoseAnalyzer().analyze(hand({0, 1, 2, 3}))
    assert states.index == FingerState.EXTENDED
    curled = FingerPoseAnalyzer().analyze(hand(set()))
    assert curled.index == FingerState.CURLED


def test_recognizer_open_palm_and_fist():
    recognizer = GestureRecognizer()
    assert recognizer.recognize(hand({0, 1, 2, 3})).gesture == GestureType.OPEN_PALM
    assert recognizer.recognize(hand(set())).gesture == GestureType.FIST


def test_dispatcher_start_hold_and_end():
    recognizer = GestureRecognizer()
    events = []
    dispatcher = GestureEventDispatcher(hold_threshold=.2)
    dispatcher.on("fist", events.append)
    result = recognizer.recognize(hand(set()))
    dispatcher.update("left", result, 0)
    dispatcher.update("left", result, .25)
    dispatcher.end("left", .3)
    assert [event.state for event in events] == [EventState.START, EventState.HOLD, EventState.END]


class StaticAnalyzer:
    def __init__(self, states): self.states = states
    def analyze(self, _hand): return self.states


def recognize_for(states, pinch=False, vertical=None):
    point = hand(set())
    if pinch:
        points = list(point.landmarks)
        points[4] = points[8]
        point = HandLandmarks(tuple(points), point.handedness, BoundingBox.from_landmarks(points))
    if vertical is not None:
        points = list(point.landmarks)
        points[4] = Landmark3D(points[4].x, vertical, points[4].z)
        point = HandLandmarks(tuple(points), point.handedness, BoundingBox.from_landmarks(points))
    return GestureRecognizer(StaticAnalyzer(states)).recognize(point).gesture


def test_all_standard_gestures_and_custom_registration():
    e, c = FingerState.EXTENDED, FingerState.CURLED
    cases = [
        (GestureType.OPEN_PALM, [e,e,e,e,e]), (GestureType.FIST, [c,c,c,c,c]),
        (GestureType.PINCH, [c,c,c,c,c]), (GestureType.PEACE, [c,e,e,c,c]),
        (GestureType.POINTING, [c,e,c,c,c]), (GestureType.THUMBS_UP, [e,c,c,c,c]),
        (GestureType.THUMBS_DOWN, [e,c,c,c,c]), (GestureType.OK_SIGN, [c,c,e,e,e]),
        (GestureType.ROCK_ON, [c,e,c,c,e]), (GestureType.CALL_ME, [e,c,c,c,e]),
    ]
    for expected, values in cases:
        states = FingerStates(*values)
        vertical = -.1 if expected == GestureType.THUMBS_UP else .1 if expected == GestureType.THUMBS_DOWN else None
        assert recognize_for(states, expected == GestureType.PINCH, vertical) == expected
    recognizer = GestureRecognizer(StaticAnalyzer(FingerStates(c,c,c,c,c)))
    recognizer.register_custom_gesture("secret", GestureDefinition(states={"thumb": c, "index": c}))
    assert recognizer.recognize(hand(set())).gesture == "secret"


def test_dispatcher_debounce_reset_and_multi_hand_isolation():
    result = GestureRecognizer().recognize(hand(set()))
    events = []
    dispatcher = GestureEventDispatcher(debounce_time=.1, hold_threshold=10)
    dispatcher.on_gesture_start(events.append)
    dispatcher.on_gesture_end(events.append)
    dispatcher.update("hand_1", result, 0)
    dispatcher.update("hand_1", GestureRecognizer().recognize(hand({0})), .05)
    assert [e.state for e in events] == [EventState.START]
    dispatcher.update("hand_2", result, .2)
    assert len(events) == 2
    dispatcher.reset(hand_id="hand_1", timestamp=.3)
    assert events[-1].hand_id == "hand_1" and events[-1].state == EventState.END
    dispatcher.reset()
    assert events[-1].hand_id == "hand_2" and events[-1].state == EventState.END
