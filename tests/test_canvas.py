import numpy as np
from handtracking.gestures import AirCanvas

def test_air_canvas_records_strokes_and_clear():
    canvas=AirCanvas(color=(1,2,3), thickness=4)
    canvas.update((.1,.1)); canvas.update((.2,.2)); canvas.end_stroke()
    assert len(canvas.strokes)==1 and len(canvas.strokes[0].points)==2
    canvas.set_color((4,5,6)); canvas.clear(); assert canvas.strokes==[]
    assert canvas.render(np.zeros((20,20,3),dtype=np.uint8)) is not None
