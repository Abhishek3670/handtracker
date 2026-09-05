# HandTracking 🚀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ModernGL](https://img.shields.io/badge/ModernGL-5.12%2B-orange.svg)](https://github.com/moderngl/moderngl)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-teal.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-92%2F92%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HandTracking** is a high-performance, real-time hand tracking, gesture recognition, and interactive 3D physics engine designed for standard webcam feeds. Built with a zero-lag asynchronous capture architecture, dynamic 1-Euro adaptive jitter filtering, and a hardware-accelerated **ModernGL GPU shader pipeline**, HandTracking delivers photorealistic 3D rendering, intuitive spatial air gestures, an interactive Air Canvas, and a touchless OS media controller with near-zero latency.

---

## 🌟 Key Features

- **⚡ Zero-Lag Asynchronous Capture**: Dedicated multi-threaded background frame grabber drains OpenCV driver queues and eliminates stale frame buildup.
- **🖐️ 21 3D Landmark Estimation**: Full 3D joint landmark detection per hand via MediaPipe with real-time monocular hand depth ($Z$) estimation and palm span tracking.
- **🎯 1-Euro Adaptive Jitter Filter**: Dynamic cutoff low-pass filtering eliminates high-frequency tremor on resting hands while preserving instantaneous response during fast motion.
- **✌️ 10 Static + 7 Temporal Gestures**: Instant rule-based geometric classification (`open_palm`, `fist`, `peace_sign`, `pointing`, `thumbs_up`, `pinch`, etc.) and spatio-temporal trajectory tracking (`swipes`, `circles`, `waves`).
- **🎨 Interactive Air Canvas**: Draw in real-time in 3D air using index-thumb pinch gestures, switch among 4 vibrant color palettes, and clear drawings on demand.
- **🎵 Touchless Media Controller**: Background state machine (`SLEEPING` $\leftrightarrow$ `ACTIVE`) with configurable wake gestures, radial circular volume dials, track skipping, play/pause, and native OS media key synthesis.
- **🏀 3D AR Physics Simulation Engine**: Real-time spring-damper hand collision, palm plane bounce deflection, pinch grab-and-throw across 3D depth, gravity toggling, and angular velocity spin integration.
- **🌌 3D Cyber-Space Environment**: Immersive digital 3D perspective cyber-room with holographic hand skeleton rendering, depth-aware shadows, and room boundary reflections.
- **✨ ModernGL GPU Shader Engine**: Hardware-accelerated OpenGL/GLSL rendering with **3-Point Studio Lighting** (Key, Fill, Rim, Hemispheric ambient), micro-pebble normal bump mapping, recessed rubber seams, and 4 PBR skins (**Basketball**, **Chrome**, **Tennis**, **Neon**).
- **📊 Real-Time HUD & Telemetry**: Millisecond latency breakdown across all pipeline stages (Capture, Inference, Smoothing, Gestures, Physics, GPU Render) with an interactive on-screen cheat sheet card.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Capture["1. Async Capture Engine"]
        Cam[Webcam DirectShow/MSMF] -->|Threaded Grab| RingBuf[Latest Frame Buffer]
    end

    subgraph Inference["2. GPU Landmark & Depth Engine"]
        RingBuf -->|Zero-Copy| MediaPipe[MediaPipe 3D Landmark Detector]
        MediaPipe --> RawPoints[21 Raw 3D Landmarks]
        RawPoints --> DepthEst[Monocular Palm Depth Z Estimation]
    end

    subgraph Signal["3. Signal Processing & Recognition"]
        DepthEst --> OneEuro[Adaptive 1-Euro Filter]
        OneEuro --> SmoothPoints[Smoothed 3D Keypoints]
        SmoothPoints --> GeomGestures[10 Geometric Gesture Classifiers]
        SmoothPoints --> TempGestures[Temporal Trajectory Recognizer]
    end

    subgraph Controllers["4. Interactive Controllers & Physics"]
        GeomGestures --> MediaSM[Media Controller State Machine]
        TempGestures --> MediaSM
        GeomGestures --> AirCanvas[Air Canvas Drawing Engine]
        SmoothPoints --> ARPhysics[3D AR Spring-Damper Physics Engine]
    end

    subgraph Rendering["5. Dual-Engine Presentation"]
        ARPhysics --> ModernGL[ModernGL GPU Shader Pipeline\n(3-Point Lighting + PBR Skins)]
        ARPhysics -.->|CPU Fallback| CVRenderer[OpenCV 2.5D Renderer]
        ModernGL --> Compositor[Alpha Frame Compositor]
        AirCanvas --> Compositor
        Compositor --> TelemetryHUD[Real-Time HUD & Cheat Sheet]
    end
```

---

## 📦 Installation & Setup

### Prerequisites
- **Python**: `3.10`, `3.11`, or `3.12`
- **Webcam**: Any standard USB or built-in webcam
- **GPU**: Optional OpenGL 3.3+ compatible GPU for hardware shader acceleration (graceful CPU fallback included)

### 1. Clone the Repository
```bash
git clone https://github.com/Abhishek3670/handtracker.git
cd handtracker
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt

# Or editable developer installation with all extras
pip install -e ".[all]"
```

---

## 🚀 Quickstart & Usage

### One-Click Launchers (Windows)
```cmd
# Command Prompt
run_demo.bat

# PowerShell
.\run_demo.ps1
```

### Command-Line Execution

```bash
# 1. Standard Live Hand Tracking with HUD
python -m handtracking

# 2. Photorealistic 3D AR Ball with ModernGL GPU Shaders
python -m handtracking --gpu-render --ar-skin basketball

# 3. 3D Digital Cyber-Space Room with AR Physics
python -m handtracking --virtual-room --ar-ball

# 4. Touchless Media Controller Mode
python -m handtracking --media --config config.yaml

# 5. Air Canvas Finger Painting Mode
python -m handtracking --canvas

# 6. Full Interactive Experience (All Features)
python -m handtracking --gpu-render --ar-ball --media --canvas

# 7. Headless Synthetic Performance Benchmark (500 frames)
python -m handtracking --benchmark 500 --headless
```

---

## ⚙️ CLI Options Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--camera` | `int` \| `str` | `0` | Camera device index (e.g. `0`, `1`) or video file path |
| `--width` | `int` | `1280` | Capture frame width in pixels |
| `--height` | `int` | `720` | Capture frame height in pixels |
| `--model-complexity` | `0` \| `1` | `1` | MediaPipe model complexity (`0` = ultra-fast, `1` = full accuracy) |
| `--gpu-render`, `--gpu` | Flag | `False` | Enable ModernGL hardware GPU shader rendering for 3D room & ball |
| `--ar-ball`, `--ar` | Flag | `False` | Enable interactive 3D AR physics ball |
| `--virtual-room`, `-vr` | Flag | `False` | Render 3D digital cyber-space room instead of raw webcam feed |
| `--ar-skin` | `str` | `basketball` | Ball material skin: `basketball`, `chrome`, `tennis`, `neon` |
| `--canvas` | Flag | `False` | Enable touchless Air Canvas drawing mode |
| `--media` | Flag | `False` | Enable touchless OS media & entertainment controller |
| `--config` | `str` | `config.yaml` | Path to media controller configuration YAML/JSON file |
| `--no-smoothing` | Flag | `False` | Disable 1-Euro adaptive temporal landmark smoothing |
| `--no-mirror` | Flag | `False` | Disable horizontal webcam mirror mode |
| `--headless` | Flag | `False` | Run in headless mode without opening a GUI window |
| `--benchmark N` | `int` | `None` | Run synthetic benchmark of `N` frames and print latency profile |

---

## ⌨️ Interactive Keyboard Shortcuts

When running the live GUI window, you can control all features in real-time:

| Hotkey | Action | Subsystem |
|:---:|---|---|
| <kbd>q</kbd> / <kbd>ESC</kbd> | Exit application | System |
| <kbd>h</kbd> | Toggle on-screen **Help & Controls Cheat Sheet** | HUD |
| <kbd>u</kbd> | Toggle **ModernGL GPU Hardware Shaders** (on/off) | GPU Renderer |
| <kbd>v</kbd> | Toggle **3D Cyber-Space Environment** (Webcam $\leftrightarrow$ Cyber-Room) | 3D Room |
| <kbd>b</kbd> | **Reset AR Ball** position to center screen | AR Physics |
| <kbd>s</kbd> | **Cycle Ball Skin** (`Basketball` $\to$ `Chrome` $\to$ `Tennis` $\to$ `Neon`) | Materials |
| <kbd>g</kbd> | Toggle **Gravity** (Zero-G float $\leftrightarrow$ Earth 9.8m/s²) | AR Physics |
| <kbd>w</kbd> | Force Toggle **Media Controller State** (`SLEEPING` $\leftrightarrow$ `ACTIVE`) | Media Controller |
| <kbd>m</kbd> | Toggle **Media Controller HUD overlay** | Media Controller |
| <kbd>c</kbd> | **Clear Air Canvas** drawing layer | Air Canvas |
| <kbd>1</kbd> - <kbd>4</kbd> | Switch Air Canvas Color (<kbd>1</kbd> Green, <kbd>2</kbd> Blue, <kbd>3</kbd> Red, <kbd>4</kbd> Yellow) | Air Canvas |

---

## 🖐️ Gesture Recognition Cheat Sheet

### Static Geometric Gestures (10 Gestures)

| Gesture Name | Hand Pose Description | Primary Action / Integration |
|---|---|---|
| `open_palm` | All 5 fingers extended outward | Wake media controller / Deflect AR ball |
| `closed_fist` | All 5 fingers curled tight into palm | Mute audio / Stop interactions |
| `pointing_up` | Index extended, other fingers folded | Precision cursor tracking |
| `peace_sign` | Index & Middle extended (V-shape) | Play / Pause media toggle |
| `thumbs_up` | Thumb pointing up, fingers folded | Confirmation / Positive action |
| `thumbs_down` | Thumb pointing down, fingers folded | Reject / Negative action |
| `ok_sign` | Thumb & Index tips touching in circle | Selection confirmation |
| `rock_on` | Index & Pinky extended (Horns) | Media shortcut trigger |
| `pinch_index_thumb` | Index tip & Thumb tip within 40px | **Draw on Canvas** / **Grab & Throw AR Ball** |
| `three_fingers` | Index, Middle, & Ring fingers extended | Extended mode toggle |

### Dynamic Temporal Gestures (7 Gestures)

| Gesture Name | Motion Trajectory | Trigger Duration | Default Action |
|---|---|---|---|
| `swipe_right` | Fast horizontal hand swipe to the right | $< 0.5\text{s}$ | Next Track (`>>`) |
| `swipe_left` | Fast horizontal hand swipe to the left | $< 0.5\text{s}$ | Previous Track (`<<`) |
| `swipe_up` | Fast upward vertical hand swipe | $< 0.5\text{s}$ | Volume Up |
| `swipe_down` | Fast downward vertical hand swipe | $< 0.5\text{s}$ | Volume Down |
| `circle_cw` | Clockwise continuous fingertip circle | Continuous | Smooth Radial Volume Up |
| `circle_ccw` | Counter-clockwise continuous fingertip circle | Continuous | Smooth Radial Volume Down |
| `wave` | Rapid alternating left-right oscillation | $> 2\text{ cycles}$ | Wake / Greeting Event |

---

## 🎵 Touchless Media Controller

HandTracking includes a touchless media controller configured via [`config.yaml`](file:///W:/Aatish/Stuff/HandTracking/config.yaml):

```yaml
wake_gesture: open_palm       # Hold gesture to wake controller
wake_duration_s: 1.0          # Seconds to hold wake gesture
idle_timeout_s: 4.0           # Seconds of inactivity before sleeping
volume_step: 2                # Volume delta per radial circle interval
actions:
  circle_cw: volume_up        # Continuous clockwise circle -> increase volume
  circle_ccw: volume_down     # Continuous counter-clockwise circle -> decrease volume
  peace_sign: play_pause      # Peace sign -> toggle play/pause
  swipe_right: next_track     # Swipe right -> next song
  swipe_left: prev_track      # Swipe left -> previous song
  fist: mute                  # Closed fist -> mute
```

### State Machine Lifecycle
1. **`SLEEPING`**: Gestures are ignored, preventing unintended triggers during normal conversation. Hold `open_palm` for 1.0s to wake.
2. **`ACTIVE`**: Visual neon badge appears on HUD. Gestures synthesize native OS media keys (`VK_VOLUME_UP`, `VK_MEDIA_PLAY_PAUSE`, etc.). Automatically returns to `SLEEPING` after 4.0s of idle time.

---

## 🏀 ModernGL GPU Shader Engine & Materials

The GPU rendering pipeline uses offscreen Framebuffer Objects (FBO) with depth testing and Blinn-Phong GLSL shaders:

### 1. 3-Point Cinematic Studio Lighting
- **Key Light**: Warm top-right directional light ($I = 1.45$, `#FFF7EB`) providing primary geometric shape.
- **Fill Light**: Soft cool cyan-blue front-left light ($I = 0.85$, `#BFE0FF`) illuminating dark side shadows.
- **Rim / Ground Bounce**: Purple bounce light ($I = 0.55$, `#D999FF`) creating edge contrast against the cyber-room floor.
- **Hemispheric Ambient**: Sky-to-ground vertical ambient gradient ensuring smooth curved contours.

### 2. Procedural PBR Material Skins
- **🏀 Basketball**: Wilson Evolution / NBA authentic leather with micro-pebble normal bump mapping, recessed black rubber channels, beveled groove highlights, and golden Fresnel grazing sheen.
- **🪞 Chrome**: Dynamic cyber-space room environment reflections, chromatic dispersion, and high-frequency dual specular glints.
- **🎾 Tennis**: Micro-fiber fuzzy felt velvet shading with embossed white seam curves and velvety inverted Fresnel glow.
- **⚡ Neon Cyber Core**: Multi-layered pulsing plasma core with animated internal energy waves, hexagonal shield grid, and multi-color rim bloom.

---

## 🧪 Testing & Verification

The test suite contains **92 automated unit and integration tests** covering all subsystems with zero regressions:

```bash
# Run complete test suite
pytest -v

# Run tests with execution speed profile
pytest --durations=10
```

### Test Suite Breakdown
| Test Module | Tests | Subsystem Covered |
|---|:---:|---|
| `test_async_cam.py` | 6 | Threaded capture, lock-free ring buffer, frame draining |
| `test_detector.py` | 5 | MediaPipe landmark extraction, handedness, bounding boxes |
| `test_depth_estimation.py` | 6 | Monocular palm depth $Z$ estimation, span scaling |
| `test_filtering.py` | 5 | 1-Euro adaptive filter mathematical convergence & latency |
| `test_gestures.py` | 6 | 10 static geometric gesture classifiers & state tracking |
| `test_temporal_gestures.py` | 8 | Trajectory recognizer, swipes, circles, angular delta |
| `test_canvas.py` | 1 | Real-time finger drawing, strokes, color palettes, clearing |
| `test_state_machine.py` | 5 | Media controller wake/sleep transitions & timeout timer |
| `test_media_controller.py` | 4 | Config YAML parsing, gesture-to-action dispatching |
| `test_ar_physics.py` | 7 | 3D spring-damper collisions, palm bounce, pinch grab & throw |
| `test_ar_renderer.py` | 4 | OpenCV fallback rendering & skin cycling |
| `test_virtual_room.py` | 5 | 3D perspective cyber-room projection, shadows, hand joints |
| `test_gpu_renderer.py` | 13 | ModernGL FBO, GLSL shaders, 3-point lighting, bump mapping |
| `test_pipeline.py` | 4 | End-to-end multi-stage pipeline integration & context managers |
| `test_visualization.py` | 7 | HUD telemetry profiler, stage latency meters, help overlay |
| `test_config.py` | 6 | Configuration schema validation, defaults, serialization |
| **Total** | **92** | **100% Passed** |

---

## 📂 Project Structure

```
HandTracking/
├── handtracking/
│   ├── __init__.py               # Package metadata & version
│   ├── __main__.py               # Python module entry point (python -m handtracking)
│   ├── demo.py                   # Main CLI application & interactive keyboard loop
│   ├── pipeline.py               # End-to-end frame processing pipeline
│   ├── ar/
│   │   ├── colliders.py          # 3D palm plane & fingertip collision primitives
│   │   ├── gpu_renderer.py       # ModernGL GPU shader engine & GLSL lighting
│   │   ├── physics.py            # 3D AR physics engine (bouncing, grab, throw, spin)
│   │   ├── renderer.py           # OpenCV 2.5D fallback renderer
│   │   └── room.py               # 3D digital cyber-space perspective environment
│   ├── capture/
│   │   └── async_cam.py          # Lock-free multi-threaded webcam capture
│   ├── config/
│   │   └── settings.py           # YAML/JSON configuration parser & defaults
│   ├── controllers/
│   │   ├── media.py              # Touchless media controller coordinator
│   │   ├── state_machine.py      # Wake/Sleep controller state machine
│   │   └── synthesizer.py        # OS media key synthesizer
│   ├── filtering/
│   │   └── one_euro_filter.py    # 1-Euro adaptive jitter filter
│   ├── gestures/
│   │   ├── canvas.py             # Air Canvas finger painting
│   │   ├── events.py             # Gesture event data classes
│   │   ├── finger_state.py       # Finger extension & curl analyzer
│   │   ├── recognizer.py         # 10 geometric gesture classifiers
│   │   └── temporal.py           # Dynamic temporal gesture recognizer
│   ├── inference/
│   │   ├── depth.py              # Monocular palm depth (Z) estimator
│   │   ├── detector.py           # MediaPipe landmark detector wrapper
│   │   └── models.py             # Landmark data models & normalization
│   └── visualization/
│       ├── hud.py                # Telemetry HUD, stage timers & help overlay
│       ├── media_hud.py          # Media controller status badges & action toasts
│       └── telemetry.py          # Rolling latency profiler & FPS calculator
├── tests/                        # 92 Automated unit and integration tests
├── config.yaml                   # Default media controller configuration
├── pyproject.toml                # Build packaging, dependencies & scripts
├── requirements.txt              # Production dependency list
├── run_demo.bat                  # Windows batch launcher
├── run_demo.ps1                  # PowerShell launcher
├── AGENTS.md                     # StackMind multi-agent governance rules
├── CHANGELOG.md                  # Semantic version changelog
├── PLAN.md                       # Architectural design & milestone roadmap
└── README.md                     # Production documentation
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

