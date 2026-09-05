# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-05

### Added
- **Asynchronous Video Capture (`handtracking.capture`)**: Multi-threaded, lock-free camera capture engine with DirectShow/MSMF backends that drains OpenCV driver queues and eliminates latency buildup.
- **MediaPipe Landmark Inference (`handtracking.inference`)**: Real-time 21 3D hand keypoints detection with normalized bounding boxes, handedness classification, and dynamic monocular hand depth ($Z$) estimation.
- **Adaptive Jitter Filter (`handtracking.filtering`)**: Real-time 1-Euro adaptive low-pass filter with dynamic cutoff frequencies for smooth landmark tracking without perceptual latency.
- **10 Geometric Gesture Classifiers (`handtracking.gestures`)**: Real-time recognition of `open_palm`, `closed_fist`, `pointing_up`, `peace_sign`, `thumbs_up`, `thumbs_down`, `ok_sign`, `rock_on`, `pinch_index_thumb`, and `three_fingers`.
- **Dynamic Temporal Gesture Engine (`handtracking.gestures.temporal`)**: Trajectory-based gesture recognition supporting `swipe_left`, `swipe_right`, `swipe_up`, `swipe_down`, `circle_cw`, `circle_ccw`, and `wave`.
- **Air Canvas (`handtracking.gestures.canvas`)**: Real-time virtual fingertip drawing with pinch-to-draw activation, color palette switching (1-4 keys), and canvas clear.
- **Touchless Media Controller (`handtracking.controllers`)**: State machine (`SLEEPING` $\leftrightarrow$ `ACTIVE`) with configurable wake gestures, radial volume control (circular gesture synthesis), play/pause, track skipping, and OS media key synthesis.
- **3D AR Physics Simulation (`handtracking.ar.physics`)**: Real-time physics engine with spring-damper hand collision, palm plane bounce, pinch grab-and-throw, depth throwing ($V_z$), gravity toggle, and angular ball spin.
- **3D Cyber-Space Environment (`handtracking.ar.room`)**: 3D perspective grid cyber-room with holographic hand skeleton rendering, depth-aware shadows, and room boundary reflections.
- **Hardware GPU Shader Engine (`handtracking.ar.gpu_renderer`)**: ModernGL offscreen FBO rendering with GLSL Blinn-Phong shaders, 3-point cinematic studio lighting, micro-pebble bump mapping, recessed rubber seams, and PBR materials (Basketball, Chrome, Tennis, Neon).
- **Digital AR Baby-Pink Heart (`handtracking.ar.heart`)**: Interactive floating baby-pink heart hovering directly over the palmar pad with continuous real-time scaling based on palm openness (curling fingers shrinks heart to a seed, opening palm blooms to full 2x size), dorsal side (back-of-hand) suppression, open-palm-only initial activation, ECG `lub_dub` heartbeat pulse, 3-layer concentric glowing aura, and orbiting sparkles.
- **HUD & Telemetry Profiler (`handtracking.visualization`)**: Real-time FPS, per-stage latency breakdown (Capture, Inference, Smoothing, Gestures, Physics, Rendering), and interactive on-screen help card.
- **Comprehensive Test Suite (`tests/`)**: 104 unit tests covering all components with 100% pass rate in headless environments.

