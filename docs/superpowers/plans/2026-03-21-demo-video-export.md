# Demo Video Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible demo export path that renders a fixed-camera obstacle-avoidance episode to a clean `mp4` video without MuJoCo UI panels.

**Architecture:** Keep the existing simulation loop intact for control logic, and add a separate offscreen rendering path driven by `mujoco.Renderer`. Camera settings and gentle motion live in a small helper layer so the interactive viewer and video export can share the same framing ideas without coupling the control loop to recording details.

**Tech Stack:** Python, MuJoCo 3.3.x, OpenCV (`cv2`) for mp4 writing, unittest stubs for regression tests

---

## File Map

- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/mujoco_env.py`
  - Add reusable viewer/camera helpers and offscreen renderer support hooks.
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/test.py`
  - Add CLI flags and the demo export execution path.
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/tests/test_viewer_reload.py`
  - Keep existing viewer regression coverage green if helper behavior changes.
- Create: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/tests/test_demo_video.py`
  - Cover demo config defaults and mp4 writer/offscreen export orchestration.

## Chunk 1: Demo Config and Failing Tests

### Task 1: Add a failing test for fixed demo camera config

**Files:**
- Create: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/tests/test_demo_video.py`

- [ ] **Step 1: Write the failing test**
  - Assert a helper returns the expected paper-style camera defaults and hides viewer UI by default.

- [ ] **Step 2: Run test to verify it fails**
  - Run: `python -m unittest tests/test_demo_video.py`
  - Expected: FAIL because helper/config is missing.

### Task 2: Add a failing test for mp4 export orchestration

**Files:**
- Create: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/tests/test_demo_video.py`

- [ ] **Step 1: Write the failing test**
  - Stub environment, renderer, and writer; assert frames are rendered and writer is closed.

- [ ] **Step 2: Run test to verify it fails**
  - Run: `python -m unittest tests/test_demo_video.py`
  - Expected: FAIL because export function is missing.

## Chunk 2: Minimal Implementation

### Task 3: Add reusable demo camera helpers

**Files:**
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/mujoco_env.py`

- [ ] **Step 1: Implement minimal config helpers**
  - Add a small preset helper for fixed lookat, distance, azimuth, elevation, UI visibility, and gentle camera motion.

- [ ] **Step 2: Re-run targeted tests**
  - Run: `python -m unittest tests/test_demo_video.py`
  - Expected: camera-config test passes or gets closer to green.

### Task 4: Add offscreen mp4 export path

**Files:**
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/test.py`
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/mujoco_env.py`

- [ ] **Step 1: Implement offscreen rendering loop**
  - Use `mujoco.Renderer` for RGB frames.
  - Use OpenCV `VideoWriter` to produce `mp4`.
  - Reuse the obstacle controller loop and stop when success or time budget is reached.

- [ ] **Step 2: Add CLI flags**
  - Add flags for `--record-demo`, `--output`, `--fps`, `--width`, `--height`, `--demo-seconds`, and optional `--hide-ui`.

- [ ] **Step 3: Re-run targeted tests**
  - Run: `python -m unittest tests/test_demo_video.py tests/test_viewer_reload.py`
  - Expected: PASS.

## Chunk 3: Verification

### Task 5: Syntax and smoke verification

**Files:**
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/test.py`
- Modify: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/mujoco_env.py`
- Create: `/Users/zoe/workplace/research stuff/coding/BCLearning_with_nmpc/tests/test_demo_video.py`

- [ ] **Step 1: Verify Python syntax**
  - Run: `python -m py_compile mujoco_env.py test.py tests/test_demo_video.py tests/test_viewer_reload.py`
  - Expected: PASS.

- [ ] **Step 2: Verify unit tests**
  - Run: `python -m unittest tests/test_demo_video.py tests/test_viewer_reload.py`
  - Expected: PASS.

- [ ] **Step 3: Provide user smoke command**
  - `mjpython test.py --mode obstacles --episodes 1 --record-demo --output demo_obstacles.mp4 --width 1920 --height 1080 --fps 30 --demo-seconds 20`

Plan complete and saved to `docs/superpowers/plans/2026-03-21-demo-video-export.md`. Ready to execute.
