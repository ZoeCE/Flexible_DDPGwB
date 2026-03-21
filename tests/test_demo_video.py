import importlib.util
import pathlib
import unittest
from unittest import mock

import numpy as np

import mujoco_env


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_PY = ROOT / "test.py"


def load_test_module():
    spec = importlib.util.spec_from_file_location("demo_test_module", TEST_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoCameraConfigTests(unittest.TestCase):
    def test_default_demo_camera_hides_ui_and_has_gentle_motion(self):
        config = mujoco_env.get_demo_camera_config()

        self.assertEqual(config["name"], "paper_orbit")
        self.assertFalse(config["show_left_ui"])
        self.assertFalse(config["show_right_ui"])
        np.testing.assert_allclose(config["lookat"], np.array([0.22, 0.16, 0.62]))
        self.assertAlmostEqual(config["distance"], 1.85)
        self.assertAlmostEqual(config["azimuth"], 140.0)
        self.assertAlmostEqual(config["elevation"], -18.0)
        self.assertAlmostEqual(config["orbit_azimuth"], 6.0)
        self.assertAlmostEqual(config["dolly_distance"], -0.08)

    def test_build_xml_with_obstacles_adds_offscreen_framebuffer_size(self):
        xml = mujoco_env._build_xml_with_obstacles(
            base_xml_content="<mujoco><worldbody><geom name=\"floor\" size=\"0 0 0.05\" type=\"plane\" material=\"groundplane\"/>\n\n    <!--  固定钢筋 --></worldbody></mujoco>",
            obstacles=[],
            offscreen_width=1920,
            offscreen_height=1080,
        )

        self.assertIn("<visual>", xml)
        self.assertIn('offwidth="1920"', xml)
        self.assertIn('offheight="1080"', xml)


class DemoExportTests(unittest.TestCase):
    def test_export_demo_stops_each_episode_on_success_and_saves_faster_playback(self):
        test_module = load_test_module()

        class FakeEnv:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.dt = 0.1
                self.model = object()
                self.data = object()
                self.target_pos = np.array([0.5, 0.5])
                self.reset_calls = 0
                self.step_calls = 0

            def reset(self):
                self.reset_calls += 1
                self.model = object()
                self.data = object()
                return np.zeros(10, dtype=np.float32)

            def get_obstacles(self):
                return [(0.3, 0.3, 0.02)]

            def get_planned_path(self):
                return np.array([[0.2, 0.2], [0.5, 0.5]], dtype=np.float32)

            def step(self, action):
                self.step_calls += 1
                return np.zeros(10, dtype=np.float32), 1.0, True, True

        class FakeController:
            def __init__(self):
                self.trajectory = None

            def set_trajectory(self, trajectory):
                self.trajectory = trajectory

            def get_tracking_action(self, _state):
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        class FakeRenderer:
            def __init__(self, model, height, width):
                self.model = model
                self.height = height
                self.width = width
                self.updated = []
                self.closed = False

            def update_scene(self, data, camera=None, scene_option=None):
                self.updated.append((data, camera, scene_option))

            def render(self):
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            def close(self):
                self.closed = True

        class FakeWriter:
            def __init__(self, path, fourcc, fps, size):
                self.path = path
                self.fourcc = fourcc
                self.fps = fps
                self.size = size
                self.frames = []
                self.released = False

            def isOpened(self):
                return True

            def write(self, frame):
                self.frames.append(frame)

            def release(self):
                self.released = True

        fake_writers = []
        fake_envs = []

        def build_env(**kwargs):
            env = FakeEnv(**kwargs)
            fake_envs.append(env)
            return env

        def build_writer(path, fourcc, fps, size):
            writer = FakeWriter(path, fourcc, fps, size)
            fake_writers.append(writer)
            return writer

        with mock.patch.object(test_module, "CableRobotEnvWithObstacles", side_effect=build_env), \
             mock.patch.object(test_module, "NMPCTrajectoryTracker", FakeController), \
             mock.patch.object(test_module.mujoco, "Renderer", FakeRenderer), \
             mock.patch.object(test_module.cv2, "VideoWriter_fourcc", return_value=1234), \
             mock.patch.object(test_module.cv2, "VideoWriter", side_effect=build_writer):
            result = test_module.export_obstacles_demo_video(
                output_path="demo.mp4",
                width=640,
                height=360,
                fps=24,
                duration_seconds=2.0,
                demo_episodes=3,
                playback_speed=1.25,
                n_obstacles=3,
                obstacle_seed=42,
            )

        self.assertEqual(result["output_path"], "demo.mp4")
        self.assertLess(result["frames_written"], 24 * 2 * 3)
        self.assertTrue(fake_writers[0].released)
        self.assertEqual(fake_writers[0].size, (640, 360))
        self.assertAlmostEqual(fake_writers[0].fps, 30.0)
        self.assertEqual(result["episodes_recorded"], 3)
        self.assertEqual(fake_envs[0].reset_calls, 3)


if __name__ == "__main__":
    unittest.main()
