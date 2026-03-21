import importlib
import os
import sys
import tempfile
import types
import unittest


class FakeBody:
    def __init__(self, name):
        self.name = name
        self.id = {"mocap": 0, "prefab": 1, "rebar_base": 2}[name]
        self.jntadr = [3] if name == "prefab" else [0]
        self.mocapid = [0] if name == "mocap" else [0]


class FakeModel:
    def __init__(self):
        self.opt = types.SimpleNamespace(timestep=0.0)

    @classmethod
    def from_xml_path(cls, _path):
        return cls()

    def body(self, name):
        return FakeBody(name)


class FakeData:
    def __init__(self, _model):
        self.qpos = [0.0] * 8
        self.qvel = [0.0] * 8
        self.mocap_pos = [[0.0, 0.0, 0.0]]


class FakeRunningViewer:
    def __init__(self):
        self.close_calls = 0

    def is_running(self):
        return True

    def close(self):
        self.close_calls += 1


class ViewerReloadTests(unittest.TestCase):
    def setUp(self):
        self._old_mujoco = sys.modules.get("mujoco")
        self._old_mujoco_viewer = sys.modules.get("mujoco.viewer")
        self._old_numpy = sys.modules.get("numpy")
        self._old_mujoco_env = sys.modules.pop("mujoco_env", None)

        fake_mujoco = types.ModuleType("mujoco")
        fake_viewer_module = types.ModuleType("mujoco.viewer")
        self.launched_viewers = []

        def launch_passive(model, data):
            viewer = FakeRunningViewer()
            self.launched_viewers.append((model, data, viewer))
            return viewer

        fake_viewer_module.launch_passive = launch_passive
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = tuple
        fake_numpy.float32 = float

        fake_mujoco.MjModel = FakeModel
        fake_mujoco.MjData = FakeData
        fake_mujoco.viewer = fake_viewer_module

        sys.modules["mujoco"] = fake_mujoco
        sys.modules["mujoco.viewer"] = fake_viewer_module
        sys.modules["numpy"] = fake_numpy

        self.mujoco_env = importlib.import_module("mujoco_env")

    def tearDown(self):
        if self._old_mujoco is not None:
            sys.modules["mujoco"] = self._old_mujoco
        else:
            sys.modules.pop("mujoco", None)

        if self._old_mujoco_viewer is not None:
            sys.modules["mujoco.viewer"] = self._old_mujoco_viewer
        else:
            sys.modules.pop("mujoco.viewer", None)

        if self._old_numpy is not None:
            sys.modules["numpy"] = self._old_numpy
        else:
            sys.modules.pop("numpy", None)

        if self._old_mujoco_env is not None:
            sys.modules["mujoco_env"] = self._old_mujoco_env
        else:
            sys.modules.pop("mujoco_env", None)

    def test_reload_model_with_running_viewer_relaunches_viewer(self):
        env = self.mujoco_env.CableRobotEnvWithObstacles.__new__(
            self.mujoco_env.CableRobotEnvWithObstacles
        )
        old_viewer = FakeRunningViewer()

        with tempfile.TemporaryDirectory() as tmpdir:
            env._base_xml_content = "<mujoco><asset/></mujoco>"
            env._assets2_dir = tmpdir
            env.physics_dt = 0.002
            env.dt = 0.02
            env.render_mode = True
            env.viewer = old_viewer
            env._temp_xml_path = None
            env._reresolve_ids = lambda: None

            env._reload_model_with_obstacles(
                obstacles=[(0.1, 0.2, 0.03)],
                path_points=None,
                start_xy=None,
                goal_xy=None,
            )

            self.assertTrue(os.path.exists(env._temp_xml_path))

        self.assertEqual(old_viewer.close_calls, 1)
        self.assertIsNot(env.viewer, old_viewer)
        self.assertEqual(len(self.launched_viewers), 1)


if __name__ == "__main__":
    unittest.main()
