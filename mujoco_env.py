import mujoco
import mujoco.viewer
import numpy as np
import os

class CableRobotEnv:
    def __init__(self, render=False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "assets2/demo_fourCable_withSteel_withSensor_cylinder.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.dt = 0.02
        self.model.opt.timestep = 0.002 
        self.sim_steps = int(self.dt / self.model.opt.timestep)
        
        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, 'mocapid'):
            mids = mocap_body.mocapid
            if isinstance(mids, np.ndarray) or isinstance(mids, list):
                self.mocap_id = mids[0]
            else:
                self.mocap_id = mids
        else:
            raise ValueError("Model does not contain a mocap body named 'mocap'")

        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.target_body_id = self.model.body("rebar_base").id
        
        self.state_dim = 10 
        self.action_dim = 2 
        self.action_space_high = 0.5 
        
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        self.default_target = np.array([-0.2, 0.3])
        self.target_pos = self.default_target.copy()
        self.max_steps = 200 
        self.current_step = 0
        
        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        noise_target = np.random.uniform(-0.1, 0.1, size=2)
        self.target_pos = self.default_target + noise_target
        self.model.body_pos[self.target_body_id][:2] = self.target_pos
        
        start_x = 0.2 + np.random.uniform(-0.05, 0.05)
        start_y = 0.3 + np.random.uniform(-0.05, 0.05)
        
        q_idx = self.prefab_jnt_id
        self.data.qpos[q_idx] = start_x
        self.data.qpos[q_idx+1] = start_y
        
        self.data.mocap_pos[self.mocap_id][0] = start_x
        self.data.mocap_pos[self.mocap_id][1] = start_y
        self.data.mocap_pos[self.mocap_id][2] = 1.0
        
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_mocap_vel = np.zeros(3)
        
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        obs_tmp = self._get_obs()
        dist_xy = np.linalg.norm(obs_tmp[8:10])
        vel_xy = np.linalg.norm(obs_tmp[6:8])
        
        if len(action) == 2:
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay = action
            
            if dist_xy < 0.05 and vel_xy < 0.15:
                target_vz = -0.2
                current_vz = self.current_mocap_vel[2]
                az = 2.0 * (target_vz - current_vz)
            else:
                target_z = 1.0
                current_z = self.current_mocap_pos[2]
                current_vz = self.current_mocap_vel[2]
                az = 5.0 * (target_z - current_z) - 2.0 * current_vz
        else:
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay, az = action

        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt + 0.5 * ax * self.dt**2
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt + 0.5 * ay * self.dt**2
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt + 0.5 * az * self.dt**2
        
        self.current_mocap_vel[0] += ax * self.dt
        self.current_mocap_vel[1] += ay * self.dt
        self.current_mocap_vel[2] += az * self.dt
        
        if self.current_mocap_pos[2] < 0.4: 
             self.current_mocap_pos[2] = 0.4
             self.current_mocap_vel[2] = 0

        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos
        
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
            
        if self.render_mode and self.viewer:
            self.viewer.sync()
            
        self.current_step += 1
        obs = self._get_obs()
        reward, done, success = self._compute_reward(obs)
        
        if self.current_step >= self.max_steps:
            done = True
            
        return obs, reward, done, success

    def _get_obs(self):
        px, py = self.current_mocap_pos[0], self.current_mocap_pos[1]
        vpx, vpy = self.current_mocap_vel[0], self.current_mocap_vel[1]
        
        q_idx = self.prefab_jnt_id
        qx = self.data.qpos[q_idx]
        qy = self.data.qpos[q_idx+1]
        
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        vqx = self.data.qvel[dof_idx]
        vqy = self.data.qvel[dof_idx+1]
        
        tx = self.target_pos[0] - qx
        ty = self.target_pos[1] - qy
        
        return np.array([px, py, vpx, vpy, qx, qy, vqx, vqy, tx, ty], dtype=np.float32)

    def _compute_reward(self, obs):
        dist_xy = np.linalg.norm(obs[8:10])
        payload_vel = np.linalg.norm(obs[6:8])
        q_z = self.data.qpos[self.prefab_jnt_id + 2]
        
        success = False
        reward = 0.0
        
        # 【关键修改】引入时间惩罚 (Step Cost)
        # 每走一步扣 0.001 分，鼓励快速完成
        step_penalty = -0.001
        reward += step_penalty
        
        if dist_xy < 0.03 and payload_vel < 0.1 and q_z < 0.15:
            reward += 1.0 # 成功大奖
            success = True
            
        return reward, success, success