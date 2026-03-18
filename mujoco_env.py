import mujoco
import mujoco.viewer
import numpy as np
import os
from collections import deque

class CableRobotEnv:
    def __init__(self, render=False, latency_steps=1, force_noise_level=0.1,
                 control_freq_hz=10, init_velocity_scale=0.15, init_position_range=0.08,
                 enable_init_randomization=True, enable_process_noise=True):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "assets2/demo_fourCable_withSteel_withSensor_cylinder.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.physics_dt = 0.002  
        self.control_freq_hz = control_freq_hz
        self.control_dt = 1.0 / control_freq_hz 
        self.frame_skip = int(self.control_dt / self.physics_dt)
        self.dt = self.control_dt
        self.model.opt.timestep = self.physics_dt
        
        mocap_body = self.model.body("mocap")
        mids = mocap_body.mocapid
        self.mocap_id = mids[0] if isinstance(mids, (np.ndarray, list)) else mids

        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id
        
        self.state_dim = 10
        self.action_dim = 2
        self.action_space_high = 0.5
        self.last_action = np.zeros(3)
        
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        self.default_target = np.array([-0.2, 0.3])
        self.target_pos = self.default_target.copy()
        self.max_steps = 200 
        self.current_step = 0
        
        self.latency_steps = latency_steps
        self.action_buffer = deque(maxlen=latency_steps + 1)
        self.force_noise_level = force_noise_level
        self.init_velocity_scale = init_velocity_scale
        self.init_position_range = init_position_range
        
        self.enable_init_randomization = enable_init_randomization
        self.enable_process_noise = enable_process_noise
        
        self.current_init_dist = 0.0
        self.accumulated_swing = 0.0
        self.swing_steps = 0
        
        # 【新增】Z轴平滑状态机，初始高度为 1.0
        self.target_z_state = 1.0

        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        q_idx = self.prefab_jnt_id
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]

        if self.enable_init_randomization:
            noise_target = np.random.uniform(-0.1, 0.1, size=2)
            self.target_pos = self.default_target + noise_target
            
            start_x = 0.2 + np.random.uniform(-self.init_position_range, self.init_position_range)
            start_y = 0.3 + np.random.uniform(-self.init_position_range, self.init_position_range)
            start_z = 0.6 + np.random.uniform(-0.05, 0.05)
            
            if q_idx + 6 < len(self.data.qpos):
                angle_perturbation = np.random.uniform(-0.17, 0.17, size=3)
                for i in range(3):
                    if q_idx + 3 + i < len(self.data.qpos):
                        self.data.qpos[q_idx + 3 + i] += angle_perturbation[i]
            
            mocap_offset_x = np.random.uniform(-0.03, 0.03)
            mocap_offset_y = np.random.uniform(-0.03, 0.03)
            mocap_z = 1.0 + np.random.uniform(-0.05, 0.05)
            
            self.data.qvel[dof_idx] = np.random.uniform(-self.init_velocity_scale, self.init_velocity_scale)
            self.data.qvel[dof_idx+1] = np.random.uniform(-self.init_velocity_scale, self.init_velocity_scale)
            self.data.qvel[dof_idx+2] = np.random.uniform(-self.init_velocity_scale*0.5, self.init_velocity_scale*0.5)
            if dof_idx + 5 < len(self.data.qvel):
                self.data.qvel[dof_idx+3] = np.random.uniform(-0.15, 0.15)
                self.data.qvel[dof_idx+4] = np.random.uniform(-0.15, 0.15)
                self.data.qvel[dof_idx+5] = np.random.uniform(-0.1, 0.1)
                
            self.current_mocap_vel = np.random.uniform(-0.05, 0.05, size=3)
        else:
            self.target_pos = self.default_target.copy()
            start_x, start_y, start_z = 0.2, 0.3, 0.6
            if q_idx + 6 < len(self.data.qpos):
                for i in range(3):
                    if q_idx + 3 + i < len(self.data.qpos):
                        self.data.qpos[q_idx + 3 + i] = 0.0
            mocap_offset_x, mocap_offset_y, mocap_z = 0.0, 0.0, 1.0
            for i in range(6):
                if dof_idx + i < len(self.data.qvel):
                    self.data.qvel[dof_idx+i] = 0.0
            self.current_mocap_vel = np.zeros(3)

        self.model.body_pos[self.target_body_id][:2] = self.target_pos
        self.data.qpos[q_idx] = start_x
        self.data.qpos[q_idx+1] = start_y
        self.data.qpos[q_idx+2] = start_z
        
        self.data.mocap_pos[self.mocap_id][0] = start_x + mocap_offset_x
        self.data.mocap_pos[self.mocap_id][1] = start_y + mocap_offset_y
        self.data.mocap_pos[self.mocap_id][2] = mocap_z

        self.current_init_dist = np.linalg.norm([start_x - self.target_pos[0], start_y - self.target_pos[1]])
        self.accumulated_swing = 0.0
        self.swing_steps = 0
        
        # 【重置】Z轴状态机
        self.target_z_state = 1.0

        self.action_buffer.clear()
        for _ in range(self.latency_steps + 1):
            if self.enable_init_randomization:
                self.action_buffer.append(np.random.uniform(-0.05, 0.05, size=3))
            else:
                self.action_buffer.append(np.zeros(3))
        
        for _ in range(30):
            if self.enable_process_noise:
                noise = np.random.normal(0, self.force_noise_level * 0.5, 3)
                self.data.xfrc_applied[self.prefab_body_id][:3] = noise
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.prefab_body_id][:3] = 0
        
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_step = 0
        self.last_action = np.zeros(3)
        
        return self._get_obs()

    def step(self, action):
        processed_action = np.zeros(3)
        obs_tmp = self._get_obs()
        dist_xy = np.linalg.norm(obs_tmp[8:10])
        vel_xy = np.linalg.norm(obs_tmp[6:8])
        
        if len(action) == 2:
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay = action
            
            # ==========================================
            # 【核心修复】平滑的 Z 轴状态机控制
            # ==========================================
            if dist_xy < 0.05 and vel_xy < 0.15:
                # 满足条件，期望高度设为装配高度 (假设为 0.6)
                desired_z = 0.6
            else:
                # 不满足条件，期望高度保持在运输高度 (1.0)
                desired_z = 1.0
                
            # 限制目标高度的变化率 (每步最多变 0.02m，实现平滑下降/上升)
            dz = np.clip(desired_z - self.target_z_state, -0.02, 0.02)
            self.target_z_state += dz
            
            # 使用统一的 PD 控制器追踪平滑的 target_z_state
            current_z = self.current_mocap_pos[2]
            current_vz = self.current_mocap_vel[2]
            # 增大 PD 参数让它跟得更紧，因为 target_z_state 是平滑的，所以不会突变
            az = 10.0 * (self.target_z_state - current_z) - 4.0 * current_vz
            
            az = np.clip(az, -5.0, 5.0) 
            processed_action = np.array([ax, ay, az])
        else:
            processed_action = np.clip(action, -self.action_space_high, self.action_space_high)
        
 
        current_action = processed_action.copy()
        self.action_diff = np.linalg.norm(current_action[:2] - self.last_action[:2])
        self.last_action = current_action # 更新 last_action
        
        self.action_buffer.append(processed_action)
        ax_exec, ay_exec, az_exec = self.action_buffer[0]
        

        self.current_mocap_vel[0] += ax_exec * self.dt
        self.current_mocap_vel[1] += ay_exec * self.dt
        self.current_mocap_vel[2] += az_exec * self.dt
        
        max_vel = 1.5 if self.control_freq_hz >= 10 else 1.0
        self.current_mocap_vel = np.clip(self.current_mocap_vel, -max_vel, max_vel)

        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt
        
        self.current_mocap_pos[0] = np.clip(self.current_mocap_pos[0], -1.0, 1.0)
        self.current_mocap_pos[1] = np.clip(self.current_mocap_pos[1], -1.0, 1.0)
        self.current_mocap_pos[2] = np.clip(self.current_mocap_pos[2], 0.4, 1.5)

        q_idx = self.prefab_jnt_id
        payload_pos = self.data.qpos[q_idx:q_idx+2]
        mocap_pos = self.current_mocap_pos[:2]
        
        if np.linalg.norm(payload_pos - mocap_pos) > 0.4:
            safe_obs = self._get_obs()
            return safe_obs, -5.0, True, {"success": False, "physics_crash": True}

        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos
        
        for sim_step in range(self.frame_skip):
            if self.enable_process_noise:
                noise = np.random.normal(0, self.force_noise_level, 3)
                noise = np.clip(noise, -0.5, 0.5)
                self.data.xfrc_applied[self.prefab_body_id][:3] = noise
            
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.prefab_body_id][:3] = 0
            
            if sim_step == self.frame_skip - 1 and self.render_mode and self.viewer:
                self.viewer.sync()
            
        self.current_step += 1
        obs = self._get_obs()
        
        if np.isnan(obs).any() or np.isinf(obs).any():
            safe_obs = np.zeros_like(obs)
            return safe_obs, -5.0, True, {"success": False, "physics_crash": True}

        qx, qy = obs[4], obs[5]
        px, py = obs[0], obs[1]
        self.accumulated_swing += np.linalg.norm([qx - px, qy - py])
        self.swing_steps += 1

        reward, done, success = self._compute_reward(obs)
        if self.current_step >= self.max_steps:
            done = True
            
        info = {
            "success": success,
            "physics_crash": False 
        }
            
        return obs, reward, done, info

    def get_avg_swing(self):
        return self.accumulated_swing / max(1, self.swing_steps)

    def _get_obs(self):
        px, py = self.current_mocap_pos[0], self.current_mocap_pos[1]
        vpx, vpy = self.current_mocap_vel[0], self.current_mocap_vel[1]
        q_idx = self.prefab_jnt_id
        qx, qy = self.data.qpos[q_idx], self.data.qpos[q_idx+1]
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        vqx, vqy = self.data.qvel[dof_idx], self.data.qvel[dof_idx+1]
        tx, ty = self.target_pos[0] - qx, self.target_pos[1] - qy
        return np.array([px, py, vpx, vpy, qx, qy, vqx, vqy, tx, ty], dtype=np.float32)

    def _compute_reward(self, obs):
        dist_xy = np.linalg.norm(obs[8:10])
        payload_vel = np.linalg.norm(obs[6:8])
        q_z = self.data.qpos[self.prefab_jnt_id + 2]
        
        success = False
        reward = -0.001  
        
        # 1. 加速度大小惩罚 (限制绝对力量)
        acc_xy = np.linalg.norm(self.last_action[:2])
        acc_penalty_weight = 0.02
        reward -= acc_penalty_weight * acc_xy
        
        # 2. 【新增】动作平滑度惩罚 (限制高频抖动/Jerk)
        # 权重可以设为 0.05，强迫智能体输出连续平滑的动作
        smoothness_weight = 0.05
        if hasattr(self, 'action_diff'):
            reward -= smoothness_weight * self.action_diff
        
        if dist_xy < 0.03 and payload_vel < 0.1 and q_z < 0.15:
            reward += 1.0
            success = True
            
        return reward, success, success