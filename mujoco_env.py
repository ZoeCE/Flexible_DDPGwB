import mujoco
import mujoco.viewer
import numpy as np
import os
from collections import deque

class CableRobotEnv:
    def __init__(self, render=False, latency_steps=1, force_noise_level=0.1,
                 control_freq_hz=10, init_velocity_scale=0.15, init_position_range=0.08):
        """
        Cable Robot Environment with Sim2Real Features
        
        Args:
            render: Enable visualization
            latency_steps: Action buffer delay (steps), simulates transmission latency
            force_noise_level: Random force disturbance magnitude (N)
            control_freq_hz: Control frequency (Hz), physics runs at 500Hz (0.002s timestep)
            init_velocity_scale: Initial velocity randomization scale (m/s)
            init_position_range: Initial position randomization range (m)
        """
        # --- 1. 加载模型 ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "assets2/demo_fourCable_withSteel_withSensor_cylinder.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # --- 2. 系统参数 ---
        self.physics_dt = 0.002  
        self.control_freq_hz = control_freq_hz
        self.control_dt = 1.0 / control_freq_hz 
        self.frame_skip = int(self.control_dt / self.physics_dt)
        self.dt = self.control_dt
        self.model.opt.timestep = self.physics_dt
        
        # --- 3. 获取对象 ID ---
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
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id
        
        # --- 4. 状态与动作 ---
        self.state_dim = 10 
        self.action_dim = 2 
        self.action_space_high = 0.5 
        
        # --- 5. 任务参数 ---
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        self.default_target = np.array([-0.2, 0.3])
        self.target_pos = self.default_target.copy()
        self.max_steps = 200 
        self.current_step = 0
        
        # --- 6. Sim2Real 参数 ---
        self.latency_steps = latency_steps
        self.action_buffer = deque(maxlen=latency_steps + 1)
        self.force_noise_level = force_noise_level
        self.init_velocity_scale = init_velocity_scale
        self.init_position_range = init_position_range
        
        # --- 7. 渲染 ---
        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        """
        Enhanced reset with comprehensive randomization to prevent overfitting
        
        Randomization includes:
        - Target position
        - Payload initial position (XY plane)
        - Payload initial velocity (all 6 DOF: 3 linear + 3 angular)
        - Mocap initial position with offset
        - Mocap initial velocity
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 1. 随机化目标位置 (Target randomization)
        noise_target = np.random.uniform(-0.1, 0.1, size=2)
        self.target_pos = self.default_target + noise_target
        self.model.body_pos[self.target_body_id][:2] = self.target_pos
        
        # 2. 随机化负载初始位置 (Payload position randomization)
        # 扩大范围，防止模型记住固定起点
        start_x = 0.2 + np.random.uniform(-self.init_position_range, self.init_position_range)
        start_y = 0.3 + np.random.uniform(-self.init_position_range, self.init_position_range)
        start_z = 0.6 + np.random.uniform(-0.05, 0.05)  # Z轴也加入随机
        
        q_idx = self.prefab_jnt_id
        self.data.qpos[q_idx] = start_x
        self.data.qpos[q_idx+1] = start_y
        self.data.qpos[q_idx+2] = start_z
        
        # 3. 随机化负载姿态 (Payload orientation randomization)
        # 四元数随机扰动，模拟初始摆动角度
        if q_idx + 6 < len(self.data.qpos):  # 确保有姿态自由度
            # 小角度随机旋转 (±10度)
            angle_perturbation = np.random.uniform(-0.17, 0.17, size=3)  # ~10 degrees
            for i in range(3):
                if q_idx + 3 + i < len(self.data.qpos):
                    self.data.qpos[q_idx + 3 + i] += angle_perturbation[i]
        
        # 4. Mocap 初始化 (带随机偏移，模拟控制不精确)
        # Mocap 不完全对齐负载，产生初始张力差异
        mocap_offset_x = np.random.uniform(-0.03, 0.03)
        mocap_offset_y = np.random.uniform(-0.03, 0.03)
        mocap_z = 1.0 + np.random.uniform(-0.05, 0.05)
        
        self.data.mocap_pos[self.mocap_id][0] = start_x + mocap_offset_x
        self.data.mocap_pos[self.mocap_id][1] = start_y + mocap_offset_y
        self.data.mocap_pos[self.mocap_id][2] = mocap_z
        
        # 5. 随机化负载速度 (Payload velocity randomization - 6 DOF)
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        
        # XY 平面线速度 (主要运动方向)
        self.data.qvel[dof_idx] = np.random.uniform(-self.init_velocity_scale, self.init_velocity_scale)
        self.data.qvel[dof_idx+1] = np.random.uniform(-self.init_velocity_scale, self.init_velocity_scale)
        
        # Z 方向速度 (垂直摆动)
        self.data.qvel[dof_idx+2] = np.random.uniform(-self.init_velocity_scale*0.5, self.init_velocity_scale*0.5)
        
        # 角速度 (旋转扰动) - 模拟负载自旋和摆动
        # 【降低角速度】从 ±0.3 降到 ±0.15 rad/s，防止仿真崩溃
        if dof_idx + 5 < len(self.data.qvel):
            # Roll, Pitch, Yaw 角速度
            self.data.qvel[dof_idx+3] = np.random.uniform(-0.15, 0.15)  # Roll (降低)
            self.data.qvel[dof_idx+4] = np.random.uniform(-0.15, 0.15)  # Pitch (降低)
            self.data.qvel[dof_idx+5] = np.random.uniform(-0.1, 0.1)   # Yaw (降低)
        
        # 6. Mocap 初始速度随机化 (模拟控制器启动瞬间的速度)
        initial_mocap_vel = np.random.uniform(-0.05, 0.05, size=3)
        self.current_mocap_vel = initial_mocap_vel.copy()
        
        # 7. 重置 Action Buffer (用小随机动作填充，而非零)
        self.action_buffer.clear()
        for _ in range(self.latency_steps + 1):
            # 用小随机动作初始化 buffer，模拟系统启动前的微小抖动
            random_init_action = np.random.uniform(-0.05, 0.05, size=3)
            self.action_buffer.append(random_init_action)
        
        # 8. 预热仿真 (让随机初始化稳定下来)
        # 在预热期间施加小随机力，模拟环境噪声
        for _ in range(30):  # 增加预热步数
            noise = np.random.normal(0, self.force_noise_level * 0.5, 3)
            self.data.xfrc_applied[self.prefab_body_id][:3] = noise
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.prefab_body_id][:3] = 0
        
        # 9. 同步 Mocap 状态
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        processed_action = np.zeros(3)
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
            
            processed_action = np.array([ax, ay, az])
        else:
            processed_action = np.clip(action, -self.action_space_high, self.action_space_high)

        # Action Buffer
        self.action_buffer.append(processed_action)
        delayed_action = self.action_buffer[0]
        
        ax_exec, ay_exec, az_exec = delayed_action

        # --- 动力学积分 (带安全钳制) ---
        # 1. 更新速度
        self.current_mocap_vel[0] += ax_exec * self.dt
        self.current_mocap_vel[1] += ay_exec * self.dt
        self.current_mocap_vel[2] += az_exec * self.dt
        
        # 【关键安全阀】限制最大速度，防止物理引擎崩溃
        # 根据控制频率动态调整速度限制
        max_vel = 1.5 if self.control_freq_hz >= 10 else 1.0
        self.current_mocap_vel = np.clip(self.current_mocap_vel, -max_vel, max_vel)

        # 2. 更新位置
        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt
        
        # 【关键安全阀】限制活动范围，防止飞出宇宙
        self.current_mocap_pos[0] = np.clip(self.current_mocap_pos[0], -1.0, 1.0)
        self.current_mocap_pos[1] = np.clip(self.current_mocap_pos[1], -1.0, 1.0)
        self.current_mocap_pos[2] = np.clip(self.current_mocap_pos[2], 0.4, 1.5) # 高度限制

        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos
        
        # Frame Skip
        for sim_step in range(self.frame_skip):
            # 过程噪声
            noise = np.random.normal(0, self.force_noise_level, 3)
            self.data.xfrc_applied[self.prefab_body_id][:3] = noise
            
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.prefab_body_id][:3] = 0
            
            if sim_step == self.frame_skip - 1 and self.render_mode and self.viewer:
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
        
        step_penalty = -0.001
        reward += step_penalty
        
        if dist_xy < 0.03 and payload_vel < 0.1 and q_z < 0.15:
            reward += 1.0 
            success = True
            
        return reward, success, success