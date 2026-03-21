import torch
import numpy as np
import argparse
import time
import os
import sys
import cv2
import mujoco

# 引入环境和控制器
from mujoco_env import (
    CableRobotEnv,
    CableRobotEnvWithObstacles,
    apply_camera_config,
    get_demo_camera_config,
)
from nmpc_controller import NMPCController, NMPCTrajectoryTracker
# 引入 Agent 网络定义 (必须，否则 torch.load 报错)
from agent import FastActor 

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _build_video_writer(output_path, fps, width, height):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")
    return writer


def _render_rgb_frame(renderer, data, camera, camera_config, frame_idx, total_frames):
    progress = 0.0 if total_frames <= 1 else frame_idx / float(total_frames - 1)
    apply_camera_config(camera, camera_config, progress=progress)
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def export_obstacles_demo_video(output_path, mode="obstacles", n_obstacles=3, obstacle_seed=42,
                                width=1920, height=1080, fps=24, duration_seconds=18.0,
                                demo_episodes=3,
                                playback_speed=1.25,
                                post_success_hold_seconds=0.75,
                                payload_radius=0.10, planning_margin=0.10, planning_grid_res=0.02,
                                default_start_xy=None, default_target_xy=None):
    camera_config = get_demo_camera_config()
    env = CableRobotEnvWithObstacles(
        render=False,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.00,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.01, 0.02),
        path_width=0.12,
        obstacle_seed=obstacle_seed,
        default_start_xy=default_start_xy or [0.2, 0.2],
        default_target_xy=default_target_xy or [0.5, 0.5],
        payload_radius=payload_radius,
        planning_margin=planning_margin,
        planning_grid_res=planning_grid_res,
        offscreen_width=width,
        offscreen_height=height,
        viewer_show_left_ui=camera_config["show_left_ui"],
        viewer_show_right_ui=camera_config["show_right_ui"],
        viewer_camera_config=camera_config,
    )

    if mode == "obstacles_base":
        controller = NMPCController()
    else:
        controller = NMPCTrajectoryTracker()

    frames_per_episode = max(1, int(round(duration_seconds * fps)))
    frames_per_step = max(1, int(round(fps * getattr(env, "dt", 1.0 / fps))))
    hold_frames_after_success = max(0, int(round(post_success_hold_seconds * fps)))
    output_fps = float(fps) * float(playback_speed)
    frame_count = 0
    total_steps = 0
    success = False
    episode_collision = False
    camera = mujoco.MjvCamera()
    writer = None
    episodes_recorded = 0

    try:
        writer = _build_video_writer(output_path, output_fps, width, height)
        episodes_to_record = max(1, int(demo_episodes))

        for ep_idx in range(episodes_to_record):
            obs = env.reset()
            obstacles = env.get_obstacles()
            if mode == "obstacles" and hasattr(env, "get_planned_path"):
                planned_path = env.get_planned_path()
                controller.set_trajectory(planned_path if planned_path is not None else [])

            try:
                renderer = mujoco.Renderer(env.model, height=height, width=width)
            except Exception as exc:
                raise RuntimeError(
                "Failed to create the offscreen MuJoCo renderer. On macOS, run this "
                    "from a normal desktop Terminal session, preferably with `mjpython` "
                    "inside the `robot_lab` environment. Original error: "
                    f"{exc}"
                ) from exc

            try:
                segment_frames = frames_per_episode
                segment_frame_idx = 0
                step = 0
                last_rgb = None
                success = False
                episode_collision = False

                while segment_frame_idx < segment_frames:
                    nmpc_state = obs[:8]
                    if mode == "obstacles_base":
                        action = controller.get_action(nmpc_state, env.target_pos)
                    else:
                        action = controller.get_tracking_action(nmpc_state)
                    obs, reward, done, success = env.step(action)
                    step += 1
                    total_steps += 1

                    qx, qy = obs[4], obs[5]
                    for ox, oy, r in obstacles:
                        if np.hypot(qx - ox, qy - oy) < r:
                            episode_collision = True
                            break

                    repeat = min(frames_per_step, segment_frames - segment_frame_idx)
                    for _ in range(repeat):
                        last_rgb = _render_rgb_frame(renderer, env.data, camera, camera_config, segment_frame_idx, segment_frames)
                        writer.write(last_rgb[:, :, ::-1].copy())
                        frame_count += 1
                        segment_frame_idx += 1

                    if success:
                        break
                    if done or step >= 500:
                        break

                if last_rgb is None:
                    last_rgb = _render_rgb_frame(renderer, env.data, camera, camera_config, 0, segment_frames)

                if success:
                    remaining_frames = max(0, segment_frames - segment_frame_idx)
                    hold_frames = min(hold_frames_after_success, remaining_frames)
                    for _ in range(hold_frames):
                        writer.write(last_rgb[:, :, ::-1].copy())
                        frame_count += 1
                        segment_frame_idx += 1
            finally:
                renderer.close()
            episodes_recorded += 1
    finally:
        if writer is not None:
            writer.release()

    return {
        "output_path": output_path,
        "frames_written": frame_count,
        "success": success,
        "collision": episode_collision,
        "steps": total_steps,
        "episodes_recorded": episodes_recorded,
        "output_fps": output_fps,
    }


def run_test(mode, log_dir, n_episodes, render, device_id=0,
             viewer_show_left_ui=True, viewer_show_right_ui=True, viewer_camera_config=None):
    """
    统一的测试主循环
    """
    # 1. 初始化环境
    env = CableRobotEnv(
        render=render,
        viewer_show_left_ui=viewer_show_left_ui,
        viewer_show_right_ui=viewer_show_right_ui,
        viewer_camera_config=viewer_camera_config,
    )
    
    # 2. 初始化策略 (Actor 或 Base)
    actor_model = None
    nmpc_controller = None
    
    if mode == 'actor':
        print(f"Loading Actor model from {log_dir}/actor.pt ...")
        device = get_device(device_id)
        model_path = os.path.join(log_dir, 'actor.pt')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        try:
            # weights_only=False 解决 PyTorch 2.6+ 兼容性
            actor_model = torch.load(model_path, map_location=device, weights_only=False)
            actor_model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    elif mode == 'base':
        print("Initializing NMPC Controller (Base)...")
        nmpc_controller = NMPCController()
    
    # 3. 开始测试循环
    success_count = 0
    total_steps_success = 0
    
    print("*******************************************")
    print(f"Start Testing [{mode.upper()}] for {n_episodes} episodes...")
    print(f"Render: {'ON' if render else 'OFF'}")
    print("*******************************************")
    
    start_time = time.time()
    
    for i in range(n_episodes):
        obs = env.reset()
        target_pos = env.target_pos # 仅 Base 需要用到绝对坐标
        step = 0
        episode_reward = 0
        
        while True:
            # --- 策略决策 ---
            if mode == 'actor':
                # RL Agent: 输入 State -> 输出 2D Action
                s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor_model(s_tensor).cpu().numpy()[0]
            else:
                # NMPC Base: 输入 State + Target -> 输出 2D Action
                nmpc_state = obs[:8]
                action = nmpc_controller.get_action(nmpc_state, target_pos)
            
            # --- 环境交互 ---
            # 无论是 Actor 还是 Base，都发送 2D 动作
            # 环境会自动判断是否满足下降条件
            next_obs, reward, done, success = env.step(action)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # 渲染延时，方便肉眼观察
            if render:
                time.sleep(0.02)
            
            # --- 结束判定 ---
            if done or step >= 500:
                # 只有在渲染模式下才打印每一局的详情，避免刷屏
                if render:
                    print(f"Ep {i+1}: Steps={step}, R={reward:.2f}, Success={success}")
                
                if success: 
                    success_count += 1
                    total_steps_success += step
                break
        
        # 进度条 (非渲染模式下显示)
        if not render and (i+1) % 10 == 0:
            print(f"Progress: {i+1}/{n_episodes} | Current SR: {success_count/(i+1)*100:.1f}%")

    end_time = time.time()
    avg_steps = total_steps_success / success_count if success_count > 0 else 0
    
    print("\n" + "="*30)
    print(f"Final Result [{mode.upper()}]:")
    print(f"Total Episodes: {n_episodes}")
    print(f"Success Rate:   {success_count}/{n_episodes} ({success_count/n_episodes*100:.2f}%)")
    print(f"Avg Steps:      {avg_steps:.1f}")
    print(f"Time Elapsed:   {end_time - start_time:.2f}s")
    print("="*30 + "\n")


def run_test_obstacles(mode, n_episodes=10, render=False, n_obstacles=3,
                       obstacle_seed=42, save_paths_dir=None,
                       payload_radius=0.2, planning_margin=0.2, planning_grid_res=0.02,
                       default_start_xy=None, default_target_xy=None,
                       viewer_show_left_ui=True, viewer_show_right_ui=True,
                       viewer_camera_config=None):
    """带障碍物避碰的 NMPC 测试：CableRobotEnvWithObstacles + NMPCControllerObstacles。"""
    env = CableRobotEnvWithObstacles(
        render=render,
        latency_steps=1,
        force_noise_level=0.08,
        control_freq_hz=10,
        init_velocity_scale=0.08,
        init_position_range=0.00,
        n_obstacles=n_obstacles,
        obstacle_radius_range=(0.01, 0.02),
        path_width=0.12,
        obstacle_seed=obstacle_seed,
        default_start_xy=default_start_xy or [0.2, 0.2],
        default_target_xy=default_target_xy or [0.5, 0.5],
        payload_radius=payload_radius,
        planning_margin=planning_margin,
        planning_grid_res=planning_grid_res,
        viewer_show_left_ui=viewer_show_left_ui,
        viewer_show_right_ui=viewer_show_right_ui,
        viewer_camera_config=viewer_camera_config,
    )

    if mode == 'obstacles_base':
        print("Initializing Basic NMPC Controller (Blindly aiming for target)...")
        nmpc_controller = NMPCController()
    else:
        print("Initializing NMPC Trajectory Tracker (Following A* path)...")
        nmpc_controller = NMPCTrajectoryTracker()

    if save_paths_dir is not None:
        os.makedirs(save_paths_dir, exist_ok=True)

    success_count = 0
    total_steps_success = 0
    collision_count = 0

    print("*******************************************")
    print("NMPC with Obstacle Avoidance")
    print(f"Episodes: {n_episodes}, Render: {render}, Obstacles: {n_obstacles}")
    print("*******************************************")

    start_time = time.time()
    for ep in range(n_episodes):
        obs = env.reset()
        step = 0
        episode_collision = False

        target_pos = env.target_pos # 获取绝对终点

        # 【修复3】每局开始时，必须获取当前环境生成的障碍物列表，用于后续碰撞统计
        obstacles = env.get_obstacles()

        
        if mode == 'obstacles' and hasattr(env, "get_planned_path"):
            planned_path = env.get_planned_path()
            if planned_path is not None and len(planned_path) > 0:
                nmpc_controller.set_trajectory(planned_path)
                
                if save_paths_dir is not None:
                    path_with_idx = np.column_stack([np.arange(planned_path.shape[0]), planned_path])
                    out_path = os.path.join(save_paths_dir, f"path_ep{ep+1}.csv")
                    np.savetxt(out_path, path_with_idx, delimiter=",", header="idx,x,y", comments="")
            else:
                nmpc_controller.set_trajectory([])

        while True:
            nmpc_state = obs[:8]
            if mode == 'obstacles_base':
                # 瞎子模式：直接把终点 target_pos 喂给它
                action = nmpc_controller.get_action(nmpc_state, target_pos)
            else:
                # 跟踪模式：沿路径前瞻行驶
                action = nmpc_controller.get_tracking_action(nmpc_state)
            next_obs, reward, done, success = env.step(action)
            obs = next_obs
            step += 1

            # 碰撞检测 (仅用于统计，不干涉 NMPC 动作)
            qx, qy = obs[4], obs[5]
            for (ox, oy, r) in obstacles:
                if np.hypot(qx - ox, qy - oy) < r:
                    episode_collision = True
                    break

            if render:
                time.sleep(0.02)

            if done or step >= 500:
                if episode_collision:
                    collision_count += 1
                if success:
                    success_count += 1
                    total_steps_success += step
                if render:
                    print(f"Ep {ep+1}: Steps={step}, Success={success}, Collision={episode_collision}")
                break

        if not render and (ep + 1) % 5 == 0:
            print(f"Progress: {ep+1}/{n_episodes} | SR: {success_count/(ep+1)*100:.1f}% | Collisions: {collision_count}")

    elapsed = time.time() - start_time
    avg_steps = total_steps_success / success_count if success_count > 0 else 0
    # 【新增】：根据模式定制专属的输出标题
    if mode == 'obstacles_base':
        title = "Result [OBSTACLES_BASE] (Blind NMPC Baseline):"
    else:
        title = "Result [OBSTACLES] (NMPC Trajectory Tracker):"
    print("\n" + "="*40)
    print(title)
    print(f"  Episodes:     {n_episodes}")
    print(f"  Success:      {success_count} ({success_count/n_episodes*100:.2f}%)")
    print(f"  Collisions:   {collision_count}")
    print(f"  Avg steps:    {avg_steps:.1f}")
    print(f"  Time:        {elapsed:.2f}s")
    print("="*40 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test RL Agent, NMPC Base, or NMPC with Obstacles")

    # 1. 选择模式: actor (RL)、base (NMPC)、obstacles (NMPC+障碍物)
    parser.add_argument('--mode', type=str, default='actor', choices=['actor', 'base', 'obstacles', 'obstacles_base'],
                        help='Policy: "actor", "base", "obstacles", or "obstacles_base"')

    # 2. 是否开启动画
    parser.add_argument('--render', action='store_true', help='Enable visualization')

    # 3. 测试局数
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')

    # 4. 模型路径 (仅 actor 模式)
    parser.add_argument('--dir', type=str, default='saves/nmpc_experiment', help='Directory with actor.pt')

    # 5. 障碍物模式专用参数
    parser.add_argument('--obstacles', type=int, default=3, help='[obstacles] Number of obstacles per episode')
    parser.add_argument('--seed', type=int, default=42, help='[obstacles] Obstacle RNG seed')
    parser.add_argument('--save_paths_dir', type=str, default=None,
                        help='[obstacles] Save planned 2D paths as CSV to this dir')
    parser.add_argument('--payload_radius', type=float, default=0.10, help='[obstacles] Payload safety radius (m)')
    parser.add_argument('--planning_margin', type=float, default=0.10, help='[obstacles] Planning margin (m)')
    parser.add_argument('--planning_grid_res', type=float, default=0.02, help='[obstacles] Grid resolution (m)')
    parser.add_argument('--hide-ui', action='store_true', help='Hide MuJoCo viewer side panels and use demo camera')
    parser.add_argument('--record-demo', action='store_true', help='Export a clean offscreen mp4 demo')
    parser.add_argument('--output', type=str, default='outputs/demo_obstacles.mp4', help='Output path for demo video')
    parser.add_argument('--fps', type=int, default=24, help='Demo video frame rate')
    parser.add_argument('--width', type=int, default=1920, help='Demo video width')
    parser.add_argument('--height', type=int, default=1080, help='Demo video height')
    parser.add_argument('--demo-seconds', type=float, default=18.0, help='Per-episode max duration in seconds')
    parser.add_argument('--demo-episodes', type=int, default=3, help='Number of obstacle episodes to stitch into one demo video')
    parser.add_argument('--playback-speed', type=float, default=1.25, help='Playback speed multiplier written into the saved video')

    args = parser.parse_args()
    viewer_camera_config = get_demo_camera_config() if args.hide_ui else None
    viewer_show_left_ui = not args.hide_ui
    viewer_show_right_ui = not args.hide_ui

    if args.record_demo:
        if args.mode not in ['obstacles', 'obstacles_base']:
            raise ValueError('--record-demo currently supports only obstacles modes')
        result = export_obstacles_demo_video(
            output_path=args.output,
            mode=args.mode,
            n_obstacles=args.obstacles,
            obstacle_seed=args.seed,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration_seconds=args.demo_seconds,
            demo_episodes=args.demo_episodes,
            playback_speed=args.playback_speed,
            payload_radius=args.payload_radius,
            planning_margin=args.planning_margin,
            planning_grid_res=args.planning_grid_res,
        )
        print("*******************************************")
        print("Demo Video Export Complete")
        print(f"Output:      {result['output_path']}")
        print(f"Frames:      {result['frames_written']}")
        print(f"Episodes:    {result['episodes_recorded']}")
        print(f"Steps:       {result['steps']}")
        print(f"Video FPS:   {result['output_fps']:.2f}")
        print(f"Success:     {result['success']}")
        print(f"Collision:   {result['collision']}")
        print("*******************************************")
        sys.exit(0)

    if args.mode in ['obstacles', 'obstacles_base']:
        run_test_obstacles(
            mode=args.mode,  # <--- 新增透传 mode
            n_episodes=args.episodes,
            render=args.render,
            n_obstacles=args.obstacles,
            obstacle_seed=args.seed,
            save_paths_dir=args.save_paths_dir,
            payload_radius=args.payload_radius,
            planning_margin=args.planning_margin,
            planning_grid_res=args.planning_grid_res,
            viewer_show_left_ui=viewer_show_left_ui,
            viewer_show_right_ui=viewer_show_right_ui,
            viewer_camera_config=viewer_camera_config,
        )
    else:
        run_test(
            args.mode,
            args.dir,
            args.episodes,
            args.render,
            viewer_show_left_ui=viewer_show_left_ui,
            viewer_show_right_ui=viewer_show_right_ui,
            viewer_camera_config=viewer_camera_config,
        )
