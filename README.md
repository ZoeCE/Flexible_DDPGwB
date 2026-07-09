# Flexible_DDPGwB - Cable-Suspended Payload RL Framework

> Current descent mainline (2026-07-05): **trainable 8D CableEncoder** with
> PID+residual RL (`descent_rl.obs_dim=52`). Legacy default before this
> promotion was **frozen 32D CableEncoder**; reproduce it explicitly with
> `--cable-encoder-output-dim 32 --frozen-cable-encoder`.

> Paper-output rule (2026-07-05): all formal test/result tables should be
> generated in LaTeX paper style. Use
> `test_results/paper_final_experiment_tables_20260705.tex` as the current
> template and status map.

> Current 5Hz-observation / 10Hz-control ablation status (2026-07-09):
> use the L8-selected checkpoint triplet from
> `saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706`.
> Do **not** use `latest` or `final` for the next test stage; training degraded
> after the L8 selection point.

> Paper main-figure rule (2026-07-09): the current binned controlled-variable
> descent comparison is confirmed and frozen as the main comparison figure.
> Use C1-C4 from the formal interval ladder, not the older mixed-factor L1-L5
> ladder, for the paper-facing headline plot.

## Current Test-Ready Checkpoints (2026-07-09)

### Full8D + ObsPred + CableLatPred, 10Hz control / 5Hz observation

Run directory:

```text
saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706
```

Use these files as one matched triplet:

```text
ckpt_best_l8.pt
ckpt_best_l8_obs_predictor.pt
ckpt_best_l8_cable_latent_predictor.pt
```

Training status:

- PPO completed normally at `8,000,000` env steps with `8` envs.
- W&B run:
  `https://wandb.ai/xz573-university-of-cambridge/phase_rl_v11_ablation/runs/13kawnws`
- `ckpt_best_l8_meta.json` selects the L8 checkpoint at
  `total_steps=5,083,134`, `episode=57,750`, `level_idx=8`.
- Selection metric: `sr`; recorded L8 SR is `0.644`, with current-window
  `cur_sr=0.72` at the selection point.
- Final/latest is worse: the final CSV row at about `8.0M` steps reports
  `sr=0.518`, `cur_sr=0.600`. For testing this branch, prefer `best_l8`.

Configuration summary:

- Policy starts from the untrained Full8D init checkpoint, not from the mature
  trained Full8D policy.
- `control_freq_hz=10`.
- `observation_predictor.enabled=True`, `measurement_period_steps=2`, so the
  policy receives one true observation every two 10Hz control steps.
- `obs_predictor.target_mode=non_cable_latent`.
- `cable_latent_predictor.enabled=True`, `latent_dim=8`.
- `cable_encoder.output_dim=8`, `cable_encoder.trainable=True`.
- Training used `--disable-vision` and `--rope-marker-feature-source site`.
- Descent still uses PID base residual mode:
  `pid_residual_mode=True`, `include_pid_base_obs=True`.

Recommended smoke command before formal testing:

```bash
cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
conda activate vsdrl_env_5060

RUN=saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706
OUT=test_results/5hzobs_10hzctrl_l8best_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python test_phase.py \
  --phase descent --algo ppo \
  --ckpt "$RUN/ckpt_best_l8.pt" \
  --episodes 32 \
  --seed 270909 \
  --eval-curriculum-level max \
  --wind-speed-max 10 \
  --control-freq-hz 10 \
  --disable-vision \
  --cable-encoder-output-dim 8 \
  --trainable-cable-encoder \
  --obs-predictor \
  --obs-predictor-ckpt "$RUN/ckpt_best_l8_obs_predictor.pt" \
  --obs-predictor-target-mode non_cable_latent \
  --obs-period 2 \
  --cable-latent-predictor \
  --cable-latent-predictor-ckpt "$RUN/ckpt_best_l8_cable_latent_predictor.pt" \
  --cable-latent-use-rope-markers \
  --rope-marker-feature-source site \
  --summary-out "$OUT/summary.json" \
  2>&1 | tee "$OUT/eval.log"
```

Testing policy:

- First run the `site` marker version above. It checks the model and predictor
  pipeline without vision detection errors.
- Only after the `site` smoke is sane, repeat with
  `--rope-marker-feature-source opencv_rgbd` and vision flags for real/sim2real
  stress testing.
- Keep `--obs-period 2`; changing it changes the ablation being tested.
- Do not pair this policy with `latest` or `final` predictors unless the test is
  explicitly a checkpoint-selection diagnostic.

## Paper Main Figure and Ablation Plan (Frozen 2026-07-09)

This section is the current planning source of truth for paper-facing descent
experiments. It supersedes the older mixed-factor five-level ladder as the main
plot design, while keeping older results as supporting or diagnostic evidence.

### Main Comparison Figure: Confirmed

The main comparison figure should use the controlled-variable descent ladder:

- x-axis: C1, C2, C3, C4.
- methods: `PID`, `Damped-PD`, `MPC`, `Residual RL Full8D`.
- primary y-axis: broad success, where broad success is strict insertion plus
  lucky insertion.
- strict success: show as a marker, annotation, or companion table, because
  classical controllers can look acceptable under broad success while having
  near-zero clean insertion.
- C0 is optional supplement only. It is useful for sanity checking but is too
  easy/saturated for the headline figure.

Canonical result sources:

- `test_results/paper_controlled_ladder/formal_interval_20260705_interval_wind_n512/formal_interval_current_results.tex`
- `test_results/paper_controlled_ladder/formal_interval_20260705_interval_wind_n512/formal_interval_summary.md`
- `test_results/paper_final_experiment_tables_20260705.tex`

Current n=512 headline values, broad / strict success:

| Level | Added factor | PID | Damped-PD | MPC | Residual RL Full8D |
|---|---|---:|---:|---:|---:|
| C1 | Initial randomization | 58.8 / 0.4 | 65.6 / 0.8 | 83.0 / 72.5 | 100.0 / 100.0 |
| C2 | 10-segment rope | 25.8 / 0.4 | 24.0 / 0.4 | 0.0 / 0.0 | 100.0 / 100.0 |
| C3 | High wind, 6-10 m/s | 25.2 / 0.8 | 24.0 / 0.4 | 0.0 / 0.0 | 92.8 / 92.0 |
| C4 | Ratio-4 precision | 1.2 / 0.0 | 0.4 / 0.0 | 0.0 / 0.0 | 77.3 / 68.2 |

Supporting panels should report failure composition and stability:

- C4 termination composition: strict, lucky, timeout, early stop, instability,
  stuck-on-rebar.
- Stability: average / p95 swing kinetic energy and swing angle.
- Do not redesign the main figure unless a later formal re-run invalidates one
  of the C1-C4 rows.

### What Recent Work Has Already Established

- The controlled-variable n=512 main comparison is complete enough for the
  current main plot. Residual RL is the only method that remains strong across
  10-segment rope, high wind, and precision insertion.
- Cable information is already validated as important. In wind-bin tests,
  removing cable latent sharply increases stuck-on-rebar failures once wind
  exceeds about 4 m/s.
- Trainable 8D CableEncoder is promoted to the current true-observation
  mainline. Full16D remains useful evidence, but 8D is better in the high-wind
  bins and has lower swing energy/angle.
- The C4 high-wind ablation is complete for Full8D, NoCable, Full16D, and
  MLP8D. Full8D has the best strict success and stability among the main
  recurrent cable-latent variants.
- The `5Hz observation / 10Hz control` branch with ObsPred + CableLatPred has a
  completed 8M PPO run and a selected L8 checkpoint triplet. It is ready for
  smoke testing and then formal evaluation.
- The independent `5Hz observation / 5Hz control` true-observation branch
  completed 8M steps, but the final log reports `sr=0.000` and there is no
  `ckpt_best_l8_meta.json`. Treat it as a negative/diagnostic result until a
  direct eval confirms otherwise.
- Existing no-PID / pure-RL attempts also show `sr=0.000` in their logs and did
  not produce a usable L8-best row. This is already negative evidence, but the
  paper table still needs either one clean final confirmation or a clear
  fail-to-train statement.

### Ablation Block 1: Cable Latent Capacity

Question: how much compact cable-state information is needed for robust
insertion, and where does extra latent capacity stop helping?

Fairness rule: all nonzero rows should use the same PPO/LSTM architecture,
PID residual base, reward, curriculum, train budget, seed convention, true
observation setting, and trainable CableEncoder. Legacy frozen 32D can be
reported only as a historical supplement unless a matched trainable 32D row is
run.

| Variant | Current status | Next action |
|---|---|---|
| No cable latent | Done in wind-bin and C4 ablation evidence | Consolidate into final cable-latent table |
| 2D cable latent | Missing | Train from scratch with matched 8M/L8-best rule, then C4 + wind-bin eval |
| 4D cable latent | Missing | Train from scratch with matched 8M/L8-best rule, then C4 + wind-bin eval |
| 8D cable latent | Current mainline, done | Use as anchor row |
| 16D cable latent | Done as supporting capacity row | Reuse existing L8-best evidence; re-run only if table formatting needs exact matching |
| 32D cable latent | Legacy frozen 32D exists historically; fair trainable 32D row missing | Either run trainable 32D or label frozen 32D as historical, not capacity-matched |

Minimum report:

- C4 broad / strict success and termination composition.
- Wind bins `0-2`, `2-4`, `4-6`, `6-8`, `8-10` m/s.
- Stuck-on-rebar count, swing kinetic energy, swing angle, and average steps.

### Ablation Block 2: Residual RL Structure

Question: is the hybrid structure necessary, or can either the classical
controller or pure RL solve the insertion task alone?

| Variant | Definition | Current status | Next action |
|---|---|---|---|
| Residual RL | PID base + PPO residual, trainable 8D CableEncoder | Mainline complete | Use current C1-C4 and C4/wind evidence |
| Pure PID | Classical PID without learned residual | Completed as baseline in controlled ladder | Keep as the non-learning controller row; Damped-PD and MPC remain supporting baselines |
| Pure RL | PID base disabled, full-action PPO/no-PID | Multiple attempts have `sr=0.000`; no usable L8-best | Run one clean final confirmation or document fail-to-train with existing logs |

Minimum report:

- C1-C4 broad / strict success for `Residual`, `Pure PID`, and `Pure RL`.
- If pure RL does not reach L8 under the matched budget, record it as
  fail-to-train instead of extending the budget only for that row.
- Keep `Damped-PD` and `MPC` in the main comparison figure, but the residual
  structure ablation should focus on the three rows above.

### Ablation Block 3: World Model / Low-Frequency Observation

Question: can prediction compensate for lower observation frequency while
preserving a faster control loop?

Reference row:

- `10Hz observation / 10Hz control`: true-observation Full8D mainline, no
  ObsPred, no CableLatPred, no vision. This is the upper bound.

Planned rows:

| Variant | Definition | Current status | Next action |
|---|---|---|---|
| 5Hz obs / 5Hz control | `--control-freq-hz 5`, true observation every control step, no ObsPred | 8M run completed but final `sr=0.000`; no L8-best metadata | Audit `ckpt_best.pt` with a small eval; if still zero, record as low-frequency-control negative result |
| 5Hz obs / 10Hz control | 10Hz control with `obs_period=2`, ObsPred for hidden non-cable state, CableLatPred for cable latent | L8-selected triplet is ready | Run site-marker smoke, then formal C1-C4/C4 evaluation; only then test `opencv_rgbd` |

For the `5Hz obs / 10Hz control` row, always keep the matched triplet together:

```text
saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706/
  ckpt_best_l8.pt
  ckpt_best_l8_obs_predictor.pt
  ckpt_best_l8_cable_latent_predictor.pt
```

Report:

- broad / strict success, failure composition, and stability metrics.
- predictor hidden-step fraction, ObsPred RMSE if available, CableLatPred
  latent error if available.
- runtime overhead: `obs_pred_ms`, `clp_ms`, vision/rope-marker valid rate when
  the RGB-D path is enabled.

### Ablation Block 4: Wind-Bin Robustness

Question: where is each method's practical wind boundary, and which component
mainly reduces high-wind insertion failures?

Use the same bins unless there is a specific reason to change them:

```text
0-2, 2-4, 4-6, 6-8, 8-10 m/s
```

Existing evidence:

- NoCable vs Full16D wind bins are complete.
- Full16D vs Full8D wind bins are complete.
- C4 high-wind 8-10 m/s stress is complete for Full8D, NoCable, and MLP8D.

Next actions:

- Merge existing wind-bin evidence into one canonical table/plot.
- Add missing cable-latent capacities after 2D/4D/32D runs exist.
- Add residual-structure rows only after the pure-RL row has a clean status.
- Add world-model rows only after `5Hz/10Hz` smoke and formal eval are sane.
- Use `n=256` per bin for screening and `n=512` for the final high-wind stress
  rows that go into the paper.

### Immediate Execution Order

1. Smoke test the `5Hz obs / 10Hz control` L8 triplet with `site` rope markers.
2. If sane, run its formal C1-C4 eval; only then repeat with `opencv_rgbd`.
3. Audit `5Hz obs / 5Hz control` using `ckpt_best.pt` to decide whether it is a
   confirmed negative row.
4. Decide whether existing no-PID logs are enough for the pure-RL row; if not,
   launch one clean matched no-PID confirmation.
5. Train the missing cable-latent capacity rows: 2D, 4D, and trainable 32D.
6. Regenerate paper tables and figures from the canonical result files, keeping
   the main comparison figure fixed.

> 当前主线: **cruise** 负责 NMPC 抬升+平移到钢筋上方, **descent** 负责 PID+residual RL 对准、下降和物理插入。
> 本 README 已合并原 `Architecture.md` 的架构说明和常用指令, 作为当前唯一主文档维护。

---

## 1. 快速开始

先进入 WSL 项目目录和 conda 环境:

```bash
cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
conda activate vsdrl_env_5060
```

最常用的 pipeline 测试:

```bash
# cruise 用 expert/NMPC, descent 用 PPO residual RL
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

加入可调恒定风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

`--wind-speed` 单位是 m/s; `--wind-dir` 是弧度。`0.0` 约为 +x 方向, `1.5708` 约为 +y 方向。不写 `--wind-dir` 时使用随机方向。

---

## 2. 当前架构

### 2.1 Cruise: NMPC 抬升 + 平移

Cruise 现在合并了原来的 lift 和水平 cruise:

```text
payload 起点: (start_xy, z ~= 0.11)
    |
    | NMPC 垂直抬升
    v
(start_xy, z_cruise ~= 0.25)
    |
    | NMPC 水平平移
    v
(target_xy, z_cruise)  # 钢筋正上方
```

核心控制路径:

```python
expert.compute_delta_q_target(obs, current_q, residual_acc=res3)
```

当 `cruise-algo expert` 时, `residual_acc=None`, 即纯 NMPC/expert。
当 `cruise-algo ppo/sac` 时, residual RL 输出小幅 3D 加速度残差, 叠加在 NMPC 输出之前。

当前 NMPC 的关键优化:

- 代价函数中加入 `U_prev` 和 jerk 惩罚, 抑制控制量第一拍突变和来回翻转。
- 平滑前瞻参考点 `ref_smoothing_alpha`, 避免 waypoint 切换导致参考点跳变。
- action smoothing + rate limit, 限制 NMPC 输出小范围高速抖动。
- 终点附近 settle deadband, 在 payload 足够接近、速度和摆动都较小时压掉微小控制量。
- pipeline 中的 cruise expert 已与单独 `--phase cruise --algo expert` 对齐, 不再使用 pipeline-only controller/ee_control 覆盖。

### 2.2 Descent: PID base + residual RL 插入

Descent 从钢筋上方开始:

```text
payload 起点: (target_xy + small noise, z_cruise)
    |
    | PID base 控制下降和对准
    | PPO/SAC residual RL 修正 xy/z
    v
物理插入钢筋
```

成功判定已从“只到达钢筋位置”改为更接近真实任务:

- xy、z、tilt、yaw 等指标在容差内;
- 并且检测到训练 reward 使用的物理插入/地面接触成功条件;
- 若 payload 卡在钢筋上且不再产生有效插入进展, 会提前判负。

这保证渲染里能看到吊装物真正插入钢筋, 而不是停在钢筋上方。

### 2.3 Pipeline

Pipeline 当前是两阶段:

```text
cruise:  NMPC/expert 或 NMPC + residual RL
handoff: 到达钢筋上方后切换
descent: PID + residual RL 或纯 expert
```

Pipeline 会禁用 3D 轨迹里的主动下降段:

```python
config["planning"]["disable_descent_segment"] = True
```

也就是说, cruise 的目标就是“稳定到达钢筋上方”, 下降和插入由 descent 阶段接管。

---

## 3. Reward 与 RL 设计

### 3.1 Cruise residual RL

Cruise residual RL 的目标不是替代 NMPC, 而是在 NMPC 出现风扰、小幅抖动、控制环效应时做小幅补偿。

观测中包含:

- 常规 EE / payload 状态;
- NMPC 当前 base action;
- 240 维绳索观测, 经过 cable encoder;
- 风力观测。

Reward 重点:

- `swing_energy_penalty`: 抑制 payload 摆动动能。
- `cable_ke_penalty`: 抑制绳索高频振动。
- `swing_improve`: 奖励每一步相对上一时刻的消摆改进。
- `action_rms_free`: 给小 residual 一个免费区间, 让 RL 有空间介入。
- `loop_counter_reward`: 当 NMPC base action 翻转/抖动时, 奖励 residual 反向抵消。
- `rel_vel_damping_reward`: 奖励 residual 阻尼 payload 和 EE 的相对速度。
- `loop_jitter_penalty`: 显式惩罚 NMPC base action 的抖动。

碰撞障碍物只给很小惩罚, 不再把 collision reward 与高度强绑定。原因是 cruise collision 多数是 base NMPC 轨迹/控制表现导致, 不希望 RL 学成“为了避免碰撞强行改变高度或破坏稳定性”。

### 3.2 Descent residual RL

Descent 的设计保持当前成功版本:

- PID 提供稳定下降和粗对准;
- residual RL 保持足够控制权威, 不随 PID 收敛而消失;
- reward 主要关注插入误差、xy 精度、z 进度、姿态稳定、物理插入;
- 当前 descent 测试表现很好, 后续默认不要轻易改 reward 和成功判定。

### 3.3 课程学习

Cruise 使用连续风力课程, 从极小风力开始逐步增加, 让 residual RL 先学“在近似无扰下不破坏 NMPC”, 再学“有风时消摆和阻尼”。

Descent 使用精度固定的课程, 重点保持最终 5mm 对准能力, 风力逐步增强。

### 3.4 历史方案: descent 5档难度分档

2026-07-09 状态: 本 L1-L5 mixed-factor 方案只保留为历史参考和
supplement 说明。paper 主对比图已经冻结为上文的 C1-C4
controlled-variable ladder。后续正式主图、主表和新增 ablation 不再用
本节作为实验设计源头。

原计划中, traditional expert 对比和 residual RL 必要性实验统一使用下面
5 档递进任务。风力上限固定为 `8 m/s`, 正式 ladder 中只使用
`0/2/4/6 m/s`; 不再用 `10 m/s` 或更高风速作为主实验难度来源。钢筋半径按当前配置 `2.5 mm` 计算, 即钢筋直径 `5 mm`。表里的孔径是 paper 分档的 nominal gate/传统 expert 化简目标; 对成熟 pretrained RL 主链路, 不直接改 XML 的 socket 物理孔径。

| 档位 | 实验目的 | 钢筋数 | nominal gate | gate/钢筋直径 | 绳索 | 风速 | 初始扰动建议 | 成功判据建议 |
|------|----------|--------|------|----------------|------|------|--------------|--------------|
| L1 | 基础可完成, 保持4维rebar误差语义一致 | 4 | 46 mm | 9.2 | 4 段 | 0 m/s | `init_xy=10 mm`, `init_tilt=0` | `xy_tol=20 mm`, yaw 宽松, `tilt_tol=0.90` |
| L2 | 引入四孔几何和 yaw/tilt 耦合 | 4 | 30 mm | 6.0 | 6 段 | 0 m/s | `init_xy=12 mm`, `init_tilt=0.002` | `xy_tol=12 mm`, `yaw_tol=0.70`, `tilt_tol=0.30` |
| L3 | 中等精度 + 轻风扰动 | 4 | 24 mm | 4.8 | 6 段 | 2 m/s | `init_xy=14 mm`, `init_tilt=0.002` | `xy_tol=10 mm`, `yaw_tol=0.35`, `tilt_tol=0.18` |
| L4 | 默认柔性复杂度 + 中等风扰 | 4 | 24 mm | 4.8 | 10 段 | 4 m/s | `init_xy=14-16 mm`, `init_tilt=0.003` | `xy_tol=10 mm`, `yaw_tol=0.35`, `tilt_tol=0.18` |
| L5 | 高精度 real-like 压力档 | 4 | 20 mm | 4.0 | 10 段 | 6 m/s | `init_xy=16 mm`, `init_tilt=0.004` | `xy_tol=6-8 mm`, `yaw_tol=0.20`, `tilt_tol=0.12` |

分档原则:

- 所有 paper ladder 档位沿用成熟训练/测试的任务位置: `default_target_xy=[-0.30, 0.20]`, `default_start_xy=[0.30, 0.15]`, `payload_z_cruise=0.25`, `descent max_steps=300`。这组位置与 RGB-D 相机 `camera_lookat` 和已有 vision checkpoint 对齐, 不要为了简化任务把目标点移出相机工作区。
- 所有 paper ladder 档位的 payload mass 保持成熟训练默认 `4.7 kg`。低难度只通过 nominal gate、绳索段数/阻尼、风力和评价 tolerance 简化, 不额外改变 payload 动力学分布。
- 不把 10 段绳索放在最前面。实测 10 段 24 mm 无风下 PID/PD 已经很弱, 过早引入会让 traditional expert 直接贴近地板, 不利于观察逐步下降。
- 所有档位保持四孔/四钢筋几何, L1 通过 nominal 大 gate 和简化绳索降低难度。这样 residual RL 的 `rebar_errors[4]` 输入在五档中语义一致, 避免一根钢筋时 `[e1,0,0,0]` 造成 OOD。
- L3/L4/L5 的风力分别固定为 `2/4/6 m/s`, 难度主要来自 nominal gate 精度、四孔姿态耦合和绳索柔性复杂度, 而不是极端风力。
- L5 不把默认 14 mm 物理孔径直接作为 traditional expert 的主实验终点。默认孔径对 traditional expert 过难, 容易全 0; 主实验终点按 `20 mm nominal gate + 10 段 + 6 m/s + 更严格 tolerance` 设计, 更适合作为 residual RL 必要性的边界任务。
- 每档 traditional expert 允许在固定 preset grid 内选该档最优超参; 如果充分调参后仍随档位明显下降, 论文论证会更强。
- 初步开发每档至少 `50 episodes`; 论文表格建议每档 `100 episodes`, 并报告 SR、平均插装误差、worst rebar error、timeout/early_stop/instability 终止原因。

Traditional expert baseline 当前定位:

- `PID`: virtual-EE velocity + IK, 使用 payload XY 误差、payload 速度、摆偏、相对速度和积分项生成横向速度; 下降由 XY/姿态 gate 控制; yaw 使用带角速度阻尼的 P/D 控制。
- `Damped-PD`: 与 PID 同结构, 但去掉积分项, 强化速度阻尼和摆动抑制。
- `MPC`: 轻量 shooting MPC, 枚举候选 XY 速度并用简化摆模型预测 `N` 步; yaw/下降 gate 仍是轻量反馈控制, 不引入复杂 NMPC/MPPI。
- 为避免 L1 baseline 过弱但又不把 expert 做成 SOTA, 当前只加入低复杂度工程增强: terminal XY alignment、terminal yaw boost、基于四孔几何的姿态下降 gate。固定 preset 仍为 PID `2`, Damped-PD `4`, MPC `1`。

### 3.5 Sim2Real 完整评测 pipeline 检查点

- 成熟 residual RL 性能评测使用 `ckpt_best.pt` + `ckpt_best_cable_latent_predictor.pt`, 不使用 `ckpt_latest` 临时代替。
- 主性能链路应为 `--vision --vision-period 1 --cable-latent-predictor --cable-latent-use-rope-markers --rope-marker-feature-source opencv_rgbd`, 并保持 `--wind-speed-max 10` 作为已训练策略的 wind obs 归一化。论文任务实际风速仍按五档限制在 `0/0/2/4/6 m/s`。
- 当前成熟高成功率链路没有启用 ObsPred: `observation_predictor/enabled=False`, `obs_pred_ms=0`。`--obs-predictor` 只用于后续单独 ablation/real 补偿实验, 不应混入主 ladder 性能评测。
- `--vision-shadow-eval` 只用于相机诊断。它会把 payload vision 设为 shadow-only, 不能代表真实 deploy pipeline 的成功率。
- 不要在 paper profile 里把 `scene.n_obstacles` 设为 `0`。env 初始化时它决定 raw obs 里的 obstacle 槽位数量; 成熟 policy 依赖默认 3 个槽位, 下游 `OBS_EE_Z/OBS_PL_Z/OBS_CABLE_START` 是固定索引。需要无障碍物任务时使用 `--obstacles 0`, 让 env 仍按 3 个槽位构造、实际生成 0 个障碍物。
- 不要为成熟 pretrained RL 主链路改 `insertion.xy_tolerance_train_end/success_z_tolerance/tilt_tolerance/yaw_tolerance`。这些值会进入 `build_descent_obs` 的 `insertion_state` 归一化, 改了会让 policy 输入 OOD。paper-specific gate 应单独记录或用于传统 expert/重新训练任务。
- 不要为成熟 pretrained RL 主链路改 `prefab.socket_hole_size` 或 payload mass。当前高成功率 checkpoint 是在默认 `14 mm` 物理孔、`4.7 kg` payload、默认四孔/四钢筋布局上训练/测试的; 若论文最终要评估物理孔径 ladder, 需要单独 finetune 或作为 traditional expert 的简化对比任务。
- 论文 ladder 的五档任务都保持 4 个孔和 4 根钢筋。不要再用一根钢筋加补零做低难度档, 否则 `rebar_errors[4]` 的后 3 维会变成训练外语义。
- 任务 XY 和高度必须保持在训练相机工作区: target 在 `[-0.30, 0.20]` 附近, descent 起始高度 `payload_z_cruise=0.25`。RGB-D payload vision 和 rope marker 视野都是围绕该区域调好的。
- `rope_markers` 默认按绳索归一化弧长放置: 当前 5 个 marker 位于整根绳索的 `0.555/0.655/0.755/0.855/0.955` 处。因此在 4/6/10 段绳索下 marker 的物理位置一致, 但绳索动力学和 raw cable 分布仍会随段数变化。
- raw cable 观测保持固定布局: `4 根绳 × 10 槽位 × 6 维 = 240`。少于 10 段时每根绳只填真实段, 后续槽位补零, 不允许把下一根绳的数据紧凑拼到前一根绳后面。
- CableLatPred 的 deployable 输入为 `core(41) + wind(3) + rope_marker_features(80) = 124`; marker feature 顺序固定为 `CABLE_NAMES` 和 marker index, 每个 marker 是 `[rel_x, rel_y, rel_z, valid_mask]`。
- ObsPred 的 `measurement_period_steps=2` 表示每 2 个控制步才返回一次真实观测, 中间步由预测补偿。画图和实验说明里应把它画成低频补偿分支, 不要画成每一步都重新观测；但它不是当前成熟 ladder 主性能链路的一部分。
- Payload AprilTag/OpenCV vision 和 rope marker vision 是两条路径。完整 sim2real 评测前先看 `Vision valid` 和 `rope_marker_valid_rate`; 如果 payload vision 失效, 不能直接把成功率下降归因给 RL 策略。

---

## 4. 日志与 W&B 指标

为了能看出 RL 是否真的改善移动过程稳定性, 训练日志窗口已加大:

- `train.log_smooth_win = 200`
- `train.log_trend_windows = [50, 200, 500]`
- `train.log_baseline_episodes = 50`

重点关注这些指标:

```text
stab/{phase}/avg_ke_mJ
stab/{phase}/max_ke_mJ
stab/{phase}/p95_ke_mJ
stab/{phase}/integral_ke_mJs
stab/{phase}/avg_angle_deg
stab/{phase}/max_angle_deg
stab/{phase}/p95_angle_deg
stab/{phase}/cable_ke_peak
stab/{phase}/cable_ke_avg
stab/{phase}/pl_vel_peak
stab/{phase}/avg_acc
stab/{phase}/max_acc
stab/{phase}/rl_action_mag_mean
stab/{phase}/rl_action_mag_peak
```

Cruise reward 还会记录:

```text
cruise/rew/action_rms_norm
cruise/rew/loop_counter_reward
cruise/rew/rel_vel_damping_reward
cruise/rew/loop_jitter_penalty
cruise/rew/swing_energy_penalty
cruise/rew/cable_ke_penalty
```

判断 cruise residual RL 是否有效时, 不只看平均 reward, 更要看 `max/p95 swing angle`, `integral_ke_mJs`, `cable_ke_peak`, `pl_vel_peak` 是否随训练下降。

---

## 5. 常用命令

### 5.1 RL 训练

Cruise PPO:

```bash
python train_phase.py --phase cruise \
  --algo ppo \
  --n-envs 8 \
  --timesteps 2500000 \
  --log-dir saves/cruise_ppo_next
```

Descent PPO:

```bash
python train_phase.py --phase descent \
  --algo ppo \
  --n-envs 8 \
  --timesteps 3000000 \
  --log-dir saves/descent_ppo_next
```

注意: 当前成功的 descent checkpoint 在 `saves/descent_ppo/`, 不要无意中覆盖。新训练建议写入 `saves/descent_ppo_next` 或其他新目录。

### 5.2 单阶段测试

Cruise 纯 expert/NMPC:

```bash
python test_phase.py --phase cruise \
  --algo expert \
  --episodes 30 --render --wait-for-space
```

Cruise residual RL:

```bash
python test_phase.py --phase cruise \
  --algo ppo \
  --ckpt saves/cruise_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space
```

Descent residual RL:

```bash
python test_phase.py --phase descent \
  --algo ppo \
  --ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space
```

Descent 纯 expert/PID:

```bash
python test_phase.py --phase descent \
  --algo expert \
  --episodes 30 --render --wait-for-space
```

### 5.3 阶段风力测试

Cruise expert 加风:

```bash
python test_phase.py --phase cruise \
  --algo expert \
  --episodes 30 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

Cruise RL 加风:

```bash
python test_phase.py --phase cruise \
  --algo ppo \
  --ckpt saves/cruise_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

Descent RL 加风:

```bash
python test_phase.py --phase descent \
  --algo ppo \
  --ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 30 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

风力扫描:

```bash
for wind in 0 2 4 6 8; do
  python test_phase.py --phase descent \
    --algo ppo \
    --ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 30 \
    --wind-speed $wind
done
```

### 5.4 Pipeline 测试

Cruise expert + descent RL:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

Cruise expert + descent RL + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

全程 RL, 即 cruise residual RL + descent residual RL:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo ppo \
  --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space
```

全程 RL + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo ppo \
  --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

纯 expert 全流程 + 风力:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo expert \
  --episodes 10 --render --wait-for-space \
  --wind-speed 8 --wind-dir 0.0
```

Pipeline 风力扫描, cruise expert + descent RL:

```bash
for wind in 0 2 4 6 8; do
  python test_phase.py --phase pipeline \
    --cruise-algo expert \
    --descent-algo ppo \
    --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 10 \
    --wind-speed $wind
done
```

Pipeline 风力扫描, 全程 RL:

```bash
for wind in 0 2 4 6 8; do
  python test_phase.py --phase pipeline \
    --cruise-algo ppo \
    --cruise-ckpt saves/cruise_ppo/ckpt_latest.pt \
    --descent-algo ppo \
    --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
    --episodes 10 \
    --wind-speed $wind
done
```

### 5.5 可视化暂停

只有在命令中显式加入下面参数时才会暂停:

```bash
--wait-for-space
```

必须同时开 `--render`。每个 episode 初始化和第一帧同步后, 点击 MuJoCo 渲染窗口并按 `Space` 才开始执行。默认不等待, 保持原来的自动运行行为。

### 5.6 插入目标和 cruise 高度调节

Pipeline 默认使用:

```text
--pipeline-cruise-z 0.25
--pipeline-insert-target-z 0.10
--pipeline-insert-depth-min 0.025
```

测试阶段不再追加独立收尾控制段; 若要观察更深插入, 应调低训练/测试共用的目标 z, 例如:

```bash
python test_phase.py --phase pipeline \
  --cruise-algo expert \
  --descent-algo ppo \
  --descent-ckpt saves/descent_ppo/ckpt_latest.pt \
  --episodes 10 --render --wait-for-space \
  --pipeline-insert-target-z 0.09
```

---

## 6. 主要文件

```text
config.py             # 全局配置: controller/reward/curriculum/train/test
controller.py         # JointSpaceExpert, NMPCTrajectoryTracker, NMPCController4D
ee_acc_controller.py  # EE acceleration controller, cruise z/yaw PID, swing damping
mujoco_env_new.py     # MuJoCo 环境、路径生成、风力、接触、物理 step
phase_agent.py        # PPO/SAC agent, cable encoder, phase obs 构建
phase_reward.py       # cruise/descent reward 和 success 判定
train_phase.py        # 训练入口、reset_for_phase、课程、W&B logging
test_phase.py         # 单阶段测试、pipeline 测试、渲染暂停、风力测试
vec_env.py            # 并行环境, 与单环境训练路径保持一致
stability_metrics.py  # 稳定性指标统计
```

---

## 7. 最近关键修改记录

| 模块 | 修改 |
|------|------|
| `controller.py` | NMPC 加入 U_prev jerk 惩罚、参考点低通、action rate limit、终点 settle deadband。 |
| `test_phase.py` | pipeline cruise expert 与单独 cruise expert 对齐, 删除 pipeline-only controller 覆盖。 |
| `test_phase.py` | 新增 `--wait-for-space`, 渲染初始化后按空格开始。 |
| `phase_reward.py` | cruise reward 新增 base action 相关项: loop counter, relative velocity damping, loop jitter。 |
| `phase_reward.py` | descent 成功判定保留物理插入/地面接触逻辑。 |
| `phase_agent.py` | cruise 观测包含 NMPC action、绳索、风力; 支持 residual RL。 |
| `train_phase.py` | W&B 平均窗口和趋势窗口增大, 更容易观察稳定性指标改善。 |
| `vec_env.py` | 并行训练路径同步 cruise NMPC residual 和 reward 所需 base action。 |

---

## 8. 使用建议

1. 改 cruise NMPC 前, 先跑:

```bash
python test_phase.py --phase cruise --algo expert --episodes 30 --render
```

如果单段 cruise expert 表现好, pipeline 表现差, 优先检查 pipeline 是否又引入了额外 controller 覆盖或 handoff 条件过严。

2. 改 descent reward 前, 先保护当前成功 ckpt:

```bash
cp -r saves/descent_ppo saves/descent_ppo_backup
```

Descent 当前表现很好, 默认只做测试和小范围可视化参数调整。

3. 观察 RL 是否有帮助时, 不只看 SR。Cruise 尤其要看:

```text
p95/max swing angle
integral_ke_mJs
cable_ke_peak
pl_vel_peak
loop_counter_reward
rel_vel_damping_reward
```

4. 如果 VS Code/WSL 出问题, 最稳的打开方式是:

```bash
cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
code .
```
