

#### 训练时指定扰动配置

```bash
# 完整扰动 (默认)
python learn.py --seed 1 --enable_init_rand 1 --enable_process_noise 1

# 仅初始化扰动
python learn.py --seed 1 --enable_init_rand 1 --enable_process_noise 0

# 仅过程噪声
python learn.py --seed 1 --enable_init_rand 0 --enable_process_noise 1

# 无扰动 (理想情况)
python learn.py --seed 1 --enable_init_rand 0 --enable_process_noise 0
```

日志会自动保存到不同目录:
```
saves/nmpc_experiment/initRand_procNoise/seed_1/     # 完整扰动
saves/nmpc_experiment/initRand_noProcNoise/seed_1/  # 仅初始化
saves/nmpc_experiment/noInitRand_procNoise/seed_1/  # 仅过程噪声
saves/nmpc_experiment/noInitRand_noProcNoise/seed_1/# 无扰动
```

#### 测试时指定扰动配置

```bash
# 测试 Actor 模型
## 无渲染
mjpython test.py --mode actor \
    --dir saves/nmpc_experiment/initRand_procNoise/seed_1 \
    --enable_init_rand 1 --enable_process_noise 1 --episodes 100

# 注意：因为你关闭了过程噪声，所以模型保存在 initRand_noProcNoise 文件夹下
python test.py --mode actor --dir saves/nmpc_experiment/initRand_noProcNoise/seed_1 --enable_init_rand 1 --enable_process_noise 0 --episodes 100

## 有渲染
mjpython test.py --mode actor \
    --dir saves/nmpc_experiment/initRand_procNoise/seed_1 \
    --enable_init_rand 1 --enable_process_noise 1 --render --episodes 10


python test.py --mode actor --dir saves/nmpc_experiment/initRand_noProcNoise/seed_1 --enable_init_rand 1 --enable_process_noise 0 --render --episodes 10

# 测试 NMPC 基线
## 无渲染
mjpython test.py --mode base --enable_init_rand 1 --enable_process_noise 1 --episodes 100

## 有渲染
mjpython test.py --mode base --enable_init_rand 1 --enable_process_noise 1 --render --episodes 10
python test.py --mode base --enable_init_rand 1 --enable_process_noise 1 --render --episodes 10

```

#### 运行对比实验

```bash
# 自动测试 4 种配置并生成对比报告
python test_disturbance_separation.py
```

```
