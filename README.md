# InvAD Flow：AI 接手与运行手册

本文是本项目的入口文档，目标是让新的 AI 或工程师无需阅读历史对话即可理解设计、约束、当前状态和下一步工作。

## 1. 项目目标与不可违反的边界

本项目将原 InvAD 的 DDPM/DDIM 反演替换为：

> 冻结 EfficientNet-B4 特征提取器 + OT-CFM 连续速度场 + 原生几何异常算子

核心目标是减少推理 NFE，同时保留或提升 MVTec-AD、VisA 的图像级和像素级异常检测性能。

必须遵守：

- 当前项目根目录：本仓库根目录
- 原始项目：本地环境中的 `invad_v1`（不随本仓库上传）
- **严禁修改、覆盖、清理或在 `invad_v1` 中生成文件。** 它只能作为只读参考。
- 训练数据只能使用正常样本；评测标签和 mask 只能用于最终指标计算，不能进入异常分数计算。
- Flow 方向固定为 `x0=Gaussian noise, t=0` 到 `x1=normal feature, t=1`。不要反转方向。
- 不要重新引入 DDPM scheduler、DDIM inversion 或旧版 `max-min+sum` 打分。

## 2. 当前实现概览

```text
正常训练图像
  -> 冻结 EfficientNet-B4（FP32）
  -> 多层特征对齐并拼接为 [272, 16, 16]
  -> 特征以 FP16 缓存；逐类别 mean/std 以 FP32 缓存
  -> 逐类别标准化
  -> OT-CFM 构造 (t, x_t, u_t)
  -> DiT 回归速度 v_theta(x_t, t)
  -> MSE(v_theta, u_t)

测试图像
  -> 同一个冻结 backbone 和同一份缓存统计量
  -> 确定性 Gaussian anchor（由 seed + 文件名生成）
  -> Angular 1-NFE 或 Curvature 2-NFE
  -> FP32 双线性插值 + Gaussian blur + Top-K pooling
  -> anomaly map、image score、最终指标
```

### OT-CFM 训练约定

- 使用 `torchcfm.conditional_flow_matching.ExactOptimalTransportConditionalFlowMatcher`。
- `flow.sigma` 必须为 `0.0`，对应直线 OT-CFM。
- DiT 接收连续 `t in [0,1]`，`FlowDiT` 内部映射为 `t * time_scale`；默认 `time_scale=999`，用于复用原 DiT 时间嵌入尺度。
- DiT 允许从零初始化；当前输出层沿用 DiT 的零初始化，因此第 0 步 `norm(v_theta)=0`、`cos=0` 是预期现象。
- 当前 `class_conditioned=false`：DiT 不使用类别 embedding，但特征标准化仍然是逐类别的，因此类别编号与缓存必须一致。

### 异常算子

- `angular`：在插值轨迹探测点比较模型速度和 anchor→feature 位移方向，输出余弦距离；严格 1-NFE。
- `curvature`：在两个时间点计算速度变化率；严格 2-NFE。
- `both`：复用第一次速度前向，同时产生两个分数，总计 2-NFE。
- 只有 DiT 前向使用 BF16/FP16 autocast。余弦、曲率、插值、滤波和 Top-K 必须保持 FP32，避免分数排序被低精度量化。

## 3. 目录与文件职责

```text
invad_flow/
├── cache_features.py          # 单卡提取训练正常特征，计算统计量并执行硬校验
├── inspect_cache.py           # 独立复检已有缓存，不运行 backbone
├── train.py                   # 单卡/DDP OT-CFM 训练、EMA、诊断和 checkpoint
├── verify_fidelity.py         # 仅用正常验证特征进行 5/20 步 Euler 保真度检查
├── eval.py                    # Angular/Curvature 评测与 raw score 保存
├── run_ddp_smoke.sh           # 固定 GPU 4,5,6,7 的 5-epoch MVTec smoke launcher
├── configs/
│   ├── mvtec_flow.yml
│   └── visa_flow.yml
├── src/
│   ├── backbones/             # 冻结 EfficientNet-B4
│   ├── datasets/              # MVTec/VisA 数据读取
│   ├── models/dit.py          # DiT 主体
│   ├── flow_model.py          # 连续时间适配、构建模型、EMA、DDPM warm start
│   ├── feature_cache.py       # 缓存 Dataset、Normalizer、兼容性检查
│   ├── operators.py           # 确定性 anchor、两种算子和后处理
│   └── adeval/                # AUROC/AP/AU-PRO 指标
├── tests/test_core.py         # Normalizer、anchor、算子的核心测试
├── cache/                     # 缓存产物，不提交源码仓库
└── results/                   # checkpoint、指标、JSONL、NCCL 日志
```

不要随意修改以下契约：

- MVTec 输入 `256x256` 时，缓存特征必须为 `[N,272,16,16]`。
- 缓存 `features` 必须为 FP16；`mean/std` 必须为 FP32，形状为 `[num_classes,272,1,1]`。
- cache 中的 `labels`、`filenames`、`split` 和 `features` 第一维必须完全一致。
- checkpoint 至少包含 `model`、`ema`、`normalizer`、`config`、`feature_shape`、`epoch`、`global_step`。
- `eval.py` 默认加载 EMA；只有明确做消融时才使用 `--no-ema`。

## 4. 环境

已验证环境：

- NVIDIA H20 96 GB
- Python 3.10
- PyTorch `2.10.0+cu128`
- torchvision `0.25.0+cu128`
- torchcfm `1.0.7`
- POT `0.9.7.post1`
- BF16 可用

```bash
cd /path/to/invad_flow
conda activate invad_flow
python -m pip install torchcfm==1.0.7 POT==0.9.7.post1
python -m pip check

CUDA_VISIBLE_DEVICES=5 python - <<'PY'
import torch, torchcfm, ot
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torchcfm.__version__, ot.__version__)
PY
```

设置 `CUDA_VISIBLE_DEVICES=5` 后，物理 GPU 5 在 Python 内是 `cuda:0`。

## 5. 标准运行流程

### 5.1 MVTec 特征缓存：物理 GPU 5

```bash
cd /path/to/invad_flow
conda activate invad_flow
CUDA_VISIBLE_DEVICES=5 python cache_features.py \
  --config configs/mvtec_flow.yml

CUDA_VISIBLE_DEVICES=5 python inspect_cache.py \
  --config configs/mvtec_flow.yml
```

缓存脚本必须立即中止的情况：

- backbone 输出或 FP16 转换出现 NaN/Inf；
- 任意类别缺失训练样本；
- 任意类别的任意通道标准差为 0 或非有限值；
- 样本数、标签、文件名或 split 长度不一致；
- 特征、统计量 dtype 不符合契约。

当前 MVTec 缓存路径：`cache/mvtec_train.pt`，逐类诊断为 `cache/mvtec_train.pt.diagnostics.json`。

### 5.2 四卡 DDP smoke train

推荐直接运行：

```bash
./run_ddp_smoke.sh
```

等价核心命令：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29501 train.py \
  --config configs/mvtec_flow.yml \
  --epochs 5 \
  --log_interval 20 \
  --save_diagnostics
```

脚本还会设置 `TORCH_DISTRIBUTED_DEBUG`、`NCCL_DEBUG` 和 `TORCH_NCCL_ASYNC_ERROR_HANDLING`，将 stdout/stderr 与每个 rank 的 NCCL 日志写入 `results/mvtec_flow/diagnostics/`。

### 5.3 正式训练

移除 `--epochs 5` 即使用配置中的 `optimizer.num_epochs=300`：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29501 train.py \
  --config configs/mvtec_flow.yml \
  --log_interval 20 \
  --save_diagnostics
```

注意：

- `flow_latest.pth` 会被新训练覆盖；需要保留 smoke checkpoint 时先另存。
- 当前 `logging.save_optimizer=false`，checkpoint 约 9.2 GiB，只保存 model+EMA，不支持精确恢复 optimizer/scheduler。
- 若正式长训必须断点续训，训练前将 `save_optimizer` 改为 `true`；checkpoint 会显著增大。
- Rank 0 单独维护 EMA 和执行 checkpoint 保存，其余 rank 不持有 EMA 副本。

### 5.4 正常特征保真度诊断

```bash
CUDA_VISIBLE_DEVICES=5 python verify_fidelity.py \
  --config configs/mvtec_flow.yml \
  --steps 5,20
```

该脚本不读取测试异常标签，仅使用缓存中的正常 validation split。默认自动读取 `<logging.save_dir>/flow_latest.pth`，并保存 `fidelity.json`。

重点检查：

- 所有生成特征有限；
- 正常特征空间均值和 log-std 是否接近；
- 5 步与 20 步 Euler 是否一致；
- 生成特征到正常特征的 SWD 是否明显优于初始噪声。

若需要让未通过直接返回非零退出码，加 `--strict`。

### 5.5 异常检测评测

```bash
# 默认 Angular，1-NFE
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --config configs/mvtec_flow.yml \
  --save_raw_scores

# Curvature，2-NFE
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --config configs/mvtec_flow.yml \
  --operator curvature \
  --save_raw_scores

# 同时评测两个算子，共享第一次前向，总计 2-NFE
CUDA_VISIBLE_DEVICES=5 python eval.py \
  --config configs/mvtec_flow.yml \
  --operator both \
  --save_raw_scores
```

输出包含每类和宏平均的 I-AUROC、I-AP、I-F1Max、P-AUROC、P-AP、P-F1Max、AU-PRO，以及排除 DataLoader 的模型计算 FPS。

`--save_raw_scores` 保存 `[N,16,16]` 原始算子图、image score、类别、标签和文件名，方便后续排查或增加新算子。

### 5.6 VisA

将以上命令中的配置替换为 `configs/visa_flow.yml`。VisA 必须单独生成缓存和训练 checkpoint，不能复用 MVTec 的统计量或模型。

## 6. 训练诊断与阻断规则

`train.py --save_diagnostics` 将以下全局多卡均值写入 `train_metrics.jsonl`：

- `loss`：`MSE(v_theta, u_t)`；
- `norm_u_t`、`norm_v_theta`；
- `cos_sim`；
- `rms_u_t`、`rms_v_theta`；
- `global_grad_norm`：梯度裁剪前的全局范数；
- epoch、iteration、global step、learning rate。

前 100 步内：

- 如果 `cos_sim` 100 步始终小于等于 0，写入 alert；
- 如果任一步 `global_grad_norm > 1e4`，立即写入 alert；
- alert 同时记录 DiT 最后一层梯度的 min/max/mean/std/L2/finite fraction。

异常和 Python traceback 写入 `ddp_error_rank<N>.log`。端口占用、广播超时及 NCCL 通信信息还会出现在 `torchrun_smoke.log` 和 `nccl_<host>_<pid>.log`。

## 7. 当前已完成的验证状态

以下是 2026-09-11 的 **5-epoch smoke 结果，不是最终模型性能**：

- MVTec 缓存：3629 个唯一正常样本；train/val = 3267/362；所有检查通过。
- DDP：4 张 H20，510 steps，正常完成；checkpoint 记录 `world_size=4`。
- epoch loss：`1.9532 -> 1.6977 -> 1.3387 -> 1.1735 -> 1.0734`。
- 已记录 cos：`0.0000 -> 0.6960`；最大已记录 grad norm 为 `1.5134`。
- 无训练 alert、无 rank traceback、无 NCCL timeout。
- Fidelity：5/20 Euler 相对误差 `0.000729`，但 mean MAE `0.2097`、SWD ratio `0.9369`，整体未通过。
- Angular：1-NFE，约 `175.2 FPS`，I-AUROC `0.6705`，P-AUROC `0.4629`，AU-PRO `0.1288`。
- Curvature：2-NFE，约 `92.8 FPS`，I-AUROC `0.4460`，P-AUROC `0.4554`，AU-PRO `0.1080`。

正确解读：数据、DDP、梯度、EMA、checkpoint、Euler 和两个算子链路已经跑通；5 epochs 的 EMA 尚未学会完整正常分布，因此当前低检测指标不能用来否定或确认最终方案效果。

当前主要产物：

```text
cache/mvtec_train.pt
cache/mvtec_train.pt.diagnostics.json
results/mvtec_flow/flow_latest.pth
results/mvtec_flow/fidelity.json
results/mvtec_flow/eval_angular.json
results/mvtec_flow/eval_angular.raw_scores.npz
results/mvtec_flow/eval_curvature.json
results/mvtec_flow/eval_curvature.raw_scores.npz
results/mvtec_flow/diagnostics/train_metrics.jsonl
results/mvtec_flow/diagnostics/torchrun_smoke.log
results/mvtec_flow/diagnostics/nccl_*.log
```

## 8. 常见错误与快速排查

### CUDA 不可用

先检查 `torch.__version__`、`torch.version.cuda`、驱动支持范围和 `torch.cuda.is_available()`。本机曾因 CUDA 13 构建高于驱动支持范围导致不可用，已验证的组合是 PyTorch CUDA 12.8。

### 第一步 cos 为 0

DiT 最终线性层零初始化导致第一步速度为 0，这是预期行为。若前 20–100 步仍不转正，再检查 flow 方向、normalizer、时间映射和梯度。

### loss 看似下降但 fidelity 不改善

依次检查：

1. 是否错误反转 `x0/x1`；
2. 训练与评测是否使用同一份逐类别统计量；
3. 类别编号是否与缓存一致；
4. 是否错误使用未训练 model 而非 EMA；
5. EMA decay 是否在短训练中造成过强滞后；
6. 是否只训练了 smoke epochs。

### 像素指标极低

重点检查：

- feature map 是否保持 `16x16` 空间对应关系；
- 算子是否沿 channel 维计算，而不是把空间维一起压平；
- 是否先双线性上采样，再 Gaussian blur；
- 几何与 Top-K 是否为 FP32；
- mask 是否使用 nearest-neighbor resize；
- anomaly map 是否发生符号反转或每图错误归一化。

### DDP 卡住或退出

- 启动前检查 `ss -ltn | grep 29501`；
- 检查四张卡是否空闲；
- 查看 `ddp_error_rank<N>.log`、`torchrun_smoke.log` 和四份 NCCL 日志；
- `NCCL INFO ... Abort COMPLETE` 在正常销毁 communicator 时可以出现，不等同于训练失败；应结合 torchrun 退出码和 traceback 判断。

## 9. 修改代码后的最低验证要求

每次修改训练、算子、normalizer 或缓存格式后至少执行：

```bash
python -m compileall -q cache_features.py inspect_cache.py train.py \
  verify_fidelity.py eval.py src
python -m pytest -q tests/test_core.py
ruff check cache_features.py inspect_cache.py train.py verify_fidelity.py \
  eval.py src/feature_cache.py src/flow_model.py src/operators.py tests/test_core.py
```

若 `invad_flow` 环境没有 pytest，可以在已安装 pytest 且依赖兼容的环境执行测试，但真实 CUDA 前向、缓存、训练和评测必须回到 `invad_flow` 环境。

## 10. 推荐的下一步

1. 决定是否保留当前 smoke checkpoint；正式训练会覆盖 `flow_latest.pth`。
2. 将 `save_optimizer=true` 后启动 MVTec 300-epoch DDP 正式训练，以支持可靠断点续训。
3. 定期运行正常-only fidelity；fidelity 未通过前，不要过度调 anomaly operator。
4. fidelity 通过后，固定 checkpoint，扫描 `probe_t`、`curvature_dt`、Gaussian sigma 和 Top-K fraction。
5. 完成 MVTec 后，再独立缓存和训练 VisA。
6. 新增算子应放在 `src/operators.py`，并在 `eval.py` 中复用同一 backbone、normalizer、anchor 和后处理协议。

任何 AI 接手时，应先阅读本 README、当前 config、最近的 `train_metrics.jsonl` 和 `fidelity.json`，再决定是否修改或启动长任务。
