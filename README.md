# InvAD Flow：项目接手与运行手册

本文用于让新的 AI 或工程师在没有历史对话的情况下安全接手项目。先读本文件，再改代码或启动训练。

## 1. 目标与边界

项目路径：`/data2/chenxuwu/medicalAD/invad_flow`。

目标架构：冻结 EfficientNet-B4 → 272 通道多尺度特征 → OT-CFM 连续速度场 → 原生几何异常算子。它替代原 InvAD 的 DDPM/DDIM 反演和 `max-min+sum` 打分。

硬约束：

- `/data2/chenxuwu/medicalAD/invad_v1` 只能只读参考，严禁修改、覆盖或写入产物。
- 训练只使用正常训练图；测试标签和 mask 只能参与指标计算。
- Flow 方向固定为 `x0=Gaussian noise, t=0` 到 `x1=normal feature, t=1`。
- MVTec 和 VisA 必须使用各自独立的缓存、统计量和 checkpoint。
- 当前 `class_conditioned: false` 与已检查的原 InvAD 联合类别主路径一致，不应在没有对照实验时擅自开启。
- 原 InvAD 没有对 272 通道做 z-score；本项目的逐类别 z-score 是 Flow 训练稳定化设计，不是原版行为。

## 2. 当前数据与模型契约

训练链路：

```text
normal image
  -> frozen EfficientNet-B4, FP32
  -> aligned feature [272,16,16], FP32 cache
  -> per-class FP32 mean/std normalization
  -> Exact OT-CFM: (x0, x1) -> (t, xt, ut)
  -> DiT predicts v_theta(xt,t)
  -> MSE(v_theta,ut)
```

评测链路：

```text
test image -> same FP32 backbone/normalizer
  -> shared path-independent Gaussian anchor bank
  -> Angular 1-NFE or Curvature 2-NFE per anchor
  -> average anchors (default K=1)
  -> FP32 bilinear resize -> Gaussian blur -> Top-K pooling
  -> seven AD metrics + mAD + FPS
```

关键契约：

- MVTec 256×256 输入对应 `[N,272,16,16]`。
- `features/mean/std` 全部持久化为 FP32；`mean/std` 形状为 `[classes,272,1,1]`。
- MVTec 缓存必须含 3629 个唯一正常样本，训练使用全部 3629 个。
- `split` 中保留 90/10 标记仅用于 normal-only fidelity 子集；它不再从训练中扣除 10%。
- `statistics_split: all` 表示统计量使用全部正常训练样本。
- anchor 不能依赖绝对路径、文件名或 DataLoader batch；所有测试图共享同一 anchor bank。
- 默认 `num_anchors: 1`，Angular/Curvature 分别为 1/2 NFE。增加 K 后 NFE 分别为 K/2K。
- checkpoint 内保存 `cache_contract` 与 normalizer，混用旧缓存会被拒绝。

## 3. 文件职责

```text
cache_features.py       生成原始 FP32 特征缓存并执行硬校验
inspect_cache.py        对缓存重新计数、校验 dtype/完整性/分布
train.py                单卡或 DDP OT-CFM 训练、EMA、诊断、checkpoint
verify_fidelity.py      normal-only 5/20 步 Euler 保真度检查
eval.py                 Angular/Curvature、完整指标、raw score、FPS
src/feature_cache.py    cache Dataset、normalizer、契约校验
src/operators.py        shared anchor bank、动力学算子、后处理
configs/*.yml           数据、缓存、模型、优化器、评测唯一配置源
tests/test_core.py      核心算子和 normalizer 单测
results/                checkpoint、评测 JSON、日志
```

## 4. 当前状态（2026-09-12）

第一轮 300 epoch 已完成，但它使用了旧契约：FP16 cache、3267/3629 训练样本、按文件名生成的 per-image anchor。因此该轮可用于诊断趋势，不能作为修复后的最终 Flow baseline。

第一轮产物保留在 `results/mvtec_flow/`；修复版配置写入 `results/mvtec_flow_fixed/`，两者不会相互覆盖。

第一轮 epoch 300，Angular 结果（百分制）：

| I-AUROC | I-AP | I-F1max | P-AUROC | P-AP | P-F1max | AU-PRO | mAD | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 97.36 | 99.00 | 97.27 | 96.37 | 45.46 | 50.23 | 86.12 | 81.69 | 175.7 |

参考 InvAD baseline：

| I-AUROC | I-AP | I-F1max | P-AUROC | P-AP | P-F1max | AU-PRO | mAD | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99.0 | 99.6 | 98.5 | 97.5 | 46.5 | 52.3 | 92.7 | 83.7 | 88.1 |

第一轮的主要缺口是 AU-PRO `-6.58`、P-F1max `-2.07`、I-AUROC `-1.64`、P-AUROC `-1.13`、mAD `-2.01`；速度约为参考值 2 倍。12 个 checkpoint 的指标持续上升到 epoch 300，没有出现后期回落，说明旧训练没有真正收敛到平台，也说明不能只把差距归因于单个算子超参。

已经完成的严重问题修复：

- 新 `cache/mvtec_train.pt`：3629 个唯一样本，特征与统计量均 FP32，约 965 MiB。
- 旧缓存保留为 `cache/mvtec_train_legacy_fp16_split90.pt`，没有删除。
- 训练读取 `train_split: all`，不再丢弃 10% 正常样本。
- DDP DataLoader 保留尾批，不再因 `drop_last=True` 每轮漏掉样本。
- anchor 改成共享、与路径无关的固定 bank。
- `eval.py` 输出全部 7 项指标、mAD、FPS，以及对 InvAD baseline 的百分制差值。
- `--epochs 5` 仅提前停止，学习率仍按完整 300e 计划推进，不再把 40e warmup 压成 5e。
- 末端学习率改为原 InvAD 实际使用的 `1e-6`；峰值仍为原版同值 `5e-5`。
- 新 checkpoint 默认保存 optimizer/scheduler，支持精确恢复；文件会显著变大。

旧 300e checkpoint 内是旧 normalizer。不要用新 `mvtec_train.pt` 对其做 fidelity 或 `--resume`；如需复查旧 fidelity，显式指定：

```bash
python verify_fidelity.py \
  --checkpoint results/mvtec_flow/flow_epoch_0300.pth \
  --cache cache/mvtec_train_legacy_fp16_split90.pt \
  --steps 5,20
```

## 5. 环境与缓存

已验证：NVIDIA H20、Python 3.10、PyTorch 2.10.0+cu128、torchcfm 1.0.7、POT 0.9.7.post1。

```bash
cd /data2/chenxuwu/medicalAD/invad_flow
conda activate invad_flow
python -m pip install -r requirements-flow.txt

CUDA_VISIBLE_DEVICES=5 python cache_features.py --config configs/mvtec_flow.yml
CUDA_VISIBLE_DEVICES=5 python inspect_cache.py --config configs/mvtec_flow.yml
```

缓存遇到以下情况必须失败：样本数不等于源数据集、文件名重复、NaN/Inf、零方差通道、shape/dtype 不符、类别缺失。MVTec 正确摘要应为：

```text
samples=3629
configured_training_samples=3629
feature_dtype=torch.float32
statistics_dtype=torch.float32
unique_filenames=3629
duplicate_filenames=0
feature_shape=[272,16,16]
statistics_split=all
configured_train_split=all
```

## 6. 训练

四卡正式训练：

```bash
cd /data2/chenxuwu/medicalAD/invad_flow
conda activate invad_flow
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=29501 train.py \
  --config configs/mvtec_flow.yml \
  --log_interval 20 \
  --save_diagnostics
```

smoke 只跑完整学习率计划的前 5 epoch：

```bash
torchrun --nproc_per_node=4 --master_port=29501 train.py \
  --config configs/mvtec_flow.yml --epochs 5 \
  --log_interval 20 --save_diagnostics
```

修复版配置使用独立的 `results/mvtec_flow_fixed/`，不会覆盖第一轮产物。Rank 0 独占 EMA 和 checkpoint 写入。JSONL/TensorBoard 记录 loss、`norm_u_t`、`norm_v_theta`、cosine、未裁剪 grad norm、LR。

当前优化器计划：40e 从 `1e-6` warmup 到 `5e-5`，随后 cosine 降回 `1e-6`。DiT 最终层零初始化，所以 step 0 的速度范数和 cosine 为 0 是正常现象。

## 7. Fidelity 与完整评测

```bash
CUDA_VISIBLE_DEVICES=5 python verify_fidelity.py \
  --config configs/mvtec_flow.yml --steps 5,20

CUDA_VISIBLE_DEVICES=5 python eval.py \
  --config configs/mvtec_flow.yml \
  --operator angular --save_raw_scores

CUDA_VISIBLE_DEVICES=5 python eval.py \
  --config configs/mvtec_flow.yml \
  --operator curvature --save_raw_scores
```

`eval.py` 每个算子输出：

- 每类和 15 类宏平均：I-AUROC、I-AP、I-F1Max、P-AUROC、P-AP、P-F1Max、AU-PRO；
- `mAD`：上述 7 项的算术平均；
- `summary_percent`：与 InvAD 表格同顺序的百分制 8 项 + FPS；
- `delta_vs_reference_percent`：相对配置中 reference baseline 的差值；
- FPS 口径：backbone + flow operator + FP32 后处理，不含 DataLoader 和最终指标计算。

`--save_raw_scores` 同时保存原生 16×16 算子图、最终 256×256 anomaly map、mask、image score、标签、类别和文件名，可离线复算全部指标。

FPS 只有在硬件、batch size、warmup、精度模式和计时范围一致时才能直接与论文比较。

## 8. 最低验证要求

```bash
python -m compileall -q cache_features.py inspect_cache.py train.py \
  verify_fidelity.py eval.py src tests
python -m pytest -q tests/test_core.py
ruff check cache_features.py inspect_cache.py train.py verify_fidelity.py \
  eval.py src/feature_cache.py src/operators.py tests/test_core.py
```

训练前额外确认：

```bash
python inspect_cache.py --config configs/mvtec_flow.yml
nvidia-smi
ss -ltn | grep 29501 || true
```

## 9. 下一步（按优先级）

1. 保留第一轮 `results/mvtec_flow/`；修复版只写 `results/mvtec_flow_fixed/`，不要删旧 checkpoint 或 raw scores。
2. 用修复后的 FP32/full cache 做 5e DDP smoke，确认 startup 显示 `train_samples=3629`、`train_split=all`、`schedule_epochs=300`、`stop_epoch=5`，且新 checkpoint 含 `cache_contract` 和 optimizer/scheduler。
3. smoke 只验证代码和数值健康，不比较 AD 指标。通过后立即开始新的 300e 正式训练。
4. 每 25 epoch 用 shared K=1 Angular 跑固定协议评测；以 mAD 为主，同时观察 AU-PRO，不在训练中途改 anchor/后处理。
5. 修复版正式基线锁定后，才依次做单变量实验：K=4 anchor averaging、class conditioning、逐类 vs 全局 normalization、global-batch OT pairing、probe/postprocess 参数。一次只改变一个因素。
6. MVTec 修复版锁榜后再迁移 VisA，先重建 VisA FP32/full cache，不能复用 MVTec 模型。

仍需关注但本轮未擅自改动的研究风险：DDP 每个 rank 内部独立做 OT pairing（有效 OT batch 较小）；逐类别 z-score 并非原 InvAD 设计；`class_conditioned=false` 下联合类别 Flow 是否足够；共享单 anchor 的方差。这些需要在修复后的同一工程基线上做受控验证，而不是与本轮 bug 修复混在一起。
