# Wan2.1 × VideoReward 分布内 Reward Model 验证

本目录用于回答**一个核心问题**：

> `KwaiVGI/VideoReward` 这个 reward model 在**我们自己的 Wan2.1 模型生成的视频**上，到底准不准？

只有先验证这一点，后面用 DanceGRPO/SAGE-GRPO 训练 Wan2.1 时，
reward 信号才有意义；否则训练会跑偏（reward hacking）。

---

## 1. 整体流程

```
┌────────────────────────────────────────────────────────────────┐
│ Phase A: generate.py    Wan2.1 生成视频                         │
│   20 prompts × 2 seeds = 40 个视频                              │
├────────────────────────────────────────────────────────────────┤
│ Phase B: score.py       VideoReward 自动打分                    │
│   每个视频出 VQ / MQ / TA / Overall / composite 五列            │
├────────────────────────────────────────────────────────────────┤
│ Phase C: annotate.py    出人工标注模板                          │
│   human_pointwise.csv (40 行) + human_pairwise.csv (20 对)     │
├────────────────────────────────────────────────────────────────┤
│   ===  这里停下来，开始人工标注  ===                            │
├────────────────────────────────────────────────────────────────┤
│ Phase D: accuracy.py    算最终准确率                            │
│   打分型 Spearman/Pearson  +  对比型 Pairwise Acc               │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 当前目录结构

```
posttrain/
├── DanceGRPO/                   # DanceGRPO 训练代码（参考用）
├── SAGE-GRPO/                   # SAGE-GRPO 训练代码（参考用）
└── VideoAlign/                  # VideoReward 原始仓库
    ├── inference.py             # VideoReward 推理入口（被 score.py 调用）
    ├── calc_accuracy.py         # pairwise acc 算法（被 accuracy.py 调用）
    ├── checkpoints/             # KwaiVGI/VideoReward 权重
    │   ├── model_config.json
    │   └── checkpoint-11352/
    └── wan_eval/                # ← 本目录
        ├── README.md            # 你正在看
        ├── prompts.txt          # 20 条 prompt（来自 vbench.csv）
        ├── generate.py          # Phase A
        ├── score.py             # Phase B
        ├── annotate.py          # Phase C
        ├── accuracy.py          # Phase D
        └── outputs/             # 跑完后自动生成
```

依赖（应该都装好了）：
```bash
pip install pandas scipy tqdm diffusers transformers torch decord
```

---

## 3. 默认参数概览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_seeds` (generate.py) | **2** | 每个 prompt 生成 2 个视频 |
| `--pairs_per_prompt` (annotate.py) | **1** | 每个 prompt 1 对 pair |
| Reward 权重 `(vq, mq, ta)` | **(1.0, 1.0, 1.0)** | SAGE-GRPO 代码库默认（等权平均） |

**最终标注量**：40 个视频打分 + 20 对视频对比，单人 ≈ **45 分钟**。

> **关于 reward 权重**：默认 1:1:1 跟 SAGE-GRPO 代码库默认 (`rewards.py:383`) 完全一致。
> SAGE-GRPO 论文里报的 `(0.5, 0.5, 1.0)` 是他们为 **HunyuanVideo + 特殊 SDE** 调出来的实验配置，
> 不能直接迁移到 Wan2.1。所以默认采用 1:1:1 是更稳的选择。
>
> 如果后面你训练时要改权重，比如 `(0.5, 0.5, 1.0)` 或 DanceGRPO 的 `(1, 0, 0)`，
> 在 `score.py` 命令行里加上 `--vq_coef X --mq_coef Y --ta_coef Z` 即可，
> `accuracy.py` 会自动从 `outputs/reward_scores.coefs.json` 读取一致的权重。

---

## 4. 一步一步怎么跑

### Step 1: 进目录

```bash
cd /aigc/posttrain/siyuanfu/VideoAlign/wan_eval
```

### Step 2: 生成 40 个视频（Phase A）

```bash
python generate.py \
    --prompts_file prompts.txt \
    --output_dir outputs \
    --gpu 0
    # --model_name 默认 = /aigc/posttrain/siyuanfu/models/Wan2.1 (本地路径，1.3B preset)
    # 想跑 14B：--model_name /path/to/Wan2.1-14B  (自动按 "14B" 子串切 preset)
    # 也支持 HF id：--model_name Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

**会发生：**
- 从本地路径加载 Wan2.1-T2V-1.3B
- 对 20 条 prompt 各生成 2 个视频（不同 seed）→ 共 40 个视频
- 1.3B + A100：约 **20-30 分钟**
- 中途断了可以直接重跑，**已生成的会跳过**（resume 模式）

**输出：**
```
outputs/
├── videos/
│   ├── p0000_s0.mp4    # 第 0 个 prompt，第 0 个 seed
│   ├── p0000_s1.mp4    # 第 0 个 prompt，第 1 个 seed
│   ├── p0001_s0.mp4
│   ├── p0001_s1.mp4
│   ├── ...
│   └── p0019_s1.mp4    # 第 19 个 prompt，第 1 个 seed
└── videos_meta.csv      # 元数据：video_id / prompt_id / seed / prompt / video_path
```

### Step 3: VideoReward 自动打分（Phase B）

```bash
python score.py \
    --videos_meta outputs/videos_meta.csv \
    --gpu 0
```

**会发生：**
- 加载 `../checkpoints/`（VideoReward 模型，约 7GB）
- 对 40 个视频依次算 `VQ / MQ / TA / Overall / composite`
- 约 **5-10 分钟**（A100）
- 启动时打印：`[score] composite coefs: {'vq_coef': 1.0, 'mq_coef': 1.0, 'ta_coef': 1.0}  -- SAGE-GRPO codebase default (1:1:1 averaging)`

**输出**：`outputs/reward_scores.csv`（在 `videos_meta.csv` 基础上多了 5 列）：

| video_id | prompt | ... | reward_VQ | reward_MQ | reward_TA | reward_Overall | reward_composite |
|---|---|---|---:|---:|---:|---:|---:|
| p0000_s0 | ... | ... | -0.62 | -0.41 | 1.42 | 0.39 | 0.39 |
| p0000_s1 | ... | ... | -0.48 | -0.28 | 1.51 | 0.75 | 0.75 |
| ... | | | | | | | |

> 注意：归一化后 VQ/MQ 经常是**负数**，TA 经常是正数，这是正常的（z-score 归一化导致）。
> 详见 `score.py` 的 docstring 或问 Cursor。

同时还会写一个 `outputs/reward_scores.coefs.json` 记下用了什么权重，方便 Phase D 自动对齐。

### Step 4: 生成人工标注模板（Phase C）

```bash
python annotate.py \
    --reward_scores outputs/reward_scores.csv \
    --output_dir outputs
```

**输出**两个空白 CSV：

**`outputs/human_pointwise.csv`**（打分型，**40 行**）

| video_id | prompt | video_path | human_VQ | human_MQ | human_TA |
|---|---|---|:---:|:---:|:---:|
| p0000_s0 | A stop sign... | videos/p0000_s0.mp4 | _填 1~5_ | _填 1~5_ | _填 1~5_ |
| p0000_s1 | (同上) | videos/p0000_s1.mp4 | _填 1~5_ | _填 1~5_ | _填 1~5_ |
| ... | | | | | |

**`outputs/human_pairwise.csv`**（对比型，**20 对**）

| pair_id | prompt | video_A_id | video_B_id | video_A_path | video_B_path | human_VQ | human_MQ | human_TA |
|---|---|---|---|---|---|:---:|:---:|:---:|
| 0 | A stop sign... | p0000_s0 | p0000_s1 | ... | ... | _A/B/same_ | _A/B/same_ | _A/B/same_ |
| 1 | A toilet... | p0001_s0 | p0001_s1 | ... | ... | _A/B/same_ | _A/B/same_ | _A/B/same_ |
| ... | | | | | | | | |

### Step 5: 人工标注

打开两个 CSV（Excel/WPS/Google Sheets 都行），对照 `outputs/videos/` 里的视频填空白列：

- **`human_pointwise.csv`**：每个视频看一遍，VQ/MQ/TA 各打 1~5 分（越高越好）
  - 1 = 非常差，3 = 一般，5 = 非常好
- **`human_pairwise.csv`**：每对视频对照看，每个维度选：
  - `A` = A 更好
  - `B` = B 更好
  - `same` = 差不多

> 单人约 45 分钟。建议直接在文件管理器里双击视频，跟 CSV 并排看。

### Step 6: 算最终准确率（Phase D）

标注好之后：

```bash
python accuracy.py \
    --reward_scores outputs/reward_scores.csv \
    --human_pointwise outputs/human_pointwise.csv \
    --human_pairwise  outputs/human_pairwise.csv
```

**会打印两张表 + 写一份 `outputs/accuracy_report.json`：**

```
[acc] composite coefs: {'vq_coef': 1.0, 'mq_coef': 1.0, 'ta_coef': 1.0}  -- SAGE-GRPO codebase default (1:1:1 averaging)

=== Pointwise (N videos = 40) ===
dim                                            Spearman  Pearson  Kendall  Top-1/prompt
VQ                                             0.6234    0.6010   0.4811   0.6000
MQ                                             0.5512    0.5398   0.4221   0.5500
TA                                             0.6890    0.6712   0.5489   0.7000
Overall                                        0.6701    0.6612   0.5223   0.7000
composite  *DanceGRPO signal*                  0.6789    0.6645   0.5310   0.7000   ← 训练用的就是这个

=== Pairwise (total annotated rows = 20) ===
dim                                            with_ties  without_ties  used  skipped
VQ                                             0.7000     0.7222        19    1
MQ                                             0.6500     0.6842        19    1
TA                                             0.7500     0.7778        19    1
Overall                                        0.7000     0.7222        19    1
composite  *DanceGRPO signal*                  0.7500     0.7778        19    1   ← 训练用的就是这个
```

**重点看 `composite` 行**——这就是后续训练时 GRPO 实际拿到的 reward 信号的准确率。

---

## 5. 指标怎么读

### 打分型 (Pointwise)

| 指标 | 含义 | 多少算好 |
|---|---|---|
| Spearman ρ | 模型分数和人分数的**秩相关**（最常用） | > 0.5 算可用，> 0.7 算优秀 |
| Pearson r | 模型分数和人分数的**线性相关** | 同上 |
| Kendall τ | 同秩相关，对异常值更鲁棒 | > 0.4 算可用 |
| Top-1/prompt | 模型在每个 prompt 内挑出的"最好视频"和人挑的一致的比例 | > 0.6 算可用 |

### 对比型 (Pairwise)

| 指标 | 含义 | 多少算好 |
|---|---|---|
| with_ties | 允许"差不多"的版本（VideoAlign 论文用的） | > 0.65 算可用 |
| without_ties | 只比 A vs B 决断时的准确率（更严格） | > 0.65 算可用 |

> **基线**：随机猜是 50%，所以只要明显高于 50% 就有信号。

---

## 6. 一些常用变形

### 想标注少一点（已经是最少）

默认就是最低标注量了（40 视频 + 20 对）。

### 想标注多一点（更稳的统计）

```bash
# 4 个 seed，每 prompt 抽 3 对 → 80 视频 + 60 对
python generate.py --prompts_file prompts.txt --output_dir outputs --num_seeds 4 --gpu 0
python score.py    --videos_meta outputs/videos_meta.csv --gpu 0
python annotate.py --reward_scores outputs/reward_scores.csv --output_dir outputs --pairs_per_prompt 3
# (后续标注 + accuracy.py 流程不变)
```

### 想换 reward 权重（对应不同的训练配置）

```bash
# DanceGRPO 原始风格（VQ only）
python score.py --videos_meta outputs/videos_meta.csv --vq_coef 1.0 --mq_coef 0.0 --ta_coef 0.0 --gpu 0
python accuracy.py --reward_scores outputs/reward_scores.csv ...
# accuracy.py 会自动读 coefs.json，composite 行会变成 = VQ 行

# SAGE-GRPO 论文 Setting B
python score.py --videos_meta outputs/videos_meta.csv --vq_coef 0.5 --mq_coef 0.5 --ta_coef 1.0 --gpu 0
```

### 想只跑 pointwise 或只跑 pairwise

```bash
# 只跑 pointwise
python accuracy.py --reward_scores outputs/reward_scores.csv --human_pointwise outputs/human_pointwise.csv

# 只跑 pairwise
python accuracy.py --reward_scores outputs/reward_scores.csv --human_pairwise outputs/human_pairwise.csv
```

---

## 7. 验证完了写什么给 mentor

```markdown
## Reward Model Validation on Wan2.1 Distribution

We sampled 20 prompts from VBench-T2V and generated 2 videos per prompt
with Wan2.1-T2V-1.3B (total 40 videos, 20 within-prompt pairs).
Reward is computed with VideoAlign (use_norm=True), matching the call used
by both DanceGRPO and SAGE-GRPO. The composite training signal follows the
SAGE-GRPO codebase default:
    R = 1.0·VQ + 1.0·MQ + 1.0·TA

### Pointwise (N=40)
| Dim       | Spearman | Pearson | Kendall | Top-1/prompt |
|-----------|----------|---------|---------|--------------|
| VQ        | 0.62     | 0.60    | 0.48    | 60.0%        |
| MQ        | 0.55     | 0.54    | 0.42    | 55.0%        |
| TA        | 0.69     | 0.67    | 0.55    | 70.0%        |
| Overall   | 0.67     | 0.66    | 0.52    | 70.0%        |
| **composite (training signal)** | **0.68** | **0.66** | **0.53** | **70.0%** |

### Pairwise (N=20 pairs)
| Dim       | Acc (with ties) | Acc (without ties) |
|-----------|-----------------|--------------------|
| VQ        | 70.0%           | 72.2%              |
| MQ        | 65.0%           | 68.4%              |
| TA        | 75.0%           | 77.8%              |
| Overall   | 70.0%           | 72.2%              |
| **composite (training signal)** | **75.0%** | **77.8%** |

**Conclusion**: The composite reward that will be fed into GRPO training
ranks Wan2.1-generated videos consistent with human preference on 77.8%
of within-prompt pairs (>> chance 50%), supporting VideoAlign as a
reliable training signal for our Wan2.1 fine-tuning.
```

加上之前已经跑完的 **VideoGen-RewardBench** 的数字 → 形成
**"分布外 (RewardBench) + 分布内 (Wan2.1)" 双重验证**，论证逻辑就闭环了。

---

## 8. 常见问题

**Q: VQ/MQ 是负数，是不是说视频很差？**
A: 不是。VideoReward 输出做了 z-score 归一化（用训练集的均值方差），负号只代表"低于训练集均值"，不代表"差"。GRPO 训练关心的是组内相对大小，绝对正负无关。

**Q: 为什么默认 1:1:1，不是 SAGE-GRPO 论文的 (0.5, 0.5, 1.0)？**
A: SAGE-GRPO 代码库（[`rewards.py:383`](https://github.com/Tencent-Hunyuan/SAGE-GRPO/blob/master/hyvideo/models/reward_models/rewards.py)）默认就是 1:1:1。论文的 (0.5, 0.5, 1.0) 是为 HunyuanVideo + 他们自己的 SDE 算法特调的，没必要套到 Wan2.1 上。1:1:1 更通用、更稳。

**Q: 标注完发现某些维度（比如 MQ）人都说"差不多"怎么办？**
A: 直接填 `same`，accuracy.py 会用 `calc_accuracy_with_ties` 正确处理"平局"。

**Q: 20 对会不会太少？**
A: 统计上确实噪声大（95% CI 约 ±19%）。如果数字看着合理就够用；想更稳就加 `--num_seeds 4 --pairs_per_prompt 3` 把样本翻到 60 对。

**Q: VideoReward 的 reward 计算细节？**
A: 输入 = prompt + 视频帧（fps=2.0，抽帧），过 `Qwen2-VL-2B + rm_head`，在三个特殊 token `<|VQ_reward|>`/`<|MQ_reward|>`/`<|TA_reward|>` 位置上各取一个标量，再做 z-score 归一化。Overall = 三者之和。具体见 `VideoAlign/inference.py:176` 和 `VideoAlign/trainer.py:60`。
