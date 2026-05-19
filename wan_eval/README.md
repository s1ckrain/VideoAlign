# Wan2.1 × VideoReward —— 分布内 Reward Model 验证

验证 `KwaiVGI/VideoReward` 在 **Wan2.1 自己生成的视频** 上是否仍然可靠
（DanceGRPO 训练就在这个分布上跑，所以必须先验证）。
报告 **打分型 (pointwise)** 与 **对比型 (pairwise)** 两种准确率。

```
Phase A: Wan2.1 生成视频         (generate.py)
Phase B: VideoReward 自动打分    (score.py)
Phase C: 出人工标注模板（2 个 CSV）(annotate.py)
       人工填两个 CSV  ◀── 你/同事/标注员手动做
Phase D: 算 pointwise + pairwise acc (accuracy.py)
```

---

## 0. 当前位置 & 前置条件

这套脚本现在的位置：

```
posttrain/
├── Flow-Factory/                      # （diffusers / WanPipeline 来源）
└── VideoAlign/
    ├── inference.py                   # VideoReward 推理入口（被 score.py 调用）
    ├── calc_accuracy.py               # pairwise acc 算法（被 accuracy.py 调用）
    ├── checkpoints/                   # KwaiVGI/VideoReward 权重
    │   ├── model_config.json
    │   └── checkpoint-11352/
    └── wan_eval/                      # ← 本目录
        ├── README.md
        ├── prompts.txt                # 已经从 vbench.csv 取了 20 条
        ├── generate.py                # Phase A
        ├── score.py                   # Phase B
        ├── annotate.py                # Phase C
        ├── accuracy.py                # Phase D
        └── outputs/                   # 自动生成
```

依赖（应该都已装好）：
```bash
pip install pandas scipy tqdm diffusers transformers torch decord
```

---

## 1. 完整工作流

### Phase A: Wan2.1 生成视频

```bash
cd VideoAlign/wan_eval/

python generate.py \
    --prompts_file prompts.txt \
    --output_dir outputs \
    --model_name Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num_seeds 4 \
    --gpu 0
```

**会发生什么：**
- 20 条 prompt × 4 个 seed = **80 个视频**
- 每个视频用同一个 prompt 但不同 seed，所以画面不同、质量不同——**这就是为后续做对比的素材**
- 1.3B 模型 + A100：大约 **40-60 分钟**

**输出：**
```
outputs/
├── videos/
│   ├── p0000_s0.mp4    # 第 0 个 prompt 的第 0 个 seed
│   ├── p0000_s1.mp4    # 第 0 个 prompt 的第 1 个 seed
│   ├── p0000_s2.mp4
│   ├── p0000_s3.mp4
│   ├── p0001_s0.mp4    # 第 1 个 prompt 的第 0 个 seed
│   └── ...
│   └── p0019_s3.mp4    # 第 19 个 prompt 的第 3 个 seed
└── videos_meta.csv     # 元数据
```

`videos_meta.csv` 长这样：

| video_id | prompt_id | seed_offset | seed | prompt | video_path |
|---|---|---|---|---|---|
| p0000_s0 | 0 | 0 | 42 | In a still frame, a weathered stop sign... | videos/p0000_s0.mp4 |
| p0000_s1 | 0 | 1 | 43 | (同上) | videos/p0000_s1.mp4 |
| p0000_s2 | 0 | 2 | 44 | (同上) | videos/p0000_s2.mp4 |
| p0000_s3 | 0 | 3 | 45 | (同上) | videos/p0000_s3.mp4 |
| p0001_s0 | 1 | 0 | 42 | A pristine, vintage porcelain toilet... | videos/p0001_s0.mp4 |
| ... | | | | | |

> 同一 prompt 下的 4 个视频内容主题一致但实现不同——**它们就是 reward model 要排序、人也要排序的对象**。

> 中途崩了可以直接重跑，已生成的视频会跳过（resume 模式）。

---

### Phase B: VideoReward 自动打分

```bash
python score.py \
    --videos_meta outputs/videos_meta.csv \
    --gpu 0
```

> `--videoalign_dir` 默认就是 `..`（VideoAlign 根目录），不用传。

**会发生什么：**
- 加载你之前下好的 `../checkpoints/`（VideoReward 模型）
- 对 80 个视频逐个算 4 个分数：VQ / MQ / TA / Overall
- 大约 **10-20 分钟**（A100）

**输出：** `outputs/reward_scores.csv` = `videos_meta.csv` + 4 列 reward：

| video_id | prompt_id | ... | reward_VQ | reward_MQ | reward_TA | reward_Overall |
|---|---|---|---|---|---|---|
| p0000_s0 | 0 | ... | 0.84 | -0.21 | 1.12 | 1.75 |
| p0000_s1 | 0 | ... | 1.02 | 0.34 | 0.89 | 2.25 |
| p0000_s2 | 0 | ... | -0.15 | -0.78 | 0.42 | -0.51 |
| p0000_s3 | 0 | ... | 0.66 | 0.12 | 1.34 | 2.12 |
| ... | | | | | | |

每行 4 个 reward 值就是 **VideoReward 对这个视频的"机器评分"**。

---

### Phase C: 出人工标注模板

```bash
python annotate.py \
    --reward_scores outputs/reward_scores.csv \
    --output_dir outputs
```

会生成 2 个空 CSV 文件给你/标注员填：

#### 模板 1：`outputs/human_pointwise.csv`（打分型）

每个视频一行，标注员看视频后按 1~5 分打分：

| video_id | prompt | video_path | **human_VQ** | **human_MQ** | **human_TA** |
|---|---|---|:---:|:---:|:---:|
| p0000_s0 | In a still frame, a weathered stop sign... | videos/p0000_s0.mp4 | _3_ | _4_ | _4_ |
| p0000_s1 | (同上) | videos/p0000_s1.mp4 | _5_ | _3_ | _5_ |
| p0000_s2 | (同上) | videos/p0000_s2.mp4 | _2_ | _2_ | _3_ |
| p0000_s3 | (同上) | videos/p0000_s3.mp4 | _4_ | _4_ | _4_ |
| ... | | | | | |

- **行数 = 80**（每个视频一行）
- 标注员只填 **VQ / MQ / TA** 三列，取值 **1~5 的整数**（越大越好）

#### 模板 2：`outputs/human_pairwise.csv`（对比型）

**同一个 prompt** 内两两组对，标注员选 A 更好 / B 更好 / 一样：

| pair_id | prompt | video_A_id | video_B_id | video_A_path | video_B_path | **human_VQ** | **human_MQ** | **human_TA** |
|---|---|---|---|---|---|:---:|:---:|:---:|
| 0 | In a still frame, a weathered stop sign... | p0000_s0 | p0000_s1 | videos/p0000_s0.mp4 | videos/p0000_s1.mp4 | _B_ | _A_ | _B_ |
| 1 | (同上) | p0000_s0 | p0000_s2 | ... | ... | _A_ | _A_ | _A_ |
| 2 | (同上) | p0000_s0 | p0000_s3 | ... | ... | _B_ | _same_ | _A_ |
| 3 | (同上) | p0000_s1 | p0000_s2 | ... | ... | _A_ | _A_ | _A_ |
| 4 | (同上) | p0000_s1 | p0000_s3 | ... | ... | _A_ | _A_ | _A_ |
| 5 | (同上) | p0000_s2 | p0000_s3 | ... | ... | _B_ | _B_ | _B_ |
| 6 | A pristine, vintage porcelain toilet... | p0001_s0 | p0001_s1 | ... | ... | _A_ | _A_ | _A_ |
| ... | | | | | | | | |

- **行数 = 20 prompt × C(4,2)=6 对/prompt = 120 对**
- 标注员只填 **VQ / MQ / TA** 三列，取值 **`A` / `B` / `same`**
- 想减少标注量：加 `--pair_sample 60` 随机抽 60 对

> 如果只关心 pairwise，可以加大 `--num_seeds`（比如 8），pair 数变成 20×C(8,2)=560；
> 如果只关心 pointwise，可以减小 `--num_seeds`（最低 1），就是 20 个视频。

---

### Phase D: 算两种准确率

**前置**：标注员把上面两个 CSV 填好。可以用 Excel/WPS/Google Sheets 打开（CSV 双击就能开），边播视频边填空白列。

```bash
python accuracy.py \
    --reward_scores outputs/reward_scores.csv \
    --human_pointwise outputs/human_pointwise.csv \
    --human_pairwise  outputs/human_pairwise.csv
```

**会打印两张表 + 写一份 `outputs/accuracy_report.json`：**

```
=== Pointwise (N videos = 80) ===
dim      Spearman  Pearson  Kendall  Top-1/prompt
VQ       0.6234    0.6010   0.4811   0.6000
MQ       0.5512    0.5398   0.4221   0.5500
TA       0.6890    0.6712   0.5489   0.7000
Overall  0.6701    0.6612   0.5223   0.7000

=== Pairwise (total annotated rows = 120) ===
dim      with_ties  without_ties  used  skipped
VQ       0.7167     0.7400        116   4
MQ       0.6500     0.6800        116   4
TA       0.7500     0.7800        116   4
Overall  0.7333     0.7600        116   4
```

---

## 2. 对比型到底是什么和什么对比？（你问的关键点）

**对比型不是 reward model 和 reward model 在比，而是 reward model 跟"人"在比"谁更会排序"**。流程：

```
       ┌─────────────────────────────┐
       │ 同一个 prompt 下的两个视频 A、B │
       └──────────────┬──────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────────┐         ┌─────────────────┐
│ VideoReward 打分 │         │ 人工偏好选择     │
│ score_A = 0.84   │         │                  │
│ score_B = 1.02   │         │ 谁更好? → 'B'    │
│ → 模型认为: B 更好│         │                  │
│  (因为 B 分数高)  │         │                  │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         └─────────────┬──────────────┘
                       ▼
              ┌────────────────────┐
              │ 模型和人是否一致？  │
              │ 一致 → +1, 否则 0  │
              └────────────────────┘
```

**算法上具体怎么算**（accuracy.py 第 130 行附近）：

1. 对每对 `(A, B)`，计算 reward 差：`diff = score_A - score_B`
2. `diff > 0` → 模型认为 A 更好；`diff < 0` → 模型认为 B 更好
3. 人工标签：`A` → +1，`B` → -1，`same` → 0
4. 把 `(diff, label)` 两两组对，调用 VideoAlign 自带的 `calc_accuracy_with_ties` / `calc_accuracy_without_ties`
5. 输出准确率 = **模型方向和人方向一致的比例**

**两种 acc 的区别：**
- `without_ties`：忽略人标"一样"的对，只看模型和人在"A vs B"决断上是否一致——**最严格、最常报**
- `with_ties`：同时奖励"模型也认为差不多 & 人也认为差不多"的对，需要扫一个 ε 阈值找最优解

**对比型 ≠ 用对比 reward 来训**——这只是一种**评估 reward model 可靠性的方法**。reward model 输出依然是 pointwise 标量分数，"对比"只是把两个 pointwise 分数相减来跟人偏好对齐。

**打分型 vs 对比型的区别：**

|  | 打分型 (Pointwise) | 对比型 (Pairwise) |
|---|---|---|
| 数据形式 | 每个视频一个分数（1~5） | 每对视频选 A/B/same |
| 衡量指标 | Spearman / Pearson / Top-1 相关性 | Pairwise accuracy |
| 人在做啥 | "这视频满分 5 分，我给几分？" | "A 还是 B 更好？" |
| 反映的能力 | reward 数值是否和人感受**线性对齐** | reward 排序是否和人**判断一致** |
| 标注难度 | 较难（绝对打分容易主观漂移） | 较易（两两比较更稳定） |
| 行业偏好 | 较少用（除非有校准） | **主流，论文都报这个**（包括 RewardBench） |
| 这套脚本里 | `human_pointwise.csv` + Spearman | `human_pairwise.csv` + Pairwise Acc |

---

## 3. 跑通的最快路径（推荐）

```bash
# 假设你已经在 VideoAlign/wan_eval/

# Step 1: 生成 80 个视频（1 小时左右）
python generate.py --prompts_file prompts.txt --output_dir outputs --gpu 0

# Step 2: 自动打分（15 分钟左右）
python score.py --videos_meta outputs/videos_meta.csv --gpu 0

# Step 3: 出标注模板
python annotate.py --reward_scores outputs/reward_scores.csv --output_dir outputs

# === 这里停下来，开始人工标注 ===
# 打开 outputs/human_pointwise.csv，看 outputs/videos/*.mp4，填 1~5 分
# 打开 outputs/human_pairwise.csv，看两个视频，填 A/B/same
# (80 视频 + 120 对，单人 ~2 小时)

# Step 4: 算最终 acc
python accuracy.py \
    --reward_scores outputs/reward_scores.csv \
    --human_pointwise outputs/human_pointwise.csv \
    --human_pairwise outputs/human_pairwise.csv
```

跑完得到 `accuracy_report.json`，这就是给 mentor 看的最终指标。

---

## 4. 给 mentor 的报告该怎么写

```markdown
## Reward Model Validation on Wan2.1 Distribution

We sampled 20 prompts from VBench-T2V and generated K=4 videos per prompt
with Wan2.1-T2V-1.3B (total 80 videos, 120 within-prompt pairs).

### Pointwise (打分型, N=80)
| Dim | Spearman | Pearson | Kendall | Top-1/prompt |
|-----|----------|---------|---------|--------------|
| VQ  | 0.62     | 0.60    | 0.48    | 60.0%        |
| MQ  | 0.55     | 0.54    | 0.42    | 55.0%        |
| TA  | 0.69     | 0.67    | 0.55    | 70.0%        |
| Overall | 0.67 | 0.66   | 0.52    | 70.0%        |

### Pairwise (对比型, N=120 pairs)
| Dim | Pairwise Acc (without ties) |
|-----|------------------------------|
| VQ  | 74.0%                        |
| MQ  | 68.0%                        |
| TA  | 78.0%                        |
| Overall | 76.0%                    |

Conclusion: VideoReward shows consistent ranking ability with human
preferences on Wan2.1-generated videos (Pairwise Acc > 65% on all
dimensions), making it a reliable reward signal for DanceGRPO fine-tuning.
```

加上之前 RewardBench 的数字 → 形成 **"分布外 (RewardBench) + 分布内 (Wan2.1)"双重验证**，足够说服任何 reviewer。
