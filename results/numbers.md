# JRTIP 稿件数字台账（numbers.md）

**用途**：`jrtip_budget_paper/latex/` 正文里出现的每一个数字都必须能在本表找到一行。列含义——
**值**：正文里印出的写法；**来源**：产物文件 `#` 键路径（相对 ``）；**性质**：`存` = 产物里存的字段原值，`派` = 本表给出的公式从存字段算出，`文` = 只写在台账/方案里的协议陈述（非测量值）。

正文引用规则（不得绕过本表）：

1. **毫秒一律"逐遍中位数 + 同遍控制档归一"**（§21 定的口径）。凡引用绝对毫秒，同段必须给出该遍 `static100_native` 作为分母，并写明是哪一遍。
2. **⌈1.5c⌉ 的分桶溢价不作结论**（复制遍未复现，见 §F/§G），只能作为"未复现"的实例引用。
3. `bottleneck_decomposition_v1/decomposition.json` 的 `verdict` 四项与 `resolution_response` **不引用**（它们建立在 n = 2/16/12 帧的"干净窗"边界样本上，而脚本 docstring 自己指定结论应取 `all` 列）；归因一律取 `all` 列（n = 358/尺度）。
4. 禁用字段清单见 §N，其中三条是产物里真实存在但语义重复/未定义的量。
5. "质量等价"只在 **score ≥ 0.5 工作点**意义上成立（§K 的仪表性质）。

产物根目录：`jrtip/`。协议常量见 §A，正文措辞与判据出处见 `02_JRTIP_逐帧输出侧算力预算实验方案_v2_20261122.md`（下称**方案 v2**）与结果台账 `JRTIP_服务时间归因与输出侧预算_20261122.md`（下称**台账 §n**）。

---

## A. 协议与仪器常量

| 项 | 值 | 来源 | 性质 |
|---|---|---|---|
| 开发质量契约帧数 | 2,947 | `jrtip/budget_bucket_pricing_v1/bucket_pricing.json#frames`；`mask_restore_resolution_v1/restore_quality.json#frames` | 存 |
| 计时流帧数（每档每 session） | 1,073 | `rule_timing_v1/rule_timing.json#sessions_meta[*].frames_per_arm` | 存 |
| 计时流摘要 | `553a930dc3b6…33c67` | 同上 `#timing_stream_sha256`（三 session 同值） | 存 |
| 检查点摘要 | `7c45e7eefb06…405dd` | `bucket_pricing.json#checkpoint_sha256`、`budget_policy_v1/policy_eval.json#checkpoint_sha256`、`rule_timing.json#sessions_meta[*].checkpoint_sha256` | 存 |
| cap-100 归档流摘要 | `9f42b099d72d…87cf1` | `bucket_pricing.json#control_stream_sha256` | 存 |
| 训练规模 | 种子 43、1,000 优化器更新、有效 batch 4（物理 1 × 累积 4）、FP32、通用 COCO R50-FPN 独立初始化、多尺度长边 320/480/640 | `rule_timing_v1/rule_timing.json#note`；`rule_timing_readout.json#scope` | 文 |
| 三 session 同流/同检查点/同版本 | true / true / true | `rule_timing.json#identical_timing_stream_across_sessions`、`#identical_checkpoint_across_sessions`、`#identical_versions_across_sessions` | 存 |
| 臂序轮换、warmup、按位一致门 | 每 session 臂序轮换；`warmup_requested = 50`；`gate = true` | `rule_timing.json#sessions_meta[*]`（session 1 首臂 static100_native，session 2 首臂 static20_native） | 存 |
| 版本串 | python 3.10.9、torch 2.6.0+cu124、torchvision 0.21.0+cu124、numpy 2.2.6、pycocotools 2.0.11 | `rule_timing.json#sessions_meta[*].versions` | 存 |
| 计数阈值 | 0.5（`count_score_threshold`） | `bucket_pricing.json#count_score_threshold` | 存 |
| 分桶格点 | {15, 20, 30, 50, 100} | `bucket_pricing.json#bucket_grid` | 存 |
| 预注册容差 | AP 降 ≤ 1.0 pp；计数 MAE 升 ≤ 0.2 叶 | `bucket_pricing.json#predeclared_tolerances`；`baseline_benchmark_v1/scales.json` 同名键 | 存 |
| 时间门 | 设备实测削减 ≥ 60% | 方案 v2 §7.1；读数在 `rule_timing_readout.json#section_7_1_gate.gate` | 存 |
| 动作阶梯（学习侧） | K = {100, 50, 30, 20, 15} | 方案 v2 §4；`bucket_grid` 同集合 | 存 |
| 打开的角色 / 锁定角色 | `roles_opened = [V_SHIFT, V_SOURCE]`；`locked_roles_accessed = []`，`gpu_used = false`（所有离线标价产物） | `bucket_pricing.json`、`policy_eval.json`、`restore_quality.json` 同名字段 | 存 |
| 未测清单 | queueing replay、energy、second hardware、locked test roles、36-epoch checkpoint、controller (learned) arms | `rule_timing_readout.json#not_measured` | 存 |
| 结论范围（原文） | "development roles only; 1,000-step pilot checkpoint; service time on one shared RTX 3090; point estimates, not a non-inferiority claim; no quality number here is paired with a time number from a different arm" | `rule_timing_readout.json#scope` | 存 |

**派生（脚本 `jrtip_budget_paper/scripts/derived_counts.py`，只读 `detection_budget_quality_v1/cap00100/quality_000100_S640.jsonl`）**

| 项 | 值 | 性质 |
|---|---|---|
| 每帧掩码实例数 | 中位数 100，均值 89.5192 | 派（与存字段 `#control.work_masks_per_frame = 89.519` 一致） |
| 每帧 score ≥ 0.5 实例数 | 中位数 9，均值 9.2732；p5 = 4、p25 = 6、p50 = 9、p75 = 12、p90 = 15、p95 = 16、p99 = 20 | 派（台账 §3 那句"中位数只有 9"的正式出处） |
| count > 15 / > 20 / > 25 / > 30 的帧数 | 201 (6.82%) / 19 (0.64%) / 2 (0.07%) / 0 | 派（与 `budget_p1b_v1/condition1_verdict.json` 里静态 15 档 `truncated_frames = 201` 独立对上） |
| gt_count 均值 | 9.1388 | 派 |
| **合并池**平均 \|计数误差\| | 0.773668 | 派 —— **不等于**报告的 count MAE 1.0139046 |

> ⚠️ 正文若同时出现 0.774 与 1.014，必须写明后者是"基因型×角色宏平均"、前者是"逐帧池化均值"。宏 mean 的定义由 `restore_quality.json#control.count_MAE = 1.0139046` 与 `#arms.g1.macro_by_role`（V_SHIFT 1.1642669、V_SOURCE 0.8635423，二者算术平均 = 1.0139046）确证。

---

## B. 冻结基准：三尺度服务时间与质量（正文表 T1）

来源统一为 `jrtip/baseline_benchmark_v1/scales.json#scales.{320,480,640}`；每档 3 session × 1,073 请求 = **3,219 请求**（`#requests_technical_repeats`）。

| 量 | S320 | S480 | S640 | 性质 |
|---|---|---|---|---|
| 服务时间 mean_ms | 421.4624 | 420.7275 | 416.1404 | 存 `#mean_ms` |
| p50_ms | 440.758 | 435.902 | 431.454 | 存 `#p50_ms` |
| p95_ms | 516.222 | 512.268 | 502.503 | 存 `#p95_ms` |
| 饱和吞吐 req/s | 2.3727 | 2.3768 | 2.4030 | 存 `#saturated_throughput_per_second` |
| 最大/最小 mean_ms 比 | **1.0128（= 1.28% 跨度）** | | | 派 421.4624/416.1404 |
| 宏 AP | 62.0696 | 68.4411 | **70.4608** | 存 `#quality.AP` |
| AP50 / AP75 | 91.2425 / 72.5159 | 94.3529 / 78.6531 | 95.4112 / 80.9295 | 存 `#quality.AP50`、`#quality.AP75` |
| 计数 MAE | 1.0912 | 0.9477 | 1.0139 | 存 `#quality.count_MAE` |
| 降尺度的质量代价 AP_drop_pp | 8.3911 | 2.0197 | 0.0（基准） | 存 `#equal_domain_macro.AP_drop_pp` |
| 降尺度的计数代价 | +0.0773 | −0.0662 | 0.0 | 存 `#equal_domain_macro.count_MAE_increase` |
| 容差判定 | 未过（AP 8.39 > 1.0） | 未过（2.02 > 1.0） | — | 派自 `#tolerances` + 上两行；`#point_tolerances_met = false`（320/480 两档为存字段） |
| 干扰标记 | `contention = true`，`jitter = false`（三档同） | | | 存 `#contention`、`#investigation_flags.jitter_detected_by_probe` |
| 分辨率开关的取舍结论 | 4× 像素只换 ≤1.28% 服务时间，却付 8.39/2.02 pp AP | | | 派（上面两行），台账 §5 |

正文写法约束：**"服务时间对输入分辨率近乎不敏感"只能写成 ≤1.3%（协议计时口径）或 ≤1.6%（归因口径，见 §C）**，不得写"flat/恒定"。

`#equal_domain_mean_ms`（454.724/451.539/446.868）**不与 `#mean_ms` 混用**（定义未在产物里断言），见 §N。

---

## C. 服务时间归因（正文表 T2 + 图 2 素材）

来源统一为 `jrtip/bottleneck_decomposition_v1/decomposition.json#scales.{320,480,640}.all.<field>.median`，**n = 358 帧/尺度**（`#frames_per_scale`）。取 `all` 列的理由与该脚本 docstring 一致；`clean_window` 列（n = 2/16/12，`#scales.<s>.clean_probe_frames`、`clean_probe_fraction` 0.56%/4.47%/3.35%）**不引用**。

| 分量（中位数 ms） | S320 | S480 | S640 | 键 |
|---|---|---|---|---|
| 重建服务时间 | 448.731 | 445.949 | 452.890 | `#service_reconstructed_ms.median` |
| 前向（CUDA event） | 346.256 | 340.897 | 342.873 | `#forward_event_ms.median` |
| 前向均值 | 323.404 | 321.004 | 325.596 | `#forward_event_ms.mean` |
| 其中 torchvision 掩码后处理 | 293.721（84.8%） | 287.716（84.4%） | 285.405（83.2%） | `#eval_postprocess_ms.median` |
| 我方分段后处理 | 88.866 | 90.353 | 91.764 | `#postprocess_ms.median` |
| backbone | 16.332 | 17.772 | 21.241 | `#backbone_ms.median` |
| RPN | 8.692 | 8.671 | 9.762 | `#rpn_ms.median` |
| ROI heads | 25.962 | 24.826 | 23.719 | `#roi_heads_ms.median` |
| 解码 / resize+transfer / H2D | 6.710 / 2.654 / 0.533 | 6.245 / 3.944 / 0.677 | 6.591 / 5.055 / 1.214 | `#decode_ms`、`#resize_transfer_ms`、`#host_to_device_event_ms` |
| 掩码还原到原尺寸（CPU 插值） | 35.868 | 36.753 | 36.461 | `#mask_restore_interpolate_to_cpu_ms.median` |
| 掩码阈值化 | 10.671 | 10.950 | 12.498 | `#mask_threshold_ms.median` |
| RLE 编码 | 42.875 | 43.075 | 41.139 | `#rle_encode_ms.median` |
| 序列化 | 0.895 | 0.956 | 0.956 | `#serialize_ms.median` |
| 分数回传 CPU / 框与标签 | 0.120 / 0.793 | 0.121 / 0.799 | 0.122 / 0.805 | `#scores_to_cpu_ms`、`#boxes_labels_ms` |
| 墙钟 − event（中位） | 0.227 | 0.202 | 0.231 | `#scales.<s>.wall_minus_event_median_ms` |
| 每尺度返回检测数中位 / 均值 | 100 / 89.82 | 100 / 91.21 | 100 / 91.55 | `#n_instances`、`#returned_detections` |

**派生（可写进正文）**

| 说法 | 值 | 公式 |
|---|---|---|
| 前向设备时间对像素的响应 | **最大/最小 = 1.0157（≤1.6%）**，4× 像素 | 346.256 / 340.897 |
| 掩码后处理占前向 | **83.2–84.8%** | `eval_postprocess_ms.median` ÷ `forward_event_ms.median` |
| 网络本体（backbone+RPN+ROI heads） | 51.0 / 51.3 / 54.7 ms | 三列相加 |
| 固定形状下界（整网一次前向） | 64² 42.765 / 40.871 / 38.391；256² 37.323 / 34.649 / 33.900 ms | `#scales.<s>.fixed_shape_floor.{64,256}.event_ms_per_forward`（reps = 20） |
| 墙钟与 event 一致性 | 差 0.20–0.23 ms，相对 ≤0.49% | 上表"墙钟 − event"行 |

**输出侧预算扫描**（`#scales.<s>.detection_cap_sweep.by_budget.<k>`，40 帧/格，`reference_budget = 100`）

| 量 | 320 | 480 | 640 | 键 |
|---|---|---|---|---|
| k=100 服务 | 450.15 | 442.53 | 442.71 | `#service_reconstructed_ms.median` |
| k=20 服务 | 133.59（−70.3%） | 134.01（−69.7%） | **130.05（−70.6%）** | 同上；百分比为派生 |
| k=5 服务 | 69.25 | 76.15 | 68.81 | 同上 |
| k=20 其中 eval 后处理 | 60.87 | 59.77 | 57.38 | `#eval_postprocess_ms.median` |
| 计数与参照一致的帧 | 40/40 | 40/40 | 40/40 | `#count_equal_to_reference`（k=5 为 **7 / 8 / 8** /40） |
| 实例集合与参照一致的帧 | 40/40 | 40/40 | 40/40 | `#instances_equal_to_reference`（k=20/5 为 0/40，定义使然） |
| k=100 返回检测数 | 中位 100、均值 89.42/88.53/89.80 | | | `#returned_detections` |

> 正文机制句"预算 20 时数得逐帧不变、砍到 5 才开始变"的引用面：`count_equal_to_reference` 与 40 帧样本量。跨尺度**只报中位数**，别把 40 帧扫描的 ms 与 3,219 请求的协议服务时间混列同表。

---

## D. 静态工作—质量阶梯（正文表 T3 / 图 1 左）

来源：`jrtip/budget_bucket_pricing_v1/bucket_pricing.json#static_comparators.<k>`（2,947 帧、离线前缀标价、`gpu_used = false`、81 项检查 0 失败、耗时 1,543.5 s）。控制档 `#control`：AP 70.4607911786734 / AP50 95.41115101923182 / AP75 80.92950804085432 / count_MAE 1.0139046254831632 / work 89.519。

| k（预算） | work（掩码片/帧） | ΔAP_pp | Δcount（叶） | 键 |
|---|---|---|---|---|
| 5 | 5.0 | 31.3652 | +4.0928 | `#<k>.work_masks_per_frame`、`#<k>.ap_drop_vs_control_pp`、`#<k>.count_MAE_increase_vs_control_leaf` |
| 10 | 10.0 | 6.2965 | +0.4859 | 同上 |
| 15 | 15.0 | 1.1888 | +0.0034 | 同上 |
| 16 | 16.0 | **0.9052** | −0.0008 | 同上 |
| 17 | 17.0 | 0.7145 | +0.0050 | 同上 |
| 18 | 18.0 | 0.5937 | 0.0 | 同上 |
| 19 | 19.0 | 0.4483 | −0.0015 | 同上 |
| 20 | 19.999 | 0.4128 | −0.0007 | 同上 |
| 22 | 21.999 | 0.3296 | +0.0002 | 同上 |
| 24 | 23.997 | 0.2763 | −0.0007 | 同上 |
| 25 | 24.996 | 0.2612 | −0.0002 | 同上 |
| 26 | 25.995 | 0.2384 | 0.0 | 同上 |
| 28 | 27.992 | 0.1917 | 0.0 | 同上 |
| 30 | 29.985 | 0.1635 | 0.0 | 同上 |
| 35 | 34.956 | 0.1351 | 0.0 | 同上 |
| 40 | 39.880 | 0.0952 | 0.0 | 同上 |
| 45 | 44.734 | 0.0697 | 0.0 | 同上 |
| 50 | 49.494 | 0.0631 | 0.0 | 同上 |
| 100 | 89.519 | 0.0 | 0.0 | 同上（控制档） |

> 每个静态节点自带 `#<k>.source` 字段（例如 k=26 的 `source = "policy_eval.json"`），指明该点出自哪一份标价产物 —— 出图或写表时按它溯源，不要假设 19 点同源。`#<k>.macro` 另存 AP/AP50/AP75/count_MAE 四项宏值（如 k=26：AP 70.2224323、AP50 95.0743450、AP75 80.6442119、count_MAE 1.0139046）。

补密的 10 个静态点（16,17,18,19,22,24,26,28,35,45）来自同一条 cap-100 流的额外前缀截断，纯 CPU、耗时 1,181.6 s（作业墙钟 1,190 s）：`jrtip/budget_policy_v1_extra/policy_eval.json`（69 项检查，失败仍是 §L 那三条 ±1 像素掩码串）。两个产物里同名静态点的数字必须逐位相同，出图前做一次等值断言。

| 说法 | 值 | 键 |
|---|---|---|
| 容差内前沿 | `#frontier_within_tolerances`（节点列表即图 1 的前沿折线） | 存 |
| 等量比较的窗口 | ±1.0 instances/frame | `#fill_gap_note` |

---

## E. 免标签规则族（质量侧，正文表 T3 右 / 图 1 右）

规则定义（`#single_source_of_truth.budget_semantics = src/jrtip_budget_rule_timing.py::budget_for`）：`c` = 掩码头之前可得的 `#{score ≥ 0.5}`；同步族 `k = clip(c+m, floor, 100)`；分桶族把 `k` 上取整到格点 {15,20,30,50,100}；`readfull` 族在读取上 differ（见 §F）。臂例：`arm[0].definition = "k=clip(c+10,15,100) ceiled to {15,20,30,50,100}"`（`rule_timing.json#arms[0].definition`）。

质量侧 23 档全部离线标价于 `bucket_pricing.json#arms`；**结构性不截断上报计数**：全部 23 档 `#arms.<name>.frames_with_reported_count_truncated = 0`，故 Δ计数 MAE ≡ 0.0（`#arms.<name>.count_MAE_increase_vs_control_leaf`）。

过双门的七档（时间来自设备实测、质量来自同名档离线标价）——来源 `rule_timing_v1/rule_timing_readout.json#section_7_1_gate.arms[*]`（该键自带 `quality_source = "jrtip\\budget_bucket_pricing_v1\\bucket_pricing.json"`）：

| 档 | 削减 % | ΔAP_pp | 键（同一条目内） |
|---|---|---|---|
| `static16_native` | 73.43 | 0.9052 | `#measured_time_reduction_pct`、`#ap_drop_pp_offline` |
| `static20_native` | 69.89 | 0.4128 | 同上 |
| `sync_ceil_1p5_f15` | 72.24 | 0.3845 | 同上 |
| `sync_c_plus_10_f15` | 69.76 | 0.2956 | 同上 |
| `readfull_c_plus_10_f15` | 70.38 | 0.2956 | 同上 |
| `bucket_ceil_1p5_f15` | 71.00 | 0.2837 | 同上 |
| `bucket_c_plus_10_f15` | 66.50 | 0.2042 | 同上 |

`#section_7_1_gate.arms[*].both_gates = true`（七档全真），`rule_timing_readout.json#all_gates_passed = true`。

| 说法 | 值 | 键 |
|---|---|---|
| 免标签前沿规模（互异过线点） | 45 个（按 (work, ΔAP) 折叠）；条目 48 条（15/10/23 三表）；削减 ≥60% 者折叠为 42 点 | 台账 §19.1 末；逐条清单 `jrtip/budget_p1b_v1/comparator_duplicates.json` |
| 其中削减 ≥60% | 42 点 | 同上（`comparator_duplicates.json` 内计数） |

> 48 与 45 与 42 是三个不同口径的数，正文只能各用其处，并写明口径（条目数 / 折叠点数）。差 3 的来源是两处"异名同点"：(16.99, 0.3845) = `clip(ceil(1.5*count),15,100)` ≡ `sync_ceil_1p5_f15`；(19.35, 0.2956) = `clip(count+10,15,100)` ≡ `sync_c_plus_10_f15` ≡ `readfull_c_plus_10_f15`。

---

## F. 事件时间：主三遍（正文表 T4 / 图 3）

来源 `jrtip/rule_timing_v1/rule_timing_readout.json#arms[*]`（`mean_ms` = 先算每 session 均值、再跨 session 平均；单请求永不跨 session 合并，见 `#aggregation_reused`）。11 臂 × 3 session × 1,073 请求。

| 臂 | mean_ms | 跨遍极差 ms | p95 ms | 发射片/帧 | read_ms | wrapper_ms | 削减 % | 键 |
|---|---|---|---|---|---|---|---|---|
| `static100_native` | 415.772 | 0.894 | 497.809 | 91.4846 | 0.0 | 0.0 | 0.0 | `#mean_ms`、`#spread_ms`、`#p95_ms`、`#emitted_per_frame`、`#read_ms`、`#wrapper_ms`、`#reduction_vs_static100_pct` |
| `static100_wrap` | 411.764 | 3.651 | 489.555 | 91.4846 | 0.0 | 6.33615 | 0.96 | 同上 |
| `static15_native` | 105.931 | 1.870 | 121.472 | 15.0 | 0.0 | 0.0 | 74.52 | 同上 |
| `static16_native` | 110.486 | 1.685 | 126.523 | 16.0 | 0.0 | 0.0 | 73.43 | 同上 |
| `static20_native` | 125.195 | 1.630 | 142.974 | 20.0 | 0.0 | 0.0 | 69.89 | 同上 |
| `static20_wrap` | 125.499 | 1.790 | 142.332 | 20.0 | 0.0 | 4.73611 | 69.82 | 同上 |
| `sync_ceil_1p5_f15` | 115.406 | 1.663 | 147.334 | 17.096 | 0.26102 | 5.02119 | 72.24 | 同上 |
| `sync_c_plus_10_f15` | 125.740 | 4.101 | 156.772 | 19.7139 | 0.26344 | 5.03091 | 69.76 | 同上 |
| `readfull_c_plus_10_f15` | 123.157 | 1.902 | 152.817 | 19.7139 | 0.14562 | 4.93316 | 70.38 | 同上 |
| `bucket_ceil_1p5_f15` | 120.587 | 1.213 | 170.859 | 18.4296 | 0.25819 | 5.07304 | 71.00 | 同上 |
| `bucket_c_plus_10_f15` | 139.286 | 3.140 | 182.883 | 23.2106 | 0.26372 | 5.18384 | 66.50 | 同上 |

**静态斜率（局部 15→20）** 4.1161 ms/片；分段：15→16 段 4.555、16→20 段 3.6772、20→100 段 4.0649 ms/片 —— 全部存于 `rule_timing_readout.json#static_cost_slope`（`#segments[*].ms_per_instance`、`#ms_per_instance_local_15_to_20`）。

**包装器开销**：`#wrapper_overhead.at_cap_100` = wrapper 411.764 vs native 415.772 → **−4.008 ms**（`within_spread = false`，即符号为负且超出 session 带宽）；`#wrapper_overhead.at_cap_20` = +0.304 ms（`within_spread = true`）。同值亦存 `rule_timing.json#implementation_overhead_decomposition`。正文只能写"包装一层固定 cap 的 Python 侧代价在本机测不出正号，落在 −4.0…+0.3 ms 之间"。

**决策读取代价**：`#decision_read_cost`：`sync_read_ms_mean = 0.26344`、`readfull_read_ms_mean = 0.14562`、`read_ms_as_pct_of_service = 0.21%`、`readfull_minus_sync_ms = −2.583`、`worst_arm_session_spread_ms = 4.101`、**`resolvable = false`**、`#interpretation`（原文："两档交付同一预算，故其实测差在未超过 session 带宽前只是归因噪声"）。

**分桶溢价（主三遍，`#bucket_grid_waste`）**：`c_plus_10_f15` 桶−同步 = **13.546 ms**，浪费 3.4967 片/帧，观测 3.8739 ms/片（静态曲线 4.1161），折服务时间 **10.77%**；`ceil_1p5_f15` = 5.181 ms / 1.3336 片 / 3.885 ms/片 / **4.49%**。同值亦存 `rule_timing.json#sync_vs_bucket`（其中 `ms_per_extra_masked_instance` = 3.874 / 3.885）。
> ⚠️ 正文引用这两行时只能引 `c_plus_10_f15` 为结论，`ceil_1p5_f15` 那条按 §G 降为"未复现"。

**干扰（同遍探针 vs 服务时间）**：`#interference_effect`：`arms_compared = 11`、`difference_pct_median = 1.094`、`difference_pct_max = 1.609`、`contention_true_in_any_session = true`、`sessions_recorded = 3`；`#per_arm[*].inflation_max` 跨 1.13–1.77 而 `difference_pct` 全 ≤1.61%；`#reading` 原文说明"探针的膨胀不转化为 1,073 请求均值的可比位移"。最干净/最脏 session 之差最极端的一行可引用：`static100_native` inflation 1.0816→1.7661、均值差 0.053 ms（0.013%）。

**等量对（规则档 vs 静态定律）**：`#equal_work_pairs`：`static15_anchor_ms = 105.931`、`ms_per_instance = 4.1161`；预测−实测残差（实测 − 预测）：`sync_c_plus_10` **+0.406**、`sync_ceil_1p5` **+0.848**、`readfull_c_plus_10` **−2.177**、`bucket_c_plus_10` **−0.441**、`bucket_ceil_1p5` **+0.539** ms；主三遍最大残差 **2.177 ms**。预测式与含义见该键的 `#pairs[*].note`。

---

## F′. 逐遍中位数与同遍归一（**正文表 T4′，引用毫秒的首选形式**）

来源：`rule_timing_v1/rule_timing.json#arms[*].session_means_ms`（三 session 均值）与同文件 `rule_timing_v1_rep/rule_timing.json`。`med` = 对该臂三遍 session 均值取中位数；`norm` = med ÷ **同遍** `static100_native` 的 med（主遍 416.035、复制遍 411.792）。`mean` 列存 `#mean_ms_session_averaged`，只用于与台账 §11/§18 已有说法对齐。

主三遍（control med = 416.035 ms）：

| 臂 | med ms | 三遍 session 均值 | mean ms | norm | 发射片/帧 | 削减 %（存、按均值算） |
|---|---|---|---|---|---|---|
| `static15_native` | 105.452 | 105.452 / 107.106 / 105.236 | 105.931 | 0.2535 | 15.0000 | 74.52 |
| `static16_native` | 110.981 | 111.081 / 110.981 / 109.396 | 110.486 | 0.2668 | 16.0000 | 73.43 |
| `sync_ceil_1p5_f15` | 115.763 | 115.763 / 116.059 / 114.395 | 115.406 | 0.2783 | 17.0960 | 72.24 |
| `bucket_ceil_1p5_f15` | 120.513 | 121.231 / 120.513 / 120.018 | 120.587 | 0.2897 | 18.4296 | 71.00 |
| `readfull_c_plus_10_f15` | 122.893 | 122.338 / 122.893 / 124.240 | 123.157 | 0.2954 | 19.7139 | 70.38 |
| `static20_wrap` | 125.150 | 124.778 / 126.568 / 125.150 | 125.499 | 0.3008 | 20.0000 | 69.82 |
| `static20_native` | 125.380 | 125.380 / 125.917 / 124.287 | 125.195 | 0.3014 | 20.0000 | 69.89 |
| `sync_c_plus_10_f15` | 126.428 | 127.447 / 126.428 / 123.346 | 125.740 | 0.3039 | 19.7139 | 69.76 |
| `bucket_c_plus_10_f15` | 139.733 | 139.733 / 140.632 / 137.492 | 139.286 | 0.3359 | 23.2106 | 66.50 |
| `static100_wrap` | 412.442 | 412.442 / 409.599 / 413.250 | 411.764 | 0.9914 | 91.4846 | 0.96 |
| `static100_native` | 416.035 | 416.088 / 416.035 / 415.194 | 415.772 | 1.0000 | 91.4846 | 0.00 |

复制三遍（control med = 411.792 ms）：

| 臂 | med ms | 三遍 session 均值 | mean ms | norm | 削减 %（存、按均值算） |
|---|---|---|---|---|---|
| `static15_native` | 107.455 | 109.769 / 106.747 / 107.455 | 107.990 | 0.2609 | 73.81 |
| `static16_native` | 110.972 | 112.254 / 110.685 / 110.972 | 111.303 | 0.2695 | 73.01 |
| `sync_ceil_1p5_f15` | 116.852 | 122.025 / 115.263 / 116.852 | 118.047 | 0.2838 | 71.37 |
| `bucket_ceil_1p5_f15` | 121.981 | 121.981 / 123.100 / 119.683 | 121.588 | 0.2962 | 70.52 |
| `static20_wrap` | 125.545 | 125.545 / 127.604 / 125.409 | 126.186 | 0.3049 | 69.40 |
| `readfull_c_plus_10_f15` | 125.795 | 125.545 / **139.863** / 125.795 | 130.401 | 0.3055 | 68.38 |
| `sync_c_plus_10_f15` | 126.437 | 126.437 / 126.931 / 126.238 | 126.536 | 0.3070 | 69.32 |
| `static20_native` | 126.501 | 127.398 / 126.501 / 126.059 | 126.653 | 0.3072 | 69.29 |
| `bucket_c_plus_10_f15` | 141.086 | 140.191 / 141.086 / **156.392** | 145.890 | 0.3426 | 64.62 |
| `static100_native` | 411.792 | 413.854 / 411.792 / 411.500 | 412.382 | 1.0000 | 0.00 |
| `static100_wrap` | 412.982 | 414.455 / 412.982 / 403.296 | 410.244 | 1.0029 | 0.52 |

主遍 → 复制遍（**中位数、同遍归一**后）的位移：

| 臂 | norm → norm | 归一位移 | 绝对 med 位移 |
|---|---|---|---|
| `static15_native` | 0.2535 → 0.2609 | **+2.95%** | +1.90% |
| `static16_native` | 0.2668 → 0.2695 | +1.02% | −0.01% |
| `sync_ceil_1p5_f15` | 0.2783 → 0.2838 | +1.98% | +0.94% |
| `bucket_ceil_1p5_f15` | 0.2897 → 0.2962 | +2.26% | +1.22% |
| `readfull_c_plus_10_f15` | 0.2954 → 0.3055 | **+3.42%** | +2.36% |
| `static20_native` | 0.3014 → 0.3072 | +1.93% | +0.89% |
| `sync_c_plus_10_f15` | 0.3039 → 0.3070 | +1.04% | +0.01% |
| `bucket_c_plus_10_f15` | 0.3359 → 0.3426 | +2.01% | +0.97% |
| `static20_wrap` | 0.3008 → 0.3049 | +1.35% | +0.32% |
| `static100_wrap` | 0.9914 → 1.0029 | +1.16% | +0.13% |

**正文可写的三条**（都由上表直接支撑，且比按均值的说法更干净）：①按中位数、同遍归一后，除控制档自身外 **10/10 个臂同向变慢，跨度 +1.02% … +3.42%**（均值口径是 +0.83% 跨臂中位，`replication_compare.json#summary.mean_shift_pct_median_across_arms`）；②控制档的绝对 med 反而 −1.02%（416.035 → 411.792），**正因如此归一比移普遍大于绝对位移** —— 这句是同遍归一口径存在的理由，要写出来；③两遍里最贵的两档各自含一个明显的 session 离群（复制遍 `readfull` 第 2 遍 139.863、`bucket_c+10` 第 3 遍 156.392），中位数把它们挡在外面，均值被推高 3.6% 与 3.4%（派生：130.401/125.795 − 1、145.890/141.086 − 1）。

`#time_reduction_vs_static100_native` 与 `#mean_ms_session_averaged` 同源（按均值算），故上表"削减 %"列与 §E 门表一致；**若正文改以中位数报削减，必须重算并注明"按逐遍中位数"**，不得把存字段的中位数说法与均值说法混在同一列。

---



## G. 复制三遍与引用口径（正文 §Reproducibility / 图 4）

来源 `jrtip/rule_timing_v1_rep/replication_compare.json`（`#summary`、`#static_line_residuals`、`#per_arm`）与 `rule_timing_v1_rep/rule_timing_readout.json`。

| 说法 | 值 | 键 |
|---|---|---|
| 结构一致性 | 11 臂 × 3 session × 1,073 请求；三遍计时前门全过 | `#summary.arms`、`#sessions_per_arm`、`#frames_per_arm`、`#gates_passed_before_timing = [true, true, true]` |
| 绝对平移（跨臂中位） | **+0.83%** | `#summary.mean_shift_pct_median_across_arms` |
| 平移最大的一臂 | `readfull_c_plus_10_f15` | `#summary.median_shift_pct_worst_across_arms` |
| 跨遍极差 ÷ 自身均值：复制遍最大 | **11.1%** | `#summary.replication_spread_pct_of_own_mean_max` |
| 同量：主三遍最大 | **3.26%** | `#summary.primary_spread_pct_of_own_mean_max` |
| 落在自身主遍带外的臂 | 5 个：`bucket_c_plus_10_f15`、`readfull_c_plus_10_f15`、`static100_native`、`static15_native`、`sync_ceil_1p5_f15` | `#summary.arms_outside_own_primary_spread` |
| 离群判定阈值 / 主遍离群 / 复制遍离群 | 份额 5.0% / 无 / 2 处：`bucket_c_plus_10_f15` 第 3 session、`readfull_c_plus_10_f15` 第 2 session | `#summary.excursion_flag_threshold_pct_share`、`#primary_excursions_arm_session`、`#replication_excursions_arm_session` |
| 七档在复制每一遍都过时间门 | `true`；**最紧一遍余量 = 1.99 个百分点削减**（`bucket_c+10` 第三遍 61.99% vs 门 60%） | `#summary.gate_arms = 7`、`#gate_arms_pass_60pct_in_every_rep_session`、`#smallest_worst_session_margin_pp` |
| 复制遍逐档读数 | `bucket_c+10` 145.89 / 64.62%、`bucket_1.5c` 121.588 / 70.52%、`readfull` 130.401 / 68.38%、`static100_native` 412.382 / 0.0%、`static100_wrap` 410.244 / 0.52%、`static15` 107.99 / 73.81%、`static16` 111.303 / 73.01%、`static20` 126.653 / 69.29%、`static20_wrap` 126.186 / 69.4%、`sync_c+10` 126.536 / 69.32%、`sync_1.5c` 118.047 / 71.37% | `rule_timing_v1_rep/rule_timing_readout.json#arms[*].mean_ms`、`.reduction_vs_static100_pct` |

**静态定律在三种聚合下的残差**（`#static_line_residuals`，斜率 4.1161 ms/片、锚 `static15_native`）：

| 聚合 | 锚 ms | 五档残差 (ms) | 最大残差 | 键 |
|---|---|---|---|---|
| 主三遍·均值 | 105.931 | +0.848 / −2.177 / +0.406 / +0.539 / −0.441 | **2.177** | `#primary_mean` |
| 复制三遍·均值 | 107.99 | +1.430 / −0.857 / +3.008 / −0.519 / +4.104 | **4.104** | `#replication_mean` |
| 复制三遍·**逐遍中位数** | 107.455 | +0.770 / −0.421 / −1.063 / +0.409 / −0.165 | **1.063** | `#replication_median_of_sessions` |

→ 正文口径句：**斜率复现，前提是用逐遍中位数**（1.063 ms < 主三遍自身的 2.177 ms）；用均值会被一次离群推 4 ms。

**分桶溢价的配对复现**（`#summary.paired_bucket_premium`）：

| 族 | 浪费片/帧 | 主三遍逐遍 (ms) | 主中位 | 主 ms/片 | 复制逐遍 (ms) | 复制中位 | 复制 ms/片 | 中位位移 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| `c_plus_10` | 3.4967 | 12.286 / 14.204 / 14.146 | 14.146 | 4.046 | 13.754 / 14.155 / 30.154 | 14.155 | 4.048 | **+0.064%** | 复现 |
| `⌈1.5c⌉` | 1.3336 | 5.468 / 4.454 / 5.623 | 5.468 | 4.100 | −0.044 / 7.837 / 2.831 | 2.831 | 2.123 | **−48.226%** | 未复现 |

**逐遍中位数 + 同遍控制档归一**的写法：见 **§F′** 的两张表（主遍 / 复制遍各 11 档，含归一比值与跨遍漂移），
那是正文表 T4′ 的直接数据源，已按 `rule_timing_v1/rule_timing.json#arms[*].session_means_ms` 与 `rule_timing_v1_rep` 同名键逐档复算。
示例档 `sync_1.5c`：主遍中位 115.763 ÷ 同遍控制 416.035 = **0.2783**；复制遍 116.852 ÷ 411.792 = **0.2838**，归一漂移 **+1.98%**。
全局：10/10 非控制档归一漂移 **+1.02%…+3.42%**，而同遍控制档自身绝对毫秒快了 **−1.02%**——所以归一后仍不等于"零漂移"。
⚠️ 上表"复制遍逐档读数"与 §F 的 11 臂表都是**三遍均值**口径，只用于说明机器态与离群；正文引绝对毫秒一律走 §F′ 的中位数口径。

**未解释的仪器规律**：复制三遍中"该遍最大偏离"全部落在**臂序第 8 位**（三遍各为不同臂），主三遍无此规律（第 7/3/5 位、幅度 ≤2.24%）。来源：`replication_compare.json#per_arm` + `rule_timing(_rep).json#sessions_meta[*].arm_order`。正文只写"未解释"，两种候选解释（序号效应 / 约半小时节律的外部活动）现有日志分辨不了。

**g 计时窗的越界复核**（判据来自预注册纪律，不是事后加判据）：被污染的 session 2 的 g=0.75 批（00:15–00:26 原始 1,073 行，只重新汇总、未发新请求）读数 503.923 ms，对另两 session 均值 505.171 ms 偏离 **−0.25%**（阈 30%），探针膨胀比 1.9883 为三遍最低 → 不重测。反向证据：g=1.0 三遍膨胀比 2.6954 / **1.1302** / 2.6784，最干净的一遍反而最慢（418.71 vs 413.65 / 409.98）→ 服务时间对 GPU 干扰代理近乎不敏感（138% 探针膨胀换 2% 墙钟）。台账 §17 末 + §20。

---

## H. 掩码还原分辨率 g 轴（正文 Negative results / 表 T5）

质量：`jrtip/mask_restore_resolution_v1/restore_quality.json#arms.{g1,g0p75,g0p5}`；时间：`jrtip/mask_restore_timing_v1/restore_timing.json#arms.{g1,g0p75,g0p5}`（3 session × 1,073 请求 = 3,219）。

| 量 | g=1.0 | g=0.75 | g=0.5 | 键 |
|---|---|---|---|---|
| **端到端 mean_ms** | **414.1142** | **504.7548（+21.89%）** | **466.3193（+12.61%）** | `#mean_ms`；百分比派生，亦存为 `#service_reduction_vs_g1` 的相反数（−0.2189 / −0.1261） |
| p50 / p95 / p99 ms | 426.655 / 496.017 / 542.586 | 512.651 / 677.177 / 735.202 | 472.860 / 631.374 / 680.839 | `#p50_ms`、`#p95_ms`、`#p99_ms` |
| 饱和吞吐 req/s | 2.4150 | 1.9812 | 2.1445 | `#saturated_throughput_per_second` |
| 服务器还原段 ms | 103.4317 | 62.3889 | 30.3253 | `#stage_means_ms.restore_scores_masks_rle_serialize_sync_ms` |
| 每片掩码还原单价 ms/片 | **1.155** | **0.697** | **0.339** | 派：上行 ÷ 89.519 片/帧（`restore_quality.json#control.work_masks_per_frame`；cap 未被约束，见 `#arms.*.mean_instances_masked_per_frame = 89.519`） |
| 还原段相对 g=1.0 的节省 ms | 0 | **−41.0428** | **−73.1064** | `#restore_ms_minus_g1` |
| 客户端还原回原尺寸 ms/帧 | 不适用（requests = 0） | **138.0774** | **130.0523** | `#client_resample.mean_ms`（`#client_resample.requests = 3219`） |
| 客户端/服务器单价比 | — | **3.4×** | **1.8×** | 派：138.0774/41.0428、130.0523/73.1064（台账 §20 原文"每片单价是服务器节省的 3.4×/1.8×"用的是掩码单价口径，两处口径不同，正文只引一种并写公式） |
| 序列化字节/帧（计时流） | 20,650.4 | 18,627.7（−2,022.7） | 13,756.4（−6,894.0） | `#serialized_bytes_mean`、`#serialized_bytes_minus_g1` |
| 掩码像素/帧 | 18,824,792.0 | 10,601,035.2 | 4,706,677.4 | `restore_quality.json#arms.*.mean_mask_pixels_per_frame` |
| 宏 AP | 70.4607912 | 69.7117452 | 68.7505709 | `restore_quality.json#arms.*.macro.AP` |
| ΔAP_pp（宏） | 0.0 | **0.749**（`gate_ap_drop_within_1pp = true`） | **1.7102**（该门 = false） | `#ap_drop_vs_control_pp`、`#gate_ap_drop_within_1pp` |
| 计数代价 | 0.0 | 0.0（门 true） | 0.0（门 true） | `#count_MAE_increase_vs_control_leaf`、`#gate_count_MAE_within_0_2_leaf` |
| 越 1 pp 的基因型格 | 0/18 | **3/18**（Col-0 −1.3572、Zdr-1 −1.2356、Cvi-0 −1.1968） | **16/18**（最差 Col-0 −3.4785） | `#ap_drop_by_genotype`、`#worst_genotype.ap_drop_vs_control_pp` |
| 交付尺寸 | 全分辨率 | 客户端还原后全分辨率 | 同 | `#delivered_hw_equals_original`、`#frames_delivered_at_full_resolution`、`#resampled_client_side` |
| 产物 failures | — | — | `["g0p5_ap_drop_within_1pp"]` | `#failures` |
| 客户端还原配方（原文） | "decode RLE → float32 → F.interpolate bilinear to the declared frame → ≥0.5 → re-encode; boxes scaled by (w/w_g, h/h_g) and re-clamped" | | | `#client_resample_recipe` |
| 仪器门 | g=1.0 流 vs 归档 cap-100：2,947 帧，长度差 0、分数/框差 0、**掩码差 0 of 263,813 实例**、参照缺失 0 | | | `#g1_gate_vs_archived_cap100.{frames_compared,length,score_or_box,mask,missing_reference}` |

> ⚠️ **符号陷阱**：同一产物里 `#ap_drop_vs_control_pp` 是"降为正"（+0.749 = 掉 0.749 pp），而 `#ap_drop_by_genotype[*]` 与 `#worst_genotype.ap_drop_vs_control_pp` 是"变化量"（−1.3572 = 掉 1.36 pp）。正文写基因型句时必须取绝对值并注明符号约定，否则会把"掉"写成"升"。

**结论句**（正文按此写，两句都有实测支撑）：还原段单价确实近似随掩码面积线性下降（1.155→0.697→0.339 ms/片），**但端到端更慢 +21.89%/+12.61%**，因为服务契约要求交付全分辨率掩码；两端都按掩码片数线性伸缩 ⇒ 换任何预算档都翻不了符号 ⇒ g 不进动作空间，v2.1 的 (k,g) 二维动作与逐实例还原分辨率分配都不必再开。

---

## I. 学习型控制器（P1b）与停损判决（正文 Negative results / 表 T6）

来源 `jrtip/budget_p1b_v1/condition1_verdict.json`（含 `#criterion` 原文，判据在任何控制器数字存在前写下）、`equal_work_C1.json`、`arms_C1.json`、`controller_C1.json`、`fit_ledger.jsonl`；台账 §17、§19.1。

| 项 | 值 | 键 / 性质 |
|---|---|---|
| 等量窗口 | ±5%（`work_window = 0.05`） | 存 `#work_window` |
| 候选数 / 内部候选 / 互异策略 | 3 / **1** / 3 个摘要 `53ea…`、`5594…`、`bbb0…` | 存 `#candidates`、`#candidates_interior`、`#distinct_policy_digests` |
| 免标签对照条目 | 48（另 3 条静态档因预注册门被显式跳过） | 存 `#label_free_comparators`；跳过原因见 `#skipped` 与台账 §19.1 |
| **判决** | **`condition_1_triggered = true`，`winners = []`** | 存（同名键） |
| 内部候选读数 | work 15.405 片/帧，ΔAP **0.5994 pp**，Δ计数 MAE 0.0，截断帧 0 | 存 `#interior_row`（`#work`、`#ap_drop_pp`、`#count_leaf`、`#truncated_frames`） |
| 同窗最强免标签档 | `clip(ceil(1.25*count),15,100)`：work 15.905，ΔAP **0.4755 pp** | 存 `#interior_row.strongest_comparator` |
| 窗口 | [14.635, 16.175]，窗内 3 条对照 | 存 `#interior_row.window`、`.comparators` |
| **学习器净差** | **`label_free_gain_pp = −0.1239`**，`beats_strongest = false` | 存（不是派生） |
| 对次强档 | `clip(c+2,15,100)`：15.442 / 0.5860 pp → 仍输 0.0134 pp | 派（`equal_work_C1.json` 窗口内逐条；Δ = 0.5994 − 0.5860） |
| 边界候选 τ=−∞ | 即 cap-100 控制（work 89.519，零削减），窗内无过线免标签档 | 存 `#boundary_rows` |
| 边界候选 τ=+∞ | 逐帧均匀 15：work 15.0，ΔAP 1.1888 pp，截断 201 帧，`passes_gates = false`，输给 `clip(count+2,15,100)` 0.6028 pp | 存 `#boundary_rows`（201 与 §A 派生行独立对上） |
| 坍缩原因 | 三折 OOF：50 档与 30 档分位头 MAE **恰为 0.0**、实现超越率 0.0（名义 0.1）；20 档 0.0029/0.0032；风险质量几乎全在 15 档（N 头 MAE 0.2971）⇒ P5–P95 共 19 个阈值给出同一张预算表 | 存 `controller_C1.json` / `fit_ledger.jsonl`；台账 §17 |
| 分配确有内容（对照） | 同一预算多重集随机置换：同工作量 1.1685 pp + **187 帧截断**，比学习策略差 **0.5691 pp** | 存 `equal_work_C1.json` 的随机臂行；Δ 派生 = 1.1685 − 0.5994 |
| 并入两消融后的复核 | candidates 9 / 内部候选 3 / 互异策略 5 / `winners = []` / `condition_1_triggered = true` | 存 `jrtip/budget_p1b_v1/condition1_with_ablations.json` |
| 机器复核作业 | stage 58/59 双 rc=0（01:22:07） | 台账 §17 / `stage2_logs/queue_console.log` |

正文口径：**停损第 1 条触发 ⇒ 正式矩阵不开跑，本稿按"机制分解 + 免标签规则"的配置型论文走**；第 2 条因前置作业取消而不可判（moot），另一半"实现开销吃掉预算收益"早已可判否（决策同步 0.263 ms，占服务时间 0.2%）。三条停损原文与阈值一字未动（方案 v2 §9）。

---

## J. 消融 A1 / A3 与由它们暴露的仪表性质（正文 Interpretation bounds）

来源 `jrtip/budget_p1b_v1/arms_A1.json`、`arms_A3.json`、`equal_work_ablations.json`、`fit_ledger.jsonl`；台账 §19。

| 项 | 值 | 性质 |
|---|---|---|
| A1（去候选分数序列特征）三个策略的质量指标 | 与 C1 的三个在 **16 位有效数字全等**，但预算表不同 | 存（逐策略对比）；台账 §19 |
| A1 vs C1 预算表实际差异 | **只差一帧**：15 的帧数 −1、20 与 30 的帧数一字不动、多出一个 100 ⇒ 请求预算 +0.029 片/帧、实到掩码 +0.009 / +0.008 片/帧 | 派（`policy_C1_s*.json` vs `policy_A1_s*.json` 直方图差）；台账 §19 |
| A3（关带外保护回退）三个策略 | `digest` 与 C1 **逐位同名** | 存；台账 §19 |
| A3 的因果 | C1 在 DEV 上 `frames_sent_to_fallback = 0` ⇒ 被消融的分支在该流上从未执行 | 存 `controller_C1.json` / `arms_C1.json` 字段 |
| A3 的正确读法 | 只能读作"保护带在与 TRAIN 同分布的开发流上零触发"，不能读作"保护机制无价值" | 判读句 |
| 夜间唯一一次保护触发 | 出现在 A1：逐帧越带比例 0.0083 → 0.0090，恰好一帧越过 0.25 的门被整帧送回 k=100 | 存 `fit_ledger.jsonl` / arms 的 out-of-band 字段 |
| **仪表性质（合成用例实测）** | COCO 101 点插值 AP 对"排在分数末尾、不新增匹配"的低分假阳**精确不敏感**（尾部加两个假阳 AP 一位不变；把它们提到分数顶端 AP 由 100 → 50）；计数侧另有 `≥0.5` 门 | 存（合成用例断言输出，台账 §19）；正文按"实测的仪表性质"引用，不是推测 |
| 由此限制的两句话 | ①A1 行不能读成"特征组无信息量"，只能读成"作用范围落在两道门都看不见的尾部"；②本轮所有"过线"含义是"砍掉尾部不花质量"，**不可反过来说**"多吐尾部不花质量" | 判读句 |
| 顺带解释 | 时间按全部掩码片计价、质量只对 ≥0.5 那部分收钱 ⇒ −72% 时间与 −0.4 pp 可同时成立 | 判读句（与 §D 的 work/ΔAP 结构一致） |

---

## K. 离线成本模型（正文 Method / 图 2 右）

来源 `bucket_pricing.json#cost_model`（离线复算的**总墙钟秒**模型，不是每帧服务时间模型）。

| 量 | 值 | 键 |
|---|---|---|
| 斜率 | 11.8929 s / (instance·frame) | `#slope_seconds_per_instance_per_frame_mean` |
| 截距 | 245.749 s | `#intercept_seconds` |
| R² | 0.997599 | `#r_squared` |
| 最大绝对相对残差 | 0.04095 | `#max_abs_relative_residual` |
| 折算 | 4.036 ms / 额外掩码实例·帧 | `#ms_per_extra_masked_instance_per_frame` |
| 自洽校验（派生） | 11.8929 × 19.353 + 245.749 = **475.9 s**，等于该档 `projected_wall_seconds_cost_model`（`sync_c_plus_10_f15`） | 派 |
| 与 GPU 静态斜率的交叉核对 | 离线 4.036 vs 设备 4.1161 ms/片 ⇒ 相差 **1.95%（派生）** | 派；两个量不同单位（总墙钟 vs 每请求服务时间），正文只能写成"两条独立路径给出的每片代价相差 2%" |

---

## L. 评价机器：前缀恒等与资格门（正文 Method / 附录）

| 项 | 值 | 来源 | 性质 |
|---|---|---|---|
| 机制 | `detections_per_img` 截断发生在 NMS 之后、掩码头之前（`RoIHeads.postprocess_detections` 的 `keep = keep[: self.detections_per_img]`）⇒ 预算 k 的输出 = cap-100 输出的前 k 项 ⇒ 一次 cap-100 前向可离线标价全部输出侧预算档 | 方案 v2 §4；台账 §9 首段（读实现，非推测） | 文 |
| 引理实测（六档非对照 GPU 流 × 2,947 帧） | 长度、顺序、分数、框 **0 处不一致**；掩码 RLE 在 k=50/30/25 **0 处**，在 k=20/10/5 分别 **15 / 7 / 4 处**（= 15/58,938、7/29,470、4/14,735，各约 0.025%）；`frames_fully_identical` 2,947/2,947/2,947 与 2,932/2,940/2,943 | `jrtip/budget_policy_v1/policy_eval.json#prefix_identity.<k>` | 存 |
| 那 26 例的性质 | 解码后是真实几何差 ±1 像素（面积 2184→2183、6105→6104 等），两版本 IoU 最低 **0.9917**、其余 ≥0.9989，差异集随档嵌套 | `src/jrtip_mask_string_diff_probe.py` 产物 + 台账 §9 | 存（探针输出） |
| 指标后果 | 七档 × 四项宏指标离线复算 vs GPU 实测 `max_abs_difference_pp` = **0.000e+00**（k = 100/50/30/25/20/10/5 全 0） | `policy_eval.json#offline_vs_measured.<k>.max_abs_difference_pp` | 存 |
| 三条 FAIL 保留不改判据 | `identity_cap20_masks_match`、`identity_cap10_masks_match`、`identity_cap5_masks_match` 在 `#failures` 里 | `policy_eval.json#failures`（`budget_policy_v1_extra` 同） | 存 |
| 引理的正确表述 | **在"指标层逐位等价"意义上成立，在"字节层完全相同"意义上不成立** | 台账 §9 原句 | 文 |
| 标价路径资格门（wiring） | cap-20：GPU 实跑 vs 离线前缀重算，AP/AP50/AP75/count_MAE 差 **0.0 / 0.0 / 0.0 / 0.0 pp**，`max_abs_difference_pp = 0.0`；work 19.999321（实跑）vs 19.999（离线）差 0.0003；字段来源 `work_source = "recomputed_from:quality_000020_S640.jsonl"` | `bucket_pricing.json#wiring_check` | 存 |
| 控制档复现冻结基准 | AP 70.4607911786734 / AP50 95.41115101923182 / AP75 80.92950804085432 / count_MAE 1.0139046254831632、work 89.519，与 `baseline_benchmark_v1/quality.json` 差 `+0.000e+00` | `bucket_pricing.json#control`；台账 §6 首条 | 存 |
| 计时侧按位一致门 | 240 帧版 `all_gates_passed = true`（门 A：固定 cap=k 包装器 vs 真 cap-k 前向；门 B：变 k 规则档 vs cap-100 前缀），证据 `jrtip/rule_timing_v1/gate240/identity_gate.json`；主三遍与复制三遍实际用 `--gate-frames 24`，三遍门全过 | 台账 §11；`rule_timing.json#sessions_meta[*].gate`、`replication_compare.json#gates_passed_before_timing` | 存 |
| 合并纪律 | 先算每档 session 均值、再跨 session 平均；单请求永不跨 session 合并 | `rule_timing.json#aggregation`、`rule_timing_readout.json#aggregation_reused` | 存 |
| 训练域数据 | TRAIN 域 cap-100 标价遍 + 逐帧特征：8,070 帧 = 150 株，结构审计 0 失败 | `jrtip/budget_train_stream_v1/train_stream_audit.json`；台账 §15 | 存 |

---

## M. 台账/方案侧陈述（写作时引用，不是测量值）

| 项 | 内容 | 出处 |
|---|---|---|
| 预注册假设 | H1–H3、贡献 (a)–(d)、动作集 K、两道门、§9 三条停损在任何控制器数字存在前写下 | 方案 v2 §3–§5、§7.1、§9 |
| 停损第 1 条触发后的处置 | 本线按配置型论文走；正式矩阵不开跑；改阶梯/(k,g) 属新方案需另行批准 | 方案 v2 §9 判决段 + 台账 §17 |
| 队列账 | 21:40:44 首启 → 04:01:41 `rule replication queue finished`；GPU 计时 2 h 38 min 53 s；计算与标价 1 h 49 min 52 s；纯 CPU 校验 1 min 50 s；失败重试 55 min 52 s（全在工具侧、无一条用于改判据） | `stage2_logs/queue_console.log` + 台账 §18/§21 + 今晨纪要第二节 |
| 红线 | `C_POLICY / T_ID / T_GENO / T_EXT` 全程未加载、未访问；原始数据/旧代码/旧论文一字未改；未装包、未改权限、未改 site-packages；未上传、未发布 | 各产物 `locked_roles_accessed = []`（机器可证部分）+ 今晨纪要末段 |
| 夜间四处工具缺陷 | ①`Counter` 键解包（rc=1，数据全留、只缺聚合）；②session 3 缺 `import os`（3,219 条逐请求记录全留）；③漏传 `--gate-frames` 被守卫拒绝（无脏数据）；④合并对空输入返回 rc=0 并写出 `0 arms from 0 sessions` 的空表 | 台账 §17/§18/§21；留档文件名见该节 |
| 窗内越界一处（已记账、已复核、不重测） | 00:30 前后两次约 2 s 纯 CPU 进程跑在 session 2 计时窗内；按预写判据检验为 −0.25% 偏离、膨胀比三遍最低 ⇒ 不重测 | 台账 §17 末、§20；数字在本文 §G 末行 |

---

## N. 禁用与受限读数登记册（负清单，写稿时逐条对照）

| 项 | 状态 | 原因 |
|---|---|---|
| `rule_timing*.json#arms.*.equal_domain_mean_ms_session_averaged`（如 `bucket_c_plus_10_f15` = 149.386）、`mask_restore_timing_v1#arms.*.equal_domain_mean_ms`（441.2229 / 548.2463 / 503.3014）、`scales.json#equal_domain_mean_ms`（454.724 / 451.539 / 446.868） | **禁用** | 该"同域"聚合的定义未在产物里断言，与 `mean_ms` 不可混用；正文只引 `mean_ms` |
| `mask_restore_timing_v1#arms.*.service_ms_plus_client_resample`（642.8322 / 596.3716）与 `#net_reduction_if_client_restores_full_res`（−0.5523 / −0.4401） | **作废** | `#mean_ms` 已含客户端还原段，这两列把客户端费用重复计一次；正确读数是端到端 **+21.89% / +12.61%**（台账 §20 已就地作废） |
| `bottleneck_decomposition_v1#verdict`（四项全 false）与 `#resolution_response`（317.21/343.54/342.62、ratio 1.0830、`forward_ms_per_megapixel` 205.70/82.70） | **不引用** | 建立于"干净窗"n = 2/16/12 的边界样本；脚本 docstring 指定 `all` 列为结论口径。判决虽在两定义下同向，但正文若引 `all` 列又附 `verdict` 布尔，会出现同段两个不同样本量 |
| ⌈1.5c⌉ 的分桶溢价（5.468 → 2.831 ms、4.100 → 2.123 ms/片、位移 −48.226%） | **不作结论** | 复制遍未复现；正文那句"分桶比同步每片多付 4.49%–10.77%"必须改成只讲 `c+10`（4.49% 那头的证据已降级） |
| "static15 锚 + 4.116 ms/片 ⇒ 五档落在 ±2.2 ms 内" | **改为逐遍中位数引用** | 用均值时复制遍最大残差 4.104 ms；用中位数 1.063 ms（§G 表） |
| "分桶实现额外付费"这类泛化 | 限定为**浪费片数 × 单价**，且只由 `c+10` 支撑 | 同步实现本身不罚款（`readfull` 与 `sync` 同预算、差 −2.583 ms 且 `resolvable = false`） |
| A1 行"去掉特征后指标不变" ⇒ "特征无信息量" | **禁止该推理** | §J 仪表性质：差异落在 AP/计数两道门都看不见的尾部 |
| A3 行"逐位不变 ⇒ 保护无用" | **禁止该推理** | 被消融分支在该流上 0 次执行 |
| 任何绝对毫秒不带同遍控制档 | 禁止 | §21 口径 |
| 把质量数字与**不同档**的时间数字配对 | 禁止 | `rule_timing_readout.json#scope` 原文："no quality number here is paired with a time number from a different arm"（同名档的离线质量 + 设备时间这一对，按 §E 明确标注两个来源后方可用） |
| 把 §D 的 0.774（池化）当 count MAE 报 | 禁止 | §A 派生行的定义说明 |

---

## O. 图 — 数字对应（供任务 #30）

| 图 | 数据行 | 全部取自 |
|---|---|---|
| Fig 1 (a) 工作—质量阶梯 | 19 静态节点 + `frontier_within_tolerances` | §D |
| Fig 1 (b) 免标签 23 档叠加 | `bucket_pricing.json#arms` 的 (work, ΔAP)，七过线档高亮 | §D/§E |
| Fig 2 (a) 服务时间分解堆叠 | §C 表 `all` 列中位数（S640 一列 + 三尺度对照） | §C |
| Fig 2 (b) 每片掩码单价 / 成本模型 | 4.1161（设备）与 4.036（离线）+ R² 0.997599 与残差 | §C/§F/§K |
| Fig 3 两门散点（时间削减 × 质量代价，11 臂） | §F 主三遍表 + §E 门位 | §E/§F |
| Fig 4 (a) 主三遍 vs 复制三遍位移 | §F + §G 复制遍逐档读数 | §F/§G |
| Fig 4 (b) 静态定律残差：均值 vs 逐遍中位 | §G 三行残差表（2.177 / 4.104 / 1.063） | §G |
| Fig 4 (c) 分桶溢价配对复现（`c+10` 复现、`⌈1.5c⌉` 未复现） | §G 表 | §G |
| Fig 5 g 轴：端到端 vs 服务器还原段 vs 客户端还原 | §H 表 | §H |
| Fig 6 等量窗口内的判决图（学习 1 点 vs 窗内 3 档） | §I 表 | §I |
