# 通海港 DOM 到 2D/3D 中心线后处理流程交接 v2

更新时间：2026-05-22

本文用于下一次对话接手当前工作。由于上一轮对话包含大量截图和多轮上下文压缩，下一轮优先读本文，不要从聊天记录里反推当前状态。

## 当前结论

当前最有价值的成果不是最终 `final_delivery`，而是一个新的道岔支线中心线后处理实验：

```text
scripts/prototype_turnout_outer_rail_centerline.py
```

它针对 `AUTO_001` 到 `AUTO_007` 的自动道岔支线，用 DeepLab 概率图里的钢轨证据重新约束中心线。当前效果在人工目测里已经解决了这一轮重点暴露的问题：

- `AUTO_004` 原先在 `X=315367.091 Y=3520899.035` 附近偏左，原因是单轨 fallback 太宽松，把不该信的左侧峰当成左轨。现在该段已回到右侧双轨配对。
- `AUTO_006` 原先在 `X=315599.718 Y=3522196.978`、`X=315600.066 Y=3522197.741` 附近偏右，原因是右侧干扰轨被误配。现在该段可用 `single_left` fallback，从左手边稳定钢轨反推中心。
- 其它支线当前没有发现严重错误，但 `AUTO_005`、`AUTO_006` 的单轨 fallback 比例较高，后续仍应重点验收。

重要边界：

- 这些输出仍在 `experiments/outer_rail_all_turnouts` 下。
- 没有替换 `output/dom_centerline_strict_auto_v1/final_delivery`。
- 没有把用户截图坐标写进生产约束；这些坐标只作为 QA probe。
- 没有读取旧验收版 v15/v19/v20 作为算法输入。
- 没有读取人工反馈点作为算法输入。

## 当前核心算法

入口脚本：

```text
scripts/prototype_turnout_outer_rail_centerline.py
```

输入：

```text
output/dom_centerline_strict_auto_v1/08_auto_turnout_crossover_evidence/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson
output/dom_centerline_strict_auto_v1/06_mainline_prior/mainline_2_track_connected.geojson
output/dom_centerline_strict_auto_v1/01_dom_tiles/selected_tile_index.csv
output/dom_centerline_strict_auto_v1/02_deeplab_segmentation/probabilities/
```

处理逻辑：

1. 使用 strict-auto 当前道岔支线作为粗 corridor 和切线参考。
2. 沿支线每 `0.5m` 建立横截面 profile。
3. 从 DeepLab 概率图采样钢轨响应。
4. 生成候选中心线修正：
   - `paired_outer`：双轨配对可信时，用两根钢轨中点。
   - `single_left`：双轨疑似错配或缺失，且左侧单轨峰接近理论半轨距时，用左轨加半轨距反推中心。
   - `single_right`：同理，用右侧单轨减半轨距反推中心。
   - `invalid`：证据不足，不直接作为可靠支撑。
5. 用动态规划按连续性选择候选序列，避免逐站跳轨。
6. 对修正量做 median/gaussian 平滑。
7. 对缺口做保护：无可靠证据缺口两端修正突变过大时，不让大修正跨缺口扩散。
8. 对接主线端点做 taper，避免支线在主线相切位置突变。
9. 输出中心线、支撑钢轨证据、CSV 审计和 DOM overlay。

关键参数（当前默认）：

```text
gauge_m = 1.5
sample_step_m = 0.5
profile_step_m = 0.025
outer_search_max_m = 2.1
expected_rail_offset_tolerance_m = 0.55
expected_rail_offset_max_deviation_m = 1.35
inner_partner_tolerance_m = 0.28
single_rail_offset_tolerance_m = 0.28
min_peak_probability = 0.28
min_inner_rail_probability = 0.28
rail_continuity_penalty = 1.2
unsupported_gap_max_delta_m = 0.55
smooth_sigma_m = 3.0
median_window_m = 2.5
mainline_anchor_taper_m = 12.0
geometry_smooth_sigma_m = 5.0
geometry_smooth_end_taper_m = 8.0
```

当前最关键的经验是：不要把“双轨配对”或“单轨 fallback”当成绝对规则。它们是分层证据：

```text
双轨配对可信 -> paired_outer
双轨疑似错配，但单侧钢轨稳定且接近半轨距 -> single_left / single_right
两者都不可信 -> invalid / 低置信
```

## 最新输出

全局 7 条支线合并输出：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.shp
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.geojson
```

全局审计：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_all_turnouts_summary.csv
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_geometry_audit.csv
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_support_kind_summary.csv
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_support_kind_runs.csv
```

总览图：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_all_turnouts_contact_sheet.png
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/outer_rail_support_kind_contact_sheet.png
```

每个分支目录包含：

```text
AUTO_xxx_outer_rail_centerline.shp
AUTO_xxx_outer_rail_centerline.geojson
AUTO_xxx_outer_rail_evidence.shp
AUTO_xxx_outer_rail_evidence.geojson
AUTO_xxx_outer_rail_samples.csv
AUTO_xxx_outer_rail_centerline_overlay.png
AUTO_xxx_support_kind_overlay.png
AUTO_xxx_outer_rail_centerline_summary.json
```

说明：

- `centerline` 是最终中心线实验输出。
- `evidence` 是实际采用的支撑钢轨证据，不一定永远是外侧轨；`single_left` 时会显示左手边支撑轨。
- `samples.csv` 里有 `support_kind`，是后续 QA 最重要的字段。

## 当前审计数值

`outer_rail_all_turnouts_summary.csv` 最新摘要：

```text
AUTO_001 valid=1.0000 correction=[-0.1943, 0.1205] p95_abs=0.1741
AUTO_002 valid=0.9673 correction=[-0.3423, 0.0624] p95_abs=0.3063
AUTO_003 valid=1.0000 correction=[-0.0391, 0.1000] p95_abs=0.0870
AUTO_004 valid=1.0000 correction=[-0.2303, 0.7125] p95_abs=0.6433
AUTO_005 valid=0.9489 correction=[-0.2523, 0.1237] p95_abs=0.2360
AUTO_006 valid=0.9786 correction=[-0.0713, 0.1149] p95_abs=0.1046
AUTO_007 valid=0.8958 correction=[-0.2228, 0.2304] p95_abs=0.2189
```

`outer_rail_geometry_audit.csv` 最新摘要：

```text
AUTO_001 max_turn=0.1717 deg
AUTO_002 max_turn=0.3947 deg
AUTO_003 max_turn=2.3815 deg  # 起点端部转角，主体 p95_turn=0.1635
AUTO_004 max_turn=0.2196 deg
AUTO_005 max_turn=0.3334 deg
AUTO_006 max_turn=0.3033 deg
AUTO_007 max_turn=0.1993 deg
```

`outer_rail_support_kind_summary.csv` 最新摘要：

```text
AUTO_001 paired=136 single_left=8  single_right=6  invalid=0
AUTO_002 paired=128 single_left=2  single_right=18 invalid=5
AUTO_003 paired=144 single_left=1  single_right=10 invalid=0
AUTO_004 paired=146 single_left=0  single_right=15 invalid=0
AUTO_005 paired=93  single_left=23 single_right=14 invalid=7
AUTO_006 paired=89  single_left=21 single_right=27 invalid=3
AUTO_007 paired=309 single_left=8  single_right=27 invalid=40
```

解读：

- `AUTO_004` 当前中段已回到 `paired_outer`，解决了偏左问题。
- `AUTO_006` 仍大量使用单轨 fallback，这是预期行为，因为右侧容易受邻线干扰。
- `AUTO_005` 和 `AUTO_007` 的低置信/单轨比例较高，下一轮应优先复查。
- `AUTO_003` 的 `max_turn` 出现在端点，不代表主体弯轨突兀。

## 本轮关键 QA 点

以下坐标只用于 QA，不是算法输入：

```text
AUTO_004: X=315366.869 Y=3520898.380
问题：粗线偏左导致右轨落到旧搜索窗外。
修复：放宽搜索窗，优先使用轨距配对。

AUTO_006: X=315600.066 Y=3522197.741
AUTO_006: X=315599.718 Y=3522196.978
问题：右侧强响应来自邻线/干扰轨，严格双轨配对会拉右。
修复：允许稳定左手边单轨 fallback，反推中心。

AUTO_004: X=315367.091 Y=3520899.035
问题：single_left fallback 容差太宽，把 -0.4m 峰当左轨，压掉右侧真实双轨。
修复：单轨 fallback 单独收紧到 single_rail_offset_tolerance_m=0.28。
当前：station=37.5m 使用 paired_outer，outer=1.77m, inner=0.275m, gauge=1.495m。
```

局部 QA 图：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/AUTO_004/AUTO_004_QA_315367_091_3520899_035_closeup_after_tight_single.png
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/AUTO_006/AUTO_006_QA_315599_718_3522196_978_closeup_final.png
```

## 运行命令

单跑一个道岔：

```powershell
.\.venv\Scripts\python.exe scripts\prototype_turnout_outer_rail_centerline.py --branch-id AUTO_004 --out-dir output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/AUTO_004
```

批量重跑 7 个道岔：

```powershell
$branches = 'AUTO_001','AUTO_002','AUTO_003','AUTO_004','AUTO_005','AUTO_006','AUTO_007'
foreach ($branch in $branches) {
  .\.venv\Scripts\python.exe scripts\prototype_turnout_outer_rail_centerline.py --branch-id $branch --out-dir "output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/$branch"
  if ($LASTEXITCODE -ne 0) { throw "Failed branch $branch" }
}
```

当前合并 Shapefile、contact sheet 和 support-kind CSV 是本轮用临时 Python 片段生成的，还没有固化成正式脚本。下一轮如果要工程化，建议新增一个正式聚合脚本，例如：

```text
scripts/package_outer_rail_turnout_experiment.py
```

测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_auto_turnout_crossover_evidence.py tests\test_dom_to_3d_guided_pipeline.py tests\test_deeplab_topology_centerline_network.py
```

最新验证结果：

```text
28 passed
```

## 下一轮建议

优先级从高到低：

1. 在 QGIS/DASView 中加载 `all_outer_rail_centerlines.shp`，重点复查 `AUTO_005`、`AUTO_006`、`AUTO_007`。
2. 如果这 7 条支线验收通过，把 `prototype_turnout_outer_rail_centerline.py` 的逻辑并入正式 strict-auto pipeline，而不是停留在 experiment。
3. 固化聚合脚本，避免每轮用临时 Python 片段合并 shp 和生成 contact sheet。
4. 给 `support_kind` 做正式字段输出和 QA 规则：
   - `paired_outer`：高置信。
   - `single_left` / `single_right`：中置信，需要重点审。
   - `invalid`：低置信，不应悄悄当高质量输出。
5. 通过验收后，再考虑把 2D 结果送入 Z 高程补全，输出 3D PolyLineZ。

## 注意事项

- 不要把本轮用户截图坐标写成生产约束。
- 不要把旧验收版中心线当标准答案训练或约束。
- 不要因为某一处左轨好识别，就写成“所有该类道岔一律信左轨”。
- 单轨 fallback 必须证据驱动：峰值高、位置接近半轨距、不会造成突兀曲率。
- `AUTO_004` 的修正量最大，是因为粗线偏移较大，不应只凭 `correction_max` 判失败。
- 当前脚本是实验脚本，下一轮如果要作为正式流程，需要补正式测试和 pipeline 集成。
