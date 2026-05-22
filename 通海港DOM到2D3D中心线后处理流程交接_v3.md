# 通海港 DOM 到 2D/3D 中心线后处理流程交接 v3

更新时间：2026-05-22

这份文档给下一次新对话接手用。当前对话经历了多次上下文压缩，且包含大量截图验收反馈；下一轮应优先读本文，不要从聊天记录里反推当前状态。

## 一句话结论

当前最值得继续的成果不是旧版 `final_delivery`，而是一个新的 strict-auto 道岔支线后处理实验：

```text
scripts/prototype_turnout_outer_rail_centerline.py
```

它针对当前 DOM 内自动识别到的 `AUTO_001` 到 `AUTO_007` 道岔支线，用 DeepLab 语义分割概率图里的钢轨证据重新约束中心线。最新合并输出在：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.shp
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.geojson
```

重要边界：

- 这些结果仍在 `experiments/outer_rail_all_turnouts` 下，没有提升为正式交付。
- 没有替换 `output/dom_centerline_strict_auto_v1/final_delivery`。
- 用户截图坐标只能作为 QA probe，不能写进生产约束。
- 旧验收版本 v15/v19/v20 不能当作算法输入或标准答案。
- 用户中途指出的点位不能当作自动化约束。
- 当前要先解决 2D 中心线与拓扑，Z 高度补全放在 2D 验收后。

## 当前用户验收状态

用户前面已经明确表示整体效果接近可验收，但随后在“完全自动、不能依赖人工线或旧成果”的方向上继续推进。最近一轮主要问题集中在道岔支线的局部偏移，尤其是岔心附近多根钢轨互相干扰时，中心线容易被错误钢轨拉偏。

最近一次全局实验后，用户检查结果是：

- 大部分支线没有明显严重错误。
- 剩余重点是截图对应的某段支线，表现为局部偏左。
- 这类问题不能靠某个道岔特调解决，必须从 DeepLab 钢轨证据和候选评分逻辑上解决。

因此下一轮不要直接宣布“完成最终版本”。正确姿态是：当前实验版有明显进展，但仍需要用户重点验收 `AUTO_005`、`AUTO_006`、`AUTO_007`，再决定是否并入正式 pipeline。

## 当前核心算法

入口脚本：

```text
scripts/prototype_turnout_outer_rail_centerline.py
```

主要输入：

```text
output/dom_centerline_strict_auto_v1/08_auto_turnout_crossover_evidence/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson
output/dom_centerline_strict_auto_v1/06_mainline_prior/mainline_2_track_connected.geojson
output/dom_centerline_strict_auto_v1/01_dom_tiles/selected_tile_index.csv
output/dom_centerline_strict_auto_v1/02_deeplab_segmentation/probabilities/
data/**/dom.tif
```

处理逻辑：

1. 使用当前 strict-auto 生成的道岔支线作为粗 corridor 和切线参考。
2. 沿支线每 `0.5m` 建立横截面 profile。
3. 在 DeepLab rail probability tiles 中沿横截面采样钢轨响应。
4. 生成候选中心线修正：
   - `paired_outer`：双轨配对可信时，用两根钢轨的中点。
   - `single_left`：双轨配对不可信，但左侧单根钢轨位置接近理论半轨距时，用左轨反推中心线。
   - `single_right`：双轨配对不可信，但右侧单根钢轨位置接近理论半轨距时，用右轨反推中心线。
   - `invalid`：证据不足，不应当被当成高置信中心线。
5. 用动态规划选择连续的候选序列，避免逐站跳到相邻轨道。
6. 对修正量做 median 和 gaussian 平滑。
7. 对 unsupported gap 做保护，防止缺口两端大修正扩散。
8. 对接主线端点做 taper，避免支线在尖轨或接主线位置突变。
9. 输出中心线、支撑钢轨证据、CSV 审计和 DOM overlay。

当前关键参数默认值：

```text
gauge_m = 1.5
sample_step_m = 0.5
profile_step_m = 0.025
outer_search_min_m = 0.12
outer_search_max_m = 2.1
expected_rail_offset_tolerance_m = 0.55
expected_rail_offset_max_deviation_m = 1.35
inner_partner_tolerance_m = 0.28
single_rail_offset_tolerance_m = 0.28
min_peak_probability = 0.28
min_inner_rail_probability = 0.28
rail_continuity_penalty = 1.2
max_correction_m = 1.15
unsupported_gap_max_delta_m = 0.55
smooth_sigma_m = 3.0
median_window_m = 2.5
mainline_anchor_taper_m = 12.0
geometry_smooth_sigma_m = 5.0
geometry_smooth_end_taper_m = 8.0
```

最重要的算法经验：

```text
双轨配对可信 -> paired_outer
双轨疑似错配，但单侧钢轨稳定且接近半轨距 -> single_left / single_right
两者都不可信 -> invalid / 低置信
```

不要把 `paired_outer` 或单轨 fallback 当成绝对规则。严格双轨配对在岔心和邻线干扰区域太脆，单轨 fallback 如果太宽又会把错误峰值当成钢轨。当前可用版本的关键就是二者分层，而不是偏向某一侧。

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

字段解释：

- `centerline` 是实验输出的中心线。
- `evidence` 是实际采用的支撑钢轨证据，不一定永远是外侧轨；当 `support_kind=single_left` 时，它显示的是左手边支撑轨。
- `samples.csv` 里的 `support_kind` 是后续 QA 最重要的字段。

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
AUTO_001 max_turn=0.1717 deg p95_turn=0.1615
AUTO_002 max_turn=0.3947 deg p95_turn=0.2535
AUTO_003 max_turn=2.3815 deg p95_turn=0.1635
AUTO_004 max_turn=0.2196 deg p95_turn=0.2062
AUTO_005 max_turn=0.3334 deg p95_turn=0.2594
AUTO_006 max_turn=0.3033 deg p95_turn=0.2814
AUTO_007 max_turn=0.1993 deg p95_turn=0.1311
```

解释：`AUTO_003 max_turn=2.3815` 是端点相关，不代表主体弯轨突兀；它的 `p95_turn=0.1635` 更能代表主体。

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

- `AUTO_004` 中段已经回到 `paired_outer`，解决了之前偏左问题。
- `AUTO_006` 仍有不少单轨 fallback，这是预期行为，因为右侧容易被邻线或干扰轨误配。
- `AUTO_005`、`AUTO_006`、`AUTO_007` 单轨或 invalid 占比相对高，下一轮应重点目视验收。
- 只看 correction 最大值会误判；应结合 overlay、support kind、是否压轨和曲率连续性一起看。

## 本轮关键 QA 点

以下坐标只用于 QA，不是算法输入：

```text
AUTO_004: X=315366.869 Y=3520898.380
历史问题：搜索窗太窄，右轨落到旧搜索窗外。
处理结果：outer_search_max_m 放宽到 2.1m，优先回到轨距配对。

AUTO_004: X=315367.091 Y=3520899.035
历史问题：single_left fallback 容差太宽，把 -0.4m 峰误认为左轨，压掉右侧真实双轨证据。
处理结果：single_rail_offset_tolerance_m 收紧到 0.28；该段回到 paired_outer。
当前证据示例：station=37.5m 使用 paired_outer，outer=1.77m，inner=0.275m，gauge=1.495m。

AUTO_006: X=315599.718 Y=3522196.978
AUTO_006: X=315600.066 Y=3522197.741
历史问题：右侧强响应来自邻线或干扰轨，严格双轨配对会把中心线拉右。
处理结果：允许证据驱动的 single_left fallback，从左手边稳定钢轨反推中心。
```

局部 QA 图：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/AUTO_004/AUTO_004_QA_315367_091_3520899_035_closeup_after_tight_single.png
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/AUTO_006/AUTO_006_QA_315599_718_3522196_978_closeup_final.png
```

## 复现命令

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

测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_auto_turnout_crossover_evidence.py tests\test_dom_to_3d_guided_pipeline.py tests\test_deeplab_topology_centerline_network.py
```

最近一次记录结果：

```text
28 passed
```

注意：当前全局合并 Shapefile、contact sheet 和 support-kind 汇总 CSV 是用临时 Python 片段生成的，还没有固化为正式脚本。下一轮如果要工程化，应新增：

```text
scripts/package_outer_rail_turnout_experiment.py
```

## 与完整 DOM 到 3D 流程的关系

当前 repo 已经有从 DOM 到 strict-auto 中心线的上游结构，但本轮最活跃、最关键的改动仍然是道岔支线 2D 中心线 refinement。

可以这样理解当前流程：

```text
DOM
-> 切 tile / 建 tile index
-> DeepLab v3+ 语义分割，得到 rail probability tiles
-> mainline prior / 自动道岔支线候选
-> prototype_turnout_outer_rail_centerline.py 重算支线中心线
-> 2D shp / geojson / QA overlay / support-kind CSV
-> 2D 通过后，再进入 DSM 或 LAS 补 Z
-> 最终 3D PolyLineZ shp
```

Z 高度相关结论：

- DSM 和 LAS 都存在，但当前主战场不是 Z。
- 之前手扫 LAS 与 DOM/DSM 可能存在整体坐标偏差，不建议用 LAS 直接修 XY。
- LAS 更适合作为相对几何证据，例如轨距、股道间距、高度合理性。
- 正式补 Z 前应先冻结 2D 中心线版本。

## 已知风险

1. `AUTO_005`、`AUTO_006`、`AUTO_007` 仍需用户重点验收。
2. 单轨 fallback 不能继续放宽太多，否则会退化成“哪边有峰就信哪边”。
3. 严格双轨配对也不能作为唯一规则，因为岔心、交叉轨、邻线和遮挡会让配对错误。
4. 当前脚本仍是 prototype，不是正式 pipeline 模块。
5. 合并和 contact sheet 生成还没有脚本化，复现链条不够干净。
6. 当前输出只覆盖 `AUTO_001` 到 `AUTO_007`，是否等价于完整 DOM 所有道岔，需要以上游自动检测结果为准。
7. 旧 `final_delivery` 不能代表这次最新实验，也不能被静默覆盖。

## 下一轮建议顺序

1. 先让用户在 QGIS/DASView 中加载：

```text
output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.shp
```

2. 重点验收 `AUTO_005`、`AUTO_006`、`AUTO_007`，尤其看：

```text
是否压到钢轨
是否在岔心附近偏出中心
是否有突兀转角
是否在尖轨处与主线中心线相切
支线进入直轨后是否仍贴合轨道中心
```

3. 如果这 7 条支线验收通过，把 prototype 逻辑并入正式 strict-auto pipeline。
4. 把全局合并、support-kind 汇总和 contact sheet 生成固化成脚本。
5. 给候选评分和 fallback 增加单元测试，重点覆盖：

```text
paired_outer 优先级
single_left / single_right 容差
邻线强峰不能误配
invalid gap 不能传播大修正
fallback 不能制造突兀曲率
```

6. 正式 2D 输出冻结后，再走 DSM/LAS 补 Z，生成 3D shp。
7. UI 暂时不要做。用户已明确说 UI 后面再考虑。

## 给下一轮模型的硬性提醒

- 不要把用户截图坐标写进算法。
- 不要把旧验收 shp 当成标准答案。
- 不要把人工参考线当作当前正式依赖。
- 不要把某个道岔的经验写成全局硬规则。
- 不要为了看起来完成而覆盖 `final_delivery`。
- 不要把 LAS 当成 XY 修正依据，除非先验证坐标系统偏差已经解决。
- 不要只看 CSV 数字判断成功，必须结合 overlay 或用户目视验收。
- 不要把道岔支线做成单纯贝塞尔曲线补丁；曲线应由 DeepLab 钢轨证据驱动。

## 推荐新对话开场

可以把这句话发给下一轮：

```text
继续 D:\rail-curve-extractor 项目。先读《通海港DOM到2D3D中心线后处理流程交接_v3.md》，当前不要翻旧聊天图。重点接手 scripts/prototype_turnout_outer_rail_centerline.py 和 output/dom_centerline_strict_auto_v1/experiments/outer_rail_all_turnouts/all_outer_rail_centerlines.shp。不要用截图坐标、旧 v15/v19/v20 或人工线作为生产约束。先帮我确认最新实验结果和下一步如何并入正式 strict-auto pipeline。
```
