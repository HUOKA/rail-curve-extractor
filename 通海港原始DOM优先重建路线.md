# 通海港原始DOM优先重建路线

## 方向重置

这轮通海港中心线重建应视为一次路线重启，而不是继续微调旧结果。

旧路线的问题不是某一个阈值没调好，而是起点错了：之前大量工作基于旋转对齐、裁到只剩轨道走廊的 DOM，甚至继续围绕旧的 `final_centerline_network.shp` 或对齐走廊拓扑骨架做修补。那些结果可以帮助理解轨道结构、校准轨距、生成初始 ROI，但不能再作为最终生产流程的主干。

新的主线必须从原始生产 DOM 开始：

```text
data/生产数据/无人机数据/正射/dom.tif
```

也就是说，最终软件要能读取 DJI Terra / 大疆智图的原始成果目录，而不是依赖人工旋转裁切后的中间图。

## 旧成果的新定位

| 旧成果 | 新定位 |
| --- | --- |
| `final_centerline_network.shp` | 失败样例和验收对照，不作为真值 |
| 对齐走廊 DOM / `aligned_dom.tif` | 标注辅助、ROI footprint、模型预训练分布 |
| v7 单根铁轨模型 | 可迁移性待验证的初始模型，不等于原始 DOM 生产模型 |
| v7 道岔区域模型 | 原始 DOM 上误检多，暂不作为生产掩膜 |
| `topology_skeleton_v1/v2` | 站场结构参考，不是最终中心线 |
| 手持 LAS | 轨距、股道间距、局部高度和道岔几何校验 |
| 点云单轨补全 | 后期几何补全模块，不是全场中心线主入口 |

这些东西不删除，因为它们仍有工程价值。但后续判断“有没有做对”，必须回到原始 DOM 叠图和原始 DOM tile 推理结果上，而不是只看对齐走廊图。

## 新主流程

```text
原始 dom.tif
  -> 基于已有地理证据生成轨道走廊 ROI
  -> 导出原始方向 3072 tile
  -> 在原始 DOM tile 上跑单根钢轨/轨道语义分割
  -> 用少量人工 QA 评估迁移效果
  -> 必要时补标原始 DOM tile 并重新训练
  -> 将钢轨 mask 转为 EPSG:32651 地图坐标候选
  -> 使用 1.50 m 左右目标轨距过滤同股钢轨配对
  -> 先重建贯通主线，再重建侧线死端
  -> 道岔连接只在可信工作区内生成，并要求与主线/侧线相切
  -> 用 DSM / 手持 LAS 补 Z 和局部几何校验
  -> 输出最终中心线网络
```

## 当前已有证据

- 原始 DOM 已确认是 `36983 x 111685`，CRS 为 `EPSG:32651`，分辨率约 `0.0326 m/px`。
- 原始 DOM 轨道走廊 ROI 已生成在 `output/raw_dom_corridor_roi`，当前收敛到 `293` 个 `3072 x 3072` 候选 tile。
- 原始 DOM 抽样 QA 表明：单根铁轨模型在部分直线轨道上仍有可用信号；道岔区域模型误检明显，不能生产使用。
- 手持 LAS 全量流式诊断支持把可见钢轨候选配对目标距离先设为约 `1.50 m`，接受范围先用 `1.35 - 1.65 m`。

## 立即下一步

下一步不是继续优化旧 `topology_skeleton`，也不是继续修旧点云 pipeline，而是建立原始 DOM ROI 全量推理入口：

1. 从 `output/raw_dom_corridor_roi/raw_dom_tile_index.csv` 流式导出或直接读取 293 个原始 DOM tile。
2. 对所有 ROI tile 跑单根铁轨模型，保留 mask、概率图、叠图和 tile georef。
3. 在原始 DOM tile 上做候选线提取，输出地图坐标中的单根钢轨线段，而不是直接跳到最终中心线。
4. 先做 DOM 叠图 QA，确认单根钢轨本身是否稳；只有钢轨证据稳了，才进入轨距配对、主线贯通和道岔连接。
5. 道岔模型需要补原始 DOM 负样本后重新训练；在这之前，道岔连接只能作为低可信候选，不能自动定稿。

可直接执行的一键入口是：

```powershell
.\.venv\Scripts\python.exe scripts\run_raw_dom_roi_fullpass.py --max-tiles 2 --dry-run
```

真正全量跑时，把 `--max-tiles 2` 去掉，并保持 `--device cuda`。默认输出会写到 `output/raw_dom_roi_fullpass_v1`，其中包含原始 DOM ROI tiles、rail mask、候选线和各阶段 summary。

2026-05-18 的全量实跑已经完成，结果在 `output/raw_dom_roi_fullpass_v1`：

- 293 张原始 DOM ROI tile 全部导出并完成 rail 模型推理。
- `rail_centerline_candidates.geojson` 的 `coordinate_space` 为 `map`，说明候选已经落到 EPSG:32651 地图坐标。
- 全量候选摘要为：`90638` 个 rail center points、`42127` 个 grouped map-space centerline candidates、`607` 个 grouped track fragments。
- 这个结果还不是最终中心线，但它证明原始 DOM 主线入口和地图坐标候选链路已经跑通。

## 判断标准

最终版本必须满足现场拓扑逻辑：

- 正中间近似直线的主线要贯穿全场，不能在道岔区断掉。
- 同一横断面最多约三条主要平行中心线。
- 侧线、装卸线或停留线允许有真实死端。
- 道岔/渡线连接应从主线或侧线平滑、相切地分出，不能横切或跳接。
- 每次输出都要叠加到原始 DOM 上裁切检查，裸 Shapefile 不能作为验收依据。

## 无损验收方式

我后续会按这个顺序自己验收自己迭代：

1. 先用原始 `dom.tif` 导出 lossless 的 ROI tile。
2. 在 tile 上跑分割和候选提取。
3. 把候选中心线直接叠加回同一张无压缩 tile，上屏看线是不是压在轨道中线。
4. 重点检查两类地方：直线主线是否贯穿连续，道岔/并轨处是否是从主线或侧线顺滑切出，而不是横切或跳接。
5. 只要发现偏轨是系统性的，就回去改分割阈值、配对逻辑或岔区约束，再重跑同一套无损叠图验收。

当前已经把候选 overlay 改成 PNG，无损查看优先于 JPEG 预览。

## 2股道主线先验

用户在 QGIS 中确认，两个截图坐标连成的线就是贯穿整个港口的 `2股道` 主线：

```text
start = 315112.328, 3519475.270
end   = 315617.422, 3522319.160
```

这条线已经被落成可复用产物：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_mainline_prior.py
```

输出目录：

```text
output/raw_dom_roi_fullpass_v1/mainline_prior
```

其中：

- `mainline_2_track_guide.shp` 是用户两点直接连线。
- `mainline_2_track_connected.shp` 是连续主线骨架，基于候选线做了约 `0.0724 m` 的横向修正。
- `mainline_2_track_support_candidates.shp` 是 2 m corridor 内支持这条主线的候选碎线。

2026-05-19，用户已在 QGIS 中验收 `mainline_2_track_connected.shp`：该线贯穿始终，并且全程压在 `2股道` 中心线上。后续所有股道分层、断线补全和道岔连接都应以它作为主线基准。

这一步标志着后处理可以从“全局一堆断线”切换到“先固定 2股道贯通主线，再接侧线和道岔”的路线。

## 平行股道分层产物

基于已验收的 `2股道` 主线，新增股道分层脚本：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_track_band_priors.py
```

输出目录：

```text
output/raw_dom_roi_fullpass_v1/track_band_priors
```

其中 `track_band_centerline_priors.shp` 是当前最值得在 QGIS 中查看的后处理中间成果：

- `mainline_2_track`：已验收的 `2股道` 主线，贯穿全场。
- `parallel_minus_5m`：一侧平行股道，当前分成 4 个站程段，总长约 `2325.099 m`。
- `parallel_plus_5m`：另一侧平行股道，当前分成 2 个站程段，总长约 `1039.975 m`。
- `possible_outer_plus_10m`：很短的外侧候选，暂不自动纳入最终三股道拓扑。

2026-05-19 的人工和 DOM 叠图复查后，侧线补全策略已改成保守版：

- 侧线只在原始 DOM 候选碎线提供支撑的站程范围内生成，普通大 gap 保持断开，不再强行向前外推进道岔或并轨区域。
- 只有 `s≈1810m` 附近两处长断裂被列入白名单桥接，因为无损 DOM 裁图显示那里仍是直股道，断裂更像由护轨/内部结构或视觉干扰造成。
- `parallel_minus_5m` 的白名单桥接最大 gap 约 `125.552 m`，`parallel_plus_5m` 的白名单桥接最大 gap 约 `83.184 m`。
- 道岔/渡线仍不在这个脚本里生成。进入道岔前的侧线端点宁愿断开，也不能用平行直线补全到主线；后续必须按相切和平滑约束单独挂接。

当前可复查的无损叠图位于：

```text
output/raw_dom_roi_fullpass_v1/track_band_priors/qa_crops
```

我已重点查看 `user_gap_pair_s1814_overlay.png`、`parallel_minus_5m_end_s908_overlay.png`、`parallel_minus_5m_end_s1064_overlay.png`、`parallel_plus_5m_end_s2219_overlay.png` 等端点和断裂图。结论是：新版侧线端点没有继续冲出轨道，`s≈1810m` 两处长断裂可以作为直股道特殊桥接，但不能泛化为所有 gap 都可桥接。

这一步只负责“主线 + 平行股道带”的骨架分层，还没有生成道岔连接。道岔连接必须在这些骨架验收通过后，再按相切和平滑约束挂接。

## 道岔/渡线候选层

道岔连接不再从侧线端点直接硬拉，而是单独生成候选层：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_turnout_connector_candidates.py
```

输出目录：

```text
output/raw_dom_roi_fullpass_v1/turnout_connector_candidates
```

当前产物分为两层：

- `turnout_connector_evidence.shp`：原始 DOM 候选里检测到的斜向过渡碎线，是证据层。
- `turnout_connector_proposals.shp`：当前为 `evidence_curve_proposal`，优先贴合证据层的真实曲率；只有端点接近股道或存在人工转辙机锚点时才短距离补全。

2026-05-19，用户复核指出：`turnout_connector_evidence.shp` 虽没有覆盖所有道岔，但大体压在轨道中心；旧版 smoothstep proposal 不按钢轨真实曲率走，局部偏离轨道。当前 v2 已改为“raw evidence 曲率优先”，并把用户给出的 `315305.45, 3520562.79` 记录为 `data/manual_feedback/turnout_switch_anchors.geojson` 中的转辙机锚点。

随后用户又指出 `315297.627, 3520490.690` 往南应视为直轨。该点已记录到 `data/manual_feedback/turnout_connector_splits.geojson`，用于把 P003 的 connector 只裁到道岔曲线段；南侧直线继续交给 `parallel_minus_5m` 股道带层处理。

当前结果为 `6` 条 evidence 和 `6` 条 proposal。分数较高、优先人工复核的是：

- `P003 minus_to_main`：`s=1032.177-1104.534`，使用用户转辙机锚点短补全，并按直轨起点 split 裁掉南侧直线尾段。
- `P004 minus_to_main`：`s=1442.231-1536.050`，沿 raw evidence 保持。
- `P001 main_to_plus`：`s=1478.186-1507.350`，沿 raw evidence 保持。

`P002`、`P005`、`P006` 是较短、低分的候选，只能作为线索，不能直接进入最终拓扑。

无损 QA 图在：

```text
output/raw_dom_roi_fullpass_v1/turnout_connector_candidates/qa_crops
```

这批候选的状态仍是 `candidate_needs_dom_review`。下一步应在 QGIS 中把 `turnout_connector_proposals.shp` 和 `turnout_connector_evidence.shp` 叠到原始 DOM 上，逐条确认哪些是真实道岔/渡线；只有确认后的连接才能并入最终中心线网络。

## P003 模板套用候选层

在 `P003` 道岔曲线已被人工验收后，新一轮尝试把它作为同规格道岔模板，套用到用户给出的 9 个转辙机/相切点附近坐标。这个做法的定位是“批量生成可复核候选”，不是自动确认所有道岔。

脚本：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_turnout_template_connectors.py
```

输入：

- `data/manual_feedback/turnout_template_anchors.geojson`：9 个用户锚点。
- `output/raw_dom_roi_fullpass_v1/turnout_connector_candidates/turnout_connector_proposals.geojson`：已验收来源模板 `P003`。
- `output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson`：已验收 2股道贯通主线。
- `output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson`：原始 DOM 全量中心线候选，用于评分。

输出：

```text
output/raw_dom_roi_fullpass_v1/turnout_template_connectors
```

关键产物：

- `turnout_template_connector_proposals.shp`：每个锚点的 best proposal，共 9 条。
- `turnout_template_connector_alternatives.shp`：保留的备选方向/股道组合，共 32 条。
- `turnout_template_anchors.shp`：锚点层。
- `qa_crops/_template_contact.png`：完整裁图拼图。
- `qa_crops/_connector_zoom_contact.png`：只围绕候选线的放大拼图。
- `VISUAL_QA.md`：本轮视觉 QA 和 QGIS 查验要点。

当前 best proposal 的 raw support 分数：

| 锚点 | score | 说明 |
| --- | ---: | --- |
| `TA09` | 1.0851 | P003 模板源附近，最可信 |
| `TA06` | 1.0071 | 高支持 |
| `TA07` | 0.9896 | 高支持 |
| `TA03` | 0.8268 | 较高支持 |
| `TA05` | 0.6196 | 中等支持 |
| `TA02` | 0.6102 | 中等支持 |
| `TA04` | 0.5528 | 中等支持 |
| `TA01` | 0.5513 | 中等支持 |
| `TA08` | 0.1771 | 低支持，必须优先人工复核 |

我已把 best proposal 叠加到原始 DOM 的无损裁图，并额外生成了放大拼图。肉眼检查结论是：9 条 best proposal 没有明显冲出轨道廊道，可以作为候选继续推进；但它们仍然不是最终中心线。尤其 `TA08` 位于道路桥/遮挡附近，raw support 很低，只能保留为低可信候选。`TA01`、`TA02`、`TA04`、`TA05` 虽然没有明显偏出轨道，但需要在 QGIS 中重点核查锚点和曲线端点是否确实符合“从转辙机附近相切分出，随后进入平行直股道”的现场逻辑。

QGIS 里优先看这些字段：

- `score`：模板候选综合分。
- `sup_cov`：raw DOM 中心线候选覆盖比例。
- `sup_dist`：与 raw DOM 证据的平均距离。
- `anchor_id`：锚点编号。
- `end_role`：锚点落在主线端还是侧线端。
- `orient`：模板曲线相对主线站程方向。
- `qa_note`：自动 QA 提示。

后续只有经过 QGIS/DOM 逐条确认的 proposal，才允许提升为最终拓扑连接边；其余仍保留为候选或备选，不参与最终中心线网络。

### 用户方向反馈与过渡段约束

用户复核后指出，旧版模板套用在 3 处道岔上选错了支线展开方向，并且 `TA08` 不是普通 P003 模板形态：

- `TA01`：应向南出支线。
- `TA04`：应向南出支线。
- `TA05`：应向北出支线。
- `TA08`：应使用曲线 + 直线 + 反曲线形态。

反馈已落盘：

```text
data/manual_feedback/turnout_template_feedback.geojson
```

这次修正的关键不是简单手动换一个 alternative，而是把“过渡段证据”单独作为约束加入评分：

- `sup_cov`：整条候选线附近的 raw DOM 候选覆盖比例。
- `trans_cov`：只统计横向偏移在中间区间的过渡段支持，避免主线端和侧线端把错误方向抬高。
- `branch`：候选支线相对锚点向 `north` 还是 `south` 展开。
- `shape`：候选形态，当前有 `p003_template` 和 `curve_straight_reverse`。

新的 best candidate 已重生成到同一目录：

```text
output/raw_dom_roi_fullpass_v1/turnout_template_connectors
```

修正结果：

| 锚点 | branch | shape | score | trans_cov |
| --- | --- | --- | ---: | ---: |
| `TA01` | south | `p003_template` | 0.6128 | 0.8333 |
| `TA04` | south | `p003_template` | 0.5793 | 0.6905 |
| `TA05` | north | `p003_template` | 0.5043 | 0.7143 |
| `TA08` | south | `curve_straight_reverse` | 0.1900 | 0.2647 |

QGIS 验收时，优先在属性表里看 `branch`、`shape`、`trans_cov`。`TA08` 的 `trans_cov` 仍低，说明语义分割/中心线证据不足；它只能作为特殊候选继续人工复核，不能直接进入最终中心线网络。

## 成对渡线候选层

用户复核后确认，`TA01/TA02` 和 `TA04/TA05` 不能按四个独立道岔支线理解，而应按两条成对渡线理解：

- `CX01`：`TA02` 南端主线支线与 `TA01` 北端侧线支线是同一条物理渡线。
- `CX02`：`TA05` 南端侧线支线与 `TA04` 北端主线支线是同一条物理渡线。

这推翻了“把已验收的 `P003` 曲线模板直接照搬到每个锚点”的假设。新脚本只把成对锚点作为拓扑约束，再从原始 DOM 候选中心线中筛选斜向 transition evidence 来约束曲线：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_turnout_crossover_connectors.py
```

输出目录：

```text
output/raw_dom_roi_fullpass_v1/turnout_crossover_connectors
```

关键输出：

- `turnout_crossover_connector_proposals.shp`
- `turnout_crossover_pairs.shp`
- `summary.json`
- `VISUAL_QA.md`
- `qa_crops/CX01_fullres_bounds_overlay.png`
- `qa_crops/CX01_fullres_south_overlay.png`
- `qa_crops/CX01_fullres_middle_overlay.png`
- `qa_crops/CX01_fullres_north_overlay.png`
- `qa_crops/CX02_fullres_bounds_overlay.png`
- `qa_crops/CX02_fullres_south_overlay.png`
- `qa_crops/CX02_fullres_middle_overlay.png`
- `qa_crops/CX02_fullres_north_overlay.png`

这些 `fullres` 图都是从原始 `dom.tif` 原分辨率裁切，绿色候选线宽 3 px，不是缩略图。自检结论是：`CX01` 线形和拓扑都较稳；`CX02` 的拓扑方向正确，中段也有斜向 raw evidence 支持，但支持强度低于 `CX01`，仍需要在 QGIS 中重点复核。

当前分数：

| 渡线 | score | support_cov | trans_cov | transition_features | transition_points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CX01` | 0.5157 | 0.4066 | 0.7105 | 7 | 32 |
| `CX02` | 0.4225 | 0.2418 | 0.5385 | 4 | 27 |

后续最终网络合并时，应优先使用这个成对渡线层替代 `TA01/TA02/TA04/TA05` 在 `turnout_template_connector_proposals.shp` 里的单锚点模板候选。

## 手扫 LAS 反推钢轨验证

用户复核 `turnout_crossover_connector_proposals.shp` 后认为两条成对渡线已经能用，均落在轨道内部，但局部没有完全压在中心线上。为了不只靠 DOM 肉眼判断，新增手扫点云验证脚本：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_crossover_with_las.py --chunk-size 1500000
```

这个脚本的逻辑是：

1. 读取 `turnout_crossover_connector_proposals.geojson`。
2. 使用既有手扫 LAS 全量约束估计出的轨距 `1.6 m`。
3. 沿每条渡线中心线采样，并按局部法线反推出左右钢轨坐标。
4. 两遍分块流式扫描 `data/生产数据/轨道/Las` 下 4 个 LAS 文件。
5. 第一遍估计每个采样点附近的局部地面高度，第二遍筛选高出局部地面 `0.08-0.35 m` 的点云，并在预测钢轨附近寻找左右轨头 lateral peak。
6. 如果左右钢轨都能支撑某个采样点，就反推该点处中心线应平移多少。

输出目录：

```text
output/raw_dom_roi_fullpass_v1/turnout_crossover_las_validation
```

关键产物：

- `turnout_crossover_predicted_rails.shp`：当前中心线反推出来的两根钢轨。
- `turnout_crossover_las_observed_rails.shp`：LAS 点云峰值推断出的两根钢轨。
- `turnout_crossover_las_refined_centerlines.shp`：LAS 推断出的修正中心线候选。
- `turnout_crossover_las_endpoint_locked_centerlines.shp`：端点保留原线、中段采用 LAS 修正的推荐复核候选。
- `turnout_crossover_las_sample_diagnostics.shp`：逐采样点诊断层。
- `turnout_crossover_las_sample_diagnostics.csv`：同样的诊断表，方便排序看局部偏差。
- `summary.json` / `VISUAL_QA.md`：本轮摘要。

结果：

| 渡线 | 采样点 | 双侧钢轨支撑采样点 | 轨头像点数 | 原始中位平移 | 平滑最大平移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CX01` | 91 | 91 | 1,447,294 | `0.12 m` | `0.3029 m` |
| `CX02` | 91 | 91 | 1,705,225 | `0.09 m` | `0.2771 m` |

这说明手扫 LAS 覆盖了这两条渡线，并且确实能反推出钢轨对中心。它对用户验收结论形成了支持：原始线已经进轨道内部，但局部中心有偏差。

不过 `turnout_crossover_las_refined_centerlines.shp` 仍然不能直接升级为最终线。原因是道岔区可能有护轨、转辙机和交叉构件，点云高度筛选可能把局部结构当成普通钢轨。下一步应把 `predicted_rails`、`las_observed_rails`、`las_refined_centerlines` 和原始 DOM 同时叠加，优先检查 `sample_diagnostics` 里局部偏移较大的点。

2026-05-20 用户继续复核指出：纯 LAS 修正版在开始和终止位置表现不如上一版，已经不在中心线；中间部分更接近中心。这个反馈是合理的，因为端点附近恰好是转辙机/相切/护轨区域，LAS 的轨头高度峰值更容易被局部构件污染。

当前脚本已改为保留两种输出：

- `turnout_crossover_las_refined_centerlines.shp`：全段 LAS 修正，只作诊断层。
- `turnout_crossover_las_endpoint_locked_centerlines.shp`：推荐验收层，两端各锁定 `8 m`，之后 `12 m` 平滑过渡，中段才吃满 LAS 修正。

下一轮 QGIS 优先验收 `turnout_crossover_las_endpoint_locked_centerlines.shp`。如果中段仍比原线更居中、两端又恢复到原先可接受位置，就用这个策略进入后续最终网络合并。
