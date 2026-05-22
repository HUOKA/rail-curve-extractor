# Rail Seg v3 Centerline Result

## 结论

这轮完整 56 张 CVAT 标注已经可以支撑一版可验证的语义分割流程。当前最适合继续走的路线是：

```text
CVAT track_area / switch_area
  -> YOLO polygon dataset
  -> ignore-aware U-Net semantic segmentation
  -> ordinary rail mask
  -> row-wise rail pairing
  -> first-pass centerline candidates
```

直道钢轨本体的识别结果已经有意义；道岔区域不应该作为普通直轨学习目标，当前训练和中心线后处理都会把 `switch_area` / `ignore_area` 排除在普通轨道目标之外。

## 数据

- CVAT 导出：`data/cvat_exports/task_1_annotations_20260510_155501.zip`
- 转换数据集：`data/datasets/rail_seg_v3`
- 图像数：56 / 56 已匹配并有标注
- 标注形状：389
- `track_area`：366
- `switch_area`：23
- `ignore_area`：0

这里的 `track_area` 按你当前标注习惯解释为“可见钢轨本体”，不是轨道床、不是两根钢轨之间的整条轨道区域。

## 语义分割结果

输出目录：`output/rail_seg_semantic_unet_v3`

模型：`output/rail_seg_semantic_unet_v3/rail_semantic_unet.pt`

训练设置：

- 前景：`track_area`
- 忽略：`switch_area,ignore_area`
- 训练源图：45
- 验证源图：11
- 训练 crop：225
- 验证 crop：55
- 最佳阈值：0.90

验证 crop 指标：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.8922 |
| Recall | 0.9428 |
| F1 | 0.9168 |
| IoU | 0.8463 |

整图验证指标：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.9307 |
| Recall | 0.8671 |
| F1 | 0.8978 |
| IoU | 0.8145 |

这些指标说明模型已经能在同一批 DOM 切片风格里学到“可见钢轨本体”的像素特征。不过它还不能证明模型已经能泛化到不同线路、不同季节、不同分辨率或不同拍摄条件。

## 中心线候选结果

输出目录：`output/rail_centerline_candidates_v3`

主要产物：

- `track_centerline_candidates.csv`
- `track_centerline_candidates.geojson`
- `track_centerline_candidates.shp`
- `rail_centers.csv`
- `overlays/*.jpg`
- `contact_sheet.jpg`
- `summary.json`

本轮后处理做了三件事：

1. 从语义分割 mask 里按行提取钢轨横向中心点。
2. 在同一行内选择互不重叠的最佳左右钢轨配对，避免 1-2、2-3、3-4 全部连上的假中心线。
3. 按上下连续性把行级候选分组，过滤短碎片，并在 `switch_area` / `ignore_area` 内不生成普通直轨候选。

当前统计：

| 项目 | 数值 |
| --- | ---: |
| 有 mask 的图像 | 56 |
| 钢轨中心点 | 46,053 |
| 原始中心线候选点 | 22,999 |
| 过滤后中心线候选点 | 22,986 |
| 分组中心线候选 | 175 |
| 估计左右钢轨间距 | 46.31 px |

## 当前判断

直道区域已经能看到比较连续的中心线候选，说明“语义分割 mask -> 左右钢轨配对 -> 中心线候选”的方向是通的。

但它还不是最终的轨道中心线提取器，主要限制是：

- 多股道密集区域仍可能产生相邻股道之间的假候选。
- 道岔拓扑还没有建模，只是通过 `switch_area` 从普通直轨结果里排除。
- 当前候选线还没有做全局轨道追踪、断点连接、曲线平滑和拓扑约束。
- 56 张图来自同一批切片，验证指标会偏乐观。

## 复现命令

训练：

```powershell
.\.yolo-venv\Scripts\python.exe .\scripts\train_rail_seg_semantic.py `
  --dataset .\data\datasets\rail_seg_v3 `
  --out .\output\rail_seg_semantic_unet_v3 `
  --epochs 30 `
  --batch 2 `
  --device cuda `
  --workers 0 `
  --input-width 384 `
  --input-height 1024 `
  --tile-height 1024 `
  --tile-stride 512 `
  --base-channels 16 `
  --foreground-label track_area `
  --ignore-labels switch_area,ignore_area
```

中心线候选：

```powershell
.\.venv\Scripts\python.exe .\scripts\extract_centerline_candidates.py `
  --dataset .\data\datasets\rail_seg_v3 `
  --mask-dir .\output\rail_seg_semantic_unet_v3\predictions\masks `
  --out .\output\rail_centerline_candidates_v3 `
  --row-step 16 `
  --column-threshold 0.08 `
  --min-run-pixels 2 `
  --min-pair-gap 20 `
  --max-pair-gap 130 `
  --gap-tolerance 0.50
```

导出 DASView 可加载的 Shapefile：

```powershell
.\.venv\Scripts\python.exe .\scripts\export_centerline_shapefile.py `
  --input .\output\rail_centerline_candidates_v3\track_centerline_candidates.geojson `
  --out .\output\rail_centerline_candidates_v3\track_centerline_candidates.shp `
  --crs-raster .\data\aligned_dom\aligned_dom.tif
```

会同时生成 `.shp/.shx/.dbf/.prj/.cpg`。其中 `.prj` 来自 DOM，当前是 `EPSG:32651`。

## 下一步

下一步不建议继续无脑多标同一批图，而是先做后处理增强：

- 按整条轨道做连续追踪，而不是只看每个 tile 内的行级配对。
- 用轨距、方向连续性、线间距和曲率约束剔除假中心线。
- 对 `switch_area` 单独建模；普通直轨中心线先在非道岔区域做好。
- 把 GeoJSON 候选线和原始 DOM 放到 GIS 里抽查，人工确认哪些候选线是真正轨道中心线。

如果后续要继续标注，优先补“模型最容易混淆”的样本：多股道间距接近的直线区、遮挡/阴影/设备覆盖的钢轨、低分辨率下钢轨边界模糊但仍属于普通直轨的区域。道岔内部先保持 `switch_area`，不用强行把里面每根复杂钢轨都拆成 `track_area`。
