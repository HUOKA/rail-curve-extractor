# 通海港 DeepLab 轨距配对中心线试点说明

## 结论

本轮不是继续把 DeepLab mask 直接转中心线，而是先做了一个 TA08 局部试点：

```text
DeepLab v1 单根钢轨概率图 -> 横向概率峰值 -> 轨距成对筛选 -> 中心线候选
```

结果说明这个方向是可行的。道路标线、桥缝、护栏阴影这类细长假阳性虽然仍会出现在语义分割概率图里，但没有被直接提升成中心线；最终只生成了 2 条成对钢轨中心线候选。

## 输入数据

- DeepLab v1 概率图：
  `output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/ta08_segmentation_probability_u8.tif`
- 主线坐标基准：
  `output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson`
- 原始 DOM：
  `data/生产数据/无人机数据/正射/dom.tif`
- DSM：
  `D:\正射\lidars\terra_dsm\dsm.tif`
- 手扫 LAS 估计轨距：
  `output/handheld_las_constraints_fullpass_switch_excluded/summary.json`

DSM 已确认是 `EPSG:32651`，像素约 `0.065m`；DOM/DeepLab 证据图也是 `EPSG:32651`，像素约 `0.0326m`。当前把 DSM 作为高度脊线支持证据，不作为硬否决条件。

## 生成命令

```powershell
.\.venv\Scripts\python.exe .\scripts\build_deeplab_gauge_pair_centerlines.py
```

已通过：

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\build_deeplab_gauge_pair_centerlines.py
.\.venv\Scripts\python.exe -m pytest tests\test_deeplab_gauge_pair_centerlines.py -q
```

单测结果：`3 passed`。

## 输出文件

主输出目录：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/
```

建议在 QGIS 里优先验收：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/deeplab_gauge_pair_centerlines.shp
```

辅助证据层：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/deeplab_gauge_pair_evidence.shp
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/deeplab_gauge_pair_samples.csv
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/summary.json
```

QA 裁切图：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/qa_crops/ta08_full_dom_deeplab_gauge_pair_overlay.png
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/qa_crops/ta08_user_coord_overlay.png
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/qa_crops/GP01_mid_overlay.png
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/qa_crops/GP02_mid_overlay.png
```

图中含义：

- 黄色/红色：DeepLab v1 概率响应。
- 彩色粗线：轨距配对后的中心线候选。
- 短横线：成对钢轨横截面证据。
- 蓝点：此前人工截图定位点。

## 数值摘要

本轮使用手扫 LAS 摘要里的轨距 `1.6m` 作为约束。TA08 裁切中：

| 指标 | 数值 |
| --- | ---: |
| 概率像素数 | 126690 |
| 进入轨道横向走廊的像素数 | 101546 |
| 横向峰值数 | 668 |
| 原始成对样本数 | 304 |
| 连续中心线候选数 | 2 |

两条中心线：

| ID | 样本数 | 站位范围 m | 横向范围 m | 平均配对间距 m | 平均 DSM 支持 m | 判断 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| GP01 | 237 | 1226.625..1409.625 | -0.097..0.119 | 1.5152 | 0.9282 | 主线附近，视觉上压在轨道中心 |
| GP02 | 62 | 1274.625..1324.125 | 2.074..7.448 | 1.5015 | 0.2828 | TA08 局部支线/侧线候选，视觉上在轨道内部 |

注意：这里的“平均配对间距”低于物理轨距 `1.6m`，原因大概率是 DeepLab 分割到的是钢轨亮暗边缘或轨头内侧响应，而不是严格的钢轨几何中心。只要左右响应偏差大体对称，中点仍可用于中心线候选。

## 自检结论

我查看了原分辨率 DOM 叠加图：

- `ta08_user_coord_overlay.png`：GP01/GP02 都在轨道内部，未把道路白线生成中心线。
- `GP02_mid_overlay.png`：GP02 在 TA08 道岔局部比之前的模板线更受分割证据约束，整体可作为候选继续用。
- `ta08_full_dom_deeplab_gauge_pair_overlay.png`：长直轨区域 GP01 稳定；GP02 只在成对钢轨证据连续的局部出现，没有强行补满整条线。

这版还不是最终答案，但它比“照搬模板曲线”更接近我们要的方向：先让模型找单根钢轨，再由轨距、DSM、主线拓扑去决定哪些线可以成为中心线。

## 下一步

建议把这个逻辑扩展到所有道岔窗口：

1. 以主线和三股道先验生成每个道岔的局部窗口。
2. 在窗口内跑同样的 DeepLab 轨距配对。
3. 对成对证据连续的地方直接生成中心线。
4. 对短缺口再用相切拓扑和 DSM/LAS 支持做保守桥接。
5. 输出全道岔候选 Shapefile，再逐块原分辨率裁切自检。
