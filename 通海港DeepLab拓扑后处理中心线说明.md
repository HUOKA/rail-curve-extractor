# 通海港 DeepLab 拓扑后处理中心线说明

## 2026-05-21 v1

本轮不是把 DeepLab 输出的碎线直接合并成最终答案，而是把它作为语义分割证据层，再叠加已经确认过的站场拓扑约束：

```text
DeepLab 语义分割中心线证据
  -> 已验收 2股道贯通主线
  -> 支撑边界内的平行直股道
  -> 道岔/渡线相切连接候选
  -> 每条输出线记录 DeepLab 支持率和复核风险
```

这样做的原因是：初步中心线会有抖动、偏轨和局部误检，尤其在道岔、车辆遮挡、道路标线和桥缝附近。DeepLab 结果适合作为“哪里有轨道视觉证据”的来源，但不能单独承担站场拓扑。

## 主输出

优先在 QGIS 中加载：

```text
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/deeplab_topology_centerline_network.shp
```

配套样式：

```text
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/deeplab_topology_centerline_network.qml
```

辅助证据层：

```text
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/deeplab_topology_evidence.shp
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/deeplab_topology_evidence.qml
```

摘要文件：

```text
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/summary.json
output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1/REVIEW.md
```

生成命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\build_deeplab_topology_centerline_network.py
```

## 本轮输出结构

`deeplab_topology_centerline_network.shp` 一共 14 条线：

| 类型 | 数量 | 含义 |
| --- | ---: | --- |
| `main_through_track` | 1 | 已验收的 2股道贯通主线，作为港口可开出去的完整直轨骨架 |
| `parallel_straight_track` | 6 | 支撑边界内的平行直股道片段，不强行延伸进道岔 |
| `turnout_connector` | 7 | 道岔/渡线连接，包括 CX01、CX02、TA03、TA06、TA07、TA08、TA09 |

脚本默认排除了 `possible_outer_plus_10m`，因为它仍是诊断外侧短候选，不应自动进入最终主拓扑。

## 关键字段

QGIS 属性表里优先看这些字段：

| 字段 | 含义 |
| --- | --- |
| `net_role` | 主线、平行直股道、道岔连接 |
| `line_id` | 本轮统一编号 |
| `band_id` / `branch_id` | 股道带或道岔编号 |
| `dl_sup` | DeepLab 证据支持率，越高表示越贴近语义分割中心线证据 |
| `dl_mean` | 到 DeepLab 证据的平均距离，单位米 |
| `dl_gap` | 连续缺少 DeepLab 支持的最长采样 gap，单位米 |
| `risk` | 后处理给出的复核优先级 |
| `qa_status` | 前序原分辨率 DOM 自检状态 |

## 本轮风险点

当前自动标出的重点复核对象：

| 对象 | 原因 | 建议 |
| --- | --- | --- |
| `BAND_parallel_minus_5m_1` | `dl_sup=0.0`，约 103 m 片段缺少 DeepLab 支持，但由既有拓扑/股道带保留 | 在 QGIS 中叠加 DOM 和证据层确认是否确实是平行直股道 |
| `TA08` | 已触发通用的低支持道岔修正规则，几何改为由附近 DeepLab 轨距配对中心线约束；本轮 `dl_sup=0.8235` | 继续用 DOM 复核补全段，但不要再用普通 P003 模板强行解释 |
| `CX02` | 前序视觉可用，但本轮 DeepLab 支持偏低，`dl_sup=0.2` | 结合手扫 LAS endpoint-locked 层复核中段是否仍压在渡线中心 |

贯通主线 `mainline_2_track` 的 `dl_sup=0.9206`，说明 DeepLab 证据总体支持它；但它仍应以已验收主线为准，不因为局部 DeepLab gap 就切断。

## 当前算法边界

- 这版已经把“多股道直线并行”和“港口必须有一条贯通主线”显式写进后处理。
- 无人机 LAS 和手扫 LAS 目前没有重新全量参与本脚本；手扫 LAS 已经通过前序 `CX01/CX02 endpoint_locked` 结果间接进入渡线候选。
- 这版已经对低支持道岔增加了通用修正规则：如果附近存在连续的 DeepLab 轨距配对中心线证据，则优先用证据线重建该道岔几何，再只做短距离拓扑补全。
- 下一步应针对 `TA08`、`CX02`、`parallel_minus_5m_1` 做局部重建，而不是继续全局调 snap tolerance。

## 验证

已通过：

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\build_deeplab_topology_centerline_network.py
.\.venv\Scripts\python.exe -m pytest tests\test_deeplab_topology_centerline_network.py -q
```

测试结果：`3 passed`。
