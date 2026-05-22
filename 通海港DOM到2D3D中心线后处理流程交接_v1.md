# 通海港 DOM 到 2D/3D 中心线后处理流程交接 v1

更新时间：2026-05-22

本文用于下一次对话继续接手当前可用流程，并说明“验收版复现流程”和“strict-auto 清洗流程”的边界。UI 暂时不做；当前优先级是把后处理生产链路从人工验收点位、保留证据层、TA08 专项逻辑里清洗出来。

## 当前结论

当前有两条路径，不能混用：

1. `strict-auto`：新的默认路径，目标是只依赖当前 DOM、DeepLab 模型、DSM/LAS 和本轮自动生成的中间证据。
2. `dom-full`：旧验收版复现路径，可以复现当前满意度约 95% 的结果，但它仍使用 retained ROI/道岔证据、reviewed bridge、TA08 专项逻辑，所以不能再称为纯自动生产算法。

旧验收版 `dom-full` 流程如下：

```text
DOM
-> ROI tiles
-> DeepLabV3+ rail segmentation
-> paired-rail centerline candidates
-> semseg-auto mainline prior
-> engineering-straight track-band priors
-> topology-aware centerline network
-> semseg curvature/radius diagnostics
-> 2D Shapefile
-> LAS/DSM Z sampling
-> 3D PolyLineZ Shapefile
```

旧验收版交付文件固定在：

```text
output/dom_centerline_pipeline_v1/final_delivery/centerline_2d.shp
output/dom_centerline_pipeline_v1/final_delivery/centerline_3d.shp
```

`strict-auto` 新默认输出目录为：

```text
output/dom_centerline_strict_auto_v1/final_delivery/centerline_2d.shp
output/dom_centerline_strict_auto_v1/final_delivery/centerline_3d.shp
```

当前 `strict-auto` 默认入口不读取旧的 `v15`、`v19`、`v20_z` 结果，不读取 `data/manual_feedback`，不读取 retained turnout/crossover evidence，也不启用 TA08 专项重建。旧 `dom-full` 只作为验收版复现/对比路径保留。

## 当前边界

需要说清楚：当前 `strict-auto` 已经完成输入依赖清洗，但还不是已经视觉验收通过的最终结果。

旧 `dom-full` 仍然保留的先验：

- ROI tile index：`output/raw_dom_roi_fullpass_v1/raw_dom_roi_tiles/selected_tile_index.csv`
- 保留的道岔/渡线证据层：
  - `output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson`
  - `output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_turnouts_v1/deeplab_gauge_pair_centerlines.geojson`
  - `output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_crossovers_v1/deeplab_gauge_pair_centerlines.geojson`

`strict-auto` 已经新增两个替代阶段：

- `scripts/build_auto_dom_tile_index.py`：只根据 DOM 元数据生成全图 tile index，不读旧 ROI 或人工走廊。
- `scripts/build_auto_turnout_crossover_evidence.py`：只根据本轮 DeepLab 配对钢轨候选在 station/offset 坐标系里寻找 transition，不读人工锚点、验收点、retained evidence，也不认识 TA08。

下一步工作不是 UI，而是实际跑 `strict-auto` 并对比 DASView/QGIS 视觉效果。如果 strict-auto 自动 transition 漏检或误检，再从通用规则上调算法，不能把用户截图点位重新塞回生产约束。

## 一键运行命令

在仓库根目录运行：

```powershell
& 'D:\rail-curve-extractor\.venv\Scripts\python.exe' scripts\run_dom_to_3d_centerline_guided_pipeline.py --profile strict-auto --force
```

如果只改了后处理、没有改 DOM 切片和 DeepLab 分割，可从后半段重跑：

```powershell
& 'D:\rail-curve-extractor\.venv\Scripts\python.exe' scripts\run_dom_to_3d_centerline_guided_pipeline.py --profile strict-auto --force --start-at build_track_band_priors
```

如果只想看执行计划：

```powershell
& 'D:\rail-curve-extractor\.venv\Scripts\python.exe' scripts\run_dom_to_3d_centerline_guided_pipeline.py --profile strict-auto --dry-run
```

## 默认输入

主入口脚本：

```text
scripts/run_dom_to_3d_centerline_guided_pipeline.py
```

默认输入：

```text
DOM: data/生产数据/无人机数据/正射/dom.tif
DSM: D:/正射/lidars/terra_dsm/dsm.tif
LAS dir: D:/正射/lidars/terra_las
DeepLab model: output/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt
Fallback model: output/_archive_superseded_20260521_v20z_baseline/output_root/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt
EPSG: 32651
```

可配置参数里 UI 第一版最需要暴露：

- `--dom`
- `--dsm`
- `--las-dir`
- `--out-dir`
- `--device`
- `--threshold`
- `--force`
- `--start-at`
- `--max-tiles`
- `--skip-qa-crops`

高级/调试参数暂时可以折叠：

- `--tile-index`
- `--raw-root`
- `--deeplab-model`
- `--use-turnout-exclusions`
- `--profile accepted-baseline`

## 关键算法状态

### 1. 主线不再依赖人工参考线

脚本：

```text
scripts/build_mainline_prior.py
```

当前默认模式：

```text
mode = semseg-auto
```

它会从 DeepLab 候选里估计主方向、找平行股道 offset 峰值，然后选择左右都有邻轨支撑且贯通最长的 2 股道作为主线。人工端点模式仍保留，但只应作为显式调试：

```powershell
python scripts/build_mainline_prior.py --mode manual
```

### 2. 直股使用工程直线

脚本：

```text
scripts/build_track_band_priors.py
```

当前最新策略：

```text
fit_mode = robust_straight_line
```

含义：

- DeepLab 配对中心候选只作为证据点；
- 每个 support-bounded 直股区间拟合一条鲁棒直线；
- 输出仍可每约 10m 一个顶点，但顶点共线；
- 不再让长直股跟着局部像素/语义分割噪声产生厘米级弯曲。

这次改动是为了符合“铁路直股在工程意义上应为直线”的约束。弯股和道岔支线仍保留曲线/支线逻辑。

### 3. LAS 不默认修正 XY

手扫/轨道 LAS 在局部核查中与 DOM/语义中心存在约 0.25-0.29m 的整体平面偏差。三条相邻股道同方向偏移，轨距仍约 1.50m，因此判断为 LAS 与 DOM 的局部配准差，而不是某一条中心线单独偏。

当前定位：

- LAS/DSM 用于 Z 高度；
- LAS 可用于相对几何质检，如轨距、股道间距、是否存在成对钢轨；
- LAS 不作为默认 XY 修正源；
- 如将来要用 LAS 修正 XY，必须先做 DOM-LAS 局部配准。

## 当前验证结果

最近一次验证命令：

```powershell
& 'D:\rail-curve-extractor\.venv\Scripts\python.exe' -m pytest tests\test_mainline_prior.py tests\test_track_band_priors.py tests\test_deeplab_topology_centerline_network.py tests\test_dom_to_3d_guided_pipeline.py tests\test_centerline_z_export.py tests\test_raw_dom_roi_fullpass.py
```

结果：

```text
36 passed
```

最终输出检查：

```text
centerline_2d.shp: POLYLINE, 19 records, no duplicate line_id
centerline_3d.shp: POLYLINEZ, 19 records, no duplicate line_id
```

关键点位核查：

```text
X=315563.068, Y=3521984.893
最近线: BAND_parallel_minus_5m_3
点到线距离: 0.0104m
直股共线误差: about 1e-6m
```

```text
X=315614.729, Y=3522304.403
最近线: BAND_mainline_2_track_0
点到线距离: 0.0030m
说明: 严格直线版本不再追尾段局部 support 细微偏移，这是预期行为。
```

## 输出目录结构

主输出目录：

```text
output/dom_centerline_pipeline_v1
```

关键阶段：

```text
01_dom_tiles
02_deeplab_segmentation
03_rail_candidates
04_refined_centerline
05_deeplab_network
06_mainline_prior
07_track_band_priors
08_topology_centerline
09_semseg_radius
10_centerline_2d
11_centerline_3d
final_delivery
```

UI 可以优先展示：

- `pipeline_plan.json`
- `pipeline_summary.json`
- `07_track_band_priors/summary.json`
- `08_topology_centerline/summary.json`
- `10_centerline_2d/REVIEW.md`
- `11_centerline_3d/summary.json`
- `final_delivery/centerline_2d.shp`
- `final_delivery/centerline_3d.shp`

## UI 套壳建议

第一版 UI 目标：把现有命令行流程封装成可选输入、可看进度、可打开输出目录的桌面程序。

### 现成桌面工程

仓库里已经有一个 Electron 桌面工程，不建议下一轮从零新建 UI 项目，优先在这里扩展：

```text
desktop/package.json
desktop/src/main.ts
desktop/src/preload.cts
desktop/src/renderer/App.tsx
desktop/src/renderer/styles.css
launch_rail_curve_extractor_electron.bat
```

当前技术栈：

```text
Electron + React + Vite + TypeScript + Fluent UI
```

现有 `desktop/src/main.ts` 已经会启动本地 Python 后端：

```text
python -m rail_curve_extractor.backend.app
```

所以下一轮 UI 有两条可选路线：

1. 快速路线：在 Electron main process 里新增 IPC，直接 `spawn` 当前 CLI 脚本。
2. 稳定路线：把 CLI 调用封装进 Python backend API，再由 React 页面调用 backend。

建议先走快速路线，因为当前目标只是把已经验收的 CLI 流程封装成可操作界面，不要在第一版同时重构后端架构。

### 推荐功能

1. 输入选择
   - DOM 文件
   - DSM 文件
   - LAS 目录
   - 输出目录
   - DeepLab 模型路径，可默认隐藏

2. 运行配置
   - 设备：`cuda` / `cpu`
   - 阈值：默认 `0.50`
   - 是否强制重跑：`--force`
   - 从哪个阶段开始：默认完整流程，可选 `build_track_band_priors`、`build_mainline_prior`
   - 是否生成 QA crops

3. 进度展示
   - 读取 `pipeline_plan.json` 显示阶段列表
   - 运行时捕获子进程 stdout/stderr
   - 每完成一个 stage 更新状态

4. 输出展示
   - 显示 2D/3D shp 路径
   - 打开 `final_delivery`
   - 打开 `pipeline_summary.json`
   - 显示记录数、shape type、重复 `line_id` 检查结果

5. 错误处理
   - DOM/DSM/LAS 路径不存在时阻止运行
   - DeepLab 模型不存在时给出明确提示
   - CUDA 不可用时允许切到 CPU
   - 某 stage 失败时保留已完成阶段，不删除输出

### UI 调用方式

UI 不应直接重写算法。第一版只调用：

```text
scripts/run_dom_to_3d_centerline_guided_pipeline.py
```

推荐子进程命令模板：

```powershell
& '<python>' scripts\run_dom_to_3d_centerline_guided_pipeline.py `
  --profile dom-full `
  --dom '<dom.tif>' `
  --dsm '<dsm.tif>' `
  --las-dir '<las_dir>' `
  --out-dir '<out_dir>' `
  --device cuda `
  --threshold 0.50 `
  --force
```

如果用户只想重跑后处理：

```powershell
& '<python>' scripts\run_dom_to_3d_centerline_guided_pipeline.py `
  --out-dir '<out_dir>' `
  --force `
  --start-at build_track_band_priors
```

### UI 第一版最小实现清单

1. 在 `desktop/src/main.ts` 增加文件/目录选择 IPC：
   - `dialog:open-dom`
   - `dialog:open-dsm`
   - `dialog:open-las-dir`
   - `dialog:select-centerline-output-dir`

2. 在 `desktop/src/main.ts` 增加运行 IPC：
   - `centerline-pipeline:dry-run`
   - `centerline-pipeline:start`
   - `centerline-pipeline:cancel`

3. 运行 IPC 内部只做三件事：
   - 拼接 `scripts/run_dom_to_3d_centerline_guided_pipeline.py` 命令；
   - 捕获 stdout/stderr，按行推送给 renderer；
   - 进程退出后读取 `pipeline_summary.json` 和 `final_delivery` 检查结果。

4. 在 `desktop/src/preload.cts` 暴露对应方法，不让 renderer 直接拿 Node 权限。

5. 在 `desktop/src/renderer/App.tsx` 增加一个“DOM 到中心线”工作区：
   - 输入路径表单；
   - dry-run 计划预览；
   - stage 日志；
   - 成功后显示 `centerline_2d.shp` / `centerline_3d.shp`；
   - 失败时显示最后一个失败 stage 和 stderr。

6. UI 完成后的最小验收：
   - 能从 UI 触发 `--dry-run`；
   - 能从 UI 触发 `--force --start-at build_track_band_priors`；
   - 能显示最终 `centerline_2d.shp` 和 `centerline_3d.shp` 路径；
   - 能显示 2D 是 `POLYLINE`、3D 是 `POLYLINEZ`、记录数均为 19、无重复 `line_id`。

## 下次接手优先级

1. 先不要再改中心线算法，除非用户验收 strict straight 版本发现明显问题。
2. 先做 UI 套壳，目标是稳定调用当前 CLI。
3. UI 完成后再考虑：
   - 自动 ROI 发现，替代 retained tile index；
   - 自动道岔/渡线证据发现，替代 retained turnout evidence；
   - LAS-DOM 局部配准模块，但不作为默认 XY 修正。

## 当前需要保留的核心文件

```text
scripts/run_dom_to_3d_centerline_guided_pipeline.py
scripts/build_mainline_prior.py
scripts/build_track_band_priors.py
scripts/build_deeplab_topology_centerline_network.py
scripts/build_semseg_smooth_review_package.py
scripts/add_z_to_deeplab_topology_centerline.py
tests/test_mainline_prior.py
tests/test_track_band_priors.py
tests/test_deeplab_topology_centerline_network.py
tests/test_dom_to_3d_guided_pipeline.py
tests/test_centerline_z_export.py
tests/test_raw_dom_roi_fullpass.py
```

历史版本产物仍可留作回归参考，但下一轮 UI 不应依赖旧输出目录。
