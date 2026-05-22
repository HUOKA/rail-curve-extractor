# Rail Curve Extractor

> GitHub 复现入口：见 [docs/REPRODUCIBLE_WORKFLOW.md](docs/REPRODUCIBLE_WORKFLOW.md)。仓库只跟踪源码、测试、文档和配置样例；生产数据、模型权重、`output/` 和本地任务记录不进 Git。

面向无人机航测成果的铁路数字孪生辅助工具。项目最初以 **激光雷达点云直接提取轨道中心线** 为主，现在重构为更稳的路线：

> **正射影像负责看清轨道位置，DSM / 点云负责补高度、校验轨距和生成三维中心线。**

简单说：`DOM` 是可量测的二维底图，`DSM / LAS / 三维瓦片` 才提供高度和三维校验。正射图像素本身通常不是“每个像素存一组三维坐标”，而是通过影像的地理变换把像素行列号换算成地图坐标。

最终目标是：无人机飞完并由 DJI Terra / 大疆智图重建后，软件读取输出目录，自动找到正射图、DSM、点云等成果，辅助识别轨道、道岔和分支线路，并导出 Isaac Sim / Omniverse / RailOmniverse 可用的三维轨道中心线。

## 当前结论

- 通海港项目当前主线已经重置为“原始 DOM 优先”路线，详见 `通海港原始DOM优先重建路线.md`；旋转对齐走廊 DOM、旧 `final_centerline_network.shp` 和既有拓扑骨架都只能作为历史实验、ROI 证据或校验参考。
- 不再死磕“纯点云全自动识别轨道”作为唯一方案。
- 优先使用 DJI Terra 输出的 `GeoTIFF` 正射图做视觉识别。
- 正射图中的像素值主要是颜色；`X/Y` 来自 `GeoTIFF / TFW / PRJ` 的仿射变换和坐标系定义。
- 如果查看器在正射图上显示 `Z`，通常是叠加查询了 `DSM`、点云或 Terra 的三维瓦片 / 金字塔，而不是 DOM 像素自身带有高度。
- 使用 `DSM GeoTIFF` 或 LAS/LAZ 点云给中心线补 `Z`。
- 使用点云和铁路几何约束校验结果，例如轨距、左右钢轨连续性、高程一致性。
- `data/生产数据/轨道/Las` 中的手持式激光雷达轨道点云密度更高，可用于校准同一股轨道的轨距、相邻股道间距和局部轨道高度；使用前需把其 CGCS2000 横轴墨卡托/高斯投影坐标转换到 DOM 使用的 `EPSG:32651`。抽样诊断和全量分块流式诊断都支持把可见钢轨配对目标距离先定为约 `1.50 m`，后续可用 `1.35 - 1.65 m` 作为初始地图坐标配对约束。
- 手持点云覆盖了不少道岔，轨距标定时应排除道岔工作区；道岔区点云保留给后续局部岔道几何验证，而不混入全局轨距/股道间距统计。
- 原始 DOM 很大，正式跑语义分割前先用已有 `EPSG:32651` 中心线、人工参考线、道岔工作区和对齐走廊 footprint 反推轨道走廊 ROI；当前输出在 `output/raw_dom_corridor_roi`，把原始 DOM 预测范围收敛到 `293` 个 `3072 x 3072` 候选 tile。
- 原始 DOM 全量推理入口已补：`scripts/run_raw_dom_roi_fullpass.py` 会把 ROI tile 导出、rail 模型推理和候选提取串成一个可 dry-run 的步骤链，默认输出到 `output/raw_dom_roi_fullpass_v1`。
- 原始 DOM 抽样 QA 已跑完：单根铁轨模型可先进入全 ROI 候选提取试验；道岔区域模型在原始 DOM 上误检较多，暂不能作为生产级 LAS 排除掩膜或道岔连接依据，建议先补原始 DOM 负样本并微调。
- `scripts/extract_centerline_candidates.py` 已支持 `--target-gauge-m` / `--gauge-tolerance-m`，可在候选生成阶段用米制轨距过滤左右钢轨配对。
- 旧的点云中心线提取流程仍保留，作为局部校验、人工锚点和后处理能力；当前已支持在已有中心线连续性或锚线约束下，用单根可见钢轨和目标轨距保守反推缺失钢轨/中心点，并对这类切片降置信度。
- 最终识别路线采用“路线 B”：模型应能直接识别未旋转的原始 DOM 切片；旋转/裁切只作为标注辅助和早期数据增强手段。

## 推荐输入

推荐直接选择 DJI Terra / 大疆智图输出目录，例如：

```text
D:\正射
```

软件应优先识别以下正式成果目录：

```text
lidars/
├─ terra_dom/
│  ├─ dom.tif          # 正射影像，RGB/RGBA，带地理坐标
│  ├─ dom.tfw
│  ├─ dom.prj
│  ├─ dom_tiles/       # GeoTIFF 分幅
│  └─ tile/            # PNG 地图瓦片，可用于快速浏览
├─ terra_dsm/
│  ├─ dsm.tif          # DSM 高程栅格
│  ├─ dsm.tfw
│  ├─ dsm.prj
│  └─ dsm_tiles/       # DSM 分幅
└─ terra_las/
   ├─ cloud_merged.las # 合并点云，若存在优先使用
   └─ cloud*.las       # 点云分块
```

如果同时存在 `cloud_merged.las` 和 `cloud0.las`、`cloud1.las` 等分块，默认应优先使用 `cloud_merged.las`，避免重复加载。

## 当前样例数据确认

以当前 `D:\正射` 成果为例，已确认存在：

- `lidars/terra_dom/dom.tif`、`dom.tfw`、`dom.prj`
- `lidars/terra_dsm/dsm.tif`、`dsm.tfw`、`dsm.prj`
- `lidars/terra_las/cloud_merged.las`
- `lidars/terra_pnts/tileset.json`

其中：

- `dom.prj` 坐标系为 `WGS 84 / UTM zone 51N`，EPSG `32651`。
- `dom.tfw` 显示 DOM 分辨率约为 `0.0326 m/像素`。
- `dsm.tfw` 显示 DSM 分辨率约为 `0.0653 m/像素`，与 DOM 坐标系一致，但像元大小不同。
- DASView 鼠标悬停显示的 `X/Y` 可以由 DOM 的地理变换计算出来；`Z` 应优先理解为来自 DSM、点云或三维瓦片的查询结果。

## 正射图、DSM、点云各自作用

### 正射影像 `dom.tif`

- 像素值通常是 `RGB` 或 `RGBA`。
- 每个像素可通过 GeoTIFF / TFW / PRJ 计算出地图坐标 `X/Y`。
- 这不是给每个像素单独存一份坐标表，而是全图或切片共享一套地理变换。
- 不直接存每个像素的 `Z`。
- 如果软件在 DOM 上显示三维坐标，需要明确 `Z` 的来源：DSM、LAS/LAZ 点云或 Terra 三维瓦片。
- 适合做轨道、道岔、枕木、轨道区域的视觉识别和人工标注。

### DSM `dsm.tif`

- 像素值是高程或表面高度。
- 与正射影像处于同一坐标系时，可按 `X/Y` 查询高度 `Z`。
- DSM 分辨率可能与 DOM 不同，不能简单按相同行列号直接对应，应该按地图坐标 `X/Y` 查询或重采样。
- 适合快速给中心线补初始高度。
- DSM 是表面高程，可能落在钢轨、枕木、道砟、车辆、植被或噪声上；最终轨道高度需要点云或几何约束校验。

### 点云 `LAS/LAZ`

- 每个点有真实 `X/Y/Z`，可能还包含 RGB、强度、分类等。
- 适合校验轨距、钢轨高度、局部三维几何和道岔区域。
- 对大范围自动识别不如正射图直观，但对最终三维中心线精修很重要。

### 三维瓦片 / 金字塔 `terra_pnts`

- DJI Terra 会生成用于浏览和快速拾取的三维瓦片层级，例如 `tileset.json` 和 `.pnts` 文件。
- 这类金字塔适合在查看器中快速显示大范围点云、鼠标拾取坐标和做交互预览。
- 算法侧不应把三维瓦片当成唯一真值来源；精修和导出仍优先使用 `DSM GeoTIFF`、`LAS/LAZ` 点云和铁路几何约束。

## 新算法路线

```text
DJI Terra 输出目录
   ↓
自动识别 DOM / DSM / LAS
   ↓
原始 DOM 横平竖直重叠切片
   ↓
视觉模型识别 track_area / switch_area / rail 等候选区域
   ↓
切片像素坐标通过 tile_georef.csv 还原为地图 X/Y
   ↓
区域 mask / polygon 后处理为轨道中心线候选
   ↓
按 X/Y 查询 DSM 或点云补 Z
   ↓
点云轨距、连续性、高程一致性校验
   ↓
生成三维中心线与道岔 branch 曲线
   ↓
导出 OpenUSD / XYZ / JSON
```

### 通海港站场拓扑先验

通海港物流基地的最终中心线应采用“主线优先”的站场拓扑约束。用户现场拍摄的线路示意图只表达拓扑关系，不能按比例或折线形状直接套到现实坐标。

当前应默认：

- 正中间有一条近似直线的贯通主线，经过道岔区也不应中断。
- 同一横断面最多约三条主要平行轨道中心线。
- 除主线外，另外一到两条并行侧线、装卸线或调机停留线可以有真实首尾死端。
- 道岔/渡线应作为平滑连接边挂接到主线和侧线上，而不是让所有线段平等吸附。
- 判断最终中心线时必须叠加 DOM 裁切图核查，尤其检查主线是否贯穿、支线是否相切分出、侧线死端是否合理。

详见 `通海港线路拓扑先验.md`。当前主线重启说明见 `通海港原始DOM优先重建路线.md`；`通海港双模型中心线重建说明.md` 主要保留对齐走廊双模型实验、ROI、LAS 诊断和历史产物记录。

注意：当前 v7 通海港训练集来自旋转对齐并裁到轨道走廊后的 DOM，不是原始大正射图 `data/生产数据/无人机数据/正射/dom.tif` 的直接切片。因此当前结果主要验证“走廊图上的识别与拓扑重建”，最终生产流程仍必须回到原始 DOM 上做迁移验证或补充训练。

### 训练与预测路线

项目最终不依赖人工手动旋转每一份待识别 DOM。正式预测应直接处理 DJI Terra / 大疆智图输出的原始 `dom.tif`：

```text
原始 dom.tif
   ↓
原始方向重叠切片
   ↓
模型预测钢轨 / 轨道区 / 道岔区 mask
   ↓
按 tile_georef.csv 反算地图 X/Y
   ↓
中心线提取 + DSM/点云补 Z + 几何校验
```

旋转裁切仍然有价值，但定位为“标注辅助”：

- 用四点走廊或两点方向对齐生成更好画的辅助 DOM。
- 在 CVAT 中快速标出第一批高质量样本。
- 训练时使用旋转、翻转、随机裁切等增强，避免模型只认识南北向轨道。
- 后续必须补充一批原始 DOM 斜向切片和黑边切片，作为最终预测分布的训练/验证数据。

## 标注建议

当前最应该做的是建立自己的正射图标注集。优先标 `dom.tif` 切片，而不是原始单张 JPG。

推荐软件：

- `CVAT`：首选，适合 polygon / mask 标注，可导出 YOLO segmentation 数据。
- `Labelme`：轻量离线，适合小规模试验。
- `Roboflow`：上手快，但会上传数据，港口/铁路数据敏感时需谨慎。

第一版建议只标 3 类：

- `track_area`：轨道区域，一条轨道一个实例。
- `switch_area`：道岔、分叉、合流、尖轨等复杂区域。
- `ignore_area`：遮挡严重、阴影、水渍、看不清或不确定区域。

后续再增加：

- `rail_visible`：可见钢轨。
- `guard_rail`：护轨。
- `sleeper`：枕木。

不建议第一轮就大量标枕木，因为数量太多，标注成本高。枕木更适合作为后期辅助特征，而不是第一版主目标。

## 正射图切片建议

大正射图通常不能直接丢进标注软件。建议先切片：

- 原始 DOM 训练/预测主数据：优先 `3072 x 3072`，横向/纵向重叠率 `50%`
- 试切或 CVAT 卡顿时：降到 `2048 x 2048`，横向/纵向重叠率 `50%`
- 对齐 DOM 标注辅助数据通常是“窄而长”的走廊图，横向用走廊全宽且横向重叠率 `0%`，纵向用 `3072` 或 `4096`，纵向重叠率 `50%`
- 保留每个切片的地理变换信息
- 保存切片在原始 DOM 中的窗口位置、像素偏移和坐标变换，方便把标注结果回投到工程坐标。
- 第一批先标 `100–200` 张，不要一次性全量标
- 切片数据集要同时包含原始方向切片和少量对齐辅助切片，训练时必须启用方向增强。

切片选择要覆盖：

- 直线轨道
- 弯道
- 多轨并行
- 道岔 / 分叉 / 合流
- 轨道旁房屋、道路、护坡、堆场
- 没有轨道的负样本

### DOM 切片工具

可以双击启动：

```text
launch_dom_tiler_gui.bat
```

当前默认值按最终预测路线设置：

- 输入优先使用 `D:\轨道正射图`，不存在时再尝试 `D:\正射`。
- 输出默认到 `data/dom_tiles_raw_3072`。
- 切片尺寸默认 `3072 x 3072`。
- 原始 DOM 预设横向/纵向重叠率默认 `50%`；标注辅助对齐图默认横向 `0%`、纵向 `50%`。
- 默认跳过空白 / nodata 切片。

工具里的“标注辅助：方向对齐...”只用于人工标注阶段。它可以把一段斜向铁路走廊临时旋转成更好画的 GeoTIFF，但不要把它当成最终预测流程的必选步骤。

推荐使用四点走廊模式。黑色外围比较明显的 DJI Terra DOM 可以先点 `自动识别`，软件会从非黑有效区域估算四点；如果自动结果不贴合，再手动按下面顺序微调地图坐标：

```text
左上、右上、左下、右下
```

软件会根据四点自动计算：

- 走廊方向
- 走廊宽度
- 对齐后 GeoTIFF 的宽高
- 标注切片尺寸

自动识别的本质是识别有效 DOM 区域，不是识别钢轨本身；如果原始 ROI 本身包含站台、道路或宽度变化，输出仍会保留这些区域。它适合快速生成标注辅助图，最终预测流程仍以原始 DOM 切片为主。

对齐完成后，软件会自动把标注切片调整为能被 `32` 整除的矩形窗口，例如 `789px` 左右的走廊宽度会自动建议为 `768 x 3072`。

## 当前软件能力

当前仓库仍包含旧的点云中心线提取能力，并已开始向新架构迁移。

已具备：

- Electron + React 桌面界面。
- FastAPI 本地 Python 后端。
- 读取 DJI Terra 输出目录中的 LAS/LAZ 点云。
- 自动识别大疆点云分块并加载预览。
- 自动 LOD 点云标注画布。
- 人工路径点 / 道岔 branch 辅助模式。
- Open3D 当前视口高清查看。
- 多轨道、道岔、中心线导出相关的基础 pipeline，以及受连续性/锚线约束的单根钢轨补全。
- PySide6 DOM 切片工具，支持原始 DOM 重叠切片、跳过空白切片、导出 `tile_georef.csv/json`。
- 标注辅助方向对齐工具，支持预览点选、输入地图坐标、自动识别非黑有效区域四点并生成对齐后的 GeoTIFF。

重构中 / 待实现：

- 自动识别 `terra_dom/dom.tif` 和 `terra_dsm/dsm.tif`。
- 在 Electron 画布上叠加正射图。
- 按视口读取 GeoTIFF 分幅或瓦片。
- 标注导入、模型预测结果显示。
- 视觉 mask 转地图坐标。
- DSM / 点云补 `Z`。
- 视觉识别结果与点云轨距约束融合。
- mask / polygon 到中心线的后处理，包括骨架化、中心拟合、曲线平滑和道岔拓扑判断。

## 启动桌面端

推荐使用 Electron 入口：

```text
launch_rail_curve_extractor_electron.bat
```

或手动运行：

```powershell
cd .\desktop
npm install
npm run start
```

本地后端也可以单独启动：

```powershell
python -m rail_curve_extractor.backend.app --host 127.0.0.1 --port 8765
```

常用 API：

- `GET /api/health`
- `GET /api/config/default`
- `POST /api/point-cloud/info`
- `POST /api/point-cloud/preview`
- `POST /api/analyze`
- `POST /api/export`
- `POST /api/viewer/open`

## 本地开发安装

建议使用项目内虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[build]"
```

Open3D 使用独立 sidecar 环境，不装进主 `.venv`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_open3d_viewer_env.ps1 `
  -PythonExe C:\Path\To\Python312\python.exe
```

如果 `.open3d-venv` 不存在，主软件仍可运行，只是 `Open3D 当前视口高清` 不可用。

## Open3D 查看说明

内嵌 Open3D WebRTC 在当前 Windows / Electron 环境下会在握手阶段崩溃，默认禁用。

当前采用独立窗口方式：

- 优先读取当前画布视口。
- 最多加载约 `1200 万点`。
- 将大地坐标平移到本地原点，改善 Open3D 缩放精度。

使用建议：

1. 先在自动 LOD 画布中放大到轨道附近。
2. 再点击 `Open3D 当前视口高清`。
3. 如果直接在全图状态点击，仍然只是全图采样，视觉密度会低。

## 输出结果

旧点云 pipeline 当前可导出：

- `filtered_points.xyz`
- `rail_points.xyz`
- `centerline_points.xyz`
- `rail_centerline.usda`
- `used_config.json`
- `run_summary.json`

多轨道模式可额外导出：

- `track_1_centerline_points.xyz`
- `track_2_centerline_points.xyz`
- `track_1_rail_points.xyz`
- `track_2_rail_points.xyz`
- `rail_centerlines.usda`

道岔模式可额外导出：

- `turnout_1_main_centerline_points.xyz`
- `turnout_1_branch_centerline_points.xyz`
- `turnout_1_switch_point.xyz`
- `turnout_1_centerlines.usda`

新路线下，最终目标输出仍然是三维中心线和分支曲线，但中心线来源会变为：

```text
正射图视觉候选 + DSM/点云高度 + 轨距/几何约束
```

导出到 Isaac Sim / Omniverse 时，不建议直接把 UTM 大坐标作为场景原点。应转换到局部坐标系，例如以工程区域左下角或第一条中心线起点为 local origin，同时在元数据中保存：

- 原始坐标系 EPSG，例如 `32651`
- 本地原点偏移 `origin_x / origin_y / origin_z`
- 每条轨道中心线与道岔 branch 的置信度和人工修正记录

## Windows 便携版构建

本地构建 Windows 便携版：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_portable.ps1 `
  -PythonExe .\.venv\Scripts\python.exe `
  -Version local `
  -Platform win64
```

构建完成后生成：

- `dist\RailCurveExtractor-win64-local\`
- `dist\RailCurveExtractor-win64-portable-local.zip`

主程序：

- `RailCurveExtractor-local.exe`

## GitHub Release

仓库包含自动发布 workflow：

- `.github/workflows/release-portable.yml`
- 通过 push tag 触发，例如 `v0.1.0`
- 产物为 `RailCurveExtractor-win64-portable-v0.1.0.zip`

## 路线图

### 第一阶段：正射图接入

- 自动识别 `terra_dom`、`terra_dsm`、`terra_las`。
- 在 UI 里把正射图作为主底图。
- 点云画布退为辅助核查。
- 支持按视口读取分幅 GeoTIFF 或 PNG 瓦片。

### 第二阶段：标注与训练

- 提供正射图切片工具。
- 支持导入 CVAT / YOLO segmentation 标注。
- 训练 `track_area` / `switch_area` 初版模型。
- 训练集以原始 DOM 切片为主，对齐裁切图只作为标注辅助和补充样本。
- 训练时启用旋转、翻转、随机裁切等增强，避免方向过拟合。
- 在软件中显示模型预测结果，允许人工修正。

### 第三阶段：视觉与点云融合

- mask / polygon 转地图坐标。
- DSM 补高程。
- LAS 点云校验轨距、钢轨候选和高程连续性。
- 生成普通轨道中心线和道岔 branch 曲线。

### 第四阶段：数字孪生导出

- 导出 Isaac Sim / Omniverse 可用 OpenUSD 曲线。
- 输出每条轨道、道岔分支、置信度和人工修正记录。
- 支持大范围港区、多轨并行、连续道岔和复杂线路。

## 适用边界

当前更适合：

- DJI Terra / 大疆智图输出的正射图 + DSM + 点云成果。
- 港区、厂区、铁路专用线等无人机俯视场景。
- 人工愿意先做少量标注、逐步训练模型的工作流。

当前不适合：

- 完全没有正射图、DSM 或点云的单一数据源。
- 不做人工标注就期望复杂道岔全自动高精度输出。
- 将 YOLO 检测框直接当最终轨道中心线。
- 将查看器显示的三维坐标误认为 DOM 像素自身保存了完整 `X/Y/Z`。
