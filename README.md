# Rail Curve Extractor

> 从无人机 DOM 正射影像中自动提取铁路中心线，并可结合 DSM / LAS 点云补全 3D 高程，输出工程可用的 2D / 3D Shapefile。

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="electron" src="https://img.shields.io/badge/electron-39-47848F?logo=electron&logoColor=white">
  <img alt="react" src="https://img.shields.io/badge/react-19-61DAFB?logo=react&logoColor=white">
  <img alt="tailwind" src="https://img.shields.io/badge/tailwindcss-4-06B6D4?logo=tailwindcss&logoColor=white">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue">
  <img alt="status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

---

## 这是什么

`rail-curve-extractor` 是一个针对铁路场景的**轨道中心线自动化提取**工具。给它一张无人机正射影像（DOM），它会跑完语义分割、钢轨候选提取、轨距配对、拓扑后处理、Z 补全这一整条流水线，最后吐出 2D / 3D 的 Shapefile（`POLYLINE` / `POLYLINEZ`），可直接导入 ArcGIS、QGIS、CAD 工具继续使用。

软件由两部分组成：

- **算法核心**：纯 Python 包 `rail_curve_extractor`，配套 `scripts/` 下的命令行脚本。任何流水线步骤都可以独立跑。
- **桌面前端**：`desktop/` 下的 Electron + React 应用，把流水线封成一屏式工作台，**不会写命令行的人也能用**。

```text
┌─ 无人机 DOM ────────────────────────────────────────────────┐
│   GeoTIFF (含 CRS) ─┬─► 语义分割 (DeepLabv3+ ResNet50)       │
│                     ├─► 钢轨候选 + 轨距配对                  │
│                     ├─► 拓扑后处理 (strict-auto)             │
│                     ├─► DSM / LAS 补 Z                       │
│                     └─► centerline_2d.shp + centerline_3d.shp│
└──────────────────────────────────────────────────────────────┘
```

## 截图

> 桌面端会自动从 DOM 文件里读 EPSG，自动探测本机 GPU / CPU，一屏看完输入、流水线状态、日志、产出。

<!-- 占位：合并主分支后补两张截图到 docs/screenshots/ -->
<!-- ![Pipeline running](docs/screenshots/pipeline.png) -->
<!-- ![Light theme](docs/screenshots/light.png) -->

## 功能特性

- **DOM → 3D 中心线一条龙**：单个命令或单击按钮就能跑完全流程，不用记几十个脚本顺序
- **CRS 自动识别**：从 GeoTIFF 自带的元数据里直接读取 EPSG，不需要用户填写
- **真实硬件感知**：自动探测 NVIDIA 显卡、PyTorch CUDA 版本，没有 GPU 时给出明确建议
- **可配置 strict-auto / dom-full / accepted-baseline 三种 profile**，分别对应纯算法、保留人工证据、复现历史交付
- **桌面端工业风界面**：流水线步骤可视化、实时日志、产出文件直达资源管理器
- **深色 / 浅色 / 跟随系统**主题切换
- **可选 GPU 加速**：DeepLab 推理在 RTX 4060 Ti 上几分钟跑完一张大 DOM，CPU 也能跑只是慢得多

## 系统要求

| 组件 | 最低 | 推荐 |
| --- | --- | --- |
| 操作系统 | Windows 10 / Linux | Windows 11 |
| Python | 3.11 | 3.13 |
| 内存 | 16 GB | 32 GB |
| GPU | 无（CPU 推理） | NVIDIA 8 GB+ 显存 |
| 磁盘 | 5 GB | 20 GB+（含模型权重 + 输出） |
| Node.js | — | 20+（仅桌面端开发） |

> AMD / Intel 显卡在 Windows 下**无法用 PyTorch CUDA 加速**。如需 GPU 加速请使用 NVIDIA 显卡，或在 Linux 上配置 ROCm（仅特定型号支持）。详见桌面端的设备提示。

## 安装

### 1. 拉代码

```powershell
git clone https://github.com/HUOKA/rail-curve-extractor.git
cd rail-curve-extractor
```

### 2. 创建 Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

> Linux / macOS 把 `\.venv\Scripts\python.exe` 换成 `./.venv/bin/python`。

### 3. 按需装可选依赖

| Extras | 何时需要 |
| --- | --- |
| `[deeplab]` | 跑语义分割推理 / 训练（PyTorch + torchvision） |
| `[yolo]` | 用 YOLO 走基线分割（OpenCV + ultralytics） |
| `[viewer]` | Open3D 三维查看器 |
| `[test]` | 跑测试 |

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deeplab,viewer]"
```

GPU 用户请按 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的轮子覆盖装一次。

### 4. 准备模型权重

把训练好的 DeepLab 权重放在：

```text
models/rail_seg_deeplab_resnet50_native_v1/
  rail_semantic_deeplab_resnet50.pt
```

权重文件不进 Git（已在 `.gitignore` 中）。

### 5. （可选）装桌面端

```powershell
cd desktop
npm install
npm run build
```

## 快速开始

### 方案 A · 桌面端（推荐）

```powershell
.\launch_rail_curve_extractor_electron.bat
```

1. 选 DOM 文件 → CRS 会自动识别
2. 选 DeepLab 权重
3. 选输出目录
4. （可选）选 DSM 栅格 / LAS 目录用于补 Z
5. 点 **开始流水线**

跑的过程中会看到六步流水线的实时进度，完成后产出文件可一键在资源管理器中定位。

### 方案 B · 命令行

```powershell
.\.venv\Scripts\python.exe scripts\run_dom_to_3d_centerline_guided_pipeline.py `
    --profile strict-auto `
    --dom data\production\terra_dom\dom.tif `
    --deeplab-model models\rail_seg_deeplab_resnet50_native_v1\rail_semantic_deeplab_resnet50.pt `
    --out-dir output\dom_centerline_v1 `
    --dsm data\production\terra_dsm\dsm.tif `
    --las-dir data\production\terra_las `
    --epsg 32651 `
    --device cuda
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--profile` | `strict-auto` | `strict-auto` / `dom-full` / `accepted-baseline` |
| `--device` | `cuda` | `cuda` / `cpu` |
| `--threshold` | `0.50` | 语义分割概率阈值 |
| `--max-tiles` | `0` | 限制处理瓦片数（0 = 不限） |
| `--epsg` | `32651` | 输出 Shapefile 的目标坐标系 |
| `--force` |  | 强制重跑所有阶段 |

完整参数：`python scripts\run_dom_to_3d_centerline_guided_pipeline.py --help`

### 方案 C · 单步执行

如果只想跑某一步，调相应脚本：

```powershell
# 仅做语义分割
.\.venv\Scripts\python.exe scripts\predict_rail_seg_deeplab_images.py ...

# 仅做拓扑后处理
.\.venv\Scripts\python.exe scripts\build_deeplab_topology_centerline_network.py ...

# 仅打包 strict-auto 2D 交付
.\.venv\Scripts\python.exe scripts\package_strict_auto_global_centerline_review.py ...

# 仅给已有 2D 中心线补 Z
.\.venv\Scripts\python.exe scripts\add_z_to_deeplab_topology_centerline.py ...
```

每个脚本都支持 `--help`。

## 配置

`config.example.json` 是完整配置示例，覆盖 ROI 范围、轨距搜索、平滑参数、人工锚点、自动多轨切分等所有可调项。命令行可以通过 `--config path/to/config.json` 覆盖默认值，桌面端会在运行时把 UI 选项合并进去。

关键参数：

```jsonc
{
  "rail_pair_spacing_target": 1.5,    // 标准轨距 (m)
  "rail_pair_spacing_min": 1.2,
  "rail_pair_spacing_max": 1.8,
  "savgol_window": 9,                 // 平滑窗口
  "auto_track_split": {               // 自动切分多条轨道
    "enabled": false,
    "count": 2
  }
}
```

## 输出

```text
output/dom_centerline_strict_auto_v1/
└── final_delivery/
    ├── centerline_2d.shp           POLYLINE   (X, Y)
    ├── centerline_3d.shp           POLYLINEZ  (X, Y, Z)
    ├── centerline_evidence.shp     人工 / 自动证据图层
    └── delivery_manifest.json      产出元数据
```

也会输出中间产物（瓦片、分割概率图、骨架几何）方便审查，详见 `delivery_manifest.json`。

## 项目结构

```text
rail-curve-extractor/
├── src/rail_curve_extractor/    Python 包：算法核心 + FastAPI 后端
│   ├── pipeline.py              主流水线
│   ├── dom_tiler.py             DOM 切瓦片
│   ├── geometry.py / io.py      几何与 IO 工具
│   └── backend/app.py           桌面端调用的本地 HTTP API
├── scripts/                     45+ 个可独立运行的 CLI 脚本
├── desktop/                     Electron + React 桌面端
│   ├── src/main.ts              主进程，负责拉起后端
│   ├── src/preload.cts
│   └── src/renderer/            React UI（Tailwind 4 + framer-motion + lucide）
├── tests/                       pytest 测试
├── assets/                      应用图标 + 候选图标
├── config.example.json          配置示例
└── pyproject.toml
```

## 开发

### 跑测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 桌面端开发

```powershell
cd desktop
npm run check        # 类型检查
npm run build        # 编译 main + 渲染层
npm run start        # 构建后启动 Electron
npm run build:icons  # 重新生成应用图标 (PNG + ICO)
```

桌面端不打热重载——这是一个工程工具不是 web 站。每次改完前端 `npm run start` 重启即可。

### 添加 / 替换应用图标

`assets/icon-candidates/` 里有 6 个候选 SVG 和一个 `preview.html`。要换：

1. 把目标 SVG 复制到 `desktop/src/renderer/assets/app-icon.svg`
2. `cd desktop && npm run build:icons`

会自动生成所有 PNG 尺寸 + Windows 多分辨率 `.ico`。

## 路线图

- [ ] 在 `docs/screenshots/` 补桌面端截图
- [ ] electron-builder 配置，输出独立安装包
- [ ] 重构 `.github/workflows/release-portable.yml`（旧的 PyInstaller workflow 已失效）
- [ ] DJI Terra 工程目录直接导入
- [ ] 多 GPU 推理 / 切片并行

## 鸣谢

- 语义分割主干：[DeepLabv3+](https://arxiv.org/abs/1802.02611) / ResNet50
- 栅格 IO：[rasterio](https://github.com/rasterio/rasterio)
- 矢量 IO：[pyshp](https://github.com/GeospatialPython/pyshp) / [shapely](https://github.com/shapely/shapely)
- 点云 IO：[laspy](https://github.com/laspy/laspy)
- 桌面端：[Electron](https://www.electronjs.org/) + [React](https://react.dev/) + [Tailwind CSS](https://tailwindcss.com/) + [framer-motion](https://www.framer.com/motion/) + [lucide](https://lucide.dev/)

## License

仓库未声明开源许可证；如需在外部项目中使用，请先与作者联系。
