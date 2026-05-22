# Rail Curve Extractor

从无人机 DOM 正射影像生成铁路 2D/3D 中心线的算法工程。当前公开仓库只保留可复现的源码、测试、配置示例和一份流程说明，不包含生产数据、模型权重或已生成成果。

正式流程说明见 [docs/REPRODUCIBLE_WORKFLOW.md](docs/REPRODUCIBLE_WORKFLOW.md)。

## 仓库内容

- `src/rail_curve_extractor/`: 可复用 Python 包代码。
- `scripts/`: DOM 切片、语义分割推理、中心线后处理、Z 高程补全和 QA 脚本。
- `tests/`: 单元测试和流水线级回归测试。
- `data/README.md`: 本地数据目录说明。
- `config.example.json` / `data/config.example.json`: 配置示例。

## 不上传的内容

以下内容只应保留在本机或作为 Release/外部资产单独发布：

- 生产 DOM / DSM / LAS / LAZ 数据。
- 训练权重和模型 checkpoint。
- `output/` 下的 Shapefile、GeoJSON、叠加图和交付包。
- `.codex-tasks/`、本地实验记录和临时过程文档。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```

DeepLab 或 YOLO 相关依赖按需安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deeplab]"
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"
```

## 当前交付边界

通海港项目本地已验证的正式成果是：

- 2D 中心线：`centerline_2d.shp`
- 3D 中心线：`centerline_3d.shp`
- 3D 输出类型：`POLYLINEZ`
- Z 值来源：DSM/LAS 采样补全

这些成果文件属于生成产物，不进入 Git。公开仓库的目标是让别人 clone 后按说明准备自己的本地数据，再复现同一套算法流程。
