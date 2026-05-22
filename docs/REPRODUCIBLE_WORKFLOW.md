# Reproducible Workflow

本文是公开仓库保留的唯一正式流程说明。旧的交接记录、实验说明和截图验收记录已移入本机 `local_archive/legacy_markdown/`，该目录被 `.gitignore` 忽略，不作为 GitHub 交付内容。

## 1. 目标

从 DJI Terra 或同类航测重建成果中读取 DOM / DSM / LAS 数据，自动生成铁路中心线：

```text
DOM 正射影像
-> 轨道语义分割
-> 钢轨候选提取
-> 轨距配对与拓扑后处理
-> strict-auto 2D 中心线
-> DSM/LAS 采样补 Z
-> 2D/3D Shapefile 交付
```

当前仓库公开算法、脚本、测试和配置示例；不公开通海港生产数据、模型权重和输出成果。

## 2. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

按需安装深度学习依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deeplab]"
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"
```

## 3. 本地数据布局

生产数据不要提交到 Git。建议放在：

```text
data/production/
  terra_dom/
    dom.tif
  terra_dsm/
    dsm.tif
  terra_las/
    cloud0.las
    cloud1.las
```

如果有模型权重，建议放在本地 `models/` 或外部路径。`*.pt`、`*.pth`、`*.onnx` 已被忽略。

## 4. 核心流程

语义分割可以重新训练，也可以直接使用已有权重推理。通海港当前验收思路是：不要把人工线或截图坐标写成生产约束，主线以 DOM 视觉中心和轨距证据为主，道岔支线在尖轨区域与主线保持切向连续。

主要脚本入口：

```text
scripts/predict_rail_seg_deeplab_images.py
scripts/build_deeplab_topology_centerline_network.py
scripts/package_strict_auto_global_centerline_review.py
scripts/add_z_to_deeplab_topology_centerline.py
```

正式打包步骤示例：

```powershell
.\.venv\Scripts\python.exe scripts\package_strict_auto_global_centerline_review.py

.\.venv\Scripts\python.exe scripts\add_z_to_deeplab_topology_centerline.py `
  --input output\dom_centerline_strict_auto_v1\global_centerline_review_tangent_occlusion\global_centerline_2d.geojson `
  --output-dir output\dom_centerline_strict_auto_v1\global_centerline_review_tangent_occlusion_z
```

最终交付目录通常为：

```text
output/dom_centerline_strict_auto_v1/final_delivery/
  centerline_2d.shp
  centerline_3d.shp
  centerline_evidence.shp
  delivery_manifest.json
```

`centerline_2d.shp` 应为 2D `POLYLINE`，`centerline_3d.shp` 应为 3D `POLYLINEZ`。

## 5. 验证

完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

中心线交付相关的重点测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_strict_auto_global_centerline_review_package.py `
  tests\test_turnout_outer_rail_geometry.py `
  tests\test_centerline_z_export.py `
  tests\test_dom_to_3d_guided_pipeline.py `
  tests\test_deeplab_topology_centerline_network.py
```

## 6. Git 规则

只提交：

- 源码
- 测试
- 文档
- 配置示例
- 小型元数据

不要提交：

- `output/`
- 生产 `data/`
- `.codex-tasks/`
- `local_archive/`
- 虚拟环境
- 模型权重
- LAS/LAZ/GeoTIFF/Shapefile/GeoPackage/QGIS 工程等大型或生成文件

正式成果建议通过 Release asset、网盘或项目交付包发布，不放进普通 Git 历史。
