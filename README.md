# Rail Curve Extractor

从无人机 DOM 正射影像提取铁路中心线，并可结合 DSM/LAS 补全 3D 高程。

## 仓库内容

- `src/rail_curve_extractor/`: Python 包代码
- `scripts/`: 推理、后处理、Z 补全、QA 脚本
- `tests/`: 自动化测试
- `config.example.json`、`data/config.example.json`: 配置示例

不包含生产数据、模型权重、输出成果和本地实验记录。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

按需安装可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deeplab]"
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"
```

## 本地数据

生产数据放在被忽略的本地目录，例如：

```text
data/production/
  terra_dom/dom.tif
  terra_dsm/dsm.tif
  terra_las/cloud0.las
```

模型权重也不要提交到 Git。

## 流程

```text
DOM
-> 轨道语义分割
-> 钢轨候选提取
-> 轨距配对与拓扑后处理
-> strict-auto 2D 中心线
-> DSM/LAS 补 Z
-> 2D/3D Shapefile
```

主要脚本：

```text
scripts/predict_rail_seg_deeplab_images.py
scripts/build_deeplab_topology_centerline_network.py
scripts/package_strict_auto_global_centerline_review.py
scripts/add_z_to_deeplab_topology_centerline.py
```

## 输出

典型交付目录：

```text
output/dom_centerline_strict_auto_v1/final_delivery/
  centerline_2d.shp
  centerline_3d.shp
  centerline_evidence.shp
  delivery_manifest.json
```

`centerline_2d.shp` 为 `POLYLINE`，`centerline_3d.shp` 为 `POLYLINEZ`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```
