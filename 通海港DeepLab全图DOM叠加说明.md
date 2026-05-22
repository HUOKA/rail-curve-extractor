# 通海港 DeepLab 全图 DOM 叠加说明

## 输出文件

已经用 DeepLab v1 模型跑完整张原始 DOM，并把识别结果直接烧录到 DOM 像素里，避免 DasView 不支持透明/NoData 时出现黑底。

优先加载这个文件：

```text
D:\rail-curve-extractor\output\full_dom_deeplab_v1_overlay\deeplab_v1_full_dom_overlay_weak050_strong090.tif
```

这是 RGB GeoTIFF，不需要再叠加语义分割透明层。它保留原始 DOM 的坐标范围和投影：

```text
CRS: EPSG:32651
size: 36983 x 111685
bounds: x=314736.927..315942.573, y=3518922.397..3522563.328
```

文件里已经建立内部金字塔概览：

```text
2, 4, 8, 16, 32, 64
```

DasView 打开、缩放和漫游会比没有概览的大图稳定一些。

## 颜色含义

- 黄色：DeepLab v1 概率 `>= 0.90`，强识别区域。
- 红色：DeepLab v1 概率 `>= 0.50` 且 `< 0.90`，弱识别区域。
- 其他位置：原 DOM 像素，没有额外透明层或黑底。

## 运行信息

使用脚本：

```text
scripts/predict_deeplab_full_dom_overlay.py
```

运行命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_deeplab_full_dom_overlay.py
```

模型：

```text
D:\rail-curve-extractor\output\rail_seg_deeplab_resnet50_native_v1\rail_semantic_deeplab_resnet50.pt
```

输入 DOM：

```text
D:\rail-curve-extractor\data\生产数据\无人机数据\正射\dom.tif
```

参数：

| 参数 | 数值 |
| --- | ---: |
| device | cuda |
| crop_size | 768 |
| stride | 512 |
| window_size | 4096 |
| padding | 256 |
| batch_size | 4 |
| weak_threshold | 0.50 |
| strong_threshold | 0.90 |
| processed_windows | 280 |

整体识别占比：

| 指标 | 数值 |
| --- | ---: |
| >= 0.50 像素占比 | 0.0010177 |
| >= 0.90 像素占比 | 0.0005007 |

## 注意

这张图是视觉验收用的“模型识别效果图”，不是最终中心线，也没有经过轨距配对和拓扑约束。它适合用来判断 DeepLab 在全图上到底识别出了哪些钢轨、哪些道路标线/桥缝也被误识别。
