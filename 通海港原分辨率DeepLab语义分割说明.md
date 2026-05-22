# 通海港原分辨率 DeepLab 语义分割说明

## 为什么换模型

旧的 `SmallUNet` 训练和推理路径会把输入 crop resize 到 `384x1024`。对通海港这种 DOM 来说，单根钢轨本来就是很细的线状目标，横向压缩会直接损失钢轨像素，尤其在桥影、护轨、道岔和浅色道床附近，模型很容易只认出一部分轨道。

新的路线改成：

- 模型：`DeepLabV3-ResNet50`
- backbone：ImageNet 预训练 ResNet50
- 输入：原生像素 patch，不做比例缩放
- 当前正式 v1 patch：`768x768`
- 推理：滑窗拼接回原图

这里的“原分辨率”不是一次把整张 `3072x3072` tile 塞进模型，而是从原图裁 patch，patch 内保持 1:1 像素。这样不会再把钢轨压窄。

## 新增脚本

```text
scripts/train_rail_seg_deeplab.py
scripts/predict_rail_seg_deeplab_images.py
```

训练脚本默认冻结 BatchNorm 统计量，因为原生大 patch 下 batch 通常只能取 1，这对 DeepLab 的 BatchNorm 更稳。

## 已完成的 v1 训练

训练命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_rail_seg_deeplab.py `
  --dataset data\datasets\rail_seg_v7_tonghaigang_chinese `
  --out output\rail_seg_deeplab_resnet50_native_v1 `
  --epochs 10 `
  --batch 1 `
  --crop-size 768 `
  --stride 512 `
  --max-predict-images 6 `
  --device cuda `
  --workers 0
```

输出模型：

```text
output/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt
```

训练结果：

| 指标 | 数值 |
| --- | ---: |
| best epoch | 4 |
| best threshold | 0.90 |
| precision | 0.8055 |
| recall | 0.9113 |
| F1 | 0.8552 |
| IoU | 0.7470 |

这不是最终质量结论，只说明新模型路线能在当前数据集上正常收敛，并且没有旧 UNet 的横向压缩问题。

## TA08 局部推理结果

我把 v1 模型跑到了 TA08 附近 29 个 raw DOM ROI tile 上：

```text
output/raw_dom_roi_fullpass_v1/native_deeplab_ta08_tiles/predictions_v1/
```

并按原始 DOM 坐标重新拼成了可叠加 GeoTIFF：

```text
output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/ta08_segmentation_overlay_rgba_thr050_090.tif
output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/ta08_dom_segmentation_overlay.tif
```

如果 DasView 支持透明 GeoTIFF，优先叠加 `ta08_segmentation_overlay_rgba_thr050_090.tif`；如果透明显示不正常，就看 `ta08_dom_segmentation_overlay.tif`。

QA 裁图：

```text
output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/qa_crops/
```

你刚才指出的坐标单独裁图：

```text
output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/qa_crops/ta08_user_coord_315334_923_3520755_899_deeplab_v1_dom_seg.png
```

这个坐标 5m 半径内的统计从旧 UNet 的强识别 `293` 像素，提升到 DeepLab v1 的强识别 `2950` 像素，说明原分辨率强 backbone 对这段确实更敏感。

## 目前看到的问题

v1 已经能更连续地识别你指出的那股轨道，但它也会把一些道路白线、桥面伸缩缝、护栏阴影等细长结构识别成钢轨。这是因为当前训练集中“像钢轨但不是钢轨”的负样本还不够。

所以后续不能只拿 DeepLab mask 直接生成中心线。正确用法应该是：

1. 用 DeepLab v1 提供更完整的单根钢轨候选。
2. 用轨距/股道间距约束筛掉不成对的细长误检。
3. 用主线拓扑和道岔相切约束生成中心线。
4. 对道路白线、桥缝、护栏等误检区域补 `ignore_area` 或负样本后继续微调。

## 下一轮建议

下一步我建议不要立刻追更高 epoch，而是先做两件事：

- 把 DeepLab v1 的 TA08 证据图叠到 DasView/QGIS 里人工看一眼，确认它是否确实补到了旧 UNet 漏掉的钢轨。
- 追加负样本或 `ignore_area`：道路白线、桥缝、栏杆阴影、混凝土边缘、道床边缘。然后用同样脚本训练 v2。

如果 v1 的轨道召回确认明显变好，后处理就改成“DeepLab 单轨候选 + 成对钢轨约束 + 中心线拟合”，而不是继续依赖旧 UNet mask。

## 2026-05-20 轨距配对试点

已新增 TA08 局部轨距配对脚本：

```text
scripts/build_deeplab_gauge_pair_centerlines.py
```

输出目录：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/
```

这一步把 DeepLab v1 的单根钢轨概率响应转换为“必须成对且间距接近轨距”的中心线候选。DSM `D:\正射\lidars\terra_dsm\dsm.tif` 也已接入为高度脊线诊断证据，但当前不作为硬否决条件。

优先验收：

```text
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/deeplab_gauge_pair_centerlines.shp
output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1/qa_crops/ta08_user_coord_overlay.png
```

详细说明见：

```text
通海港DeepLab轨距配对中心线试点说明.md
```
