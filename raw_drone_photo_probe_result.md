# Raw Drone Photo Probe Result

## 结论

当前 v3 语义分割模型可以直接在这批无人机原始 JPG 上识别直道钢轨，视觉效果比预期好。原因是这些原片虽然不是 DOM 切片，但大多数仍接近俯视正射视角，钢轨、轨枕、道床的外观和训练数据差异不算太大。

不过这只是无标注视觉抽查，不是定量验证。它能说明“可以作为下一步实验输入”，不能说明已经稳定泛化。

## 输入

- 原始照片目录：`D:\BaiduNetdiskDownload\0402NT\无人机数据\DJI_202604010912_030_铁道`
- 可识别图片：436 张 JPG
- 单张尺寸：5280 x 3956

## 测试方法

使用已训练的模型：

- `output/rail_seg_semantic_unet_v3/rail_semantic_unet.pt`
- 阈值：0.90
- 抽样方式：排序后每 35 张取 1 张，最多 12 张
- 推理方式：对原始大图做滑窗推理，窗口 768 x 1024，步长 512 x 512

输出目录：

- `output/raw_drone_photo_probe/semantic_v3_sample`

主要文件：

- `contact_sheet.jpg`
- `overlays/*.jpg`
- `masks/*.png`
- `probabilities/*.png`
- `summary.json`

## 观察

视觉上模型能识别：

- 直道两侧钢轨本体
- 多股并行直道
- 旁边有农田、道路、建筑时的钢轨区域

抽样中也看到：

- 没有明显铁轨的图，模型基本不输出或输出很少。
- 有清晰直道的图，输出沿钢轨连续分布。
- 目前输出仍是“钢轨本体 mask”，还不是最终轨道中心线。

## 限制

- 当前没有原始 JPG 的人工标注，所以没有 precision / recall / IoU。
- 原始照片没有直接按 `tile_georef.csv` 对齐，输出 mask 目前是照片像素坐标，不是地图坐标。
- 如果照片倾斜明显、运动模糊、钢轨被遮挡，效果可能下降。
- 当前模型没有专门训练道路白线、护栏、楼顶线缆等干扰物，后续仍需抽查误检。

## 复现命令

```powershell
.\.yolo-venv\Scripts\python.exe .\scripts\predict_rail_seg_images.py `
  --input-dir "D:\BaiduNetdiskDownload\0402NT\无人机数据\DJI_202604010912_030_铁道" `
  --model .\output\rail_seg_semantic_unet_v3\rail_semantic_unet.pt `
  --out .\output\raw_drone_photo_probe\semantic_v3_sample `
  --device cuda `
  --sample-every 35 `
  --max-images 12 `
  --tile-width 768 `
  --tile-height 1024 `
  --tile-stride-x 512 `
  --tile-stride-y 512
```

## 下一步

如果目标是“从原始照片直接识别轨道”，下一步应该抽 20 到 30 张代表性原始照片做少量人工标注或人工验收集，再评估这个模型是否需要微调。

如果目标是“输出真实地图坐标中心线”，仍然建议优先走 DOM / 正射影像路线，因为原始照片还需要相机姿态、内参、畸变、RTK/IMU 或摄影测量流程才能精确落图。
