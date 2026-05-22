# YOLO 铁轨分割验证说明

## 结论

当前 20 张 CVAT 标注已经能跑通“标注导出 -> YOLO segmentation 训练 -> 全量切片预测”的闭环。对普通直道或清晰轨带，模型已经有可用信号；但它还不是生产模型，主要问题是漏检和分割碎片化。道岔识别不好是正常的，因为当前标注里没有道岔样本，训练目标实际只有 `track_area`。

## 数据

- 数据集：`data/datasets/rail_seg_v1`
- 图像总数：56 张 aligned DOM tile
- 已标注图像：20 张
- 标注形状：130 个
- 当前有效类别：`track_area`
- 当前未覆盖：`switch_area`、`ignore_area`

这 36 张空 label 图暂时按“未标注”处理，不当作真正负样本。否则会把还没标的轨道误教成背景。

## 训练方式

本轮使用独立深度学习环境 `.yolo-venv`，不污染主 `.venv`。

核心做法：

1. 从 `rail_seg_v1/images` 和 `rail_seg_v1/labels` 读取 CVAT 转换后的 YOLO segmentation 数据。
2. 将每张 3072 像素高的 aligned tile 按高度 1024、步长 512 纵向切成训练小片。
3. 用每 5 张取 1 张的方式拆出验证集。
4. 因为当前只有 `track_area` 标注，用 `--single-cls` 训练单类分割。
5. 关闭 mosaic，避免小数据集里把铁轨空间关系搅乱。

本轮实际拆分：

| 项目 | 数量 |
| --- | ---: |
| 源图训练 | 16 |
| 源图验证 | 4 |
| YOLO 训练小片 | 80 |
| YOLO 验证小片 | 20 |
| 未标注源图 | 36 |

## 当前结果

输出目录：

- 最佳模型：`output/rail_seg_yolo_tiled_single_v1/runs/train/weights/best.pt`
- 指标：`output/rail_seg_yolo_tiled_single_v1/metrics.json`
- 全量预测图：`output/rail_seg_yolo_tiled_single_v1/predictions/all_images`

验证指标：

| 指标 | 数值 |
| --- | ---: |
| Mask precision | 0.6023 |
| Mask recall | 0.3947 |
| Mask mAP50 | 0.4214 |
| Mask mAP50-95 | 0.1614 |
| Box precision | 0.6022 |
| Box recall | 0.3772 |
| Box mAP50 | 0.4007 |

解释：

- precision 约 0.60：模型预测出来的轨道区域有一部分已经对上标注，明显好于纯颜色基线。
- recall 约 0.39：还有不少轨道区域没找回来，20 张标注还偏少。
- mAP50 约 0.42：直道轨带可以作为候选区域使用，但不能直接当最终中心线。
- 全量 56 张图都有预测输出，说明模型倾向于在 aligned corridor 内找轨带；后续必须做置信度筛选、连通域合并和几何校验。
- 当前预测实例数量较多，说明结果是“碎片轨带”，还需要合并成连续区域后再提中心线。

## 关于直道和道岔

本轮应该主要看直道。当前标注没有道岔，因此道岔区域识别不出来不算失败，也不应该用它评价模型好坏。

对直道的判断：

- aligned tile 已经把线路方向基本摆正，适合先验证直道轨带提取。
- YOLO 已能在普通轨道区域产生稳定候选。
- 结果目前适合作为“轨道候选 mask”，不适合直接输出中心线。
- 下一步应从预测 mask 中提取长条连通区域，再按轨距、方向连续性、DSM/点云高度校验中心线。

## 复现实验

准备数据和训练：

```powershell
.\.yolo-venv\Scripts\python.exe .\scripts\train_rail_seg_yolo.py `
  --dataset .\data\datasets\rail_seg_v1 `
  --out .\output\rail_seg_yolo_tiled_single_v1 `
  --model yolov8n-seg.pt `
  --epochs 30 `
  --imgsz 1024 `
  --batch 2 `
  --device 0 `
  --workers 0 `
  --single-cls `
  --mosaic 0
```

只重新生成 YOLO 数据集，不训练：

```powershell
.\.yolo-venv\Scripts\python.exe .\scripts\train_rail_seg_yolo.py `
  --dataset .\data\datasets\rail_seg_v1 `
  --out .\output\rail_seg_yolo_tiled_single_v1 `
  --prepare-only
```

## 后续路线

1. 继续增加普通直道、弯道、多轨并行的 `track_area` 标注，先把普通轨带做稳。
2. 单独补 `switch_area` 道岔标注，至少几十个道岔实例后再评价道岔。
3. 将 YOLO 预测 mask 合并为完整轨道区域。
4. 从轨道区域提取中心线候选。
5. 用 `manifest.csv`、`tile_georef.csv` 或切片地理信息把像素坐标还原到工程 `X/Y`。
6. 用 DSM 或点云查询 `Z`，再用轨距和连续性过滤误检。

输出和模型权重在 `output/` 和 `.pt` 文件中，本地保留用于查看，但不会提交进 git。
