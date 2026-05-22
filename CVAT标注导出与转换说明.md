# CVAT 标注导出与转换说明

## 结论

当前推荐从 CVAT 导出 `CVAT for images 1.1` 格式，然后用本项目脚本转换为本地训练/回投数据集。

本机 CVAT 服务已确认可访问，版本为 `2.62.0`；任务 API 未登录会返回 `401`，所以本流程不自动处理账号和鉴权，只使用你在浏览器里下载的导出文件。

## 在 CVAT 里导出

打开你当前任务：

```text
http://localhost:8080/tasks/1/jobs/1
```

推荐步骤：

1. 回到 task 页面，或在 job 页面右上角找 `Actions` 菜单。
2. 选择 `Export task dataset` 或 `Export annotations`。
3. 格式选择 `CVAT for images 1.1`。
4. 如果有 `Save images` 选项，可以不勾选；项目本地已经有切片图片。
5. 下载得到一个 `.zip` 文件，例如 `task_1_cvat_for_images_1_1.zip`。

如果你导出的是解压后的 `annotations.xml`，也可以直接作为脚本输入。

## 转换为项目数据集

示例命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\convert_cvat_annotations.py `
  --annotations "D:\Downloads\task_1_cvat_for_images_1_1.zip" `
  --tile-georef ".\data\dom_tiles_aligned_annotation\tile_georef.csv" `
  --out ".\data\datasets\rail_seg_v1" `
  --classes "track_area,switch_area,ignore_area" `
  --overlay
```

输出目录会包含：

```text
data/datasets/rail_seg_v1/
├─ images/                 # 从 tile_georef 指向的本地切片复制而来
├─ labels/                 # YOLO segmentation 标签
├─ overlays/               # 可选，标注叠图预览
├─ annotations_map.geojson # 像素 polygon 回投后的地图坐标
├─ classes.txt             # 类别顺序
├─ manifest.csv            # 每个标注实例的索引
├─ manifest.json
└─ summary.json
```

## 对齐逻辑

脚本不是靠图片顺序猜对应关系，而是用文件名匹配：

```text
CVAT image name -> tile_georef.csv 的 tile_name / image_path
```

匹配成功后，每个 polygon 点会按切片的仿射变换计算地图坐标：

```text
X = a * col + b * row + c
Y = d * col + e * row + f
```

其中 `[a, b, c, d, e, f]` 来自 `tile_georef.csv` 的 `tile_transform` 字段。

## 常见问题

- 如果脚本报 `missing from tile_georef`，说明 CVAT 导出的图片名和当前 `tile_georef.csv` 不是同一批切片。
- 如果报 `label is not listed in classes`，说明 CVAT 里用了不在 `--classes` 中的标签；要么补进 `--classes`，要么加 `--skip-unknown-labels`。
- 第一批 20 张标注建议一定加 `--overlay`，先肉眼抽查几张叠图，确认没有错位再训练。
- 输出在 `data/datasets/...` 下，是生成数据，不建议提交进 git。
