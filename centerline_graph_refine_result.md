# Centerline Graph Refine Result

## 结论

已经做出第一版“连续主线追踪”后处理。它不再把中心线看成每张 tile 内的独立短线，而是先把候选线按几何连续性合并成轨道链，再用更宽松的主线 stitching 规则跨过断点，输出一条贯穿全段的主线候选。

这版能验证你的判断：中心线应该按连续网络处理，而不是只做局部识别。

## 输入

- 候选线：`output/rail_centerline_candidates_v3/track_centerline_candidates.geojson`
- 原始候选线数量：175
- 用于合并的候选线数量：166
- 全局纵向范围：约 2844.59 m

## 方法

脚本：`scripts/refine_centerline_graph.py`

核心流程：

1. 估计整段线路的主方向。
2. 把每条候选线投影成：

```text
s = 沿线路方向的位置
t = 横向偏移
```

3. 对候选线做第一层合并：
   - 方向接近
   - 纵向重叠或间隙很小
   - 横向偏移接近

4. 得到较长的连续轨道链。
5. 对轨道链做第二层主线 stitching：
   - 允许跨过较大断点
   - 优先选择纵向覆盖最长、方向最连续、横向跳变较小的路径

## 输出

输出目录：`output/rail_centerline_refined_v1`

主要文件：

- `main_centerline.geojson`
- `main_centerline.shp`
- `refined_centerline_network.geojson`
- `refined_centerline_network.shp`
- `refined_centerline_diagnostic.png`
- `summary.json`

其中 DASView 推荐先加载：

```text
output/rail_centerline_refined_v1/main_centerline.shp
```

如果想看其它候选链，再加载：

```text
output/rail_centerline_refined_v1/refined_centerline_network.shp
```

## 本轮结果

| 项目 | 数值 |
| --- | ---: |
| 原始候选线 | 175 |
| 使用候选线 | 166 |
| 第一层连通组件 | 20 |
| 合并后轨道链 | 16 |
| 主线 stitching 使用链数 | 7 |
| 主线长度 | 2844.641 m |
| 主线纵向覆盖 | 2844.579 m |
| 主线平均置信度 | 0.9534 |
| 最大桥接断点 | 65.200 m |
| 总桥接断点 | 270.189 m |

## 注意

这条主线是“算法选出的贯穿主线候选”，不是最终人工确认成果。尤其要检查：

- 最大桥接断点附近是否跨错线。
- 道岔附近是否把支线当成主线。
- 多股道并行区域是否选中了你实际想要的那条主线。

如果 DASView 里红线整体贴着你想要的主线，下一步就可以把这套主线约束反过来用于改进中心线提取；如果局部跨错线，就需要在这些位置加入“主线锚点”或人工指定入口/出口。

## 复现命令

生成 refined 输出：

```powershell
.\.venv\Scripts\python.exe .\scripts\refine_centerline_graph.py `
  --input .\output\rail_centerline_candidates_v3\track_centerline_candidates.geojson `
  --out .\output\rail_centerline_refined_v1 `
  --max-lateral-gap 1.8 `
  --max-longitudinal-gap 10 `
  --max-angle-deg 20 `
  --min-chain-extent 40 `
  --merge-bin-size 0.75 `
  --main-stitch-lateral-gap 4.0 `
  --main-stitch-longitudinal-gap 180 `
  --main-stitch-angle-deg 35
```

导出 SHP：

```powershell
.\.venv\Scripts\python.exe .\scripts\export_centerline_shapefile.py `
  --input .\output\rail_centerline_refined_v1\main_centerline.geojson `
  --out .\output\rail_centerline_refined_v1\main_centerline.shp `
  --crs-raster .\data\aligned_dom\aligned_dom.tif

.\.venv\Scripts\python.exe .\scripts\export_centerline_shapefile.py `
  --input .\output\rail_centerline_refined_v1\refined_centerline_network.geojson `
  --out .\output\rail_centerline_refined_v1\refined_centerline_network.shp `
  --crs-raster .\data\aligned_dom\aligned_dom.tif
```
