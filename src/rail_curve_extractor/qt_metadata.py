from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class FieldSpec:
    path: str
    label: str
    caster: Callable[[str], Any]
    description: str
    default_hint: str
    recommended_range: str
    effect: str
    allow_blank: bool = False


PATH_FIELD_HELP: dict[str, dict[str, str]] = {
    "input_path": {
        "title": "点云文件",
        "description": "要处理的点云文件。支持 LAS、LAZ、CSV、TXT、XYZ 和 NPY。建议先裁剪到轨道附近再导入。",
        "default_hint": "无默认路径",
        "recommended_range": "选择单段轨道附近的数据窗口",
        "effect": "输入越聚焦，中心线提取越稳；如果一开始就把多股道、龙门吊、地面大范围一起塞进来，误检概率会明显上升。",
    },
    "output_path": {
        "title": "输出目录",
        "description": "导出目录。点击导出后，轨道候选点、中心线点、USD 曲线和本次运行摘要都会写到这里。",
        "default_hint": "默认写到用户 Documents 下的 RailCurveExtractorOutput",
        "recommended_range": "单独目录，避免覆盖其他结果",
        "effect": "不会影响算法结果，但会影响你后续整理导出文件和批处理结果的便利性。",
    },
    "config_path": {
        "title": "配置文件",
        "description": "可选的 JSON 配置文件。加载后会覆盖界面里的参数，另存时会把当前参数写成新的 JSON。",
        "default_hint": "可为空",
        "recommended_range": "为不同场景保存不同参数模板",
        "effect": "方便在手扫、无人机、局部裁剪数据之间快速切换参数，不需要每次重新手填。",
    },
}


COMMON_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        path="height_filter.keep_top_percent",
        label="顶部保留比例",
        caster=float,
        description="保留点云中较高部分的比例。值越大，留下的高程点越多；过小可能把轨顶点滤掉，过大则会混入更多杂点。",
        default_hint="0.55",
        recommended_range="0.45 - 0.70",
        effect="偏小会丢轨顶，偏大会引入道床、龙门吊腿和周围结构。普通 CSV/XYZ 场景尤其敏感；LAS 场景如果启用了增强预处理，这个参数的影响会弱一些。",
    ),
    FieldSpec(
        path="slice_length",
        label="切片长度 (m)",
        caster=float,
        description="沿轨道主方向切片的长度。切片越短越能跟随弯道，但太短时每片点数更容易不足。",
        default_hint="0.5",
        recommended_range="0.4 - 1.0",
        effect="偏短时对弯道更灵敏，但容易导致切片检测失败；偏长时更稳定，但会把局部变化抹平。",
    ),
    FieldSpec(
        path="min_points_per_slice",
        label="每片最少点数",
        caster=int,
        description="每个切片至少要有多少点才参与轨道检测。值越大越稳，但也更容易丢掉稀疏区域。",
        default_hint="30",
        recommended_range="20 - 60",
        effect="偏低时会把噪声切片也拿去做双峰检测；偏高时对稀疏、遮挡或局部缺失区域不友好。",
    ),
    FieldSpec(
        path="rail_pair_spacing_min",
        label="最小轨距窗口 (m)",
        caster=float,
        description="双峰检测时允许的最小左右轨间距。它不是严格轨距值，而是候选窗口的下边界。",
        default_hint="1.2",
        recommended_range="1.15 - 1.35",
        effect="偏大时会漏掉真实轨对，偏小时容易把非轨道双峰误判成左右轨。",
    ),
    FieldSpec(
        path="rail_pair_spacing_max",
        label="最大轨距窗口 (m)",
        caster=float,
        description="双峰检测时允许的最大左右轨间距。过大可能误选到道床或邻近结构，过小则会漏检。",
        default_hint="1.8",
        recommended_range="1.55 - 1.90",
        effect="偏大时容易把周边结构拖进轨对候选，偏小时对宽松横向扰动和局部误差容忍度不足。",
    ),
    FieldSpec(
        path="peak_search_bins",
        label="峰搜索分箱数",
        caster=int,
        description="横向直方图的分箱数量。箱数太少会把双峰抹平，太多则会在稀疏数据上不稳定。",
        default_hint="80",
        recommended_range="40 - 96",
        effect="它直接影响左右轨双峰的可分辨性，是横截面结构是否能被看出来的关键参数之一。",
    ),
    FieldSpec(
        path="peak_window_radius",
        label="峰点窗口半径 (m)",
        caster=float,
        description="围绕左右峰位聚合轨道点时使用的半径。它决定最终被判为轨顶候选点的横向宽度。",
        default_hint="0.08",
        recommended_range="0.06 - 0.15",
        effect="偏小时会漏掉轨顶候选点，偏大时会把轨旁碎点、道床和邻近噪声一起吸进来。",
    ),
    FieldSpec(
        path="savgol_window",
        label="平滑窗口",
        caster=int,
        description="中心线 Savitzky-Golay 平滑窗口长度。值越大，曲线越顺，但可能把局部细节抹掉。",
        default_hint="9",
        recommended_range="5 - 15，且尽量用奇数",
        effect="直接影响最终曲线的顺滑程度。偏大会更稳，但会损失局部变化；偏小则容易把噪声带进导出曲线。",
    ),
    FieldSpec(
        path="xy_constraint.smooth_window",
        label="XY 平滑窗口",
        caster=int,
        description="XY 平面约束选择“平滑”时使用的横向平滑窗口。值越大，越能压掉左右抖动。",
        default_hint="21",
        recommended_range="15 - 41，且尽量用奇数",
        effect="只作用于平面横向抖动，不会抹平 Z 方向的高程起伏。直轨模式会改用直线拟合，不主要依赖这个值。",
    ),
    FieldSpec(
        path="curve_width",
        label="导出曲线宽度",
        caster=float,
        description="写入 USD BasisCurves 的曲线宽度属性。它主要影响下游显示效果，不改变中心线几何。",
        default_hint="0.05",
        recommended_range="0.03 - 0.10",
        effect="只影响下游曲线显示和可视化粗细，不会改变中心线拟合结果。",
    ),
)


ROI_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        path="roi.x_min",
        label="X min",
        caster=float,
        description="世界坐标 X 方向的最小保留边界。留空表示不限制。",
        default_hint="留空",
        recommended_range="按当前轨道窗口设置",
        effect="用来粗裁不相关区域。裁得更准，后续提轨更稳；裁太死则可能把轨道的一部分直接切掉。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.x_max",
        label="X max",
        caster=float,
        description="世界坐标 X 方向的最大保留边界。留空表示不限制。",
        default_hint="留空",
        recommended_range="按当前轨道窗口设置",
        effect="与 X min 配合控制横向或纵向裁剪范围，主要作用是减小无关结构干扰。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.y_min",
        label="Y min",
        caster=float,
        description="世界坐标 Y 方向的最小保留边界。留空表示不限制。",
        default_hint="留空",
        recommended_range="按当前轨道窗口设置",
        effect="建议和 X 范围一起使用，把处理范围尽量卡到单股道附近。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.y_max",
        label="Y max",
        caster=float,
        description="世界坐标 Y 方向的最大保留边界。留空表示不限制。",
        default_hint="留空",
        recommended_range="按当前轨道窗口设置",
        effect="和 Y min 一起决定你让算法看的场景有多大，直接影响速度和误检率。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.z_min",
        label="Z min",
        caster=float,
        description="世界坐标 Z 方向的最小保留边界。可用来剔除地面以下噪声或无关低位结构。",
        default_hint="留空",
        recommended_range="有明显低位噪声时再设",
        effect="通常在地面以下或异常低值噪声较多时才有必要；设得过高会把轨道或道床一起裁掉。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.z_max",
        label="Z max",
        caster=float,
        description="世界坐标 Z 方向的最大保留边界。可用来剔除接触网、站台边缘等高位结构。",
        default_hint="留空",
        recommended_range="有明显高位结构干扰时再设",
        effect="对龙门吊、站台、桥梁边缘等高位结构干扰很有用；设得过低会误伤轨道上方有效点。",
        allow_blank=True,
    ),
)
