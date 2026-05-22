from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .io import load_point_cloud_data
from .pipeline import PipelineResult, analyze_input, export_pipeline_result, prepare_config
from .preview import ViewBounds, build_profile_points, combine_bounds, downsample_indices, downsample_points, visible_downsample_indices
from .qt_metadata import COMMON_FIELDS, PATH_FIELD_HELP, ROI_FIELDS, FieldSpec
from .runtime import bundled_path, default_output_dir

WINDOW_TITLE = "Rail Curve Extractor"
FONT_FAMILY = "Microsoft YaHei UI"
WINDOW_BG = "#f3f6f8"
SIDEBAR_BG = "#eef4f7"
CARD_BG = "#ffffff"
CARD_ALT_BG = "#f8fafc"
CANVAS_BG = "#fbfcfd"
TEXT = "#17212b"
SUBTEXT = "#64748b"
ACCENT = "#147d8f"
ACCENT_DARK = "#0f5f6d"
RAIL = "#f97316"
MUTED = "#94a3b8"
GRID = "#e5edf2"
TRACK_COLORS = ("#147d8f", "#f97316", "#7c3aed", "#16a34a", "#dc2626", "#0891b2", "#c026d3", "#ca8a04")
TRACK_ROI_AXES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
DEFAULT_MANUAL_TRACK_COUNT = 1
MAX_UI_TRACKS = 12
RAW_PREVIEW_CACHE_LIMIT = 2_000_000
RAW_PREVIEW_LOD_LIMITS = (120_000, 500_000, RAW_PREVIEW_CACHE_LIMIT)
RAW_PREVIEW_DISPLAY_LIMIT = 45_000
RAW_PREVIEW_UPDATE_DELAY_MS = 320
ORIENTED_ROI_TARGETS = (
    ("global", "全局 ROI"),
    ("auto_tracks", "多轨总 ROI（自动拆分）"),
    *tuple((f"track_{track_index}", f"轨道 {track_index} ROI") for track_index in range(1, MAX_UI_TRACKS + 1)),
    ("turnout", "道岔 ROI"),
)
ORIENTED_ROI_COLORS = {
    "global": ACCENT_DARK,
    "auto_tracks": "#0f766e",
    "turnout": "#dc2626",
}
ANCHOR_TARGETS = (
    ("global", "中心线锚点（单轨/全局）"),
    *tuple((f"track_{track_index}", f"轨道 {track_index} 锚点") for track_index in range(1, MAX_UI_TRACKS + 1)),
    ("turnout_main", "道岔主线锚点"),
    ("turnout_branch", "道岔分支锚点"),
)
ANCHOR_COLORS = {
    "global": "#0ea5e9",
    "turnout_main": "#ef4444",
    "turnout_branch": "#06b6d4",
}


def _set_value_on_path(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _get_value_from_path(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        current = current[key]
    return current


def _anchor_points_to_config(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in points]


def _normalize_preview_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    rgb_values = np.asarray(rgb, dtype=np.float64)
    max_value = float(np.nanmax(rgb_values)) if rgb_values.size else 0.0
    scale = 256.0 if max_value > 255.0 else 1.0
    return np.clip(rgb_values / scale, 0.0, 255.0).astype(np.uint8)


def _rgb_to_qbrushes(rgb: np.ndarray) -> list[QtGui.QBrush]:
    return [QtGui.QBrush(QtGui.QColor(int(red), int(green), int(blue), 210)) for red, green, blue in rgb]


def build_oriented_roi_from_xy_points(points_xy: list[tuple[float, float]] | np.ndarray) -> dict[str, Any]:
    selected_points = np.asarray(points_xy, dtype=float)
    if selected_points.shape != (4, 2):
        raise ValueError("四点 ROI 需要正好 4 个 XY 点。")

    origin_xy = selected_points.mean(axis=0)
    centered = selected_points - origin_xy
    if float(np.max(np.linalg.norm(centered, axis=1))) <= 1e-6:
        raise ValueError("四点 ROI 的范围太小，无法生成有效方向。")

    covariance = np.cov(centered.T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_s = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    axis_s_norm = float(np.linalg.norm(axis_s))
    if axis_s_norm <= 1e-9:
        raise ValueError("四点 ROI 的长轴方向无效。")
    axis_s = axis_s / axis_s_norm

    longest_vector = _longest_pair_vector(selected_points)
    if float(np.dot(axis_s, longest_vector)) < 0.0:
        axis_s = -axis_s
    axis_t = np.array([-axis_s[1], axis_s[0]], dtype=float)

    local_s = centered @ axis_s
    local_t = centered @ axis_t
    s_min = float(local_s.min())
    s_max = float(local_s.max())
    t_min = float(local_t.min())
    t_max = float(local_t.max())
    if s_max - s_min <= 1e-3 or t_max - t_min <= 1e-3:
        raise ValueError("四点 ROI 过窄或退化，请重新点选四个角。")

    return {
        "enabled": True,
        "origin": [float(origin_xy[0]), float(origin_xy[1])],
        "axis_s": [float(axis_s[0]), float(axis_s[1])],
        "axis_t": [float(axis_t[0]), float(axis_t[1])],
        "s_min": s_min,
        "s_max": s_max,
        "t_min": t_min,
        "t_max": t_max,
        "z_min": None,
        "z_max": None,
    }


def oriented_roi_corners(oriented_roi: dict[str, Any]) -> np.ndarray:
    origin_xy = np.asarray(oriented_roi["origin"], dtype=float)
    axis_s = np.asarray(oriented_roi["axis_s"], dtype=float)
    axis_t = np.asarray(oriented_roi["axis_t"], dtype=float)
    s_min = float(oriented_roi["s_min"])
    s_max = float(oriented_roi["s_max"])
    t_min = float(oriented_roi["t_min"])
    t_max = float(oriented_roi["t_max"])
    corners_local = ((s_min, t_min), (s_min, t_max), (s_max, t_max), (s_max, t_min), (s_min, t_min))
    return np.asarray([origin_xy + local_s * axis_s + local_t * axis_t for local_s, local_t in corners_local])


def _longest_pair_vector(points_xy: np.ndarray) -> np.ndarray:
    best_vector = points_xy[1] - points_xy[0]
    best_distance = float(np.dot(best_vector, best_vector))
    for first_index in range(len(points_xy)):
        for second_index in range(first_index + 1, len(points_xy)):
            candidate = points_xy[second_index] - points_xy[first_index]
            distance = float(np.dot(candidate, candidate))
            if distance > best_distance:
                best_vector = candidate
                best_distance = distance
    return best_vector


def _is_oriented_roi_configured(oriented_roi: Any) -> bool:
    if not isinstance(oriented_roi, dict) or oriented_roi.get("enabled") is False:
        return False
    required_keys = ("origin", "axis_s", "axis_t", "s_min", "s_max", "t_min", "t_max")
    return all(oriented_roi.get(key) is not None for key in required_keys)


def _track_index_from_target_key(target_key: str) -> int | None:
    if not target_key.startswith("track_"):
        return None
    try:
        track_index = int(target_key.split("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None
    return track_index if track_index > 0 else None


def _oriented_roi_color(target_key: str) -> str:
    track_index = _track_index_from_target_key(target_key)
    if track_index is not None:
        return TRACK_COLORS[(track_index - 1) % len(TRACK_COLORS)]
    return ORIENTED_ROI_COLORS.get(target_key, ACCENT)


def _anchor_color(target_key: str) -> str:
    track_index = _track_index_from_target_key(target_key)
    if track_index is not None:
        return TRACK_COLORS[(track_index - 1) % len(TRACK_COLORS)]
    return ANCHOR_COLORS.get(target_key, "#0ea5e9")


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AlignedComboBox(QtWidgets.QComboBox):
    def __init__(self, popup_width: int = 260) -> None:
        super().__init__()
        self.popup_width = popup_width
        self.setMaxVisibleItems(18)
        self.view().setMinimumWidth(popup_width)

    def showPopup(self) -> None:
        super().showPopup()
        self._align_popup_below()
        QtCore.QTimer.singleShot(0, self._align_popup_below)

    def _align_popup_below(self) -> None:
        popup = self.view().window()
        if popup is None:
            return

        popup_width = max(self.popup_width, self.width())
        popup.resize(popup_width, popup.height())
        global_position = self.mapToGlobal(QtCore.QPoint(0, self.height() + 2))

        screen = QtGui.QGuiApplication.screenAt(global_position) or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            popup.move(global_position)
            return

        available = screen.availableGeometry()
        x_pos = min(max(global_position.x(), available.left()), max(available.left(), available.right() - popup_width))
        y_pos = global_position.y()
        if y_pos + popup.height() > available.bottom():
            combo_top = self.mapToGlobal(QtCore.QPoint(0, 0)).y()
            y_pos = max(available.top(), combo_top - popup.height() - 2)
        popup.move(x_pos, y_pos)


class ParameterInfoPopup(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setObjectName("InfoPopup")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMinimumWidth(340)
        self.setMaximumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        self.title_label = QtWidgets.QLabel("参数说明")
        self.title_label.setObjectName("PopupTitle")
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setObjectName("PopupPath")
        self.description_label = QtWidgets.QLabel("")
        self.description_label.setObjectName("PopupBody")
        self.description_label.setWordWrap(True)
        self.default_value_label = QtWidgets.QLabel("--")
        self.default_value_label.setObjectName("PopupValue")
        self.range_value_label = QtWidgets.QLabel("--")
        self.range_value_label.setObjectName("PopupValue")
        self.effect_label = QtWidgets.QLabel("")
        self.effect_label.setObjectName("PopupBody")
        self.effect_label.setWordWrap(True)

        meta_grid = QtWidgets.QGridLayout()
        meta_grid.setHorizontalSpacing(10)
        meta_grid.setVerticalSpacing(6)
        default_tag = QtWidgets.QLabel("默认值")
        default_tag.setObjectName("PopupMetaTag")
        range_tag = QtWidgets.QLabel("推荐范围")
        range_tag.setObjectName("PopupMetaTag")
        meta_grid.addWidget(default_tag, 0, 0)
        meta_grid.addWidget(self.default_value_label, 0, 1)
        meta_grid.addWidget(range_tag, 1, 0)
        meta_grid.addWidget(self.range_value_label, 1, 1)

        effect_title = QtWidgets.QLabel("影响说明")
        effect_title.setObjectName("PopupSectionTitle")

        layout.addWidget(self.title_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.description_label)
        layout.addLayout(meta_grid)
        layout.addWidget(effect_title)
        layout.addWidget(self.effect_label)

    def set_content(
        self,
        *,
        title: str,
        field_key: str,
        description: str,
        default_hint: str,
        recommended_range: str,
        effect: str,
    ) -> None:
        self.title_label.setText(title)
        self.path_label.setText(field_key)
        self.description_label.setText(description)
        self.default_value_label.setText(default_hint)
        self.range_value_label.setText(recommended_range)
        self.effect_label.setText(effect)


class AnalysisSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class AnalysisRunnable(QtCore.QRunnable):
    def __init__(self, input_path: Path, config: dict[str, Any]) -> None:
        super().__init__()
        self.input_path = input_path
        self.config = config
        self.signals = AnalysisSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = analyze_input(input_path=self.input_path, config_overrides=self.config)
        except Exception as exc:  # pragma: no cover - passed back to GUI thread
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)


class RawPreviewSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class RawPreviewRunnable(QtCore.QRunnable):
    def __init__(self, input_path: Path, cache_limit: int = RAW_PREVIEW_CACHE_LIMIT) -> None:
        super().__init__()
        self.input_path = input_path
        self.cache_limit = cache_limit
        self.signals = RawPreviewSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            point_cloud = load_point_cloud_data(self.input_path)
            sample_indices = downsample_indices(len(point_cloud.points), self.cache_limit)
            preview_points_xy = point_cloud.points[sample_indices, :2].astype(np.float32, copy=False)
            preview_rgb = None
            if point_cloud.rgb is not None and len(point_cloud.rgb) == len(point_cloud.points):
                preview_rgb = _normalize_preview_rgb(point_cloud.rgb[sample_indices])
            payload = {
                "points_xy": preview_points_xy,
                "rgb": preview_rgb,
                "has_rgb": preview_rgb is not None,
                "input_points": int(len(point_cloud.points)),
                "cache_points": int(len(preview_points_xy)),
            }
        except Exception as exc:  # pragma: no cover - passed back to GUI thread
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(payload)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, label: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setMinimumHeight(82)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(6)

        title = QtWidgets.QLabel(label)
        title.setObjectName("MetricLabel")
        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setObjectName("MetricValue")

        layout.addWidget(title)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class PlotPanel(QtWidgets.QFrame):
    def __init__(self, title: str, description: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFrame")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("CardTitle")
        description_label = QtWidgets.QLabel(description)
        description_label.setObjectName("CardCaption")
        description_label.setWordWrap(True)

        self.plot = pg.PlotWidget(background=CANVAS_BG)
        self.plot.showGrid(x=True, y=True, alpha=0.20)
        self.plot.setMinimumHeight(230)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.getAxis("left").setTextPen(pg.mkColor(TEXT))
        self.plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT))
        self.plot.getAxis("left").setPen(pg.mkPen(color=GRID))
        self.plot.getAxis("bottom").setPen(pg.mkPen(color=GRID))

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(self.plot, stretch=1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1780, 1040)
        self.setMinimumSize(1500, 860)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.result: PipelineResult | None = None
        self.field_inputs: dict[str, QtWidgets.QLineEdit] = {}
        self.track_roi_inputs: dict[tuple[int, str], QtWidgets.QLineEdit] = {}
        self.track_enabled_checks: dict[int, QtWidgets.QCheckBox] = {}
        self.track_roi_boxes: dict[int, QtWidgets.QGroupBox] = {}
        self.turnout_roi_inputs: dict[str, QtWidgets.QLineEdit] = {}
        self.field_labels: dict[str, ClickableLabel] = {}
        self.metric_cards: dict[str, MetricCard] = {}
        self.page_stack: QtWidgets.QStackedWidget | None = None
        self.nav_buttons: list[QtWidgets.QPushButton] = []
        self.page_fade_animation: QtCore.QPropertyAnimation | None = None
        self.raw_preview_points_xy: np.ndarray | None = None
        self.raw_preview_rgb: np.ndarray | None = None
        self.raw_preview_input_points = 0
        self.raw_preview_current_display_points = 0
        self.raw_preview_current_lod_points = 0
        self.raw_preview_bounds: ViewBounds | None = None
        self.raw_preview_lod_levels: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.raw_preview_scatter_item: pg.ScatterPlotItem | None = None
        self.raw_preview_active = False
        self.raw_preview_update_timer: QtCore.QTimer | None = None
        self.oriented_roi_configs: dict[str, dict[str, Any]] = {}
        self.oriented_roi_points: dict[str, list[tuple[float, float]]] = {}
        self.oriented_roi_overlay_items: list[Any] = []
        self.anchor_points: dict[str, list[tuple[float, float]]] = {}
        self.selected_field_path: str | None = None
        self.info_popup = ParameterInfoPopup(self)
        self.info_anchor_widget: QtWidgets.QWidget | None = None

        self.input_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit(str(default_output_dir()))
        self.config_edit = QtWidgets.QLineEdit()
        self.height_filter_checkbox = QtWidgets.QCheckBox("启用高度过滤")
        self.xy_constraint_combo = QtWidgets.QComboBox()
        self.xy_constraint_combo.addItem("自由：适合曲线/道岔", "free")
        self.xy_constraint_combo.addItem("平滑：压掉轻微左右抖动", "smooth")
        self.xy_constraint_combo.addItem("直轨：XY 强制直线，保留 Z 起伏", "straight")
        self.turnout_enabled_check = QtWidgets.QCheckBox("启用道岔模式")
        self.summary_label = QtWidgets.QLabel("尚未生成预览。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("SummaryLabel")
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.status_label = QtWidgets.QLabel("准备就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("StatusLabel")

        self.preview_button = QtWidgets.QPushButton("生成预览")
        self.preview_button.setObjectName("PrimaryButton")
        self.export_button = QtWidgets.QPushButton("导出 USD Curve")
        self.export_button.setObjectName("PrimaryButton")
        self.export_button.setEnabled(False)
        self.load_overview_button = QtWidgets.QPushButton("加载点云底图")
        self.pick_oriented_roi_button = QtWidgets.QPushButton("点选四点 ROI")
        self.pick_oriented_roi_button.setCheckable(True)
        self.clear_oriented_roi_button = QtWidgets.QPushButton("清除当前 ROI")
        self.anchor_target_combo = AlignedComboBox()
        for target_key, target_label in ANCHOR_TARGETS:
            self.anchor_target_combo.addItem(target_label, target_key)
        self.pick_anchor_button = QtWidgets.QPushButton("点选中心线锚点")
        self.pick_anchor_button.setCheckable(True)
        self.clear_anchor_button = QtWidgets.QPushButton("清除当前锚点")
        self.anchor_status_label = QtWidgets.QLabel("锚点未设置。")
        self.anchor_status_label.setObjectName("CardCaption")
        self.anchor_status_label.setWordWrap(True)
        self.auto_track_split_check = QtWidgets.QCheckBox("一个 ROI 内自动拆分多条轨道")
        self.auto_track_count_spin = QtWidgets.QSpinBox()
        self.auto_track_count_spin.setRange(1, MAX_UI_TRACKS)
        self.auto_track_count_spin.setValue(3)
        self.auto_track_count_spin.setSuffix(" 条")
        self.manual_track_count_spin = QtWidgets.QSpinBox()
        self.manual_track_count_spin.setRange(1, MAX_UI_TRACKS)
        self.manual_track_count_spin.setValue(DEFAULT_MANUAL_TRACK_COUNT)
        self.manual_track_count_spin.setSuffix(" 条")
        self.oriented_roi_target_combo = AlignedComboBox()
        for target_key, target_label in ORIENTED_ROI_TARGETS:
            self.oriented_roi_target_combo.addItem(target_label, target_key)
        self.oriented_roi_status_label = QtWidgets.QLabel("四点 ROI 未设置。")
        self.oriented_roi_status_label.setObjectName("CardCaption")
        self.oriented_roi_status_label.setWordWrap(True)

        self.top_panel = PlotPanel(
            title="俯视图预览",
            description="灰色为点云/过滤点，彩色为轨道候选点和中心线；可加载底图后点选四个角生成有向 ROI。",
        )
        self.profile_panel = PlotPanel(
            title="纵断面预览",
            description="横轴是累计里程，纵轴是高程，用来判断导出曲线是否连续、是否有异常跳点。",
        )
        self.raw_preview_update_timer = QtCore.QTimer(self)
        self.raw_preview_update_timer.setSingleShot(True)
        self.raw_preview_update_timer.timeout.connect(self._refresh_raw_preview_for_current_view)

        self._setup_ui()
        self._polish_form_controls()
        self._connect_signals()
        self._apply_config_to_form(prepare_config())
        self._set_metric_defaults()
        self._draw_empty_plots()
        self._log("等待输入点云。")

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        root_layout.addWidget(self._build_navigation(), stretch=0)
        self.page_stack = QtWidgets.QStackedWidget()
        self.page_stack.setObjectName("PageStack")
        self.page_stack.addWidget(
            self._build_workflow_page(
                "1. 导入数据",
                "选择点云、输出目录和配置文件；加载彩色点云底图后再去框选 ROI。",
                [self._build_file_group(), self._build_overview_group()],
            )
        )
        self.page_stack.addWidget(
            self._build_workflow_page(
                "2. ROI 与锚点",
                "先框选轨道区域，再沿目标中心线点锚点；道岔主线和分支可分别点。",
                [
                    self._build_oriented_roi_toolbar(),
                    self._build_roi_group(),
                    self._build_track_roi_group(),
                    self._build_turnout_roi_group(),
                ],
            )
        )
        self.page_stack.addWidget(
            self._build_workflow_page(
                "3. 参数",
                "常用参数集中在这里；一般只需要调切片长度、轨距范围和 XY 约束。",
                [self._build_parameter_group()],
            )
        )
        self.page_stack.addWidget(
            self._build_workflow_page(
                "4. 预览与导出",
                "生成中心线预览，确认右侧图形和摘要后再导出 USD/XYZ 文件。",
                [self._build_action_group()],
            )
        )
        self.page_stack.addWidget(
            self._build_workflow_page(
                "5. 日志",
                "查看加载、分析、导出过程中的运行信息和错误提示。",
                [self._build_log_group()],
            )
        )
        root_layout.addWidget(self.page_stack, stretch=5)
        root_layout.addWidget(self._build_preview_area(), stretch=6)
        self._set_page(0)

    def _build_navigation(self) -> QtWidgets.QWidget:
        frame = QtWidgets.QFrame()
        frame.setObjectName("NavSidebar")
        frame.setFixedWidth(190)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(10)

        product = QtWidgets.QLabel("Rail Curve")
        product.setObjectName("EyebrowLabel")
        title = QtWidgets.QLabel("轨道中心线")
        title.setObjectName("NavTitle")
        title.setWordWrap(True)
        layout.addWidget(product)
        layout.addWidget(title)
        layout.addSpacing(8)

        for index, label in enumerate(("导入数据", "ROI / 锚点", "参数", "预览导出", "日志")):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, page_index=index: self._set_page(page_index))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)
        hint = QtWidgets.QLabel("建议流程：导入 → 框 ROI → 点锚点 → 预览 → 导出")
        hint.setObjectName("NavHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return frame

    def _set_page(self, index: int) -> None:
        changed = self.page_stack is not None and self.page_stack.currentIndex() != index
        if self.page_stack is not None:
            self.page_stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        if changed:
            self._animate_current_page()

    def _animate_current_page(self) -> None:
        if self.page_stack is None:
            return
        page = self.page_stack.currentWidget()
        if page is None:
            return
        if self.page_fade_animation is not None:
            self.page_fade_animation.stop()

        effect = QtWidgets.QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QtCore.QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(0.35)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda target=page: target.setGraphicsEffect(None))
        self.page_fade_animation = animation
        animation.start()

    def _polish_form_controls(self) -> None:
        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            if isinstance(line_edit.parentWidget(), QtWidgets.QAbstractSpinBox):
                line_edit.setClearButtonEnabled(False)
                continue
            line_edit.setClearButtonEnabled(True)
            line_edit.setMinimumHeight(38)
        for combo_box in self.findChildren(QtWidgets.QComboBox):
            combo_box.setMinimumHeight(38)
        for spin_box in self.findChildren(QtWidgets.QSpinBox):
            spin_box.setMinimumHeight(38)
            spin_box.setMinimumWidth(96)

    def _manual_track_count(self) -> int:
        return max(1, min(MAX_UI_TRACKS, int(self.manual_track_count_spin.value())))

    def _update_manual_track_roi_visibility(self) -> None:
        visible_count = self._manual_track_count()
        for track_index, box in self.track_roi_boxes.items():
            box.setVisible(track_index <= visible_count)

    def _set_manual_track_count_at_least(self, track_index: int) -> None:
        if 1 <= track_index <= MAX_UI_TRACKS and track_index > self._manual_track_count():
            self.manual_track_count_spin.setValue(track_index)

    def _manual_track_count_changed(self, _value: int) -> None:
        self._update_manual_track_roi_visibility()
        self._mark_result_stale()
        self._update_oriented_roi_status()
        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()

    def _set_manual_track_count_for_target_key(self, target_key: str) -> None:
        track_index = _track_index_from_target_key(target_key)
        if track_index is not None:
            self._set_manual_track_count_at_least(track_index)

    def _track_target_is_visible(self, target_key: str) -> bool:
        track_index = _track_index_from_target_key(target_key)
        return track_index is None or track_index <= self._manual_track_count()

    def _build_workflow_page(
        self,
        title: str,
        subtitle: str,
        widgets: list[QtWidgets.QWidget],
    ) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("WorkflowScroll")
        scroll.setMinimumWidth(560)
        scroll.setMaximumWidth(760)

        container = QtWidgets.QWidget()
        container.setObjectName("WorkflowPage")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("WindowHeader")
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setObjectName("PreviewSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_sidebar(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("SidebarScroll")
        scroll.setMinimumWidth(450)
        scroll.setMaximumWidth(510)

        container = QtWidgets.QWidget()
        container.setObjectName("SidebarContainer")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        header = QtWidgets.QLabel("轨道曲线提取")
        header.setObjectName("WindowHeader")
        product = QtWidgets.QLabel("Rail Curve Extractor")
        product.setObjectName("EyebrowLabel")
        intro = QtWidgets.QLabel(
            "导入点云、调整参数并预览中心线结果。点击带下划线的参数名可查看说明。"
        )
        intro.setObjectName("IntroLabel")
        intro.setWordWrap(True)

        layout.addWidget(product)
        layout.addWidget(header)
        layout.addWidget(intro)
        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_parameter_group())
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_track_roi_group())
        layout.addWidget(self._build_turnout_roi_group())
        layout.addWidget(self._build_action_group())
        layout.addWidget(self._build_log_group(), stretch=1)

        scroll.setWidget(container)
        return scroll

    def _build_preview_area(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("导出前预览")
        title.setObjectName("WindowHeader")
        subtitle = QtWidgets.QLabel("检查俯视轨迹、纵断面连续性和关键统计，再导出 USD Curve。")
        subtitle.setObjectName("PreviewSubtitle")
        subtitle.setWordWrap(True)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.setSpacing(12)
        for key, label in (
            ("input_points", "输入点数"),
            ("filtered_points", "预览候选点"),
            ("centerline_points", "中心线点数"),
            ("curve_length_m", "曲线长度 (m)"),
        ):
            card = MetricCard(label)
            self.metric_cards[key] = card
            metric_row.addWidget(card, stretch=1)

        summary_frame = QtWidgets.QFrame()
        summary_frame.setObjectName("CardFrame")
        summary_layout = QtWidgets.QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(18, 16, 18, 16)
        summary_layout.setSpacing(10)
        summary_title = QtWidgets.QLabel("摘要")
        summary_title.setObjectName("CardTitle")
        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.summary_label)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(metric_row)
        layout.addWidget(self.top_panel, stretch=3)
        layout.addWidget(self.profile_panel, stretch=2)
        layout.addWidget(summary_frame)
        layout.addWidget(self.status_label)
        return widget

    def _build_oriented_roi_toolbar(self) -> QtWidgets.QWidget:
        frame = QtWidgets.QFrame()
        frame.setObjectName("RoiToolFrame")
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        target_label = QtWidgets.QLabel("框选目标")
        target_label.setObjectName("SmallFieldLabel")
        layout.addWidget(target_label, 0, 0)
        layout.addWidget(self.oriented_roi_target_combo, 0, 1)
        layout.addWidget(self.pick_oriented_roi_button, 0, 2)
        layout.addWidget(self.clear_oriented_roi_button, 0, 3, 1, 2)
        layout.addWidget(self.auto_track_split_check, 1, 0, 1, 3)
        count_label = QtWidgets.QLabel("框内轨道数量")
        count_label.setObjectName("SmallFieldLabel")
        layout.addWidget(count_label, 1, 3)
        layout.addWidget(self.auto_track_count_spin, 1, 4)
        layout.addWidget(self.oriented_roi_status_label, 2, 0, 1, 5)
        anchor_label = QtWidgets.QLabel("锚点目标")
        anchor_label.setObjectName("SmallFieldLabel")
        layout.addWidget(anchor_label, 3, 0)
        layout.addWidget(self.anchor_target_combo, 3, 1)
        layout.addWidget(self.pick_anchor_button, 3, 2)
        layout.addWidget(self.clear_anchor_button, 3, 3, 1, 2)
        layout.addWidget(self.anchor_status_label, 4, 0, 1, 5)

        return frame

    def _build_file_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("输入输出")
        layout.addLayout(
            self._path_row(
                "点云文件",
                self.input_edit,
                self._browse_input,
                "input_path",
            )
        )
        layout.addLayout(
            self._path_row(
                "输出目录",
                self.output_edit,
                self._browse_output,
                "output_path",
            )
        )
        layout.addLayout(
            self._path_row(
                "配置文件",
                self.config_edit,
                self._browse_config,
                "config_path",
            )
        )

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)
        load_button = QtWidgets.QPushButton("加载配置")
        load_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        load_button.clicked.connect(self._load_config_file)
        save_button = QtWidgets.QPushButton("另存参数")
        save_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        save_button.clicked.connect(self._save_config_file)
        button_row.addWidget(load_button)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)
        return group

    def _build_overview_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("彩色底图")
        hint = QtWidgets.QLabel("加载点云底图后，右侧俯视图会显示可框选的点云；如果 LAS/LAZ 带 RGB，会自动显示彩色底图。")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        self.load_overview_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView))
        layout.addWidget(hint)
        layout.addWidget(self.load_overview_button)
        return group

    def _build_parameter_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("常用参数", grid=True)
        assert isinstance(layout, QtWidgets.QGridLayout)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        checkbox_row = QtWidgets.QHBoxLayout()
        checkbox_row.setSpacing(8)
        info_button = QtWidgets.QPushButton("查看“高度过滤”说明")
        info_button.setObjectName("SecondaryLinkButton")
        info_button.clicked.connect(
            lambda: self._show_info_popup(
                title="高度过滤",
                field_key="height_filter.enabled",
                description="启用后会先按整体高程做一次全局过滤。对于普通点云，这一步可以快速压掉大量低位杂点；"
                "对于 LAS/LAZ，当前版本还会额外启用更强的局部地面归一化和主走廊识别分支。",
                default_hint="开启",
                recommended_range="普通点云建议开启，已裁得很干净的数据可视情况关闭",
                effect="关闭后会保留更多低位点，速度可能更慢，噪声也会更多；开启后更适合快速把轨顶从复杂环境里抠出来。",
                anchor_widget=info_button,
                active_path=None,
            )
        )

        self.height_filter_checkbox.stateChanged.connect(self._mark_result_stale)
        checkbox_row.addWidget(self.height_filter_checkbox)
        checkbox_row.addStretch(1)
        checkbox_row.addWidget(info_button)

        layout.addLayout(checkbox_row, 0, 0, 1, 2)

        self.xy_constraint_combo.currentIndexChanged.connect(self._mark_result_stale)
        xy_label = QtWidgets.QLabel("XY 平面约束")
        xy_label.setObjectName("SmallFieldLabel")
        layout.addWidget(xy_label, 1, 0)
        layout.addWidget(self.xy_constraint_combo, 1, 1)

        for row, field in enumerate(COMMON_FIELDS, start=2):
            label = self._make_clickable_field_label(field)
            edit = QtWidgets.QLineEdit()
            edit.textEdited.connect(self._mark_result_stale)
            self.field_inputs[field.path] = edit
            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1)

        return group

    def _build_roi_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("全局 ROI 限制（可留空）", grid=True)
        assert isinstance(layout, QtWidgets.QGridLayout)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        for index, field in enumerate(ROI_FIELDS):
            row = index // 2
            col = (index % 2) * 2
            label = self._make_clickable_field_label(field)
            edit = QtWidgets.QLineEdit()
            edit.textEdited.connect(self._mark_result_stale)
            self.field_inputs[field.path] = edit
            layout.addWidget(label, row, col)
            layout.addWidget(edit, row, col + 1)

        return group

    def _build_track_roi_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("多轨道 ROI（半自动）")
        assert isinstance(layout, QtWidgets.QVBoxLayout)
        hint = QtWidgets.QLabel("需要几条独立轨道 ROI，就把数量调到几；建议每条 ROI 只框住一股道。")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setSpacing(10)
        count_label = QtWidgets.QLabel("手动 ROI 数量")
        count_label.setObjectName("SmallFieldLabel")
        control_row.addWidget(count_label)
        control_row.addWidget(self.manual_track_count_spin)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        for track_index in range(1, MAX_UI_TRACKS + 1):
            box = QtWidgets.QGroupBox(f"轨道 {track_index}")
            box.setObjectName("TrackBox")
            box_layout = QtWidgets.QGridLayout(box)
            box_layout.setContentsMargins(12, 10, 12, 12)
            box_layout.setHorizontalSpacing(10)
            box_layout.setVerticalSpacing(8)

            enabled = QtWidgets.QCheckBox("启用")
            enabled.setChecked(True)
            enabled.stateChanged.connect(self._mark_result_stale)
            self.track_enabled_checks[track_index] = enabled
            self.track_roi_boxes[track_index] = box
            box_layout.addWidget(enabled, 0, 0, 1, 2)

            for axis_index, axis in enumerate(TRACK_ROI_AXES):
                row = axis_index // 2 + 1
                col = (axis_index % 2) * 2
                label = QtWidgets.QLabel(axis.replace("_", " ").upper())
                label.setObjectName("SmallFieldLabel")
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText("留空")
                edit.textEdited.connect(self._mark_result_stale)
                self.track_roi_inputs[(track_index, axis)] = edit
                box_layout.addWidget(label, row, col)
                box_layout.addWidget(edit, row, col + 1)

            layout.addWidget(box)
        self._update_manual_track_roi_visibility()
        return group

    def _build_turnout_roi_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("道岔模式 ROI")
        assert isinstance(layout, QtWidgets.QVBoxLayout)
        hint = QtWidgets.QLabel("启用后优先按道岔处理：在该 ROI 内提取主线与分支中心线。建议只框住一个道岔区域。")
        hint.setObjectName("CardCaption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.turnout_enabled_check.stateChanged.connect(self._mark_result_stale)
        layout.addWidget(self.turnout_enabled_check)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for axis_index, axis in enumerate(TRACK_ROI_AXES):
            row = axis_index // 2
            col = (axis_index % 2) * 2
            label = QtWidgets.QLabel(axis.replace("_", " ").upper())
            label.setObjectName("SmallFieldLabel")
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("留空")
            edit.textEdited.connect(self._mark_result_stale)
            self.turnout_roi_inputs[axis] = edit
            grid.addWidget(label, row, col)
            grid.addWidget(edit, row, col + 1)
        layout.addLayout(grid)
        return group

    def _build_action_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("操作")
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)
        self.preview_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.export_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.export_button)

        layout.addLayout(button_row)
        return group

    def _build_log_group(self) -> QtWidgets.QWidget:
        group, layout = self._new_group("运行日志")
        self.log_text.setMinimumHeight(170)
        layout.addWidget(self.log_text)
        return group

    def _new_group(
        self, title: str, grid: bool = False
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout | QtWidgets.QGridLayout]:
        frame = QtWidgets.QFrame()
        frame.setObjectName("CardFrame")
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(12)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("CardTitle")
        outer.addWidget(title_label)

        body = QtWidgets.QWidget()
        if grid:
            layout: QtWidgets.QVBoxLayout | QtWidgets.QGridLayout = QtWidgets.QGridLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            layout = QtWidgets.QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        outer.addWidget(body)
        return frame, layout

    def _path_row(
        self,
        label_text: str,
        edit: QtWidgets.QLineEdit,
        browse_callback: Any,
        info_key: str,
    ) -> QtWidgets.QLayout:
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        label = ClickableLabel(label_text)
        label.setObjectName("FieldLabel")
        label.clicked.connect(lambda: self._show_path_info(info_key, label))
        browse = QtWidgets.QPushButton("浏览")
        browse.clicked.connect(browse_callback)

        edit.textEdited.connect(self._mark_result_stale)

        layout.addWidget(label, 0, 0)
        layout.addWidget(edit, 1, 0)
        layout.addWidget(browse, 1, 1)
        return layout

    def _make_clickable_field_label(self, field: FieldSpec) -> ClickableLabel:
        label = ClickableLabel(field.label)
        label.setObjectName("FieldLabel")
        label.clicked.connect(lambda: self._show_field_spec(field, label))
        self.field_labels[field.path] = label
        return label

    def _show_path_info(self, info_key: str, anchor_widget: QtWidgets.QWidget) -> None:
        info = PATH_FIELD_HELP[info_key]
        self._show_info_popup(
            title=info["title"],
            field_key=info_key,
            description=info["description"],
            default_hint=info["default_hint"],
            recommended_range=info["recommended_range"],
            effect=info["effect"],
            anchor_widget=anchor_widget,
            active_path=None,
        )

    def _show_field_spec(self, field: FieldSpec, anchor_widget: QtWidgets.QWidget) -> None:
        self._show_info_popup(
            title=field.label,
            field_key=field.path,
            description=field.description,
            default_hint=field.default_hint,
            recommended_range=field.recommended_range,
            effect=field.effect,
            anchor_widget=anchor_widget,
            active_path=field.path,
        )

    def _show_info_popup(
        self,
        title: str,
        field_key: str,
        description: str,
        default_hint: str,
        recommended_range: str,
        effect: str,
        anchor_widget: QtWidgets.QWidget,
        active_path: str | None,
    ) -> None:
        self.info_anchor_widget = anchor_widget
        self.info_popup.set_content(
            title=title,
            field_key=field_key,
            description=description,
            default_hint=default_hint,
            recommended_range=recommended_range,
            effect=effect,
        )
        self._set_active_field(active_path)
        self._position_info_popup(anchor_widget)
        self.info_popup.show()
        self.info_popup.raise_()

    def _position_info_popup(self, anchor_widget: QtWidgets.QWidget) -> None:
        self.info_popup.adjustSize()
        anchor_top_left = anchor_widget.mapToGlobal(QtCore.QPoint(0, 0))
        anchor_rect = QtCore.QRect(anchor_top_left, anchor_widget.size())
        popup_size = self.info_popup.sizeHint()
        margin = 10

        screen = QtGui.QGuiApplication.screenAt(anchor_rect.center()) or QtGui.QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1920, 1080)

        x_pos = anchor_rect.right() + margin
        y_pos = anchor_rect.top() - 4

        if x_pos + popup_size.width() > available.right():
            x_pos = anchor_rect.left() - popup_size.width() - margin
        if x_pos < available.left():
            x_pos = max(available.left() + margin, anchor_rect.left())
        if y_pos + popup_size.height() > available.bottom():
            y_pos = max(available.top() + margin, available.bottom() - popup_size.height() - margin)

        self.info_popup.move(x_pos, y_pos)

    def _set_active_field(self, active_path: str | None) -> None:
        self.selected_field_path = active_path
        for path, label in self.field_labels.items():
            label.setProperty("active", path == active_path)
            label.style().unpolish(label)
            label.style().polish(label)
            label.update()

    def _connect_signals(self) -> None:
        self.preview_button.clicked.connect(self._start_analysis)
        self.export_button.clicked.connect(self._export_result)
        self.load_overview_button.clicked.connect(self._start_raw_preview)
        self.pick_oriented_roi_button.toggled.connect(self._toggle_oriented_roi_pick)
        self.clear_oriented_roi_button.clicked.connect(self._clear_current_oriented_roi)
        self.oriented_roi_target_combo.currentIndexChanged.connect(self._oriented_roi_target_changed)
        self.pick_anchor_button.toggled.connect(self._toggle_anchor_pick)
        self.clear_anchor_button.clicked.connect(self._clear_current_anchor_points)
        self.anchor_target_combo.currentIndexChanged.connect(self._anchor_target_changed)
        self.auto_track_split_check.stateChanged.connect(self._mark_result_stale)
        self.auto_track_split_check.stateChanged.connect(lambda _state: self._update_oriented_roi_status())
        self.auto_track_count_spin.valueChanged.connect(self._mark_result_stale)
        self.auto_track_count_spin.valueChanged.connect(lambda _value: self._update_oriented_roi_status())
        self.manual_track_count_spin.valueChanged.connect(self._manual_track_count_changed)
        self.top_panel.plot.scene().sigMouseClicked.connect(self._handle_top_plot_click)
        self.top_panel.plot.plotItem.vb.sigRangeChanged.connect(self._schedule_raw_preview_view_update)
        QtWidgets.QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if self.info_popup.isVisible():
            if event.type() in (QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.MouseButtonPress):
                global_pos = QtGui.QCursor.pos()
                if not self._cursor_inside_popup_context(global_pos):
                    self.info_popup.hide()
                    self._set_active_field(None)
            elif event.type() in (QtCore.QEvent.Type.WindowDeactivate, QtCore.QEvent.Type.Leave):
                self.info_popup.hide()
                self._set_active_field(None)
        return super().eventFilter(watched, event)

    def _cursor_inside_popup_context(self, global_pos: QtCore.QPoint) -> bool:
        popup_rect = self.info_popup.frameGeometry()
        if popup_rect.contains(global_pos):
            return True
        if self.info_anchor_widget is not None:
            anchor_rect = QtCore.QRect(self.info_anchor_widget.mapToGlobal(QtCore.QPoint(0, 0)), self.info_anchor_widget.size())
            if anchor_rect.contains(global_pos):
                return True
        return False

    def _start_raw_preview(self) -> None:
        input_text = self.input_edit.text().strip()
        if not input_text:
            QtWidgets.QMessageBox.warning(self, "缺少点云文件", "请先选择点云文件。")
            return

        input_path = Path(input_text)
        if not input_path.exists():
            QtWidgets.QMessageBox.critical(self, "文件不存在", f"点云文件不存在：\n{input_path}")
            return

        self.load_overview_button.setEnabled(False)
        self.status_label.setText("正在加载点云底图，用于四点 ROI 框选……")
        self._log(f"开始加载点云底图：{input_path}")
        task = RawPreviewRunnable(input_path=input_path)
        task.signals.finished.connect(self._raw_preview_finished)
        task.signals.failed.connect(self._raw_preview_failed)
        self.thread_pool.start(task)

    def _raw_preview_finished(self, payload: object) -> None:
        self.load_overview_button.setEnabled(True)
        payload_dict = payload if isinstance(payload, dict) else {}
        self.raw_preview_points_xy = np.asarray(payload_dict.get("points_xy", np.empty((0, 2))), dtype=float)
        raw_rgb = payload_dict.get("rgb")
        self.raw_preview_rgb = np.asarray(raw_rgb, dtype=np.uint8) if raw_rgb is not None else None
        if self.raw_preview_rgb is not None and len(self.raw_preview_rgb) != len(self.raw_preview_points_xy):
            self.raw_preview_rgb = None
        self.raw_preview_input_points = int(payload_dict.get("input_points", len(self.raw_preview_points_xy)))
        self.raw_preview_lod_levels = self._build_raw_preview_lod_levels()
        self._draw_raw_overview()
        self.status_label.setText("点云底图已加载。放大或平移时会自动提高当前视野点密度。")
        color_mode = "彩色 RGB" if self.raw_preview_rgb is not None else "灰度"
        self._log(
            f"点云底图已加载：原始 {self.raw_preview_input_points:,} 点，缓存 {len(self.raw_preview_points_xy):,} 点，"
            f"当前视野显示 {self.raw_preview_current_display_points:,} 点，模式：{color_mode}。"
        )
        self._set_page(1)

    def _build_raw_preview_lod_levels(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if self.raw_preview_points_xy is None:
            return []

        levels: list[tuple[np.ndarray, np.ndarray | None]] = []
        last_length = 0
        for limit in RAW_PREVIEW_LOD_LIMITS:
            indices = downsample_indices(len(self.raw_preview_points_xy), min(limit, len(self.raw_preview_points_xy)))
            if len(indices) == last_length:
                continue
            if len(indices) == len(self.raw_preview_points_xy):
                level_points = self.raw_preview_points_xy
                level_rgb = self.raw_preview_rgb
            else:
                level_points = self.raw_preview_points_xy[indices]
                level_rgb = self.raw_preview_rgb[indices] if self.raw_preview_rgb is not None else None
            levels.append((level_points, level_rgb))
            last_length = len(indices)
        return levels

    def _raw_preview_failed(self, message: str) -> None:
        self.load_overview_button.setEnabled(True)
        self.raw_preview_active = False
        self.status_label.setText("点云底图加载失败。")
        self._log(f"点云底图加载失败：{message}")
        QtWidgets.QMessageBox.critical(self, "点云底图加载失败", message)

    def _draw_raw_overview(self) -> None:
        if self.raw_preview_points_xy is None or len(self.raw_preview_points_xy) == 0:
            self._draw_empty_plots("点云底图为空，请检查输入文件。")
            return

        plot = self.top_panel.plot
        self.raw_preview_active = True
        plot.clear()
        plot.setLabel("bottom", "X")
        plot.setLabel("left", "Y")
        plot.setAspectLocked(True)
        self.raw_preview_scatter_item = pg.ScatterPlotItem(size=2, pen=None)
        plot.addItem(self.raw_preview_scatter_item)
        self.raw_preview_bounds = combine_bounds(self.raw_preview_points_xy)
        plot.setXRange(float(self.raw_preview_bounds.minimum[0]), float(self.raw_preview_bounds.maximum[0]), padding=0.06)
        plot.setYRange(float(self.raw_preview_bounds.minimum[1]), float(self.raw_preview_bounds.maximum[1]), padding=0.06)
        self._refresh_raw_preview_for_current_view()

        self.profile_panel.plot.clear()
        profile_text = pg.TextItem(text="底图已加载。请在俯视图点选四个角，再生成预览。", color=SUBTEXT, anchor=(0.5, 0.5))
        profile_text.setPos(0.0, 0.0)
        self.profile_panel.plot.addItem(profile_text)
        self.profile_panel.plot.setXRange(-1.0, 1.0)
        self.profile_panel.plot.setYRange(-1.0, 1.0)
        self._redraw_oriented_roi_overlay()

    def _schedule_raw_preview_view_update(self, *_args: Any) -> None:
        if not self.raw_preview_active or self.raw_preview_points_xy is None:
            return
        if self.raw_preview_update_timer is None:
            self._refresh_raw_preview_for_current_view()
            return
        self.raw_preview_update_timer.start(RAW_PREVIEW_UPDATE_DELAY_MS)

    def _refresh_raw_preview_for_current_view(self) -> None:
        if not self.raw_preview_active or self.raw_preview_points_xy is None:
            return
        if self.raw_preview_scatter_item is None:
            return

        x_range, y_range = self.top_panel.plot.plotItem.vb.viewRange()
        level_points_xy, level_rgb = self._select_raw_preview_lod(x_range, y_range)
        indices = visible_downsample_indices(
            level_points_xy,
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            limit=RAW_PREVIEW_DISPLAY_LIMIT,
        )
        display_points_xy = level_points_xy[indices]
        display_rgb = level_rgb[indices] if level_rgb is not None else None
        self.raw_preview_current_lod_points = int(len(level_points_xy))
        self.raw_preview_current_display_points = int(len(display_points_xy))
        self.raw_preview_scatter_item.setData(
            display_points_xy[:, 0],
            display_points_xy[:, 1],
            size=2,
            pen=None,
            brush=self._raw_preview_brush(display_rgb),
        )

    def _raw_preview_brush(self, rgb: np.ndarray | None) -> Any:
        if rgb is not None and len(rgb) > 0:
            return _rgb_to_qbrushes(rgb)
        return pg.mkBrush(MUTED)

    def _select_raw_preview_lod(self, x_range: list[float], y_range: list[float]) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.raw_preview_lod_levels:
            empty_points = np.empty((0, 2), dtype=np.float32)
            return empty_points, None
        if self.raw_preview_bounds is None:
            return self.raw_preview_lod_levels[-1]

        full_span = np.maximum(self.raw_preview_bounds.maximum - self.raw_preview_bounds.minimum, 1e-9)
        view_width = abs(float(x_range[1]) - float(x_range[0]))
        view_height = abs(float(y_range[1]) - float(y_range[0]))
        area_ratio = (view_width * view_height) / float(full_span[0] * full_span[1])
        if area_ratio > 0.18:
            return self.raw_preview_lod_levels[0]
        if area_ratio > 0.035 and len(self.raw_preview_lod_levels) >= 2:
            return self.raw_preview_lod_levels[1]
        return self.raw_preview_lod_levels[-1]

    def _toggle_oriented_roi_pick(self, checked: bool) -> None:
        if checked and self.raw_preview_points_xy is None and self.result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "需要俯视底图",
                "请先点击“加载点云底图”，或先生成一次预览后再点选四点 ROI。",
            )
            self.pick_oriented_roi_button.setChecked(False)
            return

        if checked:
            self.pick_anchor_button.setChecked(False)
            target_key = self._current_oriented_roi_key()
            self.oriented_roi_points[target_key] = []
            self.top_panel.plot.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            self.status_label.setText(f"正在点选 {self._current_oriented_roi_label()}：请在俯视图依次点 4 个角。")
            self._log(f"开始点选四点 ROI：{self._current_oriented_roi_label()}")
        else:
            self.top_panel.plot.unsetCursor()
        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()

    def _toggle_anchor_pick(self, checked: bool) -> None:
        if checked and self.raw_preview_points_xy is None and self.result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "需要俯视底图",
                "请先点击“加载点云底图”，或先生成一次预览后再点选中心线锚点。",
            )
            self.pick_anchor_button.setChecked(False)
            return

        if checked:
            self.pick_oriented_roi_button.setChecked(False)
            self.top_panel.plot.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            self.status_label.setText(f"正在点选 {self._current_anchor_label()}：沿目标中心线依次点击，至少 2 个点。")
            self._log(f"开始点选人工锚点：{self._current_anchor_label()}")
        else:
            self.top_panel.plot.unsetCursor()
        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()

    def _handle_top_plot_click(self, mouse_event: object) -> None:
        if not self.pick_oriented_roi_button.isChecked() and not self.pick_anchor_button.isChecked():
            return
        if not hasattr(mouse_event, "button") or mouse_event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        scene_position = mouse_event.scenePos()
        view_box = self.top_panel.plot.plotItem.vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            return

        view_position = view_box.mapSceneToView(scene_position)
        point_xy = (float(view_position.x()), float(view_position.y()))
        if self.pick_anchor_button.isChecked():
            self._append_current_anchor_point(point_xy)
            if hasattr(mouse_event, "accept"):
                mouse_event.accept()
            return

        target_key = self._current_oriented_roi_key()
        points = self.oriented_roi_points.setdefault(target_key, [])
        if len(points) >= 4:
            points.clear()
        points.append(point_xy)
        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()

        if len(points) == 4:
            self._apply_current_oriented_roi()

        if hasattr(mouse_event, "accept"):
            mouse_event.accept()

    def _apply_current_oriented_roi(self) -> None:
        target_key = self._current_oriented_roi_key()
        points = self.oriented_roi_points.get(target_key, [])
        try:
            oriented_roi = build_oriented_roi_from_xy_points(points)
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "四点 ROI 无效", str(exc))
            self._log(f"四点 ROI 无效：{exc}")
            return

        self.oriented_roi_configs[target_key] = oriented_roi
        track_index = _track_index_from_target_key(target_key)
        if track_index is not None:
            self._set_manual_track_count_at_least(track_index)
            if track_index in self.track_enabled_checks:
                self.track_enabled_checks[track_index].setChecked(True)
        elif target_key == "auto_tracks":
            self.auto_track_split_check.setChecked(True)
        elif target_key == "turnout":
            self.turnout_enabled_check.setChecked(True)

        self.pick_oriented_roi_button.setChecked(False)
        self._mark_result_stale(clear_plots=False)
        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()
        self._log(f"已设置 {self._current_oriented_roi_label()}。")

    def _clear_current_oriented_roi(self) -> None:
        target_key = self._current_oriented_roi_key()
        self.oriented_roi_configs.pop(target_key, None)
        self.oriented_roi_points.pop(target_key, None)
        self._mark_result_stale(clear_plots=False)
        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()
        self._log(f"已清除 {self._current_oriented_roi_label()}。")

    def _oriented_roi_target_changed(self) -> None:
        self._set_manual_track_count_for_target_key(self._current_oriented_roi_key())
        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()

    def _anchor_target_changed(self) -> None:
        self._set_manual_track_count_for_target_key(self._current_anchor_key())
        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()

    def _current_oriented_roi_key(self) -> str:
        value = self.oriented_roi_target_combo.currentData()
        return str(value or "global")

    def _current_oriented_roi_label(self) -> str:
        return self.oriented_roi_target_combo.currentText() or "全局 ROI"

    def _current_anchor_key(self) -> str:
        value = self.anchor_target_combo.currentData()
        return str(value or "global")

    def _current_anchor_label(self) -> str:
        return self.anchor_target_combo.currentText() or "中心线锚点"

    def _append_current_anchor_point(self, point_xy: tuple[float, float]) -> None:
        target_key = self._current_anchor_key()
        points = self.anchor_points.setdefault(target_key, [])
        points.append(point_xy)
        track_index = _track_index_from_target_key(target_key)
        if track_index is not None:
            self._set_manual_track_count_at_least(track_index)
            if track_index in self.track_enabled_checks:
                self.track_enabled_checks[track_index].setChecked(True)
        elif target_key.startswith("turnout_"):
            self.turnout_enabled_check.setChecked(True)
        self._mark_result_stale(clear_plots=False)
        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()
        self._log(f"已添加 {self._current_anchor_label()}：第 {len(points)} 个点。")

    def _clear_current_anchor_points(self) -> None:
        target_key = self._current_anchor_key()
        self.anchor_points.pop(target_key, None)
        self._mark_result_stale(clear_plots=False)
        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()
        self._log(f"已清除 {self._current_anchor_label()}。")

    def _update_oriented_roi_status(self) -> None:
        target_key = self._current_oriented_roi_key()
        target_label = self._current_oriented_roi_label()
        points = self.oriented_roi_points.get(target_key, [])
        oriented_roi = self.oriented_roi_configs.get(target_key)
        configured_count = sum(1 for key in self.oriented_roi_configs if self._track_target_is_visible(key))

        if oriented_roi is not None:
            length_m = float(oriented_roi["s_max"]) - float(oriented_roi["s_min"])
            width_m = float(oriented_roi["t_max"]) - float(oriented_roi["t_min"])
            status = f"当前：{target_label} 已设置，长约 {length_m:.2f} m，宽约 {width_m:.2f} m。"
        elif points:
            status = f"当前：{target_label} 已点 {len(points)}/4 个角。"
        else:
            status = f"当前：{target_label} 未设置。"
        if self.auto_track_split_check.isChecked():
            status += f" 自动多轨：一个总 ROI 内按 {self.auto_track_count_spin.value()} 条拆分。"
        if configured_count:
            status += f" 已设置 ROI 数：{configured_count}。"
        self.oriented_roi_status_label.setText(status)

    def _update_anchor_status(self) -> None:
        target_key = self._current_anchor_key()
        target_label = self._current_anchor_label()
        points = self.anchor_points.get(target_key, [])
        configured_count = sum(
            1
            for key, value in self.anchor_points.items()
            if len(value) >= 2 and self._track_target_is_visible(key)
        )
        if points:
            status = f"当前：{target_label} 已点 {len(points)} 个锚点。"
            if len(points) < 2:
                status += " 至少需要 2 个点才会生效。"
        else:
            status = f"当前：{target_label} 未设置。"
        if configured_count:
            status += f" 已生效锚点组：{configured_count}。"
        self.anchor_status_label.setText(status)

    def _redraw_oriented_roi_overlay(self) -> None:
        self._clear_oriented_roi_overlay()
        plot = self.top_panel.plot

        for target_key, oriented_roi in self.oriented_roi_configs.items():
            if not self._track_target_is_visible(target_key):
                continue
            if not _is_oriented_roi_configured(oriented_roi):
                continue
            color = _oriented_roi_color(target_key)
            corners = oriented_roi_corners(oriented_roi)
            curve_item = pg.PlotCurveItem(corners[:, 0], corners[:, 1], pen=pg.mkPen(color=color, width=3))
            point_item = pg.ScatterPlotItem(corners[:-1, 0], corners[:-1, 1], size=9, pen=pg.mkPen(color=color), brush=pg.mkBrush("#ffffff"))
            label_item = pg.TextItem(text=self._label_for_oriented_roi_key(target_key), color=color, anchor=(0.0, 1.0))
            label_item.setPos(float(corners[0, 0]), float(corners[0, 1]))
            plot.addItem(curve_item)
            plot.addItem(point_item)
            plot.addItem(label_item)
            self.oriented_roi_overlay_items.extend([curve_item, point_item, label_item])

        target_key = self._current_oriented_roi_key()
        partial_points = np.asarray(self.oriented_roi_points.get(target_key, []), dtype=float)
        if partial_points.size > 0:
            color = _oriented_roi_color(target_key)
            partial_item = pg.ScatterPlotItem(
                partial_points[:, 0],
                partial_points[:, 1],
                size=11,
                pen=pg.mkPen(color=color),
                brush=pg.mkBrush(color),
            )
            plot.addItem(partial_item)
            self.oriented_roi_overlay_items.append(partial_item)
            if len(partial_points) >= 2:
                line_item = pg.PlotCurveItem(partial_points[:, 0], partial_points[:, 1], pen=pg.mkPen(color=color, width=2))
                plot.addItem(line_item)
                self.oriented_roi_overlay_items.append(line_item)

        self._draw_anchor_overlays()

    def _draw_anchor_overlays(self) -> None:
        plot = self.top_panel.plot
        current_key = self._current_anchor_key()
        for target_key, points in self.anchor_points.items():
            if not self._track_target_is_visible(target_key):
                continue
            if not points:
                continue
            color = _anchor_color(target_key)
            points_array = np.asarray(points, dtype=float)
            point_item = pg.ScatterPlotItem(
                points_array[:, 0],
                points_array[:, 1],
                size=12 if target_key == current_key else 9,
                pen=pg.mkPen(color=color, width=2),
                brush=pg.mkBrush("#ffffff"),
            )
            plot.addItem(point_item)
            self.oriented_roi_overlay_items.append(point_item)
            if len(points_array) >= 2:
                line_item = pg.PlotCurveItem(
                    points_array[:, 0],
                    points_array[:, 1],
                    pen=pg.mkPen(color=color, width=3, style=QtCore.Qt.PenStyle.DashLine),
                )
                plot.addItem(line_item)
                self.oriented_roi_overlay_items.append(line_item)
            label_item = pg.TextItem(text=self._label_for_anchor_key(target_key), color=color, anchor=(0.0, 1.0))
            label_item.setPos(float(points_array[0, 0]), float(points_array[0, 1]))
            plot.addItem(label_item)
            self.oriented_roi_overlay_items.append(label_item)

    def _clear_oriented_roi_overlay(self) -> None:
        plot = self.top_panel.plot
        for item in self.oriented_roi_overlay_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self.oriented_roi_overlay_items.clear()

    def _label_for_oriented_roi_key(self, target_key: str) -> str:
        for option_key, option_label in ORIENTED_ROI_TARGETS:
            if option_key == target_key:
                return option_label
        return target_key

    def _label_for_anchor_key(self, target_key: str) -> str:
        for option_key, option_label in ANCHOR_TARGETS:
            if option_key == target_key:
                return option_label
        return target_key

    def _browse_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择点云文件",
            str(Path.home()),
            "Point Clouds (*.las *.laz *.csv *.txt *.xyz *.npy);;All Files (*.*)",
        )
        if path:
            self.input_edit.setText(path)
            self.raw_preview_points_xy = None
            self.raw_preview_rgb = None
            self.raw_preview_input_points = 0
            self.raw_preview_current_display_points = 0
            self.raw_preview_current_lod_points = 0
            self.raw_preview_bounds = None
            self.raw_preview_lod_levels = []
            self.raw_preview_scatter_item = None
            self.raw_preview_active = False
            self._mark_result_stale()

    def _browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text().strip())
        if path:
            self.output_edit.setText(path)

    def _browse_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if path:
            self.config_edit.setText(path)

    def _load_config_file(self) -> None:
        path_text = self.config_edit.text().strip()
        if not path_text:
            QtWidgets.QMessageBox.warning(self, "缺少配置文件", "请先选择一个 JSON 配置文件。")
            return

        path = Path(path_text)
        if not path.exists():
            QtWidgets.QMessageBox.critical(self, "文件不存在", f"配置文件不存在：\n{path}")
            return

        with path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        self._apply_config_to_form(prepare_config(overrides=loaded))
        self._log(f"已加载配置：{path}")
        self._mark_result_stale()

    def _save_config_file(self) -> None:
        try:
            config = self._collect_config_from_form()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "参数错误", str(exc))
            return

        default_path = bundled_path("data", "config.example.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存当前参数",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not path:
            return

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file_handle:
            json.dump(config, file_handle, ensure_ascii=False, indent=2)
        self.config_edit.setText(str(target))
        self._log(f"已保存参数：{target}")

    def _apply_config_to_form(self, config: dict[str, Any]) -> None:
        self.height_filter_checkbox.setChecked(bool(config["height_filter"]["enabled"]))
        xy_mode = str(config.get("xy_constraint", {}).get("mode", "free")).lower()
        xy_index = self.xy_constraint_combo.findData(xy_mode)
        self.xy_constraint_combo.setCurrentIndex(xy_index if xy_index >= 0 else 0)
        auto_track_split = config.get("auto_track_split", {})
        if isinstance(auto_track_split, dict):
            self.auto_track_split_check.setChecked(bool(auto_track_split.get("enabled", False)))
            try:
                auto_count = int(auto_track_split.get("count", 3))
            except (TypeError, ValueError):
                auto_count = 3
            self.auto_track_count_spin.setValue(max(1, min(MAX_UI_TRACKS, auto_count)))
        for field in COMMON_FIELDS + ROI_FIELDS:
            value = _get_value_from_path(config, field.path)
            self.field_inputs[field.path].setText("" if value is None else str(value))

        tracks = config.get("tracks") or []
        track_configs_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(tracks, list):
            for fallback_index, track_config in enumerate(tracks, start=1):
                if not isinstance(track_config, dict):
                    continue
                try:
                    track_index = int(track_config.get("id") or fallback_index)
                except (TypeError, ValueError):
                    track_index = fallback_index
                if 1 <= track_index <= MAX_UI_TRACKS:
                    track_configs_by_id[track_index] = track_config
        manual_count = max([DEFAULT_MANUAL_TRACK_COUNT, *track_configs_by_id.keys()])
        self.manual_track_count_spin.setValue(min(MAX_UI_TRACKS, manual_count))
        self._update_manual_track_roi_visibility()
        for track_index in range(1, MAX_UI_TRACKS + 1):
            track_config = track_configs_by_id.get(track_index, {})
            self.track_enabled_checks[track_index].setChecked(bool(track_config.get("enabled", track_index <= self._manual_track_count())))
            roi = track_config.get("roi", {}) if isinstance(track_config, dict) else {}
            for axis in TRACK_ROI_AXES:
                value = roi.get(axis)
                self.track_roi_inputs[(track_index, axis)].setText("" if value is None else str(value))

        turnout = config.get("turnout", {})
        self.turnout_enabled_check.setChecked(bool(turnout.get("enabled", False)))
        turnout_roi = turnout.get("roi", {}) if isinstance(turnout, dict) else {}
        for axis in TRACK_ROI_AXES:
            value = turnout_roi.get(axis)
            self.turnout_roi_inputs[axis].setText("" if value is None else str(value))
        self._apply_oriented_roi_configs_to_form(config)
        self._apply_anchor_points_to_form(config)

    def _collect_config_from_form(self) -> dict[str, Any]:
        config = prepare_config()
        config["height_filter"]["enabled"] = bool(self.height_filter_checkbox.isChecked())
        config["xy_constraint"]["mode"] = str(self.xy_constraint_combo.currentData() or "free")

        for field in COMMON_FIELDS + ROI_FIELDS:
            raw_value = self.field_inputs[field.path].text().strip()
            if not raw_value:
                if field.allow_blank:
                    value = None
                else:
                    raise ValueError(f"参数“{field.label}”不能为空。")
            else:
                try:
                    value = field.caster(raw_value)
                except ValueError as exc:
                    raise ValueError(f"参数“{field.label}”格式不正确：{raw_value}") from exc
            _set_value_on_path(config, field.path, value)

        global_anchor_points = self.anchor_points.get("global", [])
        config["manual_anchor"]["enabled"] = len(global_anchor_points) >= 2
        config["manual_anchor"]["points"] = _anchor_points_to_config(global_anchor_points)

        global_oriented_roi = self.oriented_roi_configs.get("global")
        if global_oriented_roi is not None:
            config["oriented_roi"] = dict(global_oriented_roi)

        auto_track_roi = self.oriented_roi_configs.get("auto_tracks")
        config["auto_track_split"]["enabled"] = bool(self.auto_track_split_check.isChecked())
        config["auto_track_split"]["count"] = int(self.auto_track_count_spin.value())
        if auto_track_roi is not None:
            config["auto_track_split"]["oriented_roi"] = dict(auto_track_roi)
        elif config["auto_track_split"]["enabled"] and global_oriented_roi is not None:
            config["auto_track_split"]["oriented_roi"] = dict(global_oriented_roi)
        if config["auto_track_split"]["enabled"] and not _is_oriented_roi_configured(config["auto_track_split"].get("oriented_roi")):
            raise ValueError("启用自动多轨拆分时，请先点选“多轨总 ROI（自动拆分）”。")

        tracks: list[dict[str, Any]] = []
        for track_index in range(1, self._manual_track_count() + 1):
            roi: dict[str, float | None] = {}
            has_roi_value = False
            track_oriented_roi = self.oriented_roi_configs.get(f"track_{track_index}")
            for axis in TRACK_ROI_AXES:
                raw_value = self.track_roi_inputs[(track_index, axis)].text().strip()
                if raw_value:
                    try:
                        roi[axis] = float(raw_value)
                    except ValueError as exc:
                        raise ValueError(f"轨道 {track_index} 的 {axis} 格式不正确：{raw_value}") from exc
                    has_roi_value = True
                else:
                    roi[axis] = None
            if has_roi_value or track_oriented_roi is not None:
                track_config: dict[str, Any] = {
                    "id": track_index,
                    "enabled": bool(self.track_enabled_checks[track_index].isChecked()),
                    "roi": roi,
                }
                if track_oriented_roi is not None:
                    track_config["oriented_roi"] = dict(track_oriented_roi)
                track_anchor_points = self.anchor_points.get(f"track_{track_index}", [])
                if len(track_anchor_points) >= 2:
                    track_config["manual_anchor"] = {
                        "enabled": True,
                        "points": _anchor_points_to_config(track_anchor_points),
                    }
                tracks.append(track_config)
        if config["auto_track_split"]["enabled"]:
            tracks = []
        config["tracks"] = tracks
        turnout_roi, has_turnout_roi = self._collect_axis_roi(self.turnout_roi_inputs, "道岔")
        turnout_oriented_roi = self.oriented_roi_configs.get("turnout")
        config["turnout"] = {
            "enabled": bool(self.turnout_enabled_check.isChecked()) and (has_turnout_roi or turnout_oriented_roi is not None),
            "roi": turnout_roi,
            "branch_min_separation": float(config.get("turnout", {}).get("branch_min_separation", 0.45)),
        }
        if turnout_oriented_roi is not None:
            config["turnout"]["oriented_roi"] = dict(turnout_oriented_roi)
        turnout_main_anchor = self.anchor_points.get("turnout_main", [])
        turnout_branch_anchor = self.anchor_points.get("turnout_branch", [])
        if len(turnout_main_anchor) >= 2:
            config["turnout"]["main_anchor_points"] = _anchor_points_to_config(turnout_main_anchor)
        if len(turnout_branch_anchor) >= 2:
            config["turnout"]["branch_anchor_points"] = _anchor_points_to_config(turnout_branch_anchor)

        return config

    def _apply_oriented_roi_configs_to_form(self, config: dict[str, Any]) -> None:
        self.oriented_roi_configs.clear()
        self.oriented_roi_points.clear()

        global_oriented_roi = config.get("oriented_roi")
        if _is_oriented_roi_configured(global_oriented_roi):
            self.oriented_roi_configs["global"] = dict(global_oriented_roi)
            self.oriented_roi_points["global"] = self._points_from_oriented_roi(global_oriented_roi)

        auto_track_split = config.get("auto_track_split") or {}
        if isinstance(auto_track_split, dict):
            auto_oriented_roi = auto_track_split.get("oriented_roi")
            if _is_oriented_roi_configured(auto_oriented_roi):
                self.oriented_roi_configs["auto_tracks"] = dict(auto_oriented_roi)
                self.oriented_roi_points["auto_tracks"] = self._points_from_oriented_roi(auto_oriented_roi)

        tracks = config.get("tracks") or []
        if isinstance(tracks, list):
            for fallback_index, track_config in enumerate(tracks, start=1):
                if not isinstance(track_config, dict):
                    continue
                try:
                    track_index = int(track_config.get("id") or fallback_index)
                except (TypeError, ValueError):
                    track_index = fallback_index
                oriented_roi = track_config.get("oriented_roi")
                if _is_oriented_roi_configured(oriented_roi):
                    target_key = f"track_{track_index}"
                    self.oriented_roi_configs[target_key] = dict(oriented_roi)
                    self.oriented_roi_points[target_key] = self._points_from_oriented_roi(oriented_roi)

        turnout = config.get("turnout") or {}
        if isinstance(turnout, dict):
            turnout_oriented_roi = turnout.get("oriented_roi")
            if _is_oriented_roi_configured(turnout_oriented_roi):
                self.oriented_roi_configs["turnout"] = dict(turnout_oriented_roi)
                self.oriented_roi_points["turnout"] = self._points_from_oriented_roi(turnout_oriented_roi)

        self._update_oriented_roi_status()
        self._redraw_oriented_roi_overlay()

    def _apply_anchor_points_to_form(self, config: dict[str, Any]) -> None:
        self.anchor_points.clear()

        manual_anchor = config.get("manual_anchor") or {}
        if isinstance(manual_anchor, dict) and bool(manual_anchor.get("enabled", False)):
            points = self._points_from_anchor_config(manual_anchor.get("points", []))
            if len(points) >= 2:
                self.anchor_points["global"] = points

        tracks = config.get("tracks") or []
        if isinstance(tracks, list):
            for fallback_index, track_config in enumerate(tracks, start=1):
                if not isinstance(track_config, dict):
                    continue
                try:
                    track_index = int(track_config.get("id") or fallback_index)
                except (TypeError, ValueError):
                    track_index = fallback_index
                track_anchor = track_config.get("manual_anchor") or {}
                if isinstance(track_anchor, dict) and bool(track_anchor.get("enabled", False)):
                    points = self._points_from_anchor_config(track_anchor.get("points", []))
                    if len(points) >= 2:
                        self.anchor_points[f"track_{track_index}"] = points

        turnout = config.get("turnout") or {}
        if isinstance(turnout, dict):
            main_points = self._points_from_anchor_config(turnout.get("main_anchor_points", []))
            branch_points = self._points_from_anchor_config(turnout.get("branch_anchor_points", []))
            if len(main_points) >= 2:
                self.anchor_points["turnout_main"] = main_points
            if len(branch_points) >= 2:
                self.anchor_points["turnout_branch"] = branch_points

        self._update_anchor_status()
        self._redraw_oriented_roi_overlay()

    def _points_from_oriented_roi(self, oriented_roi: dict[str, Any]) -> list[tuple[float, float]]:
        corners = oriented_roi_corners(oriented_roi)[:-1]
        return [(float(point[0]), float(point[1])) for point in corners]

    def _points_from_anchor_config(self, raw_points: Any) -> list[tuple[float, float]]:
        if not isinstance(raw_points, list):
            return []
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if isinstance(point, dict):
                x_value = point.get("x")
                y_value = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x_value = point[0]
                y_value = point[1]
            else:
                continue
            try:
                points.append((float(x_value), float(y_value)))
            except (TypeError, ValueError):
                continue
        return points

    def _collect_axis_roi(
        self,
        inputs: dict[str, QtWidgets.QLineEdit],
        label_prefix: str,
    ) -> tuple[dict[str, float | None], bool]:
        roi: dict[str, float | None] = {}
        has_value = False
        for axis in TRACK_ROI_AXES:
            raw_value = inputs[axis].text().strip()
            if raw_value:
                try:
                    roi[axis] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{label_prefix}的 {axis} 格式不正确：{raw_value}") from exc
                has_value = True
            else:
                roi[axis] = None
        return roi, has_value

    def _start_analysis(self) -> None:
        input_path_text = self.input_edit.text().strip()
        if not input_path_text:
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请先选择一个点云文件。")
            return

        input_path = Path(input_path_text)
        if not input_path.exists():
            QtWidgets.QMessageBox.critical(self, "文件不存在", f"点云文件不存在：\n{input_path}")
            return

        try:
            config = self._collect_config_from_form()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "参数错误", str(exc))
            return

        self.result = None
        self.preview_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.status_label.setText("正在分析点云并生成预览…")
        self.summary_label.setText("正在计算预览，请稍候。")
        self._set_metric_defaults()
        self._log(f"开始分析：{input_path}")

        task = AnalysisRunnable(input_path=input_path, config=config)
        task.signals.finished.connect(self._analysis_finished)
        task.signals.failed.connect(self._analysis_failed)
        self.thread_pool.start(task)

    def _analysis_finished(self, result: PipelineResult) -> None:
        self.result = result
        self.preview_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_label.setText("预览已生成，可以检查后再导出。")
        self._update_summary()
        self._update_top_plot()
        self._update_profile_plot()
        self._log("分析完成，已生成预览。")
        self._set_page(3)

    def _analysis_failed(self, message: str) -> None:
        self.preview_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.status_label.setText("分析失败。")
        self.summary_label.setText(message)
        self._draw_empty_plots("分析失败，请检查输入和参数。")
        self._log(f"分析失败：{message}")
        QtWidgets.QMessageBox.critical(self, "分析失败", message)

    def _export_result(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "没有可导出的结果", "请先生成预览。")
            return

        output_text = self.output_edit.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "缺少输出目录", "请先设置输出目录。")
            return

        output_dir = Path(output_text)
        input_path = Path(self.input_edit.text().strip()) if self.input_edit.text().strip() else None
        config_path = Path(self.config_edit.text().strip()) if self.config_edit.text().strip() else None

        try:
            self.result.config = self._collect_config_from_form()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "参数错误", str(exc))
            return

        export_pipeline_result(
            result=self.result,
            output_dir=output_dir,
            input_path=input_path,
            config_path=config_path,
        )
        self.status_label.setText("导出完成。")
        self._update_summary()
        self._log(f"已导出到：{output_dir}")
        QtWidgets.QMessageBox.information(self, "导出完成", f"已导出到：\n{output_dir}")

    def _mark_result_stale(self, *args: Any, clear_plots: bool = True) -> None:
        self.result = None
        self.export_button.setEnabled(False)
        self.status_label.setText("参数或输入已变化，请重新生成预览。")
        self.summary_label.setText("参数或输入已变化，请重新生成预览。")
        self.info_popup.hide()
        self._set_active_field(None)
        self._set_metric_defaults()
        if clear_plots:
            self._draw_empty_plots()

    def _set_metric_defaults(self) -> None:
        for card in self.metric_cards.values():
            card.set_value("--")

    def _update_summary(self) -> None:
        if self.result is None:
            self.summary_label.setText("尚未生成预览。")
            return

        summary = self.result.summary
        self.metric_cards["input_points"].set_value(f"{summary['input_points']:,}")
        self.metric_cards["filtered_points"].set_value(f"{summary['filtered_points']:,}")
        self.metric_cards["centerline_points"].set_value(f"{summary['centerline_points']:,}")
        self.metric_cards["curve_length_m"].set_value(f"{summary['curve_length_m']:.2f}")
        if summary.get("turnout_count", 0) > 0:
            turnout = summary.get("turnouts", [{}])[0]
            text = (
                "道岔模式：主线 {main_centerline_points} 点，{main_curve_length_m:.2f} m | "
                "分支 {branch_centerline_points} 点，{branch_curve_length_m:.2f} m | "
                "置信度 {confidence:.2f} | 轨道候选：{rail_points:,}"
            ).format(**turnout)
        elif summary.get("track_count", 1) > 1:
            track_text = "；".join(
                f"轨道 {item['track_id']}：{item['centerline_points']} 点，{item['curve_length_m']:.2f} m，置信度 {item['confidence']:.2f}"
                for item in summary.get("tracks", [])
            )
            text = (
                "多轨道模式：成功 {track_count} 条，失败 {failed_track_count} 条 | "
                "输入点数：{input_points:,} | 轨道候选：{rail_points:,} | 总中心线点数：{centerline_points:,} | "
                "总长度：{curve_length_m:.2f} m\n{track_text}"
            ).format(track_text=track_text, **summary)
        elif "working_points" in summary:
            text = (
                "输入点数：{input_points:,} | 预览候选：{filtered_points:,} | 内部工作点：{working_points:,} | "
                "轨道候选：{rail_points:,} | 中心线点数：{centerline_points:,} | 曲线长度：{curve_length_m:.2f} m"
            ).format(**summary)
        else:
            text = (
                "输入点数：{input_points:,} | 过滤后：{filtered_points:,} | 轨道候选：{rail_points:,} | "
                "中心线点数：{centerline_points:,} | 曲线长度：{curve_length_m:.2f} m"
            ).format(**summary)
        self.summary_label.setText(text)

    def _draw_empty_plots(self, message: str = "生成预览后会显示在这里。") -> None:
        self.raw_preview_active = False
        self.raw_preview_scatter_item = None
        self.raw_preview_current_display_points = 0
        self.raw_preview_current_lod_points = 0
        self.top_panel.plot.clear()
        self.profile_panel.plot.clear()

        top_text = pg.TextItem(text=message, color=SUBTEXT, anchor=(0.5, 0.5))
        top_text.setPos(0.0, 0.0)
        self.top_panel.plot.addItem(top_text)
        self.top_panel.plot.setXRange(-1.0, 1.0)
        self.top_panel.plot.setYRange(-1.0, 1.0)

        profile_text = pg.TextItem(text=message, color=SUBTEXT, anchor=(0.5, 0.5))
        profile_text.setPos(0.0, 0.0)
        self.profile_panel.plot.addItem(profile_text)
        self.profile_panel.plot.setXRange(-1.0, 1.0)
        self.profile_panel.plot.setYRange(-1.0, 1.0)

    def _update_top_plot(self) -> None:
        self.raw_preview_active = False
        self.raw_preview_scatter_item = None
        if self.result is None:
            self._draw_empty_plots()
            return

        plot = self.top_panel.plot
        plot.clear()
        plot.setLabel("bottom", "X")
        plot.setLabel("left", "Y")
        plot.setAspectLocked(True)

        if self.result.track_results and self.result.summary.get("track_count", 1) > 1:
            all_bounds: list[np.ndarray] = []
            for index, track in enumerate(self.result.track_results):
                color = TRACK_COLORS[index % len(TRACK_COLORS)]
                filtered_xy = downsample_points(track.filtered_points_world[:, :2], limit=2500)
                rail_xy = downsample_points(track.rail_points_world[:, :2], limit=1800)
                center_xy = track.centerline_world[:, :2]
                all_bounds.extend([filtered_xy, rail_xy, center_xy])
                plot.addItem(pg.ScatterPlotItem(filtered_xy[:, 0], filtered_xy[:, 1], size=3, pen=None, brush=pg.mkBrush(MUTED)))
                plot.addItem(pg.ScatterPlotItem(rail_xy[:, 0], rail_xy[:, 1], size=5, pen=None, brush=pg.mkBrush(color)))
                plot.addItem(pg.PlotCurveItem(center_xy[:, 0], center_xy[:, 1], pen=pg.mkPen(color=color, width=3)))
            bounds = combine_bounds(*all_bounds)
            plot.setXRange(float(bounds.minimum[0]), float(bounds.maximum[0]), padding=0.08)
            plot.setYRange(float(bounds.minimum[1]), float(bounds.maximum[1]), padding=0.08)
            self._redraw_oriented_roi_overlay()
            return

        filtered_xy = downsample_points(self.result.filtered_points_world[:, :2], limit=6000)
        rail_xy = downsample_points(self.result.rail_points_world[:, :2], limit=4200)
        center_xy = self.result.centerline_world[:, :2]
        bounds = combine_bounds(filtered_xy, rail_xy, center_xy)

        plot.addItem(
            pg.ScatterPlotItem(
                filtered_xy[:, 0],
                filtered_xy[:, 1],
                size=4,
                pen=None,
                brush=pg.mkBrush(MUTED),
            )
        )
        plot.addItem(
            pg.ScatterPlotItem(
                rail_xy[:, 0],
                rail_xy[:, 1],
                size=5,
                pen=None,
                brush=pg.mkBrush(RAIL),
            )
        )
        plot.addItem(
            pg.PlotCurveItem(
                center_xy[:, 0],
                center_xy[:, 1],
                pen=pg.mkPen(color=ACCENT, width=3),
            )
        )

        plot.setXRange(float(bounds.minimum[0]), float(bounds.maximum[0]), padding=0.08)
        plot.setYRange(float(bounds.minimum[1]), float(bounds.maximum[1]), padding=0.08)
        self._redraw_oriented_roi_overlay()

    def _update_profile_plot(self) -> None:
        if self.result is None:
            self._draw_empty_plots()
            return

        profile = build_profile_points(self.result.centerline_world)
        bounds = combine_bounds(profile)
        plot = self.profile_panel.plot
        plot.clear()
        plot.setLabel("bottom", "Arc Length")
        plot.setLabel("left", "Z")
        plot.setAspectLocked(False)

        if self.result.track_results and self.result.summary.get("track_count", 1) > 1:
            profiles: list[np.ndarray] = []
            for index, track in enumerate(self.result.track_results):
                profile = build_profile_points(track.centerline_world)
                profiles.append(profile)
                color = TRACK_COLORS[index % len(TRACK_COLORS)]
                plot.addItem(pg.PlotCurveItem(profile[:, 0], profile[:, 1], pen=pg.mkPen(color=color, width=3)))
            bounds = combine_bounds(*profiles)
            plot.setXRange(float(bounds.minimum[0]), float(bounds.maximum[0]), padding=0.08)
            plot.setYRange(float(bounds.minimum[1]), float(bounds.maximum[1]), padding=0.12)
            return

        plot.addItem(
            pg.PlotCurveItem(
                profile[:, 0],
                profile[:, 1],
                pen=pg.mkPen(color=ACCENT, width=3),
            )
        )
        plot.setXRange(float(bounds.minimum[0]), float(bounds.maximum[0]), padding=0.08)
        plot.setYRange(float(bounds.minimum[1]), float(bounds.maximum[1]), padding=0.12)

    def _log(self, message: str) -> None:
        self.log_text.appendPlainText(message)


def _build_stylesheet() -> str:
    return f"""
    QMainWindow {{
        background: {WINDOW_BG};
        color: {TEXT};
        font-family: "{FONT_FAMILY}", "Segoe UI", sans-serif;
        font-size: 13px;
    }}
    QWidget {{
        font-family: "{FONT_FAMILY}", "Segoe UI", sans-serif;
        color: {TEXT};
    }}
    QLabel#EyebrowLabel {{
        color: {ACCENT};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    QLabel#NavTitle {{
        font-size: 20px;
        font-weight: 800;
        color: {TEXT};
    }}
    QLabel#NavHint {{
        font-size: 12px;
        color: {SUBTEXT};
        background: #f8fbfc;
        border: 1px solid #d8e2ea;
        border-radius: 12px;
        padding: 10px;
    }}
    QLabel#WindowHeader {{
        font-size: 27px;
        font-weight: 800;
        color: {TEXT};
    }}
    QLabel#IntroLabel, QLabel#PreviewSubtitle, QLabel#CardCaption, QLabel#SummaryLabel, QLabel#StatusLabel {{
        font-size: 13px;
        color: {SUBTEXT};
    }}
    QScrollArea#SidebarScroll {{
        border: none;
        background: {SIDEBAR_BG};
        border-radius: 18px;
    }}
    QWidget#SidebarContainer {{
        background: {SIDEBAR_BG};
    }}
    QFrame#NavSidebar {{
        background: {SIDEBAR_BG};
        border: 1px solid #d8e2ea;
        border-radius: 18px;
    }}
    QScrollArea#WorkflowScroll {{
        border: none;
        background: transparent;
    }}
    QWidget#WorkflowPage {{
        background: transparent;
    }}
    QStackedWidget#PageStack {{
        background: transparent;
    }}
    QFrame#CardFrame, QFrame#MetricCard {{
        background: {CARD_BG};
        border: 1px solid #d8e2ea;
        border-radius: 16px;
    }}
    QFrame#MetricCard {{
        background: {CARD_ALT_BG};
        border: 1px solid #dce7ee;
    }}
    QLabel#CardTitle {{
        font-size: 16px;
        font-weight: 800;
        color: {TEXT};
    }}
    QLabel#MetricLabel {{
        font-size: 12px;
        color: {SUBTEXT};
    }}
    QLabel#MetricValue {{
        font-size: 27px;
        font-weight: 800;
        color: {TEXT};
    }}
    QLabel#FieldLabel {{
        color: {ACCENT_DARK};
        font-size: 13px;
        font-weight: 600;
        text-decoration: underline;
        padding: 2px 0;
    }}
    QLabel#FieldLabel:hover {{
        color: {ACCENT};
    }}
    QLabel#FieldLabel[active="true"] {{
        color: white;
        background: {ACCENT};
        border-radius: 8px;
        padding: 4px 8px;
        text-decoration: none;
    }}
    QLabel#SmallFieldLabel {{
        color: {SUBTEXT};
        font-size: 12px;
        font-weight: 700;
    }}
    QGroupBox#TrackBox {{
        background: #fbfdfe;
        border: 1px solid #d8e2ea;
        border-radius: 12px;
        margin-top: 10px;
        padding-top: 8px;
        font-weight: 700;
    }}
    QGroupBox#TrackBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {TEXT};
    }}
    QFrame#InfoPopup {{
        background: #ffffff;
        border: 1px solid #d8e2ea;
        border-radius: 14px;
    }}
    QLabel#PopupTitle {{
        font-size: 16px;
        font-weight: 700;
        color: {TEXT};
    }}
    QLabel#PopupPath {{
        font-size: 12px;
        color: {ACCENT_DARK};
        background: #eef8fa;
        border: 1px solid #cfe8ee;
        border-radius: 8px;
        padding: 4px 8px;
    }}
    QLabel#PopupSectionTitle {{
        font-size: 13px;
        font-weight: 700;
        color: {TEXT};
    }}
    QLabel#PopupBody {{
        font-size: 13px;
        color: {SUBTEXT};
    }}
    QLabel#PopupMetaTag {{
        font-size: 12px;
        font-weight: 700;
        color: {SUBTEXT};
    }}
    QLabel#PopupValue {{
        font-size: 13px;
        color: {TEXT};
        background: #ffffff;
        border: 1px solid #d7ddd7;
        border-radius: 8px;
        padding: 5px 8px;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f8fbfd);
        border: 1px solid #c7d9e5;
        border-radius: 13px;
        padding: 10px 12px;
        color: {TEXT};
        selection-background-color: {ACCENT};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QPlainTextEdit:hover {{
        background: #ffffff;
        border-color: #9fb7c6;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
        background: #ffffff;
        border: 2px solid {ACCENT};
        padding: 9px 11px;
    }}
    QLineEdit::clear-button {{
        padding-right: 4px;
    }}
    QComboBox {{
        padding-right: 34px;
    }}
    QComboBox::drop-down {{
        width: 32px;
        border: none;
        border-top-right-radius: 13px;
        border-bottom-right-radius: 13px;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {SUBTEXT};
        margin-right: 12px;
    }}
    QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 24px;
        border: none;
        background: #eef5f8;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background: #dcecf2;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        border-top-right-radius: 12px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        border-bottom-right-radius: 12px;
    }}
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f6f9);
        border: 1px solid #cbd8e2;
        border-radius: 11px;
        padding: 9px 15px;
        color: {TEXT};
        font-size: 13px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: #ffffff;
        border-color: #9fb7c6;
    }}
    QPushButton:pressed {{
        background: #e3eef3;
        padding-top: 10px;
        padding-bottom: 8px;
    }}
    QPushButton:disabled {{
        color: #94a3b8;
        background: #edf2f5;
        border-color: #d8e2ea;
    }}
    QPushButton#PrimaryButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1b94a8, stop:1 {ACCENT});
        color: white;
        border-color: {ACCENT_DARK};
        font-weight: 600;
    }}
    QPushButton#PrimaryButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #16889b, stop:1 {ACCENT_DARK});
    }}
    QPushButton#SecondaryLinkButton {{
        background: transparent;
        border: none;
        padding: 0;
        color: {ACCENT_DARK};
        text-align: left;
        font-size: 12px;
        text-decoration: underline;
    }}
    QPushButton#SecondaryLinkButton:hover {{
        color: {ACCENT};
        background: transparent;
    }}
    QPushButton#NavButton {{
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 11px 12px;
        color: {TEXT};
        font-weight: 700;
    }}
    QPushButton#NavButton:hover {{
        background: #f8fbfc;
        border-color: #d8e2ea;
    }}
    QPushButton#NavButton:checked {{
        background: {ACCENT};
        color: white;
        border-color: {ACCENT_DARK};
    }}
    QCheckBox {{
        color: {TEXT};
        font-size: 13px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid #b8c9d4;
        background: #ffffff;
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT};
        background: #f0fbfd;
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT_DARK};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 8px 2px 8px 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #c7d6df;
        border-radius: 5px;
        min-height: 36px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #9fb7c6;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px 8px 2px 8px;
    }}
    QScrollBar::handle:horizontal {{
        background: #c7d6df;
        border-radius: 5px;
        min-width: 36px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    """


def create_app(argv: list[str] | None = None) -> tuple[QtWidgets.QApplication, MainWindow]:
    app = QtWidgets.QApplication.instance()
    if app is None:
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QtWidgets.QApplication(argv or sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName(WINDOW_TITLE)
        app.setStyleSheet(_build_stylesheet())
    assert isinstance(app, QtWidgets.QApplication)
    icon_path = bundled_path("assets", "app_icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    window = MainWindow()
    if icon_path.exists():
        window.setWindowIcon(QtGui.QIcon(str(icon_path)))
    return app, window


def main() -> int:
    app, window = create_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
