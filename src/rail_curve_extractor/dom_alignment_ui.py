from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .dom_tiler import (
    DomAlignmentOptions,
    DomAlignmentResult,
    DomAutoCorridorResult,
    DomPreviewResult,
    align_dom_to_corridor_from_map_points,
    align_dom_to_axis,
    auto_detect_dom_corridor_points,
    create_dom_preview,
    discover_dom_file,
)


FONT_FAMILY = "Microsoft YaHei UI"


class AlignmentWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        point1_pixel: tuple[float, float] | None,
        point2_pixel: tuple[float, float] | None,
        options: DomAlignmentOptions,
        corridor_points_map: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.point1_pixel = point1_pixel
        self.point2_pixel = point2_pixel
        self.options = options
        self.corridor_points_map = corridor_points_map

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self.corridor_points_map is not None:
                result = align_dom_to_corridor_from_map_points(
                    self.input_path,
                    self.output_path,
                    self.corridor_points_map,
                    self.options,
                )
            else:
                if self.point1_pixel is None or self.point2_pixel is None:
                    raise ValueError("Two direction points are required.")
                result = align_dom_to_axis(
                    self.input_path,
                    self.output_path,
                    self.point1_pixel,
                    self.point2_pixel,
                    self.options,
                )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class AutoCorridorWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, input_path: Path) -> None:
        super().__init__()
        self.input_path = input_path

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.finished.emit(auto_detect_dom_corridor_points(self.input_path))
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class PreviewImageLabel(QtWidgets.QLabel):
    points_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 360)
        self.preview: DomPreviewResult | None = None
        self.points_preview: list[QtCore.QPointF] = []
        self.points_source: list[tuple[float, float]] = []
        self.setText("点击“加载预览”后，在轨道主线方向上点两个远一些的位置")

    def set_preview(self, preview: DomPreviewResult) -> None:
        self.preview = preview
        self.points_preview.clear()
        self.points_source.clear()
        qimage = _qimage_from_rgb(preview.image)
        self.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.setFixedSize(preview.preview_width, preview.preview_height)
        self.points_changed.emit()
        self.update()

    def clear_points(self) -> None:
        self.points_preview.clear()
        self.points_source.clear()
        self.points_changed.emit()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self.preview is None or self.pixmap() is None:
            return
        position = event.position()
        if position.x() < 0 or position.y() < 0:
            return
        if position.x() > self.preview.preview_width or position.y() > self.preview.preview_height:
            return
        if len(self.points_preview) >= 2:
            self.points_preview.clear()
            self.points_source.clear()
        self.points_preview.append(QtCore.QPointF(position.x(), position.y()))
        self.points_source.append(_preview_to_source_pixel(self.preview, position.x(), position.y()))
        self.points_changed.emit()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self.points_preview:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(0, 220, 255), 3)
        painter.setPen(pen)
        if len(self.points_preview) == 2:
            painter.drawLine(self.points_preview[0], self.points_preview[1])
        elif len(self.points_preview) == 4:
            path = QtGui.QPainterPath(self.points_preview[0])
            path.lineTo(self.points_preview[1])
            path.lineTo(self.points_preview[3])
            path.lineTo(self.points_preview[2])
            path.closeSubpath()
            painter.drawPath(path)
        painter.setBrush(QtGui.QColor(255, 80, 80))
        for index, point in enumerate(self.points_preview, start=1):
            painter.drawEllipse(point, 7, 7)
            painter.drawText(point + QtCore.QPointF(10, -10), str(index))


class DomAlignmentDialog(QtWidgets.QDialog):
    alignment_finished = QtCore.Signal(object)

    def __init__(self, input_path: Path, default_output_path: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.worker_thread: QtCore.QThread | None = None
        self.worker: AlignmentWorker | None = None
        self.auto_worker_thread: QtCore.QThread | None = None
        self.auto_worker: AutoCorridorWorker | None = None
        self.setWindowTitle("标注辅助：轨道方向对齐")
        self.resize(980, 760)
        self._build_ui(default_output_path)

    def _build_ui(self, default_output_path: Path) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("标注辅助：轨道方向对齐")
        title_font = QtGui.QFont(FONT_FAMILY, 16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        layout.addLayout(form)

        self.input_label = QtWidgets.QLabel(str(self.input_path))
        self.input_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("输入", self.input_label)

        self.output_edit = QtWidgets.QLineEdit(str(default_output_path))
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        choose_output = QtWidgets.QPushButton("选择")
        choose_output.clicked.connect(self._choose_output)
        output_row.addWidget(choose_output)
        form.addRow("输出 GeoTIFF", _wrap_layout(output_row))

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem("上下方向（推荐）", "vertical")
        self.axis_combo.addItem("左右方向", "horizontal")
        self.padding_spin = QtWidgets.QSpinBox()
        self.padding_spin.setRange(0, 4096)
        self.padding_spin.setSingleStep(32)
        self.padding_spin.setValue(32)
        self.padding_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.padding_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.crop_checkbox = QtWidgets.QCheckBox("裁掉大部分黑色无效区域")
        self.crop_checkbox.setChecked(True)
        options_row = QtWidgets.QHBoxLayout()
        options_row.addWidget(self.axis_combo)
        options_row.addWidget(QtWidgets.QLabel("边缘保留(px)"))
        options_row.addWidget(self.padding_spin)
        options_row.addWidget(self.crop_checkbox)
        options_row.addStretch(1)
        form.addRow("对齐方式", _wrap_layout(options_row))

        self.point1_map_edit = QtWidgets.QLineEdit()
        self.point1_map_edit.setPlaceholderText("例如：315119.118, 3519469.204")
        self.point2_map_edit = QtWidgets.QLineEdit()
        self.point2_map_edit.setPlaceholderText("例如：315618.293, 3522277.196")
        map_row = QtWidgets.QHBoxLayout()
        map_row.addWidget(QtWidgets.QLabel("点1"))
        map_row.addWidget(self.point1_map_edit)
        map_row.addWidget(QtWidgets.QLabel("点2"))
        map_row.addWidget(self.point2_map_edit)
        use_map_button = QtWidgets.QPushButton("使用地图坐标")
        use_map_button.clicked.connect(self._use_map_points)
        map_row.addWidget(use_map_button)
        form.addRow("地图坐标", _wrap_layout(map_row))

        self.corridor_top_left_edit = QtWidgets.QLineEdit()
        self.corridor_top_left_edit.setPlaceholderText("左上：315601.924, 3522280.143")
        self.corridor_top_right_edit = QtWidgets.QLineEdit()
        self.corridor_top_right_edit.setPlaceholderText("右上：315618.313, 3522277.196")
        self.corridor_bottom_left_edit = QtWidgets.QLineEdit()
        self.corridor_bottom_left_edit.setPlaceholderText("左下：315096.025, 3519473.561")
        self.corridor_bottom_right_edit = QtWidgets.QLineEdit()
        self.corridor_bottom_right_edit.setPlaceholderText("右下：315119.107, 3519469.146")
        corridor_grid = QtWidgets.QGridLayout()
        corridor_grid.addWidget(QtWidgets.QLabel("左上"), 0, 0)
        corridor_grid.addWidget(self.corridor_top_left_edit, 0, 1)
        corridor_grid.addWidget(QtWidgets.QLabel("右上"), 0, 2)
        corridor_grid.addWidget(self.corridor_top_right_edit, 0, 3)
        corridor_grid.addWidget(QtWidgets.QLabel("左下"), 1, 0)
        corridor_grid.addWidget(self.corridor_bottom_left_edit, 1, 1)
        corridor_grid.addWidget(QtWidgets.QLabel("右下"), 1, 2)
        corridor_grid.addWidget(self.corridor_bottom_right_edit, 1, 3)
        use_corridor_button = QtWidgets.QPushButton("使用四点走廊")
        use_corridor_button.clicked.connect(self._use_corridor_points)
        self.auto_corridor_button = QtWidgets.QPushButton("自动识别")
        self.auto_corridor_button.clicked.connect(self._auto_detect_corridor_points)
        corridor_grid.addWidget(use_corridor_button, 0, 4)
        corridor_grid.addWidget(self.auto_corridor_button, 1, 4)
        form.addRow("四点走廊", _wrap_layout(corridor_grid))

        action_row = QtWidgets.QHBoxLayout()
        self.load_preview_button = QtWidgets.QPushButton("加载预览")
        self.load_preview_button.clicked.connect(self._load_preview)
        self.clear_points_button = QtWidgets.QPushButton("清除点")
        self.clear_points_button.clicked.connect(self._clear_points)
        action_row.addWidget(self.load_preview_button)
        action_row.addWidget(self.clear_points_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.preview_label = PreviewImageLabel()
        self.preview_label.points_changed.connect(self._update_point_label)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.preview_label)
        scroll.setWidgetResizable(False)
        scroll.setMinimumHeight(420)
        layout.addWidget(scroll, 1)

        self.point_label = QtWidgets.QLabel("已选 0/2 个点")
        layout.addWidget(self.point_label)

        self.busy_status_label = QtWidgets.QLabel("")
        self.busy_status_label.setVisible(False)
        self.busy_progress_bar = QtWidgets.QProgressBar()
        self.busy_progress_bar.setRange(0, 0)
        self.busy_progress_bar.setTextVisible(False)
        self.busy_progress_bar.setMaximumHeight(10)
        self.busy_progress_bar.setVisible(False)
        busy_row = QtWidgets.QHBoxLayout()
        busy_row.addWidget(self.busy_status_label)
        busy_row.addWidget(self.busy_progress_bar, 1)
        layout.addLayout(busy_row)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(110)
        layout.addWidget(self.log_edit)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.generate_button = QtWidgets.QPushButton("生成标注辅助 DOM")
        self.generate_button.clicked.connect(self._start_alignment)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        button_row.addWidget(self.generate_button)
        layout.addLayout(button_row)

        self._log("这个功能只用于生成更好标注的辅助图；最终模型预测主流程仍直接切原始 DOM。")
        self._log("推荐输入四点走廊：左上、右上、左下、右下。软件会自动判断走廊宽度和标注切片尺寸。")
        self._log("也可以沿主线方向点两个距离尽量远的位置，或直接输入两组地图坐标。")

    def _choose_output(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "选择对齐后 GeoTIFF 输出路径",
            self.output_edit.text(),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if selected:
            self.output_edit.setText(selected)

    def _load_preview(self) -> None:
        self._set_busy(True, "正在加载预览...")
        try:
            preview = create_dom_preview(self.input_path)
        except Exception as exc:
            self._set_busy(False)
            QtWidgets.QMessageBox.critical(self, "预览失败", str(exc))
            return
        self.preview_label.set_preview(preview)
        self._set_busy(False)
        self._log(
            f"预览已加载：源图 {preview.source_width} x {preview.source_height} px，"
            f"预览 {preview.preview_width} x {preview.preview_height} px。"
        )
        if preview.preview_path != preview.source_path:
            self._log(f"使用大疆智图预览文件：{preview.preview_path}")

    def _clear_points(self) -> None:
        self.preview_label.clear_points()

    def _use_map_points(self) -> None:
        try:
            point1_map = _parse_map_point(self.point1_map_edit.text())
            point2_map = _parse_map_point(self.point2_map_edit.text())
            source_points = _map_points_to_source_pixels(self.input_path, [point1_map, point2_map])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "坐标无效", str(exc))
            return
        self.preview_label.points_source = source_points
        self.preview_label.points_preview = []
        if self.preview_label.preview is not None:
            self.preview_label.points_preview = [
                QtCore.QPointF(*_source_pixel_to_preview(self.preview_label.preview, point))
                for point in source_points
            ]
        self.preview_label.points_changed.emit()
        self.preview_label.update()
        self._log(f"已使用地图坐标：{point1_map} -> {point2_map}")

    def _use_corridor_points(self) -> None:
        try:
            points_map = self._corridor_points_from_edits(require_all=True)
            source_points = _map_points_to_source_pixels(self.input_path, points_map)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "四点坐标无效", str(exc))
            return
        self.preview_label.points_source = source_points
        self.preview_label.points_preview = []
        if self.preview_label.preview is not None:
            self.preview_label.points_preview = [
                QtCore.QPointF(*_source_pixel_to_preview(self.preview_label.preview, point))
                for point in source_points
            ]
        self.preview_label.points_changed.emit()
        self.preview_label.update()
        self._log("已使用四点走廊坐标，生成时将按四点范围裁切旋转。")

    def _auto_detect_corridor_points(self) -> None:
        if self.auto_worker_thread is not None:
            return
        self._set_busy(True, "正在自动识别 DOM 有效区域四点...")

        self.auto_worker_thread = QtCore.QThread(self)
        self.auto_worker = AutoCorridorWorker(self.input_path)
        self.auto_worker.moveToThread(self.auto_worker_thread)
        self.auto_worker_thread.started.connect(self.auto_worker.run)
        self.auto_worker.finished.connect(self._on_auto_corridor_finished)
        self.auto_worker.failed.connect(self._on_auto_corridor_failed)
        self.auto_worker.finished.connect(self.auto_worker_thread.quit)
        self.auto_worker.failed.connect(self.auto_worker_thread.quit)
        self.auto_worker_thread.finished.connect(self.auto_worker.deleteLater)
        self.auto_worker_thread.finished.connect(self.auto_worker_thread.deleteLater)
        self.auto_worker_thread.finished.connect(self._clear_auto_worker_refs)
        self.auto_worker_thread.start()

    def _on_auto_corridor_finished(self, result: DomAutoCorridorResult) -> None:
        edits = [
            self.corridor_top_left_edit,
            self.corridor_top_right_edit,
            self.corridor_bottom_left_edit,
            self.corridor_bottom_right_edit,
        ]
        for edit, point in zip(edits, result.points_map, strict=True):
            edit.setText(_format_map_point(point))
        self.preview_label.points_source = result.points_pixel
        self.preview_label.points_preview = []
        if self.preview_label.preview is not None:
            self.preview_label.points_preview = [
                QtCore.QPointF(*_source_pixel_to_preview(self.preview_label.preview, point))
                for point in result.points_pixel
            ]
        self.preview_label.points_changed.emit()
        self.preview_label.update()
        self._set_busy(False)
        self._log(
            f"自动识别完成：采样 {result.sample_width} x {result.sample_height}，"
            f"有效区域占比 {result.valid_ratio:.2%}，长轴角度 {result.angle_degrees:.1f}°。"
        )
        if self.preview_label.preview is None:
            self._log("已填入四点坐标；加载预览后可以在图上检查四点位置。")

    def _on_auto_corridor_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(f"自动识别失败：{message}")
        QtWidgets.QMessageBox.warning(self, "自动识别失败", message)

    def _update_point_label(self) -> None:
        count = len(self.preview_label.points_source)
        if count == 0:
            self.point_label.setText("已选 0/2 个点")
        else:
            points = "；".join(f"{index}: ({point[0]:.1f}, {point[1]:.1f})" for index, point in enumerate(self.preview_label.points_source, 1))
            target_count = 4 if count == 4 else 2
            self.point_label.setText(f"已选 {count}/{target_count} 个点，源图像素坐标：{points}")

    def _start_alignment(self) -> None:
        corridor_points_map = self._corridor_points_from_edits(require_all=False)
        if corridor_points_map is None and len(self.preview_label.points_source) != 2:
            QtWidgets.QMessageBox.warning(self, "缺少点位", "请先输入四点走廊，或在预览图上点击两个轨道主线方向点。")
            return
        output_path = Path(self.output_edit.text().strip())
        if not str(output_path):
            QtWidgets.QMessageBox.warning(self, "缺少输出", "请选择对齐后 GeoTIFF 输出路径。")
            return
        if output_path.suffix.lower() not in {".tif", ".tiff"}:
            output_path = output_path.with_suffix(".tif")
            self.output_edit.setText(str(output_path))

        overwrite = False
        if output_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "文件已存在",
                f"输出文件已存在，是否覆盖？\n{output_path}",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        point1: tuple[float, float] | None = None
        point2: tuple[float, float] | None = None
        if corridor_points_map is None:
            point1, point2 = self.preview_label.points_source
        options = DomAlignmentOptions(
            target_axis=str(self.axis_combo.currentData()),
            padding_pixels=self.padding_spin.value(),
            crop_to_valid_data=self.crop_checkbox.isChecked(),
            overwrite=overwrite,
        )
        self._set_running(True)
        self._log(f"开始生成标注辅助 DOM：{output_path}")
        self.worker_thread = QtCore.QThread(self)
        self.worker = AlignmentWorker(self.input_path, output_path, point1, point2, options, corridor_points_map)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker_refs)
        self.worker_thread.start()

    def _on_finished(self, result: DomAlignmentResult) -> None:
        self._set_running(False)
        self._log(f"完成：{result.output_width} x {result.output_height} px")
        self._log(f"输出：{result.output_path}")
        self._log(f"元数据：{result.metadata_path}")
        self.alignment_finished.emit(result)
        QtWidgets.QMessageBox.information(self, "对齐完成", "已生成标注辅助 GeoTIFF，主窗口输入路径已切换到该文件。")

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._log(f"失败：{message}")
        QtWidgets.QMessageBox.critical(self, "对齐失败", message)

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self.load_preview_button.setEnabled(not busy)
        self.clear_points_button.setEnabled(not busy)
        self.auto_corridor_button.setEnabled(not busy)
        self.generate_button.setEnabled(not busy)
        self.busy_status_label.setVisible(busy)
        self.busy_progress_bar.setVisible(busy)
        if busy and text:
            self.busy_status_label.setText(text)
        elif not busy:
            self.busy_status_label.setText("")
        if text:
            self._log(text)

    def _set_running(self, running: bool) -> None:
        self.load_preview_button.setEnabled(not running)
        self.clear_points_button.setEnabled(not running)
        self.auto_corridor_button.setEnabled(not running)
        self.generate_button.setEnabled(not running)
        self.generate_button.setText("正在生成..." if running else "生成标注辅助 DOM")

    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _clear_auto_worker_refs(self) -> None:
        self.auto_worker = None
        self.auto_worker_thread = None

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def _corridor_points_from_edits(self, require_all: bool) -> list[tuple[float, float]] | None:
        texts = [
            self.corridor_top_left_edit.text(),
            self.corridor_top_right_edit.text(),
            self.corridor_bottom_left_edit.text(),
            self.corridor_bottom_right_edit.text(),
        ]
        has_any = any(text.strip() for text in texts)
        if not has_any and not require_all:
            return None
        if not all(text.strip() for text in texts):
            raise ValueError("四点走廊需要填写左上、右上、左下、右下四组坐标。")
        return [_parse_map_point(text) for text in texts]


def _qimage_from_rgb(image: np.ndarray) -> QtGui.QImage:
    contiguous = np.ascontiguousarray(image)
    height, width, channels = contiguous.shape
    if channels != 3:
        raise ValueError("Preview image must be RGB.")
    bytes_per_line = channels * width
    return QtGui.QImage(
        contiguous.data,
        width,
        height,
        bytes_per_line,
        QtGui.QImage.Format.Format_RGB888,
    ).copy()


def _preview_to_source_pixel(preview: DomPreviewResult, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = preview.preview_to_source_transform
    return a * x + b * y + c, d * x + e * y + f


def _source_pixel_to_preview(preview: DomPreviewResult, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = preview.preview_to_source_transform
    determinant = a * e - b * d
    if abs(determinant) < 1e-12:
        return point[0] / preview.scale_x, point[1] / preview.scale_y
    x = (e * (point[0] - c) - b * (point[1] - f)) / determinant
    y = (-d * (point[0] - c) + a * (point[1] - f)) / determinant
    return x, y


def _parse_map_point(text: str) -> tuple[float, float]:
    normalized = text.replace("X:", "").replace("Y:", "").replace("x:", "").replace("y:", "")
    normalized = normalized.replace(",", " ")
    parts = [part for part in normalized.split() if part]
    if len(parts) != 2:
        raise ValueError("请输入两个数值，例如：315119.118, 3519469.204")
    return float(parts[0]), float(parts[1])


def _format_map_point(point: tuple[float, float]) -> str:
    return f"{point[0]:.3f}, {point[1]:.3f}"


def _map_points_to_source_pixels(input_path: Path, points_map: list[tuple[float, float]]) -> list[tuple[float, float]]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("缺少 rasterio，无法读取 GeoTIFF 坐标。") from exc

    source_path = discover_dom_file(input_path).resolve()
    with rasterio.open(source_path) as dataset:
        inverse = ~dataset.transform
        points = [inverse * point for point in points_map]
        for col, row in points:
            if col < 0 or col > dataset.width or row < 0 or row > dataset.height:
                raise ValueError("地图坐标转换后的像素点不在 DOM 范围内。")
    return [(float(col), float(row)) for col, row in points]


def _wrap_layout(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget
