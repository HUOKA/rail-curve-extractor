from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .dom_alignment_ui import DomAlignmentDialog
from .dom_tiler import DomTileOptions, DomTilingResult, stride_from_overlap, suggest_annotation_tile_size, tile_dom


DEFAULT_INPUT_CANDIDATES = [Path("D:/轨道正射图"), Path("D:/正射")]
DEFAULT_INPUT = next((path for path in DEFAULT_INPUT_CANDIDATES if path.exists()), DEFAULT_INPUT_CANDIDATES[0])
DEFAULT_OUTPUT = Path.cwd() / "data" / "dom_tiles_raw_3072"
FONT_FAMILY = "Microsoft YaHei UI"
APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", SimSun, "Segoe UI";
    background: #f7f8fa;
    color: #1f2937;
}
QWidget#inlineContainer {
    background: transparent;
}
QGroupBox QLabel, QCheckBox {
    background: transparent;
}
QLabel#titleLabel {
    color: #111827;
}
QLabel#subtitleLabel {
    color: #6b7280;
}
QLabel#modeBadge {
    background: #e6f6f2;
    border: 1px solid #a7dfd4;
    border-radius: 4px;
    color: #0f5f56;
    padding: 6px 10px;
}
QLabel#summaryLabel {
    background: #f0f7f2;
    border: 1px solid #cce6d2;
    border-radius: 4px;
    color: #235537;
    padding: 7px 10px;
}
QLabel#statusLabel {
    color: #475569;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d9dee7;
    border-radius: 6px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #334155;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #cfd6e2;
    border-radius: 4px;
    min-height: 34px;
    padding: 4px 8px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #0f766e;
}
QPlainTextEdit {
    background: #0f172a;
    color: #dbeafe;
    font-family: Consolas, "Microsoft YaHei UI";
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    min-height: 36px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #f1f5f9;
}
QPushButton:focus {
    border: 1px solid #0f766e;
}
QPushButton[role="primary"] {
    background: #0f766e;
    border: 1px solid #0f5f56;
    color: #ffffff;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background: #0f5f56;
}
QPushButton[role="quiet"] {
    background: #f8fafc;
}
QPushButton:disabled {
    color: #94a3b8;
    background: #e5e7eb;
    border-color: #d1d5db;
}
QProgressBar {
    background: #e5e7eb;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    min-height: 10px;
}
QProgressBar::chunk {
    background: #0f766e;
    border-radius: 3px;
}
QCheckBox {
    spacing: 8px;
}
"""


class TileWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, input_path: Path, output_dir: Path, options: DomTileOptions) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.options = options

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.finished.emit(tile_dom(self.input_path, self.output_dir, self.options))
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class DomTilerWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.worker_thread: QtCore.QThread | None = None
        self.worker: TileWorker | None = None
        self.setWindowTitle("DOM 切片工具")
        self.resize(920, 780)
        self.setMinimumSize(820, 700)
        self._build_ui()
        self._update_stride_preview()

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_STYLE)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("DOM GeoTIFF 切片")
        title.setObjectName("titleLabel")
        title_font = QtGui.QFont(FONT_FAMILY, 18)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QtWidgets.QLabel("原始 DOM 主流程用于训练/预测；方向对齐只用于更省力的 CVAT 标注。")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)

        title_column = QtWidgets.QVBoxLayout()
        title_column.setSpacing(3)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)

        self.mode_label = QtWidgets.QLabel("主流程：原始 DOM 切片")
        self.mode_label.setObjectName("modeBadge")

        header_row = QtWidgets.QHBoxLayout()
        header_row.addLayout(title_column, 1)
        header_row.addWidget(self.mode_label, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        source_group = self._group_box("1. 数据源与输出")
        source_form = self._form_layout()
        source_group.setLayout(source_form)

        self.input_edit = QtWidgets.QLineEdit(str(DEFAULT_INPUT if DEFAULT_INPUT.exists() else ""))
        self.input_edit.setPlaceholderText("选择 dom.tif 或 DJI Terra 工程目录")
        input_row = self._path_row(
            self.input_edit,
            [
                ("选目录", self._choose_input_dir),
                ("选 TIF", self._choose_input_file),
            ],
        )
        source_form.addRow("DOM 或工程目录", input_row)

        self.output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT))
        self.output_edit.setPlaceholderText("选择切片输出目录")
        output_row = self._path_row(self.output_edit, [("选目录", self._choose_output_dir)])
        source_form.addRow("输出目录", output_row)
        layout.addWidget(source_group)

        settings_group = self._group_box("2. 切片参数")
        settings_layout = QtWidgets.QVBoxLayout()
        settings_layout.setSpacing(10)
        settings_group.setLayout(settings_layout)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(self._preset_button("原始 DOM 3072", lambda: self._apply_preset(3072, 3072, 50.0, 50.0, "dom", "主流程：原始 DOM 切片")))
        preset_row.addWidget(self._preset_button("快速试切 2048", lambda: self._apply_preset(2048, 2048, 50.0, 50.0, "dom_test", "试切：原始 DOM 小批量", 50)))
        preset_row.addWidget(self._preset_button("标注辅助 768x3072", lambda: self._apply_preset(768, 3072, 0.0, 50.0, "aligned", "标注辅助：对齐走廊切片")))
        preset_row.addStretch(1)
        settings_layout.addLayout(preset_row)

        settings_grid = QtWidgets.QGridLayout()
        settings_grid.setHorizontalSpacing(10)
        settings_grid.setVerticalSpacing(8)
        settings_grid.setColumnStretch(0, 0)
        settings_grid.setColumnStretch(1, 0)
        settings_grid.setColumnStretch(2, 0)
        settings_grid.setColumnStretch(3, 1)
        settings_layout.addLayout(settings_grid)

        self.width_spin = self._spinbox(128, 8192, 3072, 64)
        self.height_spin = self._spinbox(128, 8192, 3072, 64)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QtWidgets.QLabel("x"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        settings_grid.addWidget(self._field_label("切片宽高(px)"), 0, 0)
        settings_grid.addWidget(self._wrap_layout(size_row), 0, 1)

        self.overlap_x_spin = self._overlap_spinbox(50.0)
        self.overlap_y_spin = self._overlap_spinbox(50.0)
        self.stride_label = QtWidgets.QLabel()
        overlap_row = QtWidgets.QHBoxLayout()
        overlap_row.addWidget(QtWidgets.QLabel("横向"))
        overlap_row.addWidget(self.overlap_x_spin)
        overlap_row.addWidget(QtWidgets.QLabel("纵向"))
        overlap_row.addWidget(self.overlap_y_spin)
        overlap_row.addWidget(self.stride_label)
        overlap_row.addStretch(1)
        settings_grid.addWidget(self._field_label("重叠率"), 0, 2)
        settings_grid.addWidget(self._wrap_layout(overlap_row), 0, 3)

        self.prefix_edit = QtWidgets.QLineEdit("dom")
        self.prefix_edit.setFixedHeight(44)
        settings_grid.addWidget(self._field_label("文件名前缀"), 1, 0)
        settings_grid.addWidget(self.prefix_edit, 1, 1, 1, 3)

        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["png", "jpg"])
        self.format_combo.setMinimumWidth(80)
        self.format_combo.setFixedHeight(44)
        self.skip_empty_checkbox = QtWidgets.QCheckBox("跳过空白切片")
        self.skip_empty_checkbox.setChecked(True)
        self.skip_empty_checkbox.setMinimumWidth(150)
        self.max_tiles_spin = self._spinbox(0, 999999, 0, 10)
        self.max_tiles_spin.setMinimumWidth(140)
        max_tiles_label = QtWidgets.QLabel("最多切片(0=不限)")
        max_tiles_label.setMinimumWidth(130)
        extra_row = QtWidgets.QHBoxLayout()
        extra_row.addWidget(self.format_combo)
        extra_row.addWidget(self.skip_empty_checkbox)
        extra_row.addWidget(max_tiles_label)
        extra_row.addWidget(self.max_tiles_spin)
        extra_row.addStretch(1)
        settings_grid.addWidget(self._field_label("输出选项"), 2, 0)
        settings_grid.addWidget(self._wrap_layout(extra_row), 2, 1, 1, 3)

        self.parameter_summary_label = QtWidgets.QLabel()
        self.parameter_summary_label.setObjectName("summaryLabel")
        self.parameter_summary_label.setWordWrap(True)
        settings_layout.addWidget(self.parameter_summary_label)
        layout.addWidget(settings_group)

        self.width_spin.valueChanged.connect(self._update_stride_preview)
        self.height_spin.valueChanged.connect(self._update_stride_preview)
        self.overlap_x_spin.valueChanged.connect(self._update_stride_preview)
        self.overlap_y_spin.valueChanged.connect(self._update_stride_preview)
        self.format_combo.currentTextChanged.connect(self._update_stride_preview)
        self.skip_empty_checkbox.stateChanged.connect(self._update_stride_preview)
        self.max_tiles_spin.valueChanged.connect(self._update_stride_preview)

        run_group = self._group_box("3. 运行与反馈")
        run_layout = QtWidgets.QVBoxLayout()
        run_layout.setSpacing(10)
        run_group.setLayout(run_layout)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(170)
        run_layout.addWidget(self.log_edit)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(10)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar, 1)
        run_layout.addLayout(status_row)

        button_row = QtWidgets.QHBoxLayout()
        self.align_button = QtWidgets.QPushButton("标注辅助：方向对齐...")
        self.align_button.setProperty("role", "quiet")
        self.align_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.align_button.clicked.connect(self._open_alignment_dialog)
        self.start_button = QtWidgets.QPushButton("开始切片")
        self.start_button.setProperty("role", "primary")
        self.start_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.start_button.clicked.connect(self._start_tiling)
        self.open_output_button = QtWidgets.QPushButton("打开输出目录")
        self.open_output_button.setProperty("role", "quiet")
        self.open_output_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.open_output_button.clicked.connect(self._open_output_dir)
        button_row.addWidget(self.align_button)
        button_row.addStretch(1)
        button_row.addWidget(self.open_output_button)
        button_row.addWidget(self.start_button)
        run_layout.addLayout(button_row)
        layout.addWidget(run_group, 1)

        self._log("推荐主流程：直接切原始 DOM，默认 3072 x 3072，横向/纵向重叠率 50%。")
        self._log("方向对齐只作为标注辅助；对齐走廊切片默认横向 0%、纵向 50%。先小批量试切可把最多切片设为 50。")

    def _path_row(self, edit: QtWidgets.QLineEdit, buttons: list[tuple[str, Any]]) -> QtWidgets.QWidget:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(edit, 1)
        for text, callback in buttons:
            button = QtWidgets.QPushButton(text)
            button.setProperty("role", "quiet")
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(callback)
            row.addWidget(button)
        return self._wrap_layout(row)

    def _group_box(self, title: str) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        return group

    def _form_layout(self) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        return form

    def _preset_button(self, text: str, callback: Any) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setProperty("role", "quiet")
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return label

    def _overlap_spinbox(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 90.0)
        spin.setSingleStep(5.0)
        spin.setDecimals(1)
        spin.setSuffix(" %")
        spin.setValue(value)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        spin.setFixedSize(112, 44)
        return spin

    def _apply_preset(
        self,
        width: int,
        height: int,
        overlap_x_percent: float,
        overlap_y_percent: float,
        prefix: str,
        mode_text: str,
        max_tiles: int | None = 0,
    ) -> None:
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.overlap_x_spin.setValue(overlap_x_percent)
        self.overlap_y_spin.setValue(overlap_y_percent)
        self.prefix_edit.setText(prefix)
        if max_tiles is not None:
            self.max_tiles_spin.setValue(max_tiles)
        self.mode_label.setText(mode_text)
        self._set_status(f"已应用预设：{width} x {height}，横向重叠 {overlap_x_percent:g}%，纵向重叠 {overlap_y_percent:g}%")
        self._update_stride_preview()

    def _spinbox(self, minimum: int, maximum: int, value: int, step: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        spin.setFixedHeight(44)
        spin.setMinimumWidth(112)
        return spin

    def _wrap_layout(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setObjectName("inlineContainer")
        widget.setMinimumHeight(48)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        widget.setLayout(layout)
        return widget

    def _choose_input_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 DJI Terra 工程目录", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)

    def _choose_input_file(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 DOM GeoTIFF",
            self.input_edit.text(),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if selected:
            self.input_edit.setText(selected)

    def _choose_output_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _open_alignment_dialog(self) -> None:
        input_text = self.input_edit.text().strip()
        if not input_text:
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请选择 DOM GeoTIFF 或 DJI Terra 工程目录。")
            return
        dialog = DomAlignmentDialog(Path(input_text), self._default_alignment_output_path(), self)
        dialog.alignment_finished.connect(self._on_alignment_finished)
        dialog.exec()

    def _default_alignment_output_path(self) -> Path:
        output_text = self.output_edit.text().strip()
        base_dir = Path(output_text).parent if output_text else Path.cwd() / "data"
        if not str(base_dir):
            base_dir = Path.cwd() / "data"
        return base_dir / "aligned_dom" / "aligned_dom.tif"

    def _on_alignment_finished(self, result: Any) -> None:
        aligned_path = Path(result.output_path)
        tile_base = aligned_path.parent.parent if aligned_path.parent.name == "aligned_dom" else aligned_path.parent
        self.input_edit.setText(str(aligned_path))
        self.output_edit.setText(str(tile_base / "dom_tiles_aligned_annotation"))
        suggestion = suggest_annotation_tile_size(result.output_width, result.output_height)
        self.width_spin.setValue(suggestion.tile_width)
        self.height_spin.setValue(suggestion.tile_height)
        self.overlap_x_spin.setValue(0.0)
        self.overlap_y_spin.setValue(suggestion.overlap_ratio * 100.0)
        self.prefix_edit.setText("aligned")
        self.max_tiles_spin.setValue(0)
        self.mode_label.setText("标注辅助：对齐走廊切片")
        self._set_status("已生成对齐 DOM，可继续切标注图")
        self._log(f"已切换输入为标注辅助对齐 DOM：{aligned_path}")
        self._log(
            f"已自动判断标注切片尺寸：{suggestion.tile_width} x {suggestion.tile_height} px，"
            f"步长 {suggestion.stride_x} x {suggestion.stride_y} px。"
        )
        self._log("这批切片适合 CVAT 标注；最终预测主流程仍建议切原始 DOM。")

    def _update_stride_preview(self, *_: Any) -> None:
        overlap_x_ratio = self.overlap_x_spin.value() / 100.0
        overlap_y_ratio = self.overlap_y_spin.value() / 100.0
        tile_width = self.width_spin.value()
        tile_height = self.height_spin.value()
        stride_x = stride_from_overlap(tile_width, overlap_x_ratio)
        stride_y = stride_from_overlap(tile_height, overlap_y_ratio)
        self.stride_label.setText(f"步长：{stride_x} x {stride_y} px")
        if not hasattr(self, "parameter_summary_label"):
            return
        image_format = self.format_combo.currentText().upper() if hasattr(self, "format_combo") else "PNG"
        skip_text = "跳过空白切片" if self.skip_empty_checkbox.isChecked() else "保留全部切片"
        limit_text = "不限数量" if self.max_tiles_spin.value() == 0 else f"最多 {self.max_tiles_spin.value()} 张"
        warning = ""
        if tile_width % 32 != 0 or tile_height % 32 != 0:
            warning = "；建议宽高使用 32 的倍数，后续训练更省心"
        self.parameter_summary_label.setText(
            f"当前：{tile_width} x {tile_height} px，横向重叠 {self.overlap_x_spin.value():g}%，纵向重叠 {self.overlap_y_spin.value():g}%，"
            f"步长 {stride_x} x {stride_y} px，输出 {image_format}，{skip_text}，{limit_text}{warning}。"
        )

    def _start_tiling(self) -> None:
        input_path = Path(self.input_edit.text().strip())
        output_dir = Path(self.output_edit.text().strip())
        if not str(input_path):
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请选择 DOM GeoTIFF 或 DJI Terra 工程目录。")
            return
        if not input_path.exists():
            QtWidgets.QMessageBox.warning(self, "输入不存在", "当前输入路径不存在，请重新选择 DOM GeoTIFF 或工程目录。")
            return
        if not str(output_dir):
            QtWidgets.QMessageBox.warning(self, "缺少输出", "请选择输出目录。")
            return

        overlap_x_ratio = self.overlap_x_spin.value() / 100.0
        overlap_y_ratio = self.overlap_y_spin.value() / 100.0
        tile_width = self.width_spin.value()
        tile_height = self.height_spin.value()
        max_tiles = self.max_tiles_spin.value() or None
        options = DomTileOptions(
            tile_width=tile_width,
            tile_height=tile_height,
            stride_x=stride_from_overlap(tile_width, overlap_x_ratio),
            stride_y=stride_from_overlap(tile_height, overlap_y_ratio),
            image_format=self.format_combo.currentText(),
            prefix=self.prefix_edit.text().strip() or "dom",
            skip_empty=self.skip_empty_checkbox.isChecked(),
            max_tiles=max_tiles,
        )

        self._set_running(True)
        self._log(f"开始切片：{input_path}")
        self._log(f"输出目录：{output_dir}")
        self._log(
            f"参数：{tile_width} x {tile_height} px，"
            f"横向重叠 {self.overlap_x_spin.value():g}%，纵向重叠 {self.overlap_y_spin.value():g}%，"
            f"格式 {options.image_format}"
        )
        self.worker_thread = QtCore.QThread(self)
        self.worker = TileWorker(input_path, output_dir, options)
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

    def _on_finished(self, result: DomTilingResult) -> None:
        self._set_running(False)
        self._set_status(f"完成：{result.tile_count} 张切片")
        self._log(f"完成：{result.tile_count} 张切片")
        self._log(f"图片目录：{result.images_dir}")
        self._log(f"坐标表：{result.csv_path}")
        self._log(f"JSON：{result.json_path}")
        QtWidgets.QMessageBox.information(self, "切片完成", f"已生成 {result.tile_count} 张切片。")

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._set_status("切片失败，请查看日志")
        self._log(f"失败：{message}")
        QtWidgets.QMessageBox.critical(self, "切片失败", message)

    def _set_running(self, running: bool) -> None:
        self.align_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.open_output_button.setEnabled(not running)
        self.start_button.setText("正在切片..." if running else "开始切片")
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, 0)
            self._set_status("正在切片，请保持窗口打开")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def _set_status(self, text: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(text)

    def _clear_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _open_output_dir(self) -> None:
        output_dir = Path(self.output_edit.text().strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(output_dir.resolve())))

    def _log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the DOM tiler UI.")
    parser.add_argument("--smoke-test", action="store_true", help="Construct the window and exit without showing it.")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont(FONT_FAMILY, 10))
    window = DomTilerWindow()
    if args.smoke_test:
        return 0
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
