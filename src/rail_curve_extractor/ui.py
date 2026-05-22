from __future__ import annotations

import ctypes
import json
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Any, Callable

import numpy as np

from .pipeline import PipelineResult, analyze_input, export_pipeline_result, prepare_config
from .preview import build_profile_points, combine_bounds, downsample_points, fit_points_to_canvas

WINDOW_BG = "#eeece4"
SIDEBAR_BG = "#e6ece6"
CARD_BG = "#fbfaf6"
CARD_ALT_BG = "#f3f1e8"
ACCENT = "#1c7a62"
ACCENT_DEEP = "#145445"
ACCENT_SOFT = "#d8ebe2"
RAIL = "#cf6a32"
MUTED = "#c3ccd1"
EDGE = "#d4d9d6"
GRID = "#dfddd1"
TEXT = "#1f2528"
SUBTEXT = "#5d6a72"
CANVAS_BG = "#f7f5ee"
TOOLTIP_BG = "#fffdf8"
TOOLTIP_EDGE = "#c9d4ce"
TOOLTIP_TEXT = "#263036"

UI_FONT_FAMILY = "Microsoft YaHei UI" if sys.platform == "win32" else "TkDefaultFont"
CODE_FONT_FAMILY = "Cascadia Mono" if sys.platform == "win32" else "TkFixedFont"

PATH_FIELD_HELP: dict[str, str] = {
    "input_path": "要处理的点云文件。支持 LAS、LAZ、CSV、TXT、XYZ 和 NPY。建议先裁剪到轨道附近再导入。",
    "output_path": "导出目录。预览不会写文件，只有点击导出时才会把点云结果、中心线和 USD Curve 写到这里。",
    "config_path": "可选的 JSON 参数文件。加载后会填充到界面输入框里，保存时会写出当前界面上的参数。",
}


@dataclass(frozen=True, slots=True)
class FieldSpec:
    path: str
    label: str
    caster: Callable[[str], Any]
    description: str
    allow_blank: bool = False


COMMON_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        path="height_filter.keep_top_percent",
        label="顶部保留比例",
        caster=float,
        description="保留点云中较高部分的比例。值越大，留下的高程点越多；过小可能把轨顶点滤掉，过大则会混入更多杂点。",
    ),
    FieldSpec(
        path="slice_length",
        label="切片长度 (m)",
        caster=float,
        description="沿轨道主方向切片的长度。切片越短越能跟随弯道，但太短时每片点数容易不足。",
    ),
    FieldSpec(
        path="min_points_per_slice",
        label="每片最少点数",
        caster=int,
        description="每个切片至少要有多少点才参与轨道检测。值越大越稳，但也更容易丢掉稀疏区域。",
    ),
    FieldSpec(
        path="rail_pair_spacing_min",
        label="最小轨距窗口 (m)",
        caster=float,
        description="双峰检测时允许的最小左右轨间距。它不是严格轨距值，而是候选窗口的下边界。",
    ),
    FieldSpec(
        path="rail_pair_spacing_max",
        label="最大轨距窗口 (m)",
        caster=float,
        description="双峰检测时允许的最大左右轨间距。过大可能误选到道床或邻近结构，过小则会漏检。",
    ),
    FieldSpec(
        path="peak_search_bins",
        label="峰搜索分箱数",
        caster=int,
        description="横向直方图的分箱数量。箱数太少会把双峰抹平，太多则在稀疏数据上不稳定。",
    ),
    FieldSpec(
        path="peak_window_radius",
        label="峰点窗口半径 (m)",
        caster=float,
        description="围绕左右峰位聚合轨道点时使用的半径。它决定最终被判为轨顶候选点的横向宽度。",
    ),
    FieldSpec(
        path="savgol_window",
        label="平滑窗口",
        caster=int,
        description="中心线 Savitzky-Golay 平滑窗口长度。值越大，曲线越顺，但可能把局部细节抹掉。",
    ),
    FieldSpec(
        path="curve_width",
        label="导出曲线宽度",
        caster=float,
        description="写入 USD `BasisCurves` 的曲线宽度属性。它主要影响下游可视化，不改变中心线几何。",
    ),
)

ROI_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        path="roi.x_min",
        label="X min",
        caster=float,
        description="世界坐标 X 方向的最小保留边界。留空表示不限制。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.x_max",
        label="X max",
        caster=float,
        description="世界坐标 X 方向的最大保留边界。留空表示不限制。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.y_min",
        label="Y min",
        caster=float,
        description="世界坐标 Y 方向的最小保留边界。留空表示不限制。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.y_max",
        label="Y max",
        caster=float,
        description="世界坐标 Y 方向的最大保留边界。留空表示不限制。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.z_min",
        label="Z min",
        caster=float,
        description="世界坐标 Z 方向的最小保留边界。可用来剔除地面以下噪声或无关低位结构。",
        allow_blank=True,
    ),
    FieldSpec(
        path="roi.z_max",
        label="Z max",
        caster=float,
        description="世界坐标 Z 方向的最大保留边界。可用来剔除接触网、站台边缘等高位结构。",
        allow_blank=True,
    ),
)

_DPI_AWARENESS_CONFIGURED = False


def _enable_high_dpi_support() -> None:
    global _DPI_AWARENESS_CONFIGURED
    if _DPI_AWARENESS_CONFIGURED or sys.platform != "win32":
        return

    _DPI_AWARENESS_CONFIGURED = True
    try:
        user32 = ctypes.windll.user32
        awareness_context = ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if set_context is not None and set_context(awareness_context):
            return
    except Exception:
        pass

    try:
        shcore = ctypes.windll.shcore
        set_awareness = getattr(shcore, "SetProcessDpiAwareness", None)
        if set_awareness is not None:
            set_awareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            return
    except Exception:
        pass

    try:
        user32 = ctypes.windll.user32
        set_dpi_aware = getattr(user32, "SetProcessDPIAware", None)
        if set_dpi_aware is not None:
            set_dpi_aware()
    except Exception:
        pass


class HoverTooltip:
    def __init__(self, root: tk.Tk, px: Callable[[float], int]) -> None:
        self.root = root
        self._px = px
        self.window: tk.Toplevel | None = None
        self.label: tk.Label | None = None

    def bind(self, widget: tk.Widget, text: str) -> None:
        if not text:
            return
        widget.bind("<Enter>", lambda event, tooltip_text=text: self.show(tooltip_text, event), add="+")
        widget.bind("<Motion>", self.move, add="+")
        widget.bind("<Leave>", lambda _event: self.hide(), add="+")
        widget.bind("<FocusIn>", lambda event, tooltip_text=text: self.show(tooltip_text, event), add="+")
        widget.bind("<FocusOut>", lambda _event: self.hide(), add="+")

    def show(self, text: str, event: tk.Event[tk.Widget]) -> None:
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.root)
            self.window.overrideredirect(True)
            self.window.attributes("-topmost", True)
            container = tk.Frame(
                self.window,
                background=TOOLTIP_BG,
                highlightbackground=TOOLTIP_EDGE,
                highlightthickness=1,
            )
            container.pack()
            self.label = tk.Label(
                container,
                background=TOOLTIP_BG,
                foreground=TOOLTIP_TEXT,
                justify="left",
                wraplength=self._px(320),
                padx=self._px(12),
                pady=self._px(10),
                font=(UI_FONT_FAMILY, 10),
            )
            self.label.pack()
        if self.label is not None:
            self.label.configure(text=text)
        self.window.deiconify()
        self.move(event)

    def move(self, event: tk.Event[tk.Widget]) -> None:
        if self.window is None or not self.window.winfo_exists() or not self.window.winfo_viewable():
            return
        self.window.update_idletasks()
        x_position = event.x_root + self._px(18)
        y_position = event.y_root + self._px(18)
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        margin = self._px(16)

        if x_position + width > screen_width - margin:
            x_position = event.x_root - width - margin
        if y_position + height > screen_height - margin:
            y_position = event.y_root - height - margin
        self.window.geometry(f"+{x_position}+{y_position}")

    def hide(self) -> None:
        if self.window is not None and self.window.winfo_exists():
            self.window.withdraw()


class RailCurveExtractorApp(tk.Tk):
    def __init__(self) -> None:
        _enable_high_dpi_support()
        super().__init__()

        self.option_add("*tearOff", False)
        self.dpi_scale = self._configure_scaling()
        self.pixel_scale = min(max(self.dpi_scale, 1.0), 1.6)
        self.tooltip = HoverTooltip(self, self._px)

        self.title("铁轨曲线提取器")
        self._set_initial_geometry()
        self.minsize(1320, 820)
        self.configure(background=WINDOW_BG)

        self.result: PipelineResult | None = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self._redraw_after_id: str | None = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.config_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="准备就绪")
        self.height_filter_enabled_var = tk.BooleanVar(value=True)
        self.summary_var = tk.StringVar(value="尚未生成预览。")
        self.metric_vars = {
            "input_points": tk.StringVar(value="--"),
            "filtered_points": tk.StringVar(value="--"),
            "centerline_points": tk.StringVar(value="--"),
            "curve_length_m": tk.StringVar(value="--"),
        }

        self.field_vars: dict[str, tk.StringVar] = {}
        self.export_button: ttk.Button | None = None
        self.top_canvas: tk.Canvas | None = None
        self.profile_canvas: tk.Canvas | None = None
        self.log_text: tk.Text | None = None

        self._configure_fonts()
        self._configure_style()
        self._build_layout()
        self._apply_config_to_form(prepare_config())
        self._set_busy(False)
        self.after(100, self._poll_worker_queue)

    def _configure_scaling(self) -> float:
        pixels_per_inch = float(self.winfo_fpixels("1i"))
        if pixels_per_inch <= 0:
            pixels_per_inch = 96.0
        try:
            self.tk.call("tk", "scaling", pixels_per_inch / 72.0)
        except tk.TclError:
            pass
        return max(pixels_per_inch / 96.0, 1.0)

    def _configure_fonts(self) -> None:
        named_fonts = {
            "TkDefaultFont": (UI_FONT_FAMILY, 10),
            "TkTextFont": (UI_FONT_FAMILY, 10),
            "TkMenuFont": (UI_FONT_FAMILY, 10),
            "TkHeadingFont": (UI_FONT_FAMILY, 11, "bold"),
            "TkCaptionFont": (UI_FONT_FAMILY, 9),
            "TkTooltipFont": (UI_FONT_FAMILY, 9),
            "TkFixedFont": (CODE_FONT_FAMILY, 10),
        }
        for name, font_value in named_fonts.items():
            try:
                tkfont.nametofont(name).configure(family=font_value[0], size=font_value[1])
                if len(font_value) > 2:
                    tkfont.nametofont(name).configure(weight=font_value[2])
            except tk.TclError:
                continue

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        header_font = (UI_FONT_FAMILY, 17, "bold")
        title_font = (UI_FONT_FAMILY, 12, "bold")
        body_font = (UI_FONT_FAMILY, 10)
        caption_font = (UI_FONT_FAMILY, 9)
        metric_value_font = (UI_FONT_FAMILY, 18, "bold")

        style.configure("Root.TFrame", background=WINDOW_BG)
        style.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("AltCard.TFrame", background=CARD_ALT_BG)
        style.configure("Section.TLabelframe", background=CARD_BG, foreground=TEXT, borderwidth=1, relief="solid")
        style.configure("Section.TLabelframe.Label", background=CARD_BG, foreground=TEXT, font=title_font)
        style.configure("Header.TLabel", background=WINDOW_BG, foreground=TEXT, font=header_font)
        style.configure("SubHeader.TLabel", background=WINDOW_BG, foreground=SUBTEXT, font=body_font)
        style.configure("Title.TLabel", background=CARD_BG, foreground=TEXT, font=title_font)
        style.configure("Body.TLabel", background=CARD_BG, foreground=TEXT, font=body_font)
        style.configure("Caption.TLabel", background=CARD_BG, foreground=SUBTEXT, font=caption_font)
        style.configure("MetricValue.TLabel", background=CARD_ALT_BG, foreground=TEXT, font=metric_value_font)
        style.configure("MetricLabel.TLabel", background=CARD_ALT_BG, foreground=SUBTEXT, font=caption_font)
        style.configure("Status.TLabel", background=SIDEBAR_BG, foreground=SUBTEXT, font=caption_font)
        style.configure(
            "Accent.TButton",
            font=(UI_FONT_FAMILY, 10, "bold"),
            padding=(self._px(14), self._px(10)),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DEEP), ("!disabled", ACCENT)],
            foreground=[("!disabled", "white")],
        )
        style.configure(
            "Secondary.TButton",
            font=(UI_FONT_FAMILY, 10),
            padding=(self._px(14), self._px(10)),
        )
        style.configure("Hint.TLabel", background=SIDEBAR_BG, foreground=ACCENT_DEEP, font=caption_font)

    def _set_initial_geometry(self) -> None:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(max(int(screen_width * 0.70), 1480), screen_width - self._px(80))
        height = min(max(int(screen_height * 0.80), 900), screen_height - self._px(96))
        x_position = max((screen_width - width) // 2, self._px(32))
        y_position = max((screen_height - height) // 2, self._px(24))
        self.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=self._pad(24, 24, 18, 24))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.columnconfigure(0, weight=1)

        preview = ttk.Frame(self, style="Root.TFrame", padding=self._pad(0, 24, 24, 24))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(2, weight=1)
        preview.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text="铁轨曲线提取器", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            sidebar,
            text="先预览点云和中心线，再决定是否导出 Isaac Sim / Omniverse 可用的 USD Curve。",
            style="SubHeader.TLabel",
            wraplength=self._px(360),
            justify="left",
        ).grid(row=1, column=0, sticky="we", pady=(self._px(8), self._px(16)))
        ttk.Label(
            sidebar,
            text="悬停在参数标签或输入框上，会显示该参数代表的含义。",
            style="Hint.TLabel",
            wraplength=self._px(360),
            justify="left",
        ).grid(row=2, column=0, sticky="we", pady=(0, self._px(18)))

        self._build_file_panel(sidebar, row=3)
        self._build_parameter_panel(sidebar, row=4)
        self._build_roi_panel(sidebar, row=5)
        self._build_action_panel(sidebar, row=6)
        self._build_log_panel(sidebar, row=7)

        ttk.Label(preview, text="导出前预览", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            preview,
            text="俯视图用于确认轨道提取是否偏到道床或旁边结构，纵断面用于确认曲线高程是否连续平顺。",
            style="SubHeader.TLabel",
            wraplength=self._px(920),
            justify="left",
        ).grid(row=1, column=0, sticky="we", pady=(self._px(8), self._px(16)))

        self._build_metrics_bar(preview, row=2)
        self._build_preview_panels(preview, start_row=3)

    def _build_metrics_bar(self, parent: ttk.Frame, row: int) -> None:
        bar = ttk.Frame(parent, style="Root.TFrame")
        bar.grid(row=row, column=0, sticky="we", pady=(0, self._px(16)))
        for index in range(4):
            bar.columnconfigure(index, weight=1)

        cards = (
            ("输入点数", self.metric_vars["input_points"]),
            ("过滤后点数", self.metric_vars["filtered_points"]),
            ("中心线点数", self.metric_vars["centerline_points"]),
            ("曲线长度 (m)", self.metric_vars["curve_length_m"]),
        )
        for column, (label, variable) in enumerate(cards):
            frame = ttk.Frame(bar, style="AltCard.TFrame", padding=self._pad(16, 14, 16, 14))
            frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else self._px(10), 0))
            frame.columnconfigure(0, weight=1)
            ttk.Label(frame, text=label, style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(frame, textvariable=variable, style="MetricValue.TLabel").grid(
                row=1, column=0, sticky="w", pady=(self._px(8), 0)
            )

    def _build_preview_panels(self, parent: ttk.Frame, start_row: int) -> None:
        top_panel = ttk.Frame(parent, style="Card.TFrame", padding=self._pad(16, 14, 16, 16))
        top_panel.grid(row=start_row, column=0, sticky="nsew")
        top_panel.columnconfigure(0, weight=1)
        top_panel.rowconfigure(1, weight=1)
        ttk.Label(top_panel, text="俯视图预览", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            top_panel,
            text="灰色为过滤点，橙色为轨道候选点，绿色为导出中心线。",
            style="Caption.TLabel",
            wraplength=self._px(900),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(self._px(2), self._px(10)))
        self.top_canvas = tk.Canvas(
            top_panel,
            background=CANVAS_BG,
            highlightbackground=EDGE,
            highlightthickness=1,
            relief="flat",
        )
        self.top_canvas.grid(row=2, column=0, sticky="nsew")
        self.top_canvas.bind("<Configure>", lambda _event: self._schedule_redraw())

        profile_panel = ttk.Frame(parent, style="Card.TFrame", padding=self._pad(16, 14, 16, 16))
        profile_panel.grid(row=start_row + 1, column=0, sticky="nsew", pady=(self._px(16), 0))
        profile_panel.columnconfigure(0, weight=1)
        profile_panel.rowconfigure(2, weight=1)
        ttk.Label(profile_panel, text="纵断面预览", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            profile_panel,
            text="横轴是累计里程，纵轴是高程。它主要用来判断导出的曲线是否连续、是否有异常跳点。",
            style="Caption.TLabel",
            wraplength=self._px(900),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(self._px(2), self._px(10)))
        self.profile_canvas = tk.Canvas(
            profile_panel,
            background=CANVAS_BG,
            highlightbackground=EDGE,
            highlightthickness=1,
            relief="flat",
        )
        self.profile_canvas.grid(row=2, column=0, sticky="nsew")
        self.profile_canvas.bind("<Configure>", lambda _event: self._schedule_redraw())

        summary_panel = ttk.Frame(parent, style="Card.TFrame", padding=self._pad(16, 14, 16, 14))
        summary_panel.grid(row=start_row + 2, column=0, sticky="ew", pady=(self._px(16), 0))
        summary_panel.columnconfigure(0, weight=1)
        ttk.Label(summary_panel, text="摘要", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            summary_panel,
            textvariable=self.summary_var,
            style="Caption.TLabel",
            justify="left",
            wraplength=self._px(900),
        ).grid(row=1, column=0, sticky="we", pady=(self._px(8), 0))

    def _build_file_panel(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="输入输出", style="Section.TLabelframe", padding=self._pad(14, 12, 14, 14))
        panel.grid(row=row, column=0, sticky="we")
        panel.columnconfigure(0, weight=1)

        self._add_path_row(panel, 0, "点云文件", self.input_var, self._browse_input_file, PATH_FIELD_HELP["input_path"])
        self._add_path_row(panel, 1, "输出目录", self.output_var, self._browse_output_dir, PATH_FIELD_HELP["output_path"])
        self._add_path_row(panel, 2, "配置文件", self.config_path_var, self._browse_config_file, PATH_FIELD_HELP["config_path"])

        button_row = ttk.Frame(panel, style="Card.TFrame")
        button_row.grid(row=3, column=0, sticky="we", pady=(self._px(10), 0))
        button_row.columnconfigure(0, weight=1)
        ttk.Button(button_row, text="加载配置", style="Secondary.TButton", command=self._load_config_file).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(button_row, text="另存参数", style="Secondary.TButton", command=self._save_current_config).grid(
            row=0, column=1, sticky="e"
        )

    def _build_parameter_panel(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="常用参数", style="Section.TLabelframe", padding=self._pad(14, 12, 14, 14))
        panel.grid(row=row, column=0, sticky="we", pady=(self._px(16), 0))
        panel.columnconfigure(1, weight=1)

        check = ttk.Checkbutton(
            panel,
            text="启用高度过滤",
            variable=self.height_filter_enabled_var,
            command=self._clear_previous_result,
        )
        check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, self._px(10)))
        self.tooltip.bind(
            check,
            "启用后会先按整体高程过滤掉较低部分的点云，通常能减轻道床和地面杂点对轨顶检测的干扰。",
        )

        for index, field in enumerate(COMMON_FIELDS, start=1):
            label = ttk.Label(panel, text=field.label, style="Body.TLabel")
            label.grid(row=index, column=0, sticky="w", pady=(self._px(5), self._px(5)))
            variable = tk.StringVar()
            entry = ttk.Entry(panel, textvariable=variable)
            entry.grid(row=index, column=1, sticky="we", pady=(self._px(5), self._px(5)), padx=(self._px(12), 0))
            entry.bind("<KeyRelease>", lambda _event: self._clear_previous_result())
            self.field_vars[field.path] = variable
            self._bind_field_tooltip(label, entry, field)

    def _build_roi_panel(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="ROI 限制（可留空）", style="Section.TLabelframe", padding=self._pad(14, 12, 14, 14))
        panel.grid(row=row, column=0, sticky="we", pady=(self._px(16), 0))
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(3, weight=1)

        for index, field in enumerate(ROI_FIELDS):
            target_row = index // 2
            target_col = (index % 2) * 2
            label = ttk.Label(panel, text=field.label, style="Body.TLabel")
            label.grid(row=target_row, column=target_col, sticky="w", pady=(self._px(5), self._px(5)))
            variable = tk.StringVar()
            entry = ttk.Entry(panel, textvariable=variable, width=10)
            entry.grid(
                row=target_row,
                column=target_col + 1,
                sticky="we",
                pady=(self._px(5), self._px(5)),
                padx=(self._px(8), self._px(14)),
            )
            entry.bind("<KeyRelease>", lambda _event: self._clear_previous_result())
            self.field_vars[field.path] = variable
            self._bind_field_tooltip(label, entry, field)

    def _build_action_panel(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.Frame(parent, style="Sidebar.TFrame")
        panel.grid(row=row, column=0, sticky="we", pady=(self._px(20), 0))
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        preview_button = ttk.Button(panel, text="生成预览", style="Accent.TButton", command=self._start_analysis)
        preview_button.grid(row=0, column=0, sticky="we", padx=(0, self._px(8)))
        self.export_button = ttk.Button(panel, text="导出 USD Curve", style="Secondary.TButton", command=self._export_result)
        self.export_button.grid(row=0, column=1, sticky="we", padx=(self._px(8), 0))

        self.tooltip.bind(preview_button, "先分析点云并在界面里显示轨道候选点和中心线。只有确认没问题后再导出。")
        self.tooltip.bind(self.export_button, "把当前预览结果导出为 XYZ 点集、USD Curve、摘要和实际使用的参数文件。")

        ttk.Label(
            parent,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=self._px(360),
            justify="left",
        ).grid(row=row + 1, column=0, sticky="we", pady=(self._px(10), 0))

    def _build_log_panel(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="运行日志", style="Section.TLabelframe", padding=self._pad(12, 10, 12, 12))
        panel.grid(row=row, column=0, sticky="nsew", pady=(self._px(16), 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            panel,
            height=10,
            background=CANVAS_BG,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            wrap="word",
            borderwidth=0,
            padx=self._px(10),
            pady=self._px(10),
            font=(CODE_FONT_FAMILY, 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.insert("1.0", "等待输入点云。\n")
        self.log_text.configure(state="disabled")

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label_text: str,
        variable: tk.StringVar,
        callback: Callable[[], None],
        help_text: str,
    ) -> None:
        container = ttk.Frame(parent, style="Card.TFrame")
        container.grid(row=row, column=0, sticky="we", pady=(0, self._px(10)))
        container.columnconfigure(0, weight=1)

        label = ttk.Label(container, text=label_text, style="Body.TLabel")
        label.grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(container, textvariable=variable)
        entry.grid(row=1, column=0, sticky="we", pady=(self._px(5), 0))
        entry.bind("<KeyRelease>", lambda _event: self._clear_previous_result())
        button = ttk.Button(container, text="浏览", style="Secondary.TButton", command=callback)
        button.grid(row=1, column=1, sticky="e", padx=(self._px(10), 0), pady=(self._px(5), 0))

        self.tooltip.bind(label, help_text)
        self.tooltip.bind(entry, help_text)
        self.tooltip.bind(button, help_text)

    def _bind_field_tooltip(self, label: ttk.Label, entry: ttk.Entry, field: FieldSpec) -> None:
        tooltip_text = f"{field.label}\n\n{field.description}"
        self.tooltip.bind(label, tooltip_text)
        self.tooltip.bind(entry, tooltip_text)

    def _browse_input_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择点云文件",
            filetypes=[
                ("点云文件", "*.las *.laz *.csv *.txt *.xyz *.npy"),
                ("全部文件", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            self._clear_previous_result()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def _browse_config_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON 文件", "*.json"), ("全部文件", "*.*")],
        )
        if path:
            self.config_path_var.set(path)

    def _load_config_file(self) -> None:
        config_path_text = self.config_path_var.get().strip()
        if not config_path_text:
            messagebox.showwarning("缺少配置文件", "请先选择一个 JSON 配置文件。")
            return

        config_path = Path(config_path_text)
        if not config_path.exists():
            messagebox.showerror("文件不存在", f"配置文件不存在：\n{config_path}")
            return

        with config_path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        config = prepare_config(overrides=loaded)
        self._apply_config_to_form(config)
        self._log(f"已加载配置：{config_path}")
        self._clear_previous_result()

    def _save_current_config(self) -> None:
        try:
            config = self._collect_config_from_form()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="保存当前参数",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path:
            return

        target = Path(path)
        with target.open("w", encoding="utf-8") as file_handle:
            json.dump(config, file_handle, ensure_ascii=False, indent=2)
        self.config_path_var.set(str(target))
        self._log(f"已保存参数：{target}")

    def _apply_config_to_form(self, config: dict[str, Any]) -> None:
        self.height_filter_enabled_var.set(bool(config["height_filter"]["enabled"]))
        for field in COMMON_FIELDS + ROI_FIELDS:
            value = _get_value_from_path(config, field.path)
            variable = self.field_vars[field.path]
            variable.set("" if value is None else str(value))

    def _collect_config_from_form(self) -> dict[str, Any]:
        config = prepare_config()
        config["height_filter"]["enabled"] = bool(self.height_filter_enabled_var.get())

        for field in COMMON_FIELDS + ROI_FIELDS:
            raw_value = self.field_vars[field.path].get().strip()
            if raw_value == "":
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

        return config

    def _start_analysis(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return

        input_path_text = self.input_var.get().strip()
        if not input_path_text:
            messagebox.showwarning("缺少输入", "请先选择一个点云文件。")
            return

        input_path = Path(input_path_text)
        if not input_path.exists():
            messagebox.showerror("文件不存在", f"点云文件不存在：\n{input_path}")
            return

        try:
            config = self._collect_config_from_form()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self.result = None
        self.status_var.set("正在分析点云并生成预览…")
        self.summary_var.set("正在计算预览，请稍候。")
        self._set_metric_defaults()
        self._set_busy(True)
        self._draw_placeholders()
        self._log(f"开始分析：{input_path}")

        def worker() -> None:
            try:
                result = analyze_input(input_path=input_path, config_overrides=config)
            except Exception as exc:  # pragma: no cover - worker result handled by GUI thread
                self.worker_queue.put(("error", str(exc)))
                return
            self.worker_queue.put(("result", result))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_worker_queue(self) -> None:
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_worker_queue)
            return

        if kind == "error":
            self.status_var.set("分析失败。")
            self.summary_var.set(str(payload))
            self._log(f"分析失败：{payload}")
            self._set_busy(False)
            messagebox.showerror("分析失败", str(payload))
        else:
            self.result = payload
            self.status_var.set("预览已生成，可以检查后再导出。")
            self._update_summary()
            self._set_busy(False)
            self._log("分析完成，已生成预览。")
            self._schedule_redraw()

        self.after(100, self._poll_worker_queue)

    def _export_result(self) -> None:
        if self.result is None:
            messagebox.showwarning("没有可导出的结果", "请先生成预览。")
            return

        output_text = self.output_var.get().strip()
        if not output_text:
            output_text = filedialog.askdirectory(title="选择输出目录")
            if not output_text:
                return
            self.output_var.set(output_text)

        output_dir = Path(output_text)
        input_path = Path(self.input_var.get().strip()) if self.input_var.get().strip() else None
        config_path = Path(self.config_path_var.get().strip()) if self.config_path_var.get().strip() else None

        try:
            self.result.config = self._collect_config_from_form()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        export_pipeline_result(
            result=self.result,
            output_dir=output_dir,
            input_path=input_path,
            config_path=config_path,
        )
        self.status_var.set("导出完成。")
        self._update_summary()
        self._log(f"已导出到：{output_dir}")
        messagebox.showinfo("导出完成", f"已导出到：\n{output_dir}")

    def _set_busy(self, busy: bool) -> None:
        if self.export_button is None:
            return
        if busy:
            self.export_button.configure(state="disabled")
        else:
            self.export_button.configure(state="normal" if self.result is not None else "disabled")

    def _clear_previous_result(self) -> None:
        self.result = None
        self.status_var.set("参数或输入已变更，请重新生成预览。")
        self.summary_var.set("参数或输入已变更，请重新生成预览。")
        self._set_metric_defaults()
        if self.export_button is not None:
            self.export_button.configure(state="disabled")
        self._schedule_redraw()

    def _set_metric_defaults(self) -> None:
        for variable in self.metric_vars.values():
            variable.set("--")

    def _update_summary(self) -> None:
        if self.result is None:
            self.summary_var.set("尚未生成预览。")
            self._set_metric_defaults()
            return

        summary = self.result.summary
        self.metric_vars["input_points"].set(f"{summary['input_points']:,}")
        self.metric_vars["filtered_points"].set(f"{summary['filtered_points']:,}")
        self.metric_vars["centerline_points"].set(f"{summary['centerline_points']:,}")
        self.metric_vars["curve_length_m"].set(f"{summary['curve_length_m']:.2f}")
        self.summary_var.set(
            "输入点数：{input_points:,} | 过滤后点数：{filtered_points:,} | 轨道候选点：{rail_points:,} | "
            "中心线点数：{centerline_points:,} | 曲线长度：{curve_length_m:.2f} m".format(**summary)
        )

    def _schedule_redraw(self) -> None:
        if self._redraw_after_id is not None:
            self.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.after(60, self._redraw_preview)

    def _redraw_preview(self) -> None:
        self._redraw_after_id = None
        if self.top_canvas is None or self.profile_canvas is None:
            return
        self._draw_top_view()
        self._draw_profile_view()

    def _draw_placeholders(self) -> None:
        placeholders = (
            (self.top_canvas, "等待预览数据"),
            (self.profile_canvas, "等待预览数据"),
        )
        for canvas, text in placeholders:
            if canvas is None:
                continue
            canvas.delete("all")
            width = max(canvas.winfo_width(), self._px(400))
            height = max(canvas.winfo_height(), self._px(240))
            canvas.create_rectangle(0, 0, width, height, fill=CANVAS_BG, outline="")
            canvas.create_text(
                width / 2,
                height / 2,
                text=text,
                fill=SUBTEXT,
                font=(UI_FONT_FAMILY, 14),
            )

    def _draw_top_view(self) -> None:
        if self.top_canvas is None:
            return
        canvas = self.top_canvas
        width = max(canvas.winfo_width(), self._px(400))
        height = max(canvas.winfo_height(), self._px(300))
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=CANVAS_BG, outline="")
        self._draw_grid(canvas, width, height)

        if self.result is None:
            canvas.create_text(width / 2, height / 2, text="生成预览后显示俯视图", fill=SUBTEXT, font=(UI_FONT_FAMILY, 13))
            return

        filtered_xy = downsample_points(self.result.filtered_points_world[:, :2], limit=6000)
        rail_xy = downsample_points(self.result.rail_points_world[:, :2], limit=4200)
        center_xy = self.result.centerline_world[:, :2]
        bounds = combine_bounds(filtered_xy, rail_xy, center_xy)

        filtered_draw = fit_points_to_canvas(filtered_xy, bounds, width, height, padding=self._px(28))
        rail_draw = fit_points_to_canvas(rail_xy, bounds, width, height, padding=self._px(28))
        center_draw = fit_points_to_canvas(center_xy, bounds, width, height, padding=self._px(28))

        self._draw_point_cloud(canvas, filtered_draw, color=MUTED, size=self._px(1.2))
        self._draw_point_cloud(canvas, rail_draw, color=RAIL, size=self._px(1.8))
        self._draw_polyline(canvas, center_draw, color=ACCENT, line_width=self._px(2.4))
        self._draw_legend(
            canvas,
            items=[
                ("过滤点", MUTED),
                ("轨道候选", RAIL),
                ("中心线", ACCENT),
            ],
        )

    def _draw_profile_view(self) -> None:
        if self.profile_canvas is None:
            return
        canvas = self.profile_canvas
        width = max(canvas.winfo_width(), self._px(400))
        height = max(canvas.winfo_height(), self._px(220))
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=CANVAS_BG, outline="")
        self._draw_grid(canvas, width, height)

        if self.result is None:
            canvas.create_text(width / 2, height / 2, text="生成预览后显示纵断面", fill=SUBTEXT, font=(UI_FONT_FAMILY, 13))
            return

        profile_points = build_profile_points(self.result.centerline_world)
        bounds = combine_bounds(profile_points)
        profile_draw = fit_points_to_canvas(profile_points, bounds, width, height, padding=self._px(28))
        self._draw_polyline(canvas, profile_draw, color=ACCENT, line_width=self._px(2.4))
        self._draw_legend(canvas, items=[("中心线纵断面", ACCENT)])

    def _draw_grid(self, canvas: tk.Canvas, width: int, height: int) -> None:
        for ratio in (0.2, 0.4, 0.6, 0.8):
            canvas.create_line(width * ratio, 0, width * ratio, height, fill=GRID, width=self._px(1))
            canvas.create_line(0, height * ratio, width, height * ratio, fill=GRID, width=self._px(1))

    def _draw_point_cloud(self, canvas: tk.Canvas, points: np.ndarray, color: str, size: int) -> None:
        radius = max(size, 1)
        for x_value, y_value in points:
            canvas.create_rectangle(
                x_value - radius,
                y_value - radius,
                x_value + radius,
                y_value + radius,
                fill=color,
                outline="",
            )

    def _draw_polyline(self, canvas: tk.Canvas, points: np.ndarray, color: str, line_width: int) -> None:
        if len(points) < 2:
            return
        canvas.create_line(*points.reshape(-1).tolist(), fill=color, width=line_width, smooth=False)

    def _draw_legend(self, canvas: tk.Canvas, items: list[tuple[str, str]]) -> None:
        x_origin = self._px(18)
        y_origin = self._px(18)
        box_size = self._px(13)
        text_offset = self._px(20)
        line_height = self._px(24)
        for index, (label, color) in enumerate(items):
            y_value = y_origin + index * line_height
            canvas.create_rectangle(x_origin, y_value, x_origin + box_size, y_value + box_size, fill=color, outline="")
            canvas.create_text(
                x_origin + text_offset,
                y_value + box_size / 2,
                text=label,
                anchor="w",
                fill=TEXT,
                font=(UI_FONT_FAMILY, 9),
            )

    def _log(self, message: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self.pixel_scale)))

    def _pad(self, left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
        return (self._px(left), self._px(top), self._px(right), self._px(bottom))


def _get_value_from_path(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        current = current[key]
    return current


def _set_value_on_path(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def main() -> int:
    app = RailCurveExtractorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
