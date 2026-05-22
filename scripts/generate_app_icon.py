from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui
from PIL import Image


CANVAS_SIZE = 1024


def build_icon_image(size: int = CANVAS_SIZE) -> QtGui.QImage:
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

    # Background
    background_rect = QtCore.QRectF(48, 48, size - 96, size - 96)
    background_gradient = QtGui.QLinearGradient(background_rect.topLeft(), background_rect.bottomRight())
    background_gradient.setColorAt(0.0, QtGui.QColor("#143b39"))
    background_gradient.setColorAt(0.42, QtGui.QColor("#1a6c60"))
    background_gradient.setColorAt(1.0, QtGui.QColor("#0f2625"))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(background_gradient)
    painter.drawRoundedRect(background_rect, 170, 170)

    # Subtle top highlight
    halo = QtGui.QRadialGradient(size * 0.28, size * 0.22, size * 0.62)
    halo.setColorAt(0.0, QtGui.QColor(255, 255, 255, 58))
    halo.setColorAt(0.5, QtGui.QColor(255, 255, 255, 10))
    halo.setColorAt(1.0, QtCore.Qt.GlobalColor.transparent)
    painter.setBrush(halo)
    painter.drawEllipse(background_rect)

    # Rails
    left_rail = QtGui.QPainterPath(QtCore.QPointF(size * 0.36, size * 0.88))
    left_rail.cubicTo(
        QtCore.QPointF(size * 0.32, size * 0.66),
        QtCore.QPointF(size * 0.33, size * 0.38),
        QtCore.QPointF(size * 0.41, size * 0.15),
    )
    right_rail = QtGui.QPainterPath(QtCore.QPointF(size * 0.64, size * 0.88))
    right_rail.cubicTo(
        QtCore.QPointF(size * 0.68, size * 0.66),
        QtCore.QPointF(size * 0.67, size * 0.38),
        QtCore.QPointF(size * 0.59, size * 0.15),
    )

    rail_pen = QtGui.QPen(QtGui.QColor("#f8f4ea"), size * 0.05, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(rail_pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(left_rail)
    painter.drawPath(right_rail)

    # Rail inner shade for depth
    inner_pen = QtGui.QPen(QtGui.QColor("#c6d6d3"), size * 0.016, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(inner_pen)
    painter.drawPath(left_rail)
    painter.drawPath(right_rail)

    # Sleepers
    sleeper_pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
    sleeper_gradient = QtGui.QLinearGradient(0, size * 0.46, 0, size * 0.83)
    sleeper_gradient.setColorAt(0.0, QtGui.QColor("#a4bbb6"))
    sleeper_gradient.setColorAt(1.0, QtGui.QColor("#d8e5e2"))
    painter.setPen(sleeper_pen)
    painter.setBrush(sleeper_gradient)
    for t in (0.20, 0.34, 0.48, 0.62, 0.75):
        point_left = left_rail.pointAtPercent(t)
        point_right = right_rail.pointAtPercent(t)
        angle = left_rail.angleAtPercent(t)
        sleeper = QtGui.QPainterPath()
        sleeper_rect = QtCore.QRectF(-size * 0.07, -size * 0.018, size * 0.14, size * 0.036)
        sleeper.addRoundedRect(sleeper_rect, 12, 12)
        transform = QtGui.QTransform()
        transform.translate((point_left.x() + point_right.x()) / 2.0, (point_left.y() + point_right.y()) / 2.0)
        transform.rotate(-angle)
        painter.drawPath(transform.map(sleeper))

    # Centerline curve
    centerline = QtGui.QPainterPath(QtCore.QPointF(size * 0.49, size * 0.85))
    centerline.cubicTo(
        QtCore.QPointF(size * 0.51, size * 0.66),
        QtCore.QPointF(size * 0.55, size * 0.44),
        QtCore.QPointF(size * 0.56, size * 0.19),
    )
    glow_pen = QtGui.QPen(QtGui.QColor(255, 186, 77, 70), size * 0.06, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(glow_pen)
    painter.drawPath(centerline)
    center_pen = QtGui.QPen(QtGui.QColor("#ffb84d"), size * 0.022, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(center_pen)
    painter.drawPath(centerline)

    # Lidar signal arcs
    painter.setPen(QtGui.QPen(QtGui.QColor("#ffb84d"), size * 0.014))
    arc_rect = QtCore.QRectF(size * 0.63, size * 0.16, size * 0.18, size * 0.18)
    for start_deg, span_deg, scale in ((20, 70, 1.00), (12, 56, 0.78), (6, 42, 0.56)):
        rect = QtCore.QRectF(
            arc_rect.center().x() - arc_rect.width() * scale / 2.0,
            arc_rect.center().y() - arc_rect.height() * scale / 2.0,
            arc_rect.width() * scale,
            arc_rect.height() * scale,
        )
        painter.drawArc(rect, start_deg * 16, span_deg * 16)

    # Sensor dot
    painter.setBrush(QtGui.QColor("#ffb84d"))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawEllipse(QtCore.QPointF(size * 0.66, size * 0.24), size * 0.018, size * 0.018)

    painter.end()
    return image


def export_assets(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_1024 = build_icon_image(1024)
    image_64 = build_icon_image(64)
    image_32 = build_icon_image(32)

    png_path = output_dir / "app_icon.png"
    png64_path = output_dir / "app_icon_64.png"
    png32_path = output_dir / "app_icon_32.png"
    ico_path = output_dir / "app_icon.ico"

    if not image_1024.save(str(png_path), "PNG"):
        raise RuntimeError("Failed to save PNG icon.")
    if not image_64.save(str(png64_path), "PNG"):
        raise RuntimeError("Failed to save 64px PNG icon.")
    if not image_32.save(str(png32_path), "PNG"):
        raise RuntimeError("Failed to save 32px PNG icon.")

    base_image = Image.open(png_path).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base_image.save(ico_path, format="ICO", sizes=sizes)

    return [png_path, png64_path, png32_path, ico_path]


def main() -> int:
    output_dir = Path(__file__).resolve().parents[1] / "assets"
    saved = export_assets(output_dir)
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
