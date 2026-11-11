#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RVC STUDIO · PySide6 客户端（靛蓝瓷设计 token，已接 rvc_engine）。

设计 token 来自歸藏 guizang-design：
  paper #f1f3f5 / ink #0a1f3d / accent #315d93 / line #d8dde3
"""
import glob
import json

import numpy as np
import os
import re
import sys
import tempfile
import time

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtCore import QPointF
from PySide6.QtGui import (
    QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

def _detect_rvc_dir():
    """优先 RVC_DIR 环境变量，其次脚本所在目录/上一级（需含 assets 与 configs）。"""
    env = os.environ.get("RVC_DIR", "").strip()
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(candidate, "assets")) and os.path.isdir(
            os.path.join(candidate, "configs")
        ):
            return candidate
    return here


RVC_DIR = _detect_rvc_dir()
WEIGHTS_DIR = os.path.join(RVC_DIR, "assets", "weights")
LOGS_DIR = os.path.join(RVC_DIR, "logs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "rvc_studio_qt_settings.json")
BUFFER_KEYS = {"block_time", "crossfade_length", "extra_time"}
if RVC_DIR not in sys.path:
    sys.path.insert(0, RVC_DIR)

PAPER = "#f1f3f5"
INK = "#0a1f3d"
ACCENT = "#315d93"
ACCENT_DARK = "#274c7a"
LINE = "#d8dde3"
TRACK = "#e6eaf0"
SECOND = "#5a6b80"
HELPER = "#8a97a8"
WHITE = "#ffffff"
GREEN = "#1f7a4d"

QSS = f"""
QWidget {{
    background: {PAPER};
    color: {INK};
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 12px;
}}
QLabel {{ background: transparent; }}
QFrame#card {{
    background: {WHITE};
    border: 1px solid {LINE};
    border-radius: 12px;
}}
QLabel#cardTitle {{
    color: {SECOND};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#fieldLabel {{ color: {SECOND}; font-size: 12px; }}
QLabel#valueLabel {{ color: {ACCENT}; font-size: 12px; font-weight: 600; }}
QLineEdit#valueEdit {{ color: {ACCENT}; font-size: 12px; font-weight: 600; background: transparent; border: none; padding: 0; }}
QLabel#chip {{
    background: {WHITE};
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 7px 16px;
    color: {SECOND};
    font-size: 12px;
}}
QLabel#hint {{ color: {HELPER}; font-size: 11px; }}
QLineEdit, QComboBox {{
    background: {WHITE};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 7px 10px;
    color: {INK};
    selection-background-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {WHITE};
    border: 1px solid {LINE};
    selection-background-color: #e6ecf4;
    color: {INK};
}}
QPushButton#primary {{
    background: {ACCENT};
    color: {PAPER};
    border: none;
    border-radius: 10px;
    padding: 12px 30px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primary:disabled {{ background: #c3ccd8; color: {WHITE}; }}
QPushButton#ghost {{
    background: {WHITE};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 7px 16px;
}}
QPushButton#ghost:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#secondary {{
    background: {WHITE};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 12px 30px;
    font-size: 13px;
}}
QPushButton#secondary:hover {{ border-color: {ACCENT}; }}
QPushButton#secondary:disabled {{ color: {HELPER}; }}
QCheckBox, QRadioButton {{ background: transparent; color: {INK}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {LINE};
    border-radius: 3px;
    background: {WHITE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url(__TICK__);
}}
QPlainTextEdit {{
    background: {WHITE};
    border: 1px solid {LINE};
    border-radius: 10px;
    color: {SECOND};
    font-family: "Consolas", "Microsoft YaHei UI";
    font-size: 11px;
}}
"""


class DesignSlider(QSlider):
    """扁平轨道 + 强调色填充 + 描边圆钮；bipolar=True 时从中心填充。"""

    def __init__(self, lo, hi, val, bipolar=False):
        super().__init__(Qt.Horizontal)
        self.setRange(lo, hi)
        self.setValue(val)
        self.bipolar = bipolar
        self._hover = False
        self.setFixedHeight(26)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception:
            pass

    def _paint(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cy = h / 2.0
        left, right = 11.0, w - 11.0
        span = max(1.0, right - left)
        rng = max(1, self.maximum() - self.minimum())
        ratio = (self.value() - self.minimum()) / rng
        pos = left + span * ratio

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(TRACK))
        p.drawRoundedRect(int(left), int(cy - 2), int(span), 4, 2, 2)

        start = left + span / 2.0 if self.bipolar else left
        p.setBrush(QColor(ACCENT))
        if abs(pos - start) > 0.5:
            x0, x1 = sorted((start, pos))
            p.drawRoundedRect(int(x0), int(cy - 2), int(x1 - x0), 4, 2, 2)

        radius = 7.5
        border = QPen(QColor(ACCENT_DARK if self._hover else ACCENT))
        border.setWidthF(2.0)
        p.setPen(border)
        p.setBrush(QColor(WHITE))
        p.drawEllipse(int(pos - radius), int(cy - radius), int(radius * 2), int(radius * 2))


def make_tick_icon(path):
    from PySide6.QtCore import QPointF

    img = QImage(14, 14, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(WHITE))
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QPointF(3.0, 7.4), QPointF(6.0, 10.4), QPointF(11.0, 4.4)])
    p.end()
    img.save(path)
    return path


def card(title):
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 14, 18, 16)
    lay.setSpacing(9)
    lab = QLabel(title)
    lab.setObjectName("cardTitle")
    lay.addWidget(lab)
    return frame, lay


def field_row(label, widget):
    row = QHBoxLayout()
    row.setSpacing(12)
    lab = QLabel(label)
    lab.setObjectName("fieldLabel")
    lab.setFixedWidth(78)
    row.addWidget(lab)
    row.addWidget(widget, 1)
    return row


class WaveformWidget(QWidget):
    """滚动波形：镜像填充 + 自适应增益 + 网格。"""

    def __init__(self, color):
        super().__init__()
        self.color = QColor(color)
        self.blocks = []
        self.max_blocks = 8
        self.enabled = False
        self.setMinimumHeight(58)

    def push(self, arr):
        try:
            a = np.asarray(arr, dtype=np.float32).ravel()
        except Exception:
            return
        if a.size == 0:
            return
        self.blocks.append(a.copy())
        if len(self.blocks) > self.max_blocks:
            self.blocks.pop(0)
        self.update()

    def clear(self):
        self.blocks = []
        self.update()

    def paintEvent(self, _event):
        try:
            self._paint()
        except Exception:
            pass

    def _paint(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return

        clip = QPainterPath()
        clip.addRoundedRect(0, 0, w, h, 9, 9)
        p.setClipPath(clip)
        p.fillRect(0, 0, w, h, QColor(WHITE))

        p.setPen(QPen(QColor("#eef1f6"), 1))
        for i in (1, 3):
            y = int(h * i / 4)
            p.drawLine(0, y, w, y)
        p.setPen(QPen(QColor("#f3f6f9"), 1))
        for i in (1, 2, 3):
            x = int(w * i / 4)
            p.drawLine(x, 0, x, h)

        p.setPen(QPen(QColor("#dde3ea"), 1))
        p.drawLine(0, h // 2, w, h // 2)

        if not self.enabled or not self.blocks:
            self._hint(p, w, h, "波形已关闭 · 性能模式" if not self.enabled else "等待音频…")
            return

        data = np.concatenate(self.blocks)
        if data.size == 0:
            self._hint(p, w, h, "等待音频…")
            return
        peak = float(np.abs(data).max())
        if peak < 1e-5:
            self._hint(p, w, h, "无输入信号")
            return

        gain = min(2000.0, 0.92 / max(peak, 1e-6))
        n = data.size
        cols = max(2, w - 2)
        step = max(1, n // cols)
        mid = h / 2.0
        amp = (h - 14) / 2.0

        tops, bottoms = [], []
        for x in range(cols):
            seg = data[x * step:(x + 1) * step]
            if seg.size == 0:
                continue
            hi = float(seg.max()) * gain
            lo = float(seg.min()) * gain
            hi = max(-1.0, min(1.0, hi))
            lo = max(-1.0, min(1.0, lo))
            tops.append((x + 1, mid - hi * amp))
            bottoms.append((x + 1, mid - lo * amp))
        if len(tops) < 2:
            return

        poly = QPolygonF(
            [QPointF(x, y) for x, y in tops]
            + [QPointF(x, y) for x, y in reversed(bottoms)]
        )
        fill = QColor(self.color)
        fill.setAlpha(62)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPolygon(poly)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self.color, 1.4))
        p.drawPolyline([QPointF(x, y) for x, y in tops])
        p.drawPolyline([QPointF(x, y) for x, y in bottoms])

        p.setPen(QColor(HELPER))
        p.setFont(QFont("Microsoft YaHei UI", 7))
        if gain > 1.5:
            p.drawText(self.rect().adjusted(8, 0, 0, -3), Qt.AlignLeft | Qt.AlignBottom,
                       "×%d" % int(round(gain)))
        secs = len(self.blocks) * 0.3
        p.drawText(self.rect().adjusted(0, 0, -8, -3), Qt.AlignRight | Qt.AlignBottom,
                   "%.1f s" % secs)

    def _hint(self, p, w, h, text):
        p.setPen(QColor(HELPER))
        p.setFont(QFont("Microsoft YaHei UI", 8))
        p.drawText(self.rect(), Qt.AlignCenter, text)


class EngineLoader(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            import rvc_engine

            engine = rvc_engine.VoiceEngine()
            self.ready.emit(engine)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


class EngineWorker(QThread):
    log = Signal(str)
    ready = Signal()
    failed = Signal(str)

    def __init__(self, engine, values):
        super().__init__()
        self.engine = engine
        self.values = values

    def run(self):
        try:
            self.engine.on_log(self.log.emit)
            if not self.engine.set_values(self.values):
                self.failed.emit("参数校验失败")
                return
            self.engine.start_vc()
            self.ready.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


class Studio(QMainWindow):
    engine_error = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC STUDIO v2.5")
        self.resize(1280, 960)
        self.running = False
        self.engine = None
        self.worker = None
        self.sliders = {}
        self.value_edits = {}

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 16)
        outer.setSpacing(12)

        outer.addLayout(self._header())
        outer.addLayout(self._body(), 1)
        outer.addWidget(self._wave_card())
        outer.addWidget(self._log_card())
        outer.addLayout(self._controls())
        outer.addWidget(self._footer())

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._poll_status)
        self.wave_timer = QTimer(self)
        self.wave_timer.setInterval(60)
        self.wave_timer.timeout.connect(self._poll_wave)
        self.refresh_devices()
        self._load_settings()

        self.engine_error.connect(self._handle_engine_error)
        self.btn_start.setEnabled(False)
        self.append_log("正在加载音频引擎…（首次约 5–10 秒）")
        self.engine_loader = EngineLoader(self)
        self.engine_loader.ready.connect(self._on_engine_ready)
        self.engine_loader.failed.connect(self._on_engine_failed)
        self.engine_loader.start()

    def _on_engine_ready(self, engine):
        self.engine = engine
        engine.on_log(self.append_log)
        engine.on_error(self._on_engine_error)
        self.btn_start.setEnabled(True)
        self.append_log("引擎就绪")

    def _on_engine_failed(self, message):
        self.append_log("引擎加载失败：%s" % message)
        QMessageBox.critical(self, "引擎加载失败", message)

    def _on_engine_error(self, message, is_cuda):
        # 引擎的声卡回调在 PortAudio 线程里触发，用信号切回界面线程
        self.engine_error.emit(str(message), bool(is_cuda))

    def _handle_engine_error(self, message, is_cuda):
        self.running = False
        self.status_timer.stop()
        self.wave_timer.stop()
        self.chip_status.setText("●  已停止")
        self.chip_status.setStyleSheet(
            "background: %s; border: 1px solid %s; border-radius: 16px;"
            " padding: 7px 16px; color: %s; font-size: 12px;" % (WHITE, LINE, SECOND))
        self.chip_infer.setText("推理 0 ms")
        self.btn_start.setEnabled(self.engine is not None)
        self.btn_stop.setEnabled(False)
        self.append_log("音频回调异常：%s" % message)
        crash_log = os.path.join(RVC_DIR, "logs", "realtime_crash.log")
        self.append_log("详细堆栈：%s" % crash_log)
        if is_cuda:
            QMessageBox.critical(
                self, "GPU 出错（音频已自动停止）",
                "声卡回调里检测到 CUDA 错误：%s\n\n"
                "GPU 上下文已经失效，本进程无法继续推理。\n"
                "请完全关闭 RVC Studio 再重新打开（不要只点开始）。\n\n"
                "完整堆栈已写入：\n%s\n\n"
                "如果反复出现，可用 go-rvc-studio-qt-safe.bat 启动（关闭 CUDA Graph）做对照排查。"
                % (message, crash_log))
        else:
            QMessageBox.warning(
                self, "音频已停止",
                "声卡回调异常：%s\n\n已自动停止音频。可以先直接重新点击“开始音频”。\n"
                "完整堆栈已写入：\n%s" % (message, crash_log))

    # ---------- header ----------
    def _header(self):
        row = QHBoxLayout()
        row.setSpacing(16)
        bar = QFrame()
        bar.setFixedSize(4, 46)
        bar.setStyleSheet("background: %s; border-radius: 2px;" % ACCENT)
        row.addWidget(bar)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        t = QLabel("RVC STUDIO")
        t.setStyleSheet("color: %s; font-size: 26px; font-weight: 700; letter-spacing: 1px;" % INK)
        s = QLabel("REAL-TIME VOICE CONVERTER   ·   v2.5")
        s.setStyleSheet("color: %s; font-size: 10px; letter-spacing: 2px;" % SECOND)
        titles.addWidget(t)
        titles.addWidget(s)
        row.addLayout(titles)
        row.addStretch(1)

        self.chip_status = QLabel("●  待机")
        self.chip_sr = QLabel("— Hz")
        self.chip_delay = QLabel("延迟 0 ms")
        self.chip_infer = QLabel("推理 0 ms")
        for c in (self.chip_status, self.chip_sr, self.chip_delay, self.chip_infer):
            c.setObjectName("chip")
            row.addWidget(c)
        return row

    # ---------- body ----------
    def _body(self):
        body = QHBoxLayout()
        body.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(self._model_card())
        left.addWidget(self._device_card())
        left.addWidget(self._status_card())
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self._params_card())
        right.addWidget(self._perf_card())
        right.addStretch(1)

        body.addLayout(left, 3)
        body.addLayout(right, 2)
        return body

    def _model_card(self):
        f, lay = card("加载模型")
        head = QHBoxLayout()
        self.chk_ckpt = QCheckBox("显示训练检查点")
        self.chk_ckpt.toggled.connect(lambda _v: self._fill_models())
        head.addWidget(self.chk_ckpt)
        head.addStretch(1)
        lay.addLayout(head)

        self.pth_combo = QComboBox()
        self.pth_combo.setEditable(True)
        self.pth_combo.currentTextChanged.connect(self._on_model_changed)
        btn1 = QPushButton("选择 .pth")
        btn1.setObjectName("ghost")
        btn1.clicked.connect(self._browse_pth)
        self.index_combo = QComboBox()
        self.index_combo.setEditable(True)
        self.index_combo.currentTextChanged.connect(
            lambda _t: self._on_model_changed(self.pth_combo.currentText()))
        btn2 = QPushButton("选择 .index")
        btn2.setObjectName("ghost")
        btn2.clicked.connect(self._browse_index)
        for combo, btn in ((self.pth_combo, btn1), (self.index_combo, btn2)):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(combo, 1)
            row.addWidget(btn)
            lay.addLayout(row)
        return f

    def _browse_pth(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 .pth 模型", WEIGHTS_DIR, "RVC 模型 (*.pth)")
        if not path:
            return
        path = os.path.normpath(path)
        name = os.path.basename(path) if path.startswith(os.path.normpath(WEIGHTS_DIR)) else path
        if self.pth_combo.findText(name) < 0:
            self.pth_combo.addItem(name)
        self.pth_combo.setCurrentText(name)

    def _browse_index(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 .index 索引", LOGS_DIR, "索引 (*.index)")
        if not path:
            return
        path = os.path.normpath(path)
        root = os.path.normpath(RVC_DIR)
        name = os.path.relpath(path, root).replace("\\", "/") if path.startswith(root) else path
        if self.index_combo.findText(name) < 0:
            self.index_combo.addItem(name)
        self.index_combo.setCurrentText(name)

    def _fill_models(self):
        weights = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WEIGHTS_DIR, "*.pth")))
        if not self.chk_ckpt.isChecked():
            weights = [w for w in weights if not re.search(r"_e\d+_s\d+\.pth$", w)]
        current = self.pth_combo.currentText()
        self.pth_combo.blockSignals(True)
        self.pth_combo.clear()
        self.pth_combo.addItems(weights)
        if current in weights:
            self.pth_combo.setCurrentText(current)
        elif weights:
            self.pth_combo.setCurrentIndex(0)
        self.pth_combo.blockSignals(False)

        cur_idx = self.index_combo.currentText()
        indexes = sorted(
            os.path.relpath(p, RVC_DIR).replace("\\", "/")
            for p in glob.glob(os.path.join(LOGS_DIR, "*", "added_*.index"))
        )
        self.index_combo.blockSignals(True)
        self.index_combo.clear()
        self.index_combo.addItem("（不使用索引）")
        self.index_combo.addItems(indexes)
        if cur_idx in indexes:
            self.index_combo.setCurrentText(cur_idx)
        self.index_combo.blockSignals(False)
        self._on_model_changed(self.pth_combo.currentText())

    def _on_model_changed(self, name):
        if hasattr(self, "stat_model"):
            self.stat_model.setText(name or "—")
        if hasattr(self, "stat_index"):
            idx = self.index_combo.currentText()
            self.stat_index.setText("关闭" if idx in ("", "（不使用索引）") else "已启用")

    def _device_card(self):
        f, lay = card("音频设备")
        self.host_combo = QComboBox()
        self.host_combo.currentIndexChanged.connect(lambda _i: self._fill_devices())
        row0 = QHBoxLayout()
        row0.setSpacing(12)
        row0.addLayout(field_row("设备类型", self.host_combo))
        self.chk_wasapi = QCheckBox("独占 WASAPI 设备")
        row0.addWidget(self.chk_wasapi)
        lay.addLayout(row0)

        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        lay.addLayout(field_row("输入设备", self.in_combo))
        lay.addLayout(field_row("输出设备", self.out_combo))

        row = QHBoxLayout()
        row.setSpacing(14)
        b = QPushButton("重载设备列表")
        b.setObjectName("ghost")
        b.clicked.connect(self.refresh_devices)
        row.addWidget(b)
        self.rb_sr_model = QRadioButton("使用模型采样率")
        self.rb_sr_dev = QRadioButton("使用设备采样率")
        self.grp_sr = QButtonGroup(self)
        self.grp_sr.addButton(self.rb_sr_model)
        self.grp_sr.addButton(self.rb_sr_dev)
        self.rb_sr_dev.setChecked(True)
        row.addWidget(self.rb_sr_model)
        row.addWidget(self.rb_sr_dev)
        self.lbl_sr = QLabel("采样率: —")
        self.lbl_sr.setObjectName("hint")
        row.addWidget(self.lbl_sr)
        row.addStretch(1)
        lay.addLayout(row)
        return f

    def refresh_devices(self):
        import sounddevice as sd

        apis = sd.query_hostapis()
        cur = self.host_combo.currentText()
        self.host_combo.blockSignals(True)
        self.host_combo.clear()
        self.host_combo.addItems([a["name"] for a in apis])
        mme = next((i for i in range(self.host_combo.count())
                    if self.host_combo.itemText(i).upper() == "MME"), -1)
        if cur:
            self.host_combo.setCurrentText(cur)
        elif mme >= 0:
            self.host_combo.setCurrentIndex(mme)
        self.host_combo.blockSignals(False)
        self._fill_devices()
        self._fill_models()

    def _fill_devices(self):
        import sounddevice as sd

        apis = sd.query_hostapis()
        idx = self.host_combo.currentIndex()
        if idx < 0 or idx >= len(apis):
            return
        api = apis[idx]
        devs = sd.query_devices()
        ins, outs = [], []
        for di in api.get("devices", []):
            d = devs[di]
            if d["max_input_channels"] > 0:
                ins.append(d["name"])
            if d["max_output_channels"] > 0:
                outs.append(d["name"])

        def pick(combo, items, prefer):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(items)
            if cur in items:
                combo.setCurrentText(cur)
            else:
                hit = next((n for n in items if prefer in n), None)
                combo.setCurrentText(hit or (items[0] if items else ""))

        pick(self.in_combo, ins, "麦克风")
        if self.in_combo.currentText() not in ins:
            pick(self.in_combo, ins, "Microphone")
        pick(self.out_combo, outs, "Voicemeeter Input")
        try:
            sr = int(sd.query_devices(kind="input").get("default_samplerate", 0))
        except Exception:
            sr = int(api.get("default_samplerate", 0) or 0)
        self.lbl_sr.setText("采样率: %s Hz" % (sr or "—"))

    def _status_card(self):
        f, lay = card("运行状态")
        self.stat_model = self._kv(lay, "当前模型", self.pth_combo.currentText() or "—", True)
        self.stat_index = self._kv(lay, "索引检索", "关闭")
        self.stat_delay = self._kv(lay, "算法延迟", "0 ms")
        self.stat_infer = self._kv(lay, "推理时间", "0 ms")
        return f

    def _kv(self, lay, key, val, strong=False):
        row = QHBoxLayout()
        k = QLabel(key)
        k.setObjectName("fieldLabel")
        k.setFixedWidth(78)
        v = QLabel(val)
        v.setStyleSheet("color: %s; font-size: 12px;%s" % (INK if strong else ACCENT,
                                                          " font-weight: 600;" if strong else ""))
        row.addWidget(k)
        row.addWidget(v, 1)
        lay.addLayout(row)
        return v

    def slider_row(self, key, label, lo, hi, val, fmt, suffix="", bipolar=False):
        row = QHBoxLayout()
        row.setSpacing(12)
        lab = QLabel(label)
        lab.setObjectName("fieldLabel")
        lab.setFixedWidth(78)
        s = DesignSlider(lo, hi, val, bipolar=bipolar)
        s.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))
        value = QLineEdit(fmt(val) + suffix)
        value.setObjectName("valueEdit")
        value.setFixedWidth(64)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setFrame(False)
        s.valueChanged.connect(lambda v, lv=value, f=fmt, sf=suffix: lv.setText(f(v) + sf))

        def _apply_text(k=key, sl=s, f=fmt, sf=suffix, lo=lo, hi=hi):
            import re as _re
            txt = value.text().replace(sf, "")
            m = _re.search(r"[-+]?\d*\.?\d+", txt)
            if not m:
                value.setText(f(sl.value()) + sf)
                return
            num = float(m.group())
            scale = {"formant": 100.0, "index_rate": 100.0, "rms_mix_rate": 100.0,
                     "deess": 100.0, "block_time": 100.0,
                     "crossfade_length": 100.0, "extra_time": 100.0}.get(k, 1.0)
            raw = max(lo, min(hi, int(round(num * scale))))
            if raw != sl.value():
                sl.setValue(raw)
            value.setText(f(sl.value()) + sf)

        value.editingFinished.connect(_apply_text)
        value.returnPressed.connect(_apply_text)
        self.sliders[key] = s
        self.value_edits[key] = value
        row.addWidget(lab)
        row.addWidget(s, 1)
        row.addWidget(value)
        return row

    def _on_slider(self, key, raw):
        if key == "voice_sim":
            if self.engine is not None:
                self.engine.set_voice_sim(raw / 100.0)
            return
        if key == "eq_high":
            if self.engine is not None:
                self.engine.set_eq_high(raw)
            return
        if self.engine is not None and key in ("input_gain", "output_gain"):
            self.engine.set_gain(
                self.sliders["input_gain"].value() / 100.0,
                self.sliders["output_gain"].value() / 100.0,
            )
        if not self.running:
            return
        if key in BUFFER_KEYS:
            self.lbl_restart.setVisible(True)
            return
        scale = {"formant": 100.0, "index_rate": 100.0, "rms_mix_rate": 100.0,
                 "deess": 100.0, "block_time": 100.0, "crossfade_length": 100.0,
                 "extra_time": 100.0}.get(key)
        if scale:
            raw = raw / scale
        try:
            self.engine._apply_slider(key, raw)
        except Exception:
            pass

    def _params_card(self):
        f, lay = card("常规设置")
        lay.addLayout(self.slider_row("threhold", "响应阈值", -60, 0, -60, lambda v: str(v), " dB"))
        lay.addLayout(self.slider_row("pitch", "音调", -24, 24, 13, lambda v: str(v), " st", True))
        lay.addLayout(self.slider_row("formant", "声线粗细", -600, 600, 10,
                                      lambda v: "%.2f" % (v / 100), "", True))
        lay.addLayout(self.slider_row("index_rate", "检索占比", 0, 100, 0, lambda v: "%.2f" % (v / 100)))
        lay.addLayout(self.slider_row("rms_mix_rate", "响度因子", 0, 100, 26, lambda v: "%.2f" % (v / 100)))
        lay.addLayout(self.slider_row("deess", "齿音抑制", 0, 100, 30, lambda v: "%.2f" % (v / 100)))
        lay.addLayout(self.slider_row("eq_high", "高音衰减", 0, 12, 0, lambda v: "%d" % v, " dB"))
        lay.addLayout(self.slider_row("voice_sim", "语音听感", 0, 100, 0, lambda v: "%d%%" % v))
        row = QHBoxLayout()
        row.setSpacing(14)
        lab = QLabel("音高算法")
        lab.setObjectName("fieldLabel")
        lab.setFixedWidth(78)
        row.addWidget(lab)
        self.rb_pm = QRadioButton("pm")
        self.rb_rmvpe = QRadioButton("rmvpe")
        self.rb_fcpe = QRadioButton("fcpe")
        self.grp_f0 = QButtonGroup(self)
        for rb in (self.rb_pm, self.rb_rmvpe, self.rb_fcpe):
            self.grp_f0.addButton(rb)
            row.addWidget(rb)
        self.rb_rmvpe.setChecked(True)
        row.addStretch(1)
        lay.addLayout(row)
        return f

    def _perf_card(self):
        f, lay = card("性能设置")
        lay.addLayout(self.slider_row("block_time", "采样长度", 2, 150, 30, lambda v: "%.2f" % (v / 100)))
        lay.addLayout(self.slider_row("crossfade_length", "淡入淡出", 1, 15, 2, lambda v: "%.2f" % (v / 100)))
        lay.addLayout(self.slider_row("extra_time", "额外推理", 5, 500, 250, lambda v: "%.2f" % (v / 100)))
        self.lbl_restart = QLabel("缓冲参数已修改，重启音频后生效")
        self.lbl_restart.setObjectName("hint")
        self.lbl_restart.setStyleSheet("color: %s; font-size: 11px;" % ACCENT)
        self.lbl_restart.setVisible(False)
        lay.addWidget(self.lbl_restart)

        lay.addLayout(self.slider_row("input_gain", "输入增益", 0, 300, 100, lambda v: "%d%%" % v))
        lay.addLayout(self.slider_row("output_gain", "输出音量", 0, 300, 100, lambda v: "%d%%" % v))

        row = QHBoxLayout()
        row.setSpacing(18)
        self.chk_in_nr = QCheckBox("输入降噪")
        self.chk_out_nr = QCheckBox("输出降噪")
        row.addWidget(self.chk_in_nr)
        row.addWidget(self.chk_out_nr)
        row.addStretch(1)
        lay.addLayout(row)
        return f

    def _log_card(self):
        f, lay = card("运行日志")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(58)
        lay.addWidget(self.log_view)
        return f

    def _wave_card(self):
        f, lay = card("波形")
        head = QHBoxLayout()
        self.chk_scope = QCheckBox("显示波形")
        self.chk_scope.toggled.connect(self._on_scope_toggled)
        head.addWidget(self.chk_scope)
        head.addStretch(1)
        lay.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(14)
        self.wave_out = WaveformWidget(ACCENT)
        self.wave_in = WaveformWidget(GREEN)
        for title, wdg in (("输出波形", self.wave_out), ("输入波形", self.wave_in)):
            box = QFrame()
            box.setObjectName("wavebox")
            box.setStyleSheet(
                "QFrame#wavebox { background: %s; border: 1px solid %s; border-radius: 10px; }" % (WHITE, LINE))
            bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 4, 8, 6)
            bl.setSpacing(2)
            lab = QLabel(title)
            lab.setObjectName("hint")
            bl.addWidget(lab)
            bl.addWidget(wdg, 1)
            row.addWidget(box, 1)
        lay.addLayout(row)
        return f

    def _on_scope_toggled(self, checked):
        self.wave_out.enabled = bool(checked)
        self.wave_in.enabled = bool(checked)
        if self.engine is not None:
            self.engine.set_scope(bool(checked))
        if not checked:
            self.wave_out.clear()
            self.wave_in.clear()
        else:
            self.wave_out.update()
            self.wave_in.update()

    def _poll_wave(self):
        if self.engine is None or not self.running:
            return
        try:
            out, inp = self.engine.get_scope()
            if self.wave_out.enabled and out is not None:
                self.wave_out.push(out)
            if self.wave_in.enabled and inp is not None:
                self.wave_in.push(inp)
        except Exception:
            return

    def _set_function(self, fn):
        if self.engine is not None:
            self.engine.set_function(fn)
        self.append_log("音频路由：%s" % ("输入监听" if fn == "im" else "输出变声"))

    def _controls(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        self.btn_start = QPushButton("开始音频转换")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_engine)
        self.btn_stop = QPushButton("停止音频转换")
        self.btn_stop.setObjectName("secondary")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_engine)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)

        self.rb_monitor = QRadioButton("输入监听")
        self.rb_vc = QRadioButton("输出变声")
        self.grp_route = QButtonGroup(self)
        self.grp_route.addButton(self.rb_monitor)
        self.grp_route.addButton(self.rb_vc)
        self.rb_vc.setChecked(True)
        self.rb_monitor.toggled.connect(
            lambda on: self._set_function("im") if on else None)
        self.rb_vc.toggled.connect(
            lambda on: self._set_function("vc") if on else None)
        row.addWidget(self.rb_monitor)
        row.addWidget(self.rb_vc)
        row.addStretch(1)
        return row

    def _footer(self):
        lab = QLabel("CABLE Output  →  游戏麦克风        ·        建议 48 kHz        ·        嘴离麦 10–15 cm")
        lab.setObjectName("hint")
        return lab

    # ---------- engine ----------
    def _ensure_engine(self):
        return self.engine

    def append_log(self, message):
        self.log_view.appendPlainText(str(message).strip())
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _abs_pth(self):
        name = self.pth_combo.currentText().strip()
        if not name:
            return ""
        return name if os.path.isabs(name) else os.path.join(WEIGHTS_DIR, name)

    def _abs_index(self):
        name = self.index_combo.currentText().strip()
        if name in ("", "（不使用索引）"):
            return ""
        return name if os.path.isabs(name) else os.path.join(RVC_DIR, name.replace("/", os.sep))

    def _load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        def set_combo(combo, value, editable):
            if not value:
                return
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif editable:
                combo.setCurrentText(value)

        set_combo(self.pth_combo, data.get("pth"), True)
        set_combo(self.index_combo, data.get("index"), True)
        set_combo(self.host_combo, data.get("host"), False)
        self._fill_devices()
        set_combo(self.in_combo, data.get("input"), True)
        set_combo(self.out_combo, data.get("output"), True)
        self.rb_sr_model.setChecked(data.get("sr_type") == "sr_model")
        self.rb_sr_dev.setChecked(data.get("sr_type") != "sr_model")
        f0 = data.get("f0method", "rmvpe")
        self.rb_pm.setChecked(f0 == "pm")
        self.rb_rmvpe.setChecked(f0 == "rmvpe")
        self.rb_fcpe.setChecked(f0 == "fcpe")
        self.chk_scope.setChecked(bool(data.get("show_scope", False)))
        self.chk_in_nr.setChecked(bool(data.get("input_nr", False)))
        self.chk_out_nr.setChecked(bool(data.get("output_nr", False)))
        self.chk_wasapi.setChecked(bool(data.get("wasapi_exclusive", False)))
        for key, value in (data.get("sliders") or {}).items():
            if key in self.sliders:
                try:
                    self.sliders[key].setValue(int(value))
                except Exception:
                    pass

    def _save_settings(self):
        data = {
            "pth": self.pth_combo.currentText(),
            "index": self.index_combo.currentText(),
            "host": self.host_combo.currentText(),
            "input": self.in_combo.currentText(),
            "output": self.out_combo.currentText(),
            "sr_type": "sr_model" if self.rb_sr_model.isChecked() else "sr_device",
            "f0method": "pm" if self.rb_pm.isChecked() else ("fcpe" if self.rb_fcpe.isChecked() else "rmvpe"),
            "show_scope": self.chk_scope.isChecked(),
            "input_nr": self.chk_in_nr.isChecked(),
            "output_nr": self.chk_out_nr.isChecked(),
            "wasapi_exclusive": self.chk_wasapi.isChecked(),
            "sliders": {k: s.value() for k, s in self.sliders.items()},
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _collect_values(self):
        eng = self._ensure_engine()
        v = dict(vars(eng.cfg))
        v.update({
            "pth_path": self._abs_pth(),
            "index_path": self._abs_index(),
            "sg_hostapi": self.host_combo.currentText(),
            "sg_wasapi_exclusive": self.chk_wasapi.isChecked(),
            "sg_input_device": self.in_combo.currentText(),
            "sg_output_device": self.out_combo.currentText(),
            "sr_model": self.rb_sr_model.isChecked(),
            "sr_device": self.rb_sr_dev.isChecked(),
            "threhold": self.sliders["threhold"].value(),
            "pitch": self.sliders["pitch"].value(),
            "formant": self.sliders["formant"].value() / 100.0,
            "index_rate": self.sliders["index_rate"].value() / 100.0,
            "rms_mix_rate": self.sliders["rms_mix_rate"].value() / 100.0,
            "deess": self.sliders["deess"].value() / 100.0,
            "block_time": self.sliders["block_time"].value() / 100.0,
            "crossfade_length": self.sliders["crossfade_length"].value() / 100.0,
            "extra_time": self.sliders["extra_time"].value() / 100.0,
            "pm": self.rb_pm.isChecked(),
            "rmvpe": self.rb_rmvpe.isChecked(),
            "fcpe": self.rb_fcpe.isChecked(),
            "I_noise_reduce": self.chk_in_nr.isChecked(),
            "O_noise_reduce": self.chk_out_nr.isChecked(),
            "show_scope": self.chk_scope.isChecked(),
        })
        return v

    def start_engine(self):
        if self.running or self.worker is not None:
            return
        if self.engine is None:
            self.append_log("引擎还在加载，请稍候…")
            return
        eng = self._ensure_engine()
        values = self._collect_values()
        eng.deesser_amount = values["deess"]
        eng.set_scope(self.chk_scope.isChecked())
        eng.set_function("im" if self.rb_monitor.isChecked() else "vc")
        eng.set_eq_high(self.sliders["eq_high"].value())
        eng.set_voice_sim(self.sliders["voice_sim"].value() / 100.0)
        eng.set_gain(self.sliders["input_gain"].value() / 100.0,
                     self.sliders["output_gain"].value() / 100.0)
        self.lbl_restart.setVisible(False)
        self._save_settings()
        self.append_log("加载模型 %s …" % os.path.basename(values["pth_path"]))
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.worker = EngineWorker(eng, values)
        self.worker.log.connect(self.append_log)
        self.worker.ready.connect(self._on_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.wait(3000)

    def _on_ready(self):
        self.running = True
        self.btn_stop.setEnabled(True)
        self.chip_status.setText("●  运行中")
        self.chip_status.setStyleSheet(
            "background: %s; border: 1px solid %s; border-radius: 16px;"
            " padding: 7px 16px; color: %s; font-size: 12px;" % (WHITE, LINE, GREEN))
        self.status_timer.start(200)
        self.wave_timer.start()
        self._poll_status()

    def _on_failed(self, message):
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log("启动失败：%s" % message)
        QMessageBox.warning(self, "启动失败", message)

    def stop_engine(self):
        if self.engine is not None:
            try:
                self.engine.stop_stream()
            except Exception as exc:  # noqa: BLE001
                self.append_log("停止异常：%s" % exc)
        self.status_timer.stop()
        self.wave_timer.stop()
        self.wave_out.clear()
        self.wave_in.clear()
        self.running = False
        self.chip_status.setText("●  待机")
        self.chip_status.setStyleSheet(
            "background: %s; border: 1px solid %s; border-radius: 16px;"
            " padding: 7px 16px; color: %s; font-size: 12px;" % (WHITE, LINE, SECOND))
        self.chip_sr.setText("— Hz")
        self.chip_delay.setText("延迟 0 ms")
        self.chip_infer.setText("推理 0 ms")
        self.stat_delay.setText("0 ms")
        self.stat_infer.setText("0 ms")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _poll_status(self):
        if self.engine is None:
            return
        try:
            st = self.engine.status()
        except Exception:
            return
        if st["running"]:
            self.chip_sr.setText("%d Hz" % st["sr"])
            self.chip_delay.setText("延迟 %d ms" % st["delay_ms"])
            self.chip_infer.setText("推理 %d ms" % st["infer_ms"])
            self.stat_delay.setText("%d ms" % st["delay_ms"])
            self.stat_infer.setText("%d ms" % st["infer_ms"])

    def closeEvent(self, event):
        self._save_settings()
        if self.running:
            self.stop_engine()
        if self.worker is not None:
            try:
                self.worker.wait(5000)
            except Exception:
                pass
        super().closeEvent(event)


ERROR_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "rvc_studio_qt_error.log")


def _install_crash_log():
    import faulthandler
    import traceback

    try:
        fh = open(ERROR_LOG, "a", encoding="utf-8")
        faulthandler.enable(fh)
    except Exception:
        pass

    def hook(exc_type, exc, tb):
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write("\n=== %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                traceback.print_exception(exc_type, exc, tb, file=f)
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


_INSTANCE_LOCK = None


def main():
    _install_crash_log()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PySide6.QtCore import QLockFile
    lock = QLockFile(os.path.join(tempfile.gettempdir(), "rvc_studio_qt.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(150):
        QMessageBox.warning(None, "RVC Studio", "已经有一个 RVC Studio 在运行，请勿重复打开。")
        return
    global _INSTANCE_LOCK
    _INSTANCE_LOCK = lock
    tick = make_tick_icon(os.path.join(tempfile.gettempdir(), "rvc_qt_tick.png"))
    app.setStyleSheet(QSS.replace("__TICK__", tick.replace("\\", "/")))
    app.setFont(QFont("Microsoft YaHei UI", 10))
    win = Studio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
