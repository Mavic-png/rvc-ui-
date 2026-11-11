#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RVC 实时变声引擎（从 realtime_gui.py 抽出，行为不变）。

只负责音频：模型加载、设备枚举、音频回调、参数下发。
界面通过 start/stop/status/set_param/list_devices 调用；日志通过 on_log 回调抛出。
"""
import json
import os
import re
import sys
import threading
import time
import traceback

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from scipy.signal import butter, lfilter, sosfilt

from configs.config import Config
from infer import rtrvc as rvc_for_realtime
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
from tools.file_io import read_text
from tools.torchgate import TorchGate

class EngineConfig:
    def __init__(self) :
        self.pth_path = ""
        self.index_path = ""
        self.pitch = 0
        self.formant=0.0
        self.sr_type = "sr_model"
        self.block_time = 0.25  # s
        self.threhold = -60
        self.crossfade_time = 0.05
        self.extra_time = 2.5
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.0
        self.f0method = "rmvpe"
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""


class DeEsser:
    def __init__(self, sr):
        from scipy.signal import butter

        self.sr = sr
        self.amount = 0.0
        self.lp = butter(2, 6000, btype="lowpass", fs=sr, output="sos")
        self.hp = butter(2, 6000, btype="highpass", fs=sr, output="sos")
        self.zi = {}
        self.env_zi = np.zeros(1, dtype=np.float64)
        self.gain_zi = np.zeros(1, dtype=np.float64)
        self.alpha = 1.0 - np.exp(-1.0 / (0.005 * sr))
        self.alpha_g = 1.0 - np.exp(-1.0 / (0.08 * sr))
        self.gain_zi[0] = 1.0 - self.alpha_g

    def _reset(self):
        self.zi = {}
        self.env_zi[:] = 0.0
        self.gain_zi[:] = 1.0 - self.alpha_g

    def set_amount(self, a):
        self.amount = float(a)
        if self.amount <= 0.001:
            self._reset()

    def process(self, x):
        if self.amount <= 0.001:
            return x
        from scipy.signal import lfilter, sosfilt

        x = np.asarray(x, dtype=np.float32)
        zi = self.zi  # 取一次引用：GUI 线程 set_amount/_reset 时不会拿到半空的 dict
        for k in ("lp1", "lp2", "hp1", "hp2"):
            if k not in zi:
                sos = self.lp if k.startswith("lp") else self.hp
                zi[k] = np.zeros((sos.shape[0], 2), dtype=np.float64)
        low, zi["lp1"] = sosfilt(self.lp, x, zi=zi["lp1"])
        low, zi["lp2"] = sosfilt(self.lp, low, zi=zi["lp2"])
        high, zi["hp1"] = sosfilt(self.hp, x, zi=zi["hp1"])
        high, zi["hp2"] = sosfilt(self.hp, high, zi=zi["hp2"])
        env, self.env_zi = lfilter(
            [self.alpha],
            [1.0, -(1.0 - self.alpha)],
            np.abs(high).astype(np.float64),
            zi=self.env_zi,
        )
        over = np.maximum(env - 0.02, 0.0)
        red = 10.0 * self.amount * over / (env + 1e-6)
        gain = np.power(10.0, -red / 20.0)
        gain, self.gain_zi = lfilter(
            [self.alpha_g],
            [1.0, -(1.0 - self.alpha_g)],
            gain,
            zi=self.gain_zi,
        )
        y = low + high * gain
        return np.clip(y, -1.0, 1.0).astype(np.float32)


class _WsolaShifter:
    """流式 WSOLA 变调（音高与共振峰一起移动，时长不变）。"""

    def __init__(self, sr, semitones, frame=2048, hop=512, search=128):
        self.sr = sr
        self.frame = frame
        self.hop = hop
        self.search = search
        self.win = np.hanning(frame)
        self.set_semitones(semitones)

    def set_semitones(self, semitones):
        self.r = 2.0 ** (float(semitones) / 12.0)
        self.inbuf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0
        self.stretch = np.zeros(0, dtype=np.float32)
        self.wbuf = np.zeros(0, dtype=np.float32)
        self.ow = 0
        self.prev_tail = None
        self.res_pos = 0.0

    def _ensure(self, n):
        if len(self.stretch) < n:
            pad = n - len(self.stretch)
            self.stretch = np.concatenate([self.stretch, np.zeros(pad, dtype=np.float32)])
            self.wbuf = np.concatenate([self.wbuf, np.zeros(pad, dtype=np.float32)])

    def _produce(self):
        while True:
            base = int(round(self.pos))
            if base + self.frame + self.search > len(self.inbuf):
                return
            best = 0
            if self.prev_tail is not None:
                best_score = -1e30
                for k in range(-self.search, self.search + 1):
                    seg = self.inbuf[base + k: base + k + self.search]
                    if len(seg) < self.search:
                        continue
                    score = float(np.dot(seg, self.prev_tail))
                    if score > best_score:
                        best_score = score
                        best = k
            seg = self.inbuf[base + best: base + best + self.frame]
            if len(seg) < self.frame:
                return
            self._ensure(self.ow + self.frame)
            self.stretch[self.ow:self.ow + self.frame] += seg * self.win
            self.wbuf[self.ow:self.ow + self.frame] += self.win
            self.prev_tail = self.stretch[self.ow + self.frame - self.search:
                                          self.ow + self.frame].copy()
            self.ow += self.hop
            self.pos += self.hop / self.r

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        L = len(x)
        self.inbuf = np.concatenate([self.inbuf, x])
        self._produce()
        need = int(np.ceil(self.res_pos + L * self.r)) + 2
        self._ensure(need)
        src = self.stretch[:need]
        w = self.wbuf[:need]
        src = np.divide(src, w, out=np.zeros_like(src), where=w > 1e-6)
        idx = self.res_pos + np.arange(L) * self.r
        out = np.interp(idx, np.arange(len(src)), src)
        self.res_pos += L * self.r
        return out.astype(np.float32)

class VoiceEngine:
    """实时变声引擎。"""

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "configs", "config.json")
        self.cfg = EngineConfig()
        self.rvc_config = Config()
        self.function = "vc"
        self.delay_time = 0.0
        self.stream = None
        self.rvc = None
        self.tg = None
        self.resampler = None
        self.resampler2 = None
        self._running = False
        self._log_cb = None
        self._prewarmed = False
        self.deesser_amount = 0.0
        self.device_latency = 0.0
        self.zc = 0
        self.stat_running = False
        self.stat_sr = 0
        self.stat_delay = 0
        self.stat_infer = 0
        self.scope_seq = 0
        self.input_gain = 1.0
        self.output_gain = 1.0
        self._hp_state = None
        self._lim_env = 1.0
        self._lim_sr = 0
        self.eq_high_db = 0.0
        self._eq_coeffs = None
        self._eq_zi = None
        self._eq_sr = 0
        self.pre_shift = 0.0
        self._pre_shifters = None
        self._pre_shift_sr = 0
        # ---- 线程/故障状态 ----
        self._cb_lock = threading.Lock()
        self._eq_lock = threading.Lock()
        self._faulted = False
        self.cuda_faulted = False
        self.last_error = None
        self._error_cb = None

        self.block_frame = None
        self.block_frame_16k = None
        self.crossfade_frame = None
        self.deesser = None
        self.extra_frame = None
        self.fade_in_window = None
        self.fade_out_window = None
        self.hostapis = None
        self.input_devices = None
        self.input_devices_indices = None
        self.input_wav = None
        self.input_wav_denoise = None
        self.input_wav_res = None
        self.nr_buffer = None
        self.output_buffer = None
        self.output_devices = None
        self.output_devices_indices = None
        self.return_length = None
        self.rms_buffer = None
        self.scope_in = None
        self.scope_out = None
        self.skip_head = None
        self.sola_buffer = None
        self.sola_buffer_frame = None
        self.sola_den_kernel = None
        self.sola_search_frame = None
        self.update_devices()

    # ---------- 日志 ----------
    def on_log(self, callback):
        self._log_cb = callback

    def on_error(self, callback):
        """回调出错（尤其 CUDA 故障）时通知界面，参数为 (摘要, 是否 CUDA)。"""
        self._error_cb = callback

    def _log(self, *args):
        msg = args[0] % args[1:] if len(args) > 1 else (args[0] if args else "")
        if self._log_cb:
            try:
                self._log_cb(str(msg))
            except Exception:
                pass

    # ---------- 故障处理 ----------
    def _crash_log_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs", "realtime_crash.log"
        )

    def _write_crash_log(self, text):
        header = "\n=== %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S")
        body = str(text).rstrip() + "\n"
        try:
            body += "cuda_faulted=%s allocated=%.1fMiB reserved=%.1fMiB f0=%s\n" % (
                self.cuda_faulted,
                torch.cuda.memory_allocated() / 2 ** 20,
                torch.cuda.memory_reserved() / 2 ** 20,
                getattr(self.cfg, "f0method", "?"),
            )
        except Exception:
            pass
        paths = [self._crash_log_path()]
        try:
            import tempfile

            paths.append(os.path.join(tempfile.gettempdir(), "rvc_realtime_crash.log"))
        except Exception:
            pass
        for path in paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(header)
                    f.write(body)
                self._log("音频回调异常，已写入 %s" % path)
                return
            except Exception:
                continue

    def _handle_callback_error(self):
        text = traceback.format_exc()
        self._faulted = True
        self._running = False
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        self.last_error = (lines[-1] if lines else "音频回调异常")[:300]
        if "CUDA" in text or "illegal memory" in text:
            self.cuda_faulted = True
            try:
                from tools.cuda_graph import disable_cuda_graphs

                disable_cuda_graphs("audio callback CUDA error")
            except Exception:
                pass
            self._log("CUDA 出错，本次运行已关闭 CUDA Graph")
        self._write_crash_log(text)
        cb = self._error_cb
        if cb is not None:
            try:
                cb(self.last_error, bool(self.cuda_faulted))
            except Exception:
                pass
        threading.Thread(target=self._deferred_stop, daemon=True).start()

    def _deferred_stop(self):
        # 不能在声卡回调线程里直接 abort/close 流
        time.sleep(0.2)
        try:
            self.stop_stream()
        except Exception:
            pass

    # ---------- 状态 ----------
    def status(self):
        delay = 0.0
        if self._running and self.stream is not None:
            try:
                delay = (
                    float(self.stream.latency[-1])
                    + float(getattr(self.cfg, "block_time", 0.0) or 0.0)
                    + float(getattr(self.cfg, "crossfade_length", 0.0) or 0.0)
                    + 0.01
                )
                if getattr(self.cfg, "I_noise_reduce", False):
                    delay += min(float(getattr(self.cfg, "crossfade_length", 0.0) or 0.0), 0.04)
            except Exception:
                delay = 0.0
        return {
            "running": bool(self._running),
            "sr": int(getattr(self.cfg, "samplerate", 0) or 0),
            "delay_ms": int(round(delay * 1000)),
            "infer_ms": int(getattr(self, "stat_infer", 0) or 0),
            "faulted": bool(self._faulted),
            "cuda_faulted": bool(self.cuda_faulted),
            "error": self.last_error,
        }

    def get_scope(self):
        return getattr(self, "scope_out", None), getattr(self, "scope_in", None)

    def set_function(self, fn):
        self.function = fn if fn in ("vc", "im") else "vc"

    def set_scope(self, enabled):
        self.show_scope = bool(enabled)

    def set_gain(self, input_gain=None, output_gain=None):
        if input_gain is not None:
            self.input_gain = float(input_gain)
        if output_gain is not None:
            self.output_gain = float(output_gain)

    def set_eq_high(self, db):
        """高架衰减：db<=0 表示不处理，正数表示在 4.5kHz 以上衰减该 dB 数。"""
        try:
            db = float(db)
        except Exception:
            db = 0.0
        with self._eq_lock:
            self.eq_high_db = max(0.0, db)
            self._eq_coeffs = None
            self._eq_zi = None



    def _apply_dc_block(self, x):
        """20Hz 一阶高通，去掉直流和低频轰鸣。"""
        import numpy as np
        from scipy.signal import lfilter
        sr = int(getattr(self.cfg, "samplerate", 0) or 0)
        if sr <= 0:
            return x
        if getattr(self, "_hp_state", None) is None or self._hp_state[0] != sr:
            a = float(np.exp(-2.0 * np.pi * 20.0 / sr))
            nch = x.shape[1] if x.ndim > 1 else 1
            self._hp_state = (sr, [a, -a], [1.0, -a], np.zeros((nch, 1)))
        sr_, b, aa, zi = self._hp_state
        if x.ndim == 1:
            y, z = lfilter(b, aa, x, zi=zi[0])
            zi[0] = z
            return y
        out = np.empty_like(x)
        for ch in range(x.shape[1]):
            y, z = lfilter(b, aa, x[:, ch], zi=zi[ch])
            out[:, ch] = y
            zi[ch] = z
        return out

    def _apply_limiter(self, x):
        """峰值限幅：-1 dBFS 上限，快攻击慢释放，防止增益/均衡后削波。"""
        import numpy as np
        sr = int(getattr(self.cfg, "samplerate", 0) or 0)
        if sr <= 0 or x.size == 0:
            return x
        peak = float(np.max(np.abs(x)))
        if peak <= 1e-9:
            return x
        thr = 0.891  # -1 dBFS
        target = 1.0 if peak <= thr else thr / peak
        n = x.shape[0] if x.ndim > 1 else x.size
        block_dur = float(n) / float(sr)
        if target < getattr(self, "_lim_env", 1.0):
            self._lim_env = target
        else:
            rel = float(np.exp(-block_dur / 0.10))
            self._lim_env = target + (getattr(self, "_lim_env", 1.0) - target) * rel
        y = x * self._lim_env
        np.clip(y, -1.0, 1.0, out=y)
        return y

    def set_pre_shift(self, semitones):
        """前置变调：先把人声整体变高（含共振峰），再送进 RVC。"""
        try:
            st = float(semitones)
        except Exception:
            st = 0.0
        self.pre_shift = max(-12.0, min(12.0, st))
        self._pre_shifters = None
        self._pre_shift_sr = 0

    def _apply_pre_shift(self, indata):
        st = float(getattr(self, "pre_shift", 0.0) or 0.0)
        if abs(st) < 0.05:
            return indata
        sr = int(getattr(self.cfg, "samplerate", 0) or 0)
        if sr <= 0:
            return indata
        if self._pre_shifters is None or self._pre_shift_sr != sr:
            self._pre_shifters = _WsolaShifter(sr, st)
            self._pre_shift_sr = sr
        try:
            return self._pre_shifters.process(indata)
        except Exception:
            return indata

    def _design_high_shelf(self, sr, gain_db, f0=4500.0, q=0.707):
        from scipy.signal import lfilter  # noqa: F401
        import math
        amp = 10.0 ** (-abs(gain_db) / 40.0)
        w0 = 2.0 * math.pi * f0 / float(sr)
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * q)
        beta = 2.0 * math.sqrt(amp) * alpha
        a0 = (amp + 1) - (amp - 1) * cos_w0 + beta
        b0 = amp * ((amp + 1) + (amp - 1) * cos_w0 + beta)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_w0)
        b2 = amp * ((amp + 1) + (amp - 1) * cos_w0 - beta)
        a1 = 2 * ((amp - 1) - (amp + 1) * cos_w0)
        a2 = (amp + 1) - (amp - 1) * cos_w0 - beta
        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]
        return b, a

    def _apply_eq(self, outdata):
        with self._eq_lock:
            return self._apply_eq_locked(outdata)

    def _apply_eq_locked(self, outdata):
        db = float(getattr(self, "eq_high_db", 0.0) or 0.0)
        if db <= 0.01:
            return outdata
        import numpy as np
        from scipy.signal import lfilter
        sr = int(getattr(self.cfg, "samplerate", 0) or 0)
        if sr <= 0:
            return outdata
        if self._eq_coeffs is None or self._eq_sr != sr or self._eq_zi is None:
            self._eq_coeffs = self._design_high_shelf(sr, db)
            self._eq_sr = sr
            channels = outdata.shape[1] if outdata.ndim > 1 else 1
            self._eq_zi = [np.zeros(2) for _ in range(channels)]
        b, a = self._eq_coeffs
        if outdata.ndim == 1:
            outdata, z = lfilter(b, a, outdata, zi=self._eq_zi[0])
            self._eq_zi[0] = z
        else:
            for ch in range(outdata.shape[1]):
                outdata[:, ch], z = lfilter(b, a, outdata[:, ch], zi=self._eq_zi[ch])
                self._eq_zi[ch] = z
        return outdata


    def load(self):
        try:
            data = json.loads(read_text(self.config_path))
            data["sr_model"] = data["sr_type"] == "sr_model"
            data["sr_device"] = data["sr_type"] == "sr_device"
            if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
                data["f0method"] = "rmvpe"
            data["pm"] = data["f0method"] == "pm"
            data["rmvpe"] = data["f0method"] == "rmvpe"
            data["fcpe"] = data["f0method"] == "fcpe"
            if data["sg_hostapi"] in self.hostapis:
                self.update_devices(hostapi_name=data["sg_hostapi"])
                if (
                    data["sg_input_device"] not in self.input_devices
                    or data["sg_output_device"] not in self.output_devices
                ):
                    self.update_devices()
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
            else:
                data["sg_hostapi"] = self.hostapis[0]
                data["sg_input_device"] = self.input_devices[
                    self.input_devices_indices.index(sd.default.device[0])
                ]
                data["sg_output_device"] = self.output_devices[
                    self.output_devices_indices.index(sd.default.device[1])
                ]
        except:
            with open(self.config_path, "w", encoding="utf8") as j:
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_hostapi": self.hostapis[0],
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ],
                    "sr_type": "sr_model",
                    "threhold": -60,
                    "pitch": 0,
                    "formant": 0.0,
                    "show_scope": True,
                    "deess": 0.0,
                    "index_rate": 0,
                    "rms_mix_rate": 0,
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                    "extra_time": 2.5,
                    "f0method": "rmvpe",
                }
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                data["pm"] = data["f0method"] == "pm"
                data["rmvpe"] = data["f0method"] == "rmvpe"
                data["fcpe"] = data["f0method"] == "fcpe"
        return data


    def set_values(self, values):
        if len(values["pth_path"].strip()) == 0:
            self._log("请选择pth文件")
            return False
        pattern = re.compile("[^\x00-\x7F]+")
        if pattern.findall(values["pth_path"]):
            self._log("pth文件路径不可包含中文")
            return False
        if pattern.findall(values["index_path"]):
            self._log("index文件路径不可包含中文")
            return False
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        # self.device_latency = values["device_latency"]
        self.cfg.sg_hostapi = values["sg_hostapi"]
        self.cfg.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
        self.cfg.sg_input_device = values["sg_input_device"]
        self.cfg.sg_output_device = values["sg_output_device"]
        self.cfg.pth_path = values["pth_path"]
        self.cfg.index_path = values["index_path"]
        self.cfg.sr_type = ["sr_model", "sr_device"][
            [
                values["sr_model"],
                values["sr_device"],
            ].index(True)
        ]
        self.cfg.threhold = values["threhold"]
        self.cfg.pitch = values["pitch"]
        self.cfg.formant = values["formant"]
        self.cfg.block_time = values["block_time"]
        self.cfg.crossfade_time = values["crossfade_length"]
        self.cfg.extra_time = values["extra_time"]
        self.cfg.I_noise_reduce = values["I_noise_reduce"]
        self.cfg.O_noise_reduce = values["O_noise_reduce"]
        self.cfg.rms_mix_rate = values["rms_mix_rate"]
        self.cfg.index_rate = values["index_rate"]
        self.cfg.f0method = ["pm", "rmvpe", "fcpe"][
            [values["pm"], values["rmvpe"], values["fcpe"]].index(True)
        ]
        return True


    def _apply_slider(self, key, v):
        if key == "threhold":
            self.cfg.threhold = v
        elif key == "pitch":
            self.cfg.pitch = v
            if hasattr(self, "rvc"):
                self.rvc.change_key(v)
        elif key == "formant":
            self.cfg.formant = v
            if hasattr(self, "rvc"):
                self.rvc.change_formant(v)
        elif key == "index_rate":
            self.cfg.index_rate = v
            if hasattr(self, "rvc"):
                self.rvc.change_index_rate(v)
        elif key == "rms_mix_rate":
            self.cfg.rms_mix_rate = v
        elif key == "deess":
            self.deesser_amount = v
            if hasattr(self, "deesser"):
                self.deesser.set_amount(v)


    def start_vc(self):
        if self.stream is not None:
            self.stop_stream()
        self._faulted = False
        self.cuda_faulted = False
        self.last_error = None
        try:
            from tools.cuda_graph import empty_cuda_cache_safely

            empty_cuda_cache_safely()
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.rvc = rvc_for_realtime.RVC(
            self.cfg.pitch,
            self.cfg.formant,
            self.cfg.pth_path,
            self.cfg.index_path,
            self.cfg.index_rate,
            self.rvc_config,
            self.rvc if hasattr(self, "rvc") else None,
        )
        self.cfg.channels = self.get_device_channels()
        dev_sr = self.get_device_samplerate()
        want_sr = (
            self.rvc.tgt_sr
            if self.cfg.sr_type == "sr_model"
            else dev_sr
        )

        def _supports(sr):
            try:
                sd.check_input_settings(
                    device=sd.default.device[0], channels=self.cfg.channels,
                    samplerate=int(sr), dtype="float32")
                sd.check_output_settings(
                    device=sd.default.device[1], channels=self.cfg.channels,
                    samplerate=int(sr), dtype="float32")
                return True
            except Exception:
                return False

        if not _supports(want_sr):
            self._log("采样率 %s 不被设备支持，回退到 %s", want_sr, dev_sr)
            want_sr = dev_sr
        self.cfg.samplerate = int(want_sr)
        self.deesser = DeEsser(self.cfg.samplerate)
        self.deesser.set_amount(self.deesser_amount)
        self.zc = self.cfg.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.cfg.block_time
                    * self.cfg.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.scope_in = np.zeros(self.block_frame, dtype=np.float32)
        self.scope_out = np.zeros(self.block_frame, dtype=np.float32)
        self.crossfade_frame = (
            int(
                np.round(
                    self.cfg.crossfade_time
                    * self.cfg.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.cfg.extra_time
                    * self.cfg.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.rvc_config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.rvc_config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.rvc_config.device, dtype=torch.float32
        )
        self.sola_den_kernel = torch.ones(
            1,
            1,
            self.sola_buffer_frame,
            device=self.rvc_config.device,
            dtype=torch.float32,
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.rvc_config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.cfg.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.rvc_config.device)
        if self.rvc.tgt_sr != self.cfg.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.cfg.samplerate,
                dtype=torch.float32,
            ).to(self.rvc_config.device)
        else:
            self.resampler2 = None
        # Bundled torch.istft is not CUDA Graph-capturable, so TorchGate
        # stays eager while resampling and RVC inference still use graphs.
        self.tg = TorchGate(
            sr=self.cfg.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.rvc_config.device)
        self.prewarm_cuda_graph()
        self.start_stream()


    def prewarm_cuda_graph(self):
        if not cuda_graph_enabled(self.rvc_config.device):
            return
        try:
            self._log("正在预热CUDA Graph")
            samples = self.input_wav_res.shape[0]
            phase = torch.arange(
                samples, device=self.rvc_config.device, dtype=torch.float32
            )
            probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
            self.input_wav_res.copy_(probe)

            if self.cfg.I_noise_reduce:
                short = self.input_wav[
                    -self.sola_buffer_frame - self.block_frame :
                ].unsqueeze(0)
                self.tg(short, self.input_wav.unsqueeze(0))

            resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
            run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )

            inferred = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.cfg.f0method,
            )
            if self.resampler2 is not None:
                inferred = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    inferred,
                )
            if self.cfg.O_noise_reduce:
                self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
            torch.cuda.synchronize(self.rvc_config.device)
            self._log("CUDA Graph预热完成")
        except Exception:
            self._log(traceback.format_exc())
        finally:
            self.input_wav.zero_()
            self.input_wav_denoise.zero_()
            self.input_wav_res.zero_()
            self.output_buffer.zero_()
            self.sola_buffer.zero_()
            self.nr_buffer.zero_()
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()


    def start_stream(self):
        if self._running:
            return
        if "WASAPI" in self.cfg.sg_hostapi and self.cfg.sg_wasapi_exclusive:
            extra_settings = sd.WasapiSettings(exclusive=True)
        else:
            extra_settings = None
        stream = sd.Stream(
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.cfg.samplerate,
            channels=self.cfg.channels,
            dtype="float32",
            extra_settings=extra_settings,
        )
        self.stream = stream
        self._faulted = False
        self.last_error = None
        self._running = True
        try:
            stream.start()
        except Exception:
            self._running = False
            self.stream = None
            raise
        self.stat_running = True
        self.stat_sr = int(self.cfg.samplerate)


    def stop_stream(self):
        stream, self.stream = self.stream, None
        self._running = False
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        # 关键：等已经进入的回调真正退出后再释放缓冲。声卡回调线程可能还在用
        # self.input_wav 等 CUDA 张量，提前换掉会让 kernel 读到已释放的显存，
        # 表现为随机的 CUDA illegal memory access（栈里看不出真正现场）。
        with self._cb_lock:
            pass
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        self.stat_running = False
        self.scope_in = None
        self.scope_out = None
        self.scope_seq = getattr(self, "scope_seq", 0) + 1


    def audio_callback(self, indata, outdata, frames, times, status):
        """声卡回调入口：异常绝不允许逃出 cffi 回调。

        以前回调里的任何异常都会直接冒到 PortAudio 的 C 回调；pythonw 没有
        控制台，于是弹 "Python-CFFI error" 框并停掉音频流，日志里什么都不留。
        现在统一转成静音 + 崩溃日志 + 通知界面。
        """
        if not self._running or self._faulted:
            outdata.fill(0.0)
            return
        if not self._cb_lock.acquire(blocking=False):
            outdata.fill(0.0)
            return
        try:
            self._process_audio(indata, outdata, frames)
        except Exception:
            try:
                outdata.fill(0.0)
            except Exception:
                pass
            self._handle_callback_error()
        finally:
            self._cb_lock.release()

    def _process_audio(self, indata, outdata, frames):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if getattr(self, "pre_shift", 0):
            indata = self._apply_pre_shift(indata)
        if getattr(self, "input_gain", 1.0) != 1.0:
            indata = indata * self.input_gain
        raw_in = indata.copy()
        if self.cfg.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.cfg.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.rvc_config.device
        )
        if getattr(self, "show_scope", True):
            self.scope_in = np.asarray(
                raw_in[-self.block_frame :], dtype=np.float32
            ).copy()
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if self.cfg.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame :] = input_wav[
                : self.block_frame
            ]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            resample_input = self.input_wav_denoise[
                -self.block_frame - 2 * self.zc :
            ]
            self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        else:
            resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.cfg.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    infer_wav,
                )
        elif self.cfg.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if self.cfg.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.cfg.rms_mix_rate < 1 and self.function == "vc":
            if self.cfg.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.rvc_config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.rvc_config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, 1.0 - self.cfg.rms_mix_rate
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                self.sola_den_kernel,
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (
            self.sola_buffer * self.fade_out_window
        )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        outdata[:] = (
            infer_wav[: self.block_frame]
            .repeat(self.cfg.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        if getattr(self, "deesser", None) is not None:
            outdata[:] = self.deesser.process(outdata[:, 0])[:, None]
        outdata = self._apply_dc_block(outdata)
        outdata = self._apply_eq(outdata)
        if getattr(self, "output_gain", 1.0) != 1.0:
            outdata *= self.output_gain
        outdata = self._apply_limiter(outdata)
        if getattr(self, "show_scope", True):
            self.scope_out = outdata[:, 0].copy()
            self.scope_seq = getattr(self, "scope_seq", 0) + 1
        total_time = time.perf_counter() - start_time
        if self._running:
            self.stat_infer = int(total_time * 1000)


    def update_devices(self, hostapi_name=None):
        """获取设备列表"""

        self._running = False
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]


    def set_devices(self, input_device, output_device):
        """设置输出设备"""
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]
        self._log("输入设备：%s:%s", str(sd.default.device[0]), input_device)
        self._log("输出设备：%s:%s", str(sd.default.device[1]), output_device)


    def get_device_samplerate(self):
        return int(
            sd.query_devices(device=sd.default.device[1])["default_samplerate"]
        )


    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])[
            "max_input_channels"
        ]
        max_output_channels = sd.query_devices(device=sd.default.device[1])[
            "max_output_channels"
        ]
        return min(max_input_channels, max_output_channels, 2)

