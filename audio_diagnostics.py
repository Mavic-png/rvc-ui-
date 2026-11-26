#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RVC Studio 音质诊断（只采集统计，不修改音频）。

在音频回调里调用 feed()，只做轻量的峰值/RMS/削波/静音/F0 统计。
界面通过 snapshot() 读取，不持有回调线程锁。
"""
import json
import math
import os
import time

import numpy as np


class AudioDiagnostics:
    def __init__(self, sample_rate=48000):
        self.sample_rate = int(sample_rate)
        self.reset()

    def reset(self):
        self.blocks = 0
        self.in_samples = 0
        self.out_samples = 0
        self.in_sq_sum = 0.0
        self.out_sq_sum = 0.0
        self.in_clip = 0
        self.out_clip = 0
        self.in_silent_blocks = 0
        self.out_silent_blocks = 0
        self.in_peak = 0.0
        self.out_peak = 0.0
        self.in_rms_last = 0.0
        self.out_rms_last = 0.0
        self.f0_valid = 0
        self.f0_total = 0
        self.f0_values = []
        self._recording = False
        self._rec_started = 0.0
        self._rec_max_seconds = 60.0
        self._rec_in = []
        self._rec_out = []
        self._rec_frames = 0

    # ---------- 回调侧 ----------
    def feed(self, indata, outdata, f0=None):
        try:
            inp = np.asarray(indata, dtype=np.float32).reshape(-1)
            out = np.asarray(outdata, dtype=np.float32).reshape(-1)
        except Exception:
            return
        if inp.size:
            ip = float(np.max(np.abs(inp)))
            ir = float(np.sqrt(np.mean(inp * inp)))
            self.in_peak = max(self.in_peak, ip)
            self.in_rms_last = ir
            self.in_clip += int(np.count_nonzero(np.abs(inp) >= 0.999))
            self.in_samples += inp.size
            self.in_sq_sum += float(np.sum(inp * inp))
            if ir < 1e-4:
                self.in_silent_blocks += 1
        if out.size:
            op = float(np.max(np.abs(out)))
            orms = float(np.sqrt(np.mean(out * out)))
            self.out_peak = max(self.out_peak, op)
            self.out_rms_last = orms
            self.out_clip += int(np.count_nonzero(np.abs(out) >= 0.999))
            self.out_samples += out.size
            self.out_sq_sum += float(np.sum(out * out))
            if orms < 1e-4:
                self.out_silent_blocks += 1
        self.blocks += 1

        if f0 is not None:
            try:
                vals = np.asarray(f0, dtype=np.float32).reshape(-1)
            except Exception:
                vals = np.array([], dtype=np.float32)
            if vals.size:
                self.f0_total += int(vals.size)
                valid = vals[vals > 0]
                self.f0_valid += int(valid.size)
                if valid.size:
                    self.f0_values.extend(float(v) for v in valid[:64])
                    if len(self.f0_values) > 2000:
                        del self.f0_values[: len(self.f0_values) - 2000]

        if self._recording:
            elapsed = time.time() - self._rec_started
            if elapsed > self._rec_max_seconds:
                self._recording = False
            else:
                self._rec_in.append(inp.copy())
                self._rec_out.append(out.copy())
                self._rec_frames += inp.size

    # ---------- 界面侧 ----------
    def snapshot(self):
        in_clip_rate = (self.in_clip / self.in_samples) if self.in_samples else 0.0
        out_clip_rate = (self.out_clip / self.out_samples) if self.out_samples else 0.0
        in_rms_avg = ((self.in_sq_sum / self.in_samples) ** 0.5
                      if self.in_samples else 0.0)
        out_rms_avg = ((self.out_sq_sum / self.out_samples) ** 0.5
                       if self.out_samples else 0.0)
        f0_ratio = (self.f0_valid / self.f0_total) if self.f0_total else 0.0
        f0 = list(self.f0_values)
        if len(f0) > 500:
            step = max(1, len(f0) // 500)
            f0 = f0[::step][-500:]
        return {
            "blocks": self.blocks,
            "in_peak": round(self.in_peak, 4),
            "in_rms": round(in_rms_avg, 4),
            "in_rms_last": round(self.in_rms_last, 4),
            "out_peak": round(self.out_peak, 4),
            "out_rms": round(out_rms_avg, 4),
            "out_rms_last": round(self.out_rms_last, 4),
            "in_clip": self.in_clip,
            "out_clip": self.out_clip,
            "in_clip_rate": round(in_clip_rate, 6),
            "out_clip_rate": round(out_clip_rate, 6),
            "in_silence_ratio": round(
                self.in_silent_blocks / self.blocks if self.blocks else 0.0, 4),
            "out_silence_ratio": round(
                self.out_silent_blocks / self.blocks if self.blocks else 0.0, 4),
            "f0_valid_ratio": round(f0_ratio, 4),
            "f0_frames": self.f0_total,
            "f0_curve": [round(v, 2) for v in f0],
            "recording": bool(self._recording),
            "record_seconds": round(self._rec_frames / max(1, self.sample_rate), 2),
        }

    def report(self, extra=None):
        data = self.snapshot()
        data["sample_rate"] = self.sample_rate
        data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["extra"] = extra or {}
        return data

    def export_report(self, path, extra=None):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.report(extra), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ---------- 录制 ----------
    def start_recording(self, max_seconds=60.0):
        self._rec_in = []
        self._rec_out = []
        self._rec_frames = 0
        self._rec_started = time.time()
        self._rec_max_seconds = float(max_seconds)
        self._recording = True

    def stop_recording(self):
        self._recording = False
        return self._rec_in, self._rec_out

    def save_recording(self, path):
        inp, out = self._rec_in, self._rec_out
        if not inp or not out:
            return False
        n = min(sum(x.size for x in inp), sum(x.size for x in out))
        a = np.concatenate(inp)[:n]
        b = np.concatenate(out)[:n]
        stereo = np.stack([a, b], axis=1)
        pcm = np.clip(stereo, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        try:
            from scipy.io import wavfile
            wavfile.write(path, self.sample_rate, pcm)
            return True
        except Exception:
            import wave
            try:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm.tobytes())
                return True
            except Exception:
                return False
