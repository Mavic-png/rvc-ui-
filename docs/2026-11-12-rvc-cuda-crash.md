# RVC Studio 实时变声崩溃排查（2026-11-12）

## 现象

直播/变声过程中弹出：

```
Python-CFFI error
Exception ignored from cffi callback <function _StreamBase._init__.<locals>.callback_ptr ...>
  File "...\rvc_engine.py", line 829, in audio_callback
    y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
RuntimeError: CUDA error: an illegal memory access was encountered
```

## 结论

1. **栈里那一行是“背锅位”**。CUDA 是异步的，真正的越界发生在更早的某个 kernel，
   错误会延迟到下一次 GPU→CPU 同步时才报出来。`line 829` 只是本块音频里第一个
   同步点（`librosa.feature.rms` 前把张量拷回 CPU），并不代表那行有问题。
2. **这不是普通的 Python 异常，而是 GPU 侧故障**。`logs\realtime_crash.log` 之外，
   Windows 系统日志同一时刻有 NVIDIA 驱动事件：
   `nvlddmkm` Event ID 153, "Error occurred on GPUID: 100"（2026-11-11 23:40:43）。
   说明确实有 kernel 做了非法访存，驱动因此判定上下文损坏。
3. **上下文一旦损坏就救不回来**：之后任何 CUDA 调用都会继续报同一个错误。
   所以必须关闭程序重开，不能只点“开始”。
4. 原来的代码把回调里的异常**直接放给 PortAudio 的 C 回调**。程序用 `pythonw.exe`
   启动（没有控制台），CPython 只能弹 `Python-CFFI error` 消息框 —— 于是表现为
   “莫名其妙弹个框、声音没了、日志里什么都没有”。

## 已做的修改

### rvc_engine.py

- `audio_callback` 变成纯兜底入口：任何异常 → 输出静音 + 写
  `logs\realtime_crash.log`（失败时回落 `%TEMP%\rvc_realtime_crash.log`）+ 通知界面
  + 延迟自动停流。**异常再也不会逃进 cffi**，不会再弹那个框。
- CUDA 类错误自动调用 `tools/cuda_graph.disable_cuda_graphs()`，本次运行后续全部
  走 eager，避免在坏上下文上继续提交 kernel。
- `stop_stream()` 会等已进入的回调真正退出（`_cb_lock`）再释放/替换 CUDA 缓冲，
  消除“回调线程还在用 `input_wav`，主线程已经把它换掉”的 use-after-free 竞态。
- `start_stream()` 先建流再置 `_running`，启动失败不会残留半开状态。
- `start_vc()` 开头先停旧流，并清空故障标志。
- 修掉两个 GUI 线程 / 回调线程抢状态的假崩溃：
  - `_apply_eq` 与 `set_eq_high` 之间 `_eq_zi = None` 的竞态（拖动“高音衰减”
    滑杆时必现 `TypeError: 'NoneType' object does not support item assignment`）；
  - `DeEsser.process` 中 `self.zi` 被 `_reset()` 换掉的竞态。
- `status()` 增加 `faulted / cuda_faulted / error` 字段。

### tools/cuda_graph.py

- 抓图有上限（默认 16 张，`RVC_CUDA_GRAPH_MAX_CAPTURES`）：拖动“声线粗细/音调”
  时不再无限抓图（一张图 ~300–400 ms，会把实时回调拖爆）。
- 淘汰旧图之前先 `torch.cuda.synchronize()`，避免图被销毁时 GPU 还在用它池子里的显存。
- 抓图/回放失败自动退回 eager；识别到 CUDA 错误就全局关闭 CUDA Graph。
- `clear_cuda_graph_cache()` 同样先同步再销毁。
- 抓图与 `empty_cache()` 用同一把锁串行化：否则会撞上 PyTorch 内部断言
  `captures_underway.empty() INTERNAL ASSERT FAILED`（实测能稳定复现，
  以前的表现是“换模型时偶发启动失败”）。

### rvc_studio_qt.py

- 引擎新增 `on_error` 回调：出错时界面线程弹一次明确提示（CUDA / 普通异常分开），
  自动把状态栏、按钮复位，并把 `logs\realtime_crash.log` 路径写进日志面板。

### 新增 go-rvc-studio-qt-safe.bat

安全模式启动（`RVC_CUDA_GRAPH=0`），用于对照排查是否与 CUDA Graph 有关。

## 已知但未改的行为

- 每抓一张 CUDA Graph，PyTorch 会为它保留一块私有显存池；换模型时旧图销毁，
  但这块池不会立刻还给系统（实测每次换 pth 约 +8.5 MiB `memory_allocated`，
  关掉 CUDA Graph 则完全不涨）。量级很小（12 GB 卡、几十次换模型撑得住），
  所以没动；如果长期不关程序且频繁换模型，重启一次即可释放。
- `rvc_engine` 里的“响度混合”仍走 `librosa`，每块音频有两次 GPU→CPU 同步
  （也就是报错栈里那一行）。这是 latency/性能问题，不是本次崩溃的原因，未改。


## 如果再次出现

1. 看 `F:\rvc\RVC20260723Nvidia\RVC20260718Nvidia\logs\realtime_crash.log`
   最后一次的完整堆栈。
2. 用 `go-rvc-studio-qt-safe.bat` 复现一次：
   - 不复现 → 大概率是 CUDA Graph 回放；把 `logs\realtime_crash.log` 发我。
   - 仍复现 → 把 `RVC_CUDA_GRAPH` 恢复默认，再临时加
     `set CUDA_LAUNCH_BLOCKING=1`（更慢，但会把出错 kernel 精确指到 Python 行）。
3. 同时确认有没有别的程序（训练脚本、游戏、Edge 硬件加速）在抢显卡；
   `nvlddmkm` 153 也可能是显卡驱动/超频不稳定的表现，可先恢复默认频率复测。

## 复现用的离线压测

`F:\zuoc\tools\stress_realtime.py`（不开声卡，直接灌音频跑回调）：

```
runtime\python.exe -u -X utf8 F:\zuoc\tools\stress_realtime.py 600 --realtime --restart 20 --switch --churn
```

- `--churn` 模拟实时拖动音调/声线粗细/齿音/高音衰减滑杆；
- `--restart N` 每 N 块停流再 `start_vc`；`--switch` 顺带切换 pth；
- `--fault N` 在第 N 块注入异常，验证兜底链路；
- `--no-graph` 关闭 CUDA Graph 对照。

改完引擎后实测（2026-11-12 00:26，RTX 4070 SUPER / torch 2.7.1+cu118）：

- 修前：拖动“高音衰减”滑杆后，第 29 块就必现 `TypeError: 'NoneType' ...`；
  换模型时偶发 `captures_underway.empty() INTERNAL ASSERT FAILED`。
- 修后：3 分钟实时节奏灌音 + 30 次停流/换 pth/重启 + 持续拖参数，0 异常、
  0 CUDA 报错，每次切换约 1 s 后音频正常恢复，只有第一次启动的抓图块超时。
- 异常注入（`--fault`）验证兜底：不弹 cffi 框、输出静音、界面收到 `cuda=True`
  通知、`logs\realtime_crash.log` 记录完整堆栈和显存信息。
