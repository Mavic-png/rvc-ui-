# RVC Studio 项目简介与交接文档

最后更新：2026-11-28
当前主线版本：v2.7 beta（界面标识）

---

## 1. 项目是什么

RVC Studio 是 RVC 实时变声引擎的 PySide6 桌面客户端。

它不包含 RVC 本体，也不是独立 exe。使用方式是把本仓库的界面、引擎与补丁文件放进一个已经装好的 RVC 根目录，由 RVC 自带的 `runtime` 提供 Python、torch、CUDA、HuBERT、RMVPE 等依赖。

```
RVC 根目录/
├─ assets/weights/             .pth 声音模型
├─ configs/                    RVC 配置（上游）
├─ infer/                      RVC 推理代码（上游）
├─ runtime/                    RVC Python 环境
├─ tools/                      RVC 工具（上游）+ 本仓库补丁（cuda_graph.py / file_io.py / torchgate/）
├─ rvc_engine.py               本仓库：实时引擎
├─ rvc_studio_qt.py            本仓库：界面
├─ audio_diagnostics.py        本仓库：音质诊断采集
├─ run.bat                     本仓库：常规启动脚本
├─ go-rvc-studio-qt-safe.bat   本仓库：安全模式启动（关闭 CUDA Graph）
└─ diagnostics/                运行时生成的诊断录音与报告
```

### 包含

- PySide6 界面（`rvc_studio_qt.py`）
- 实时音频回调、RVC 推理调用、SOLA 拼接（`rvc_engine.py`）
- 音高/共振峰/Index Rate/RMS/齿音/降噪等参数下发
- 音质诊断采集与报告导出（`audio_diagnostics.py`）
- CUDA Graph 加固、文件编码读取与降噪补丁（`tools/cuda_graph.py`、`tools/file_io.py`、`tools/torchgate/`）
- 启动脚本（`run.bat`、`go-rvc-studio-qt-safe.bat`）
- 配置持久化、设备监控、日志

### 不包含

- RVC 上游核心推理代码（`infer/`）及除上述补丁外的上游 `tools/`、`configs/`
- HuBERT / RMVPE 预训练模型
- `.pth` 声音模型、`.index` 索引
- 独立安装包或 exe

---

## 2. 仓库与当前状态

```
GitHub   https://github.com/Mavic-png/rvc-ui-
上游     https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
分支     main
代码基线 v2.7 beta（基于 2f9508d 之后的修复）
Release  v2.7-beta（tag 指向 17dd775，落后 main 多个提交）
```

### 提交历史（近期基准）

```
bde7b29  docs: 项目简介与 v2.7 beta 交接文档
2f9508d  fix(diagnostics): RMS 改为整段平均
a199b66  feat(ui): 诊断录制自动保存到 diagnostics 文件夹
614b66a  fix(ui): 诊断录制前检查音频流，补充采集块数与录音时长
40c79ea  feat(ui): 音质诊断加入朗读测试文本与复制按钮
17dd775  v2.7 beta：音质诊断、设备监控与配置安全（Release tag）
2377a28  feat(diagnostics): 音质诊断采集、F0 曲线与诊断对话框
9e45289  feat(ui): 配置原子保存、设备热插拔监控、延迟统计显示与统一日志
e8ebb77  feat(engine): 回调/推理延迟统计、溢出计数与异常日志
44c5b02  v2.6（上一版基线）
```

注意：`v2.7-beta` 的 tag 仍停在 `17dd775`，未包含自动保存、RMS 平均与本轮诊断修复；若要对外发布新 beta，需要打新 tag（如 `v2.7-beta.1`）。本机安装路径与历史 `.bak` 备份见文末「[11. 本机环境速查](#11-本机环境速查)」。

---

## 3. 文件职责、环境变量与修改红线

| 文件 / 路径 | 职责 |
|---|---|
| `rvc_studio_qt.py` | 全部界面、控件、状态卡片、诊断对话框、设置原子加载/保存 |
| `rvc_engine.py` | 音频设备、音频回调、RVC 推理、SOLA、降噪、EQ、限幅、延迟统计 |
| `audio_diagnostics.py` | 单声道峰值/RMS/削波/静音/F0 统计、录制双声道 WAV、导出 JSON |
| `tools/cuda_graph.py` | CUDA Graph 抓图上限（默认 16）、同步淘汰与失败退回 eager |
| `tools/file_io.py` | 文本读取编码兼容辅助 |
| `tools/torchgate/` | 实时降噪门限模块 |
| `run.bat` / `go-rvc-studio-qt-safe.bat` | 常规启动 / 安全模式启动（设置 `RVC_CUDA_GRAPH=0`） |
| `rvc_studio_qt_settings.json` / `.bak` | 脚本同目录下的用户配置与热备份（运行时生成） |
| `logs/rvc_studio_app.log` | 统一滚动应用日志（2 MB × 3，运行时生成） |
| `logs/realtime_crash.log` | 音频回调崩溃日志（失败时回落 `%TEMP%\rvc_realtime_crash.log`） |
| `diagnostics/` | 诊断 WAV/JSON（运行时生成） |
| `docs/` | 崩溃排查（[2026-11-12-rvc-cuda-crash.md](2026-11-12-rvc-cuda-crash.md)）与本交接文档 |

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RVC_DIR` | 自动探测脚本目录/上级 | 显式指定含 `assets/` 与 `configs/` 的 RVC 根目录 |
| `RVC_CUDA_GRAPH` | `1` | 设为 `0` 时关闭 CUDA Graph（对应 `go-rvc-studio-qt-safe.bat`） |
| `RVC_CUDA_GRAPH_MAX_CAPTURES` | `16` | CUDA Graph 最大抓图数量上限，防止滑杆拖动爆显存/卡顿 |

### 修改红线（工程约束）

详见 [2026-11-12-rvc-cuda-crash.md](2026-11-12-rvc-cuda-crash.md)，后续改动必须遵守：

1. **回调异常禁止逃出 cffi**：`audio_callback` 必须兜底捕获全部异常，输出静音、写崩溃日志并通知界面，绝不能把 Python 异常抛给 PortAudio C 回调。
2. **停流必须等待 `_cb_lock`**：`stop_stream()` 必须等正在执行的回调退出后再释放或替换 CUDA 缓冲，防止 use-after-free。
3. **CUDA Graph 抓图与 `empty_cache()` 串行**：销毁图前必须 `torch.cuda.synchronize()`，并用同一把锁串行化抓图与清缓存。
4. **诊断只读且不改上游 `infer/rtrvc.py`**：F0 通过包装 `self.rvc.get_f0` 获取，`diagnostics.feed()` 严禁修改音频样本或持有重锁。

---

## 4. 已实现功能

### 4.1 基础

- 模型 `.pth` / 索引 `.index` 选择，过滤训练检查点
- 音频设备枚举、Host API 切换
- 实时波形、输入监听/输出变声切换
- 参数热更新：音调、声线、Index Rate、RMS 混合、齿音、EQ、输入/输出增益
- 每个模型独立保存参数预设（v2.6）
- 设置持久化、崩溃日志

### 4.2 v2.7 新增

**延迟与稳定性**

- 回调耗时、推理耗时、回调周期
- 平均 / P95 / P99
- 输入溢出 / 输出欠载计数
- 状态卡片显示「处理平均 / P95 / P99 / 溢出欠载」

**配置安全**

- 设置文件临时写入 → JSON 校验 → `.bak` 备份 → `os.replace`
- 主设置损坏时自动回退 `.bak`
- 统一滚动日志：`logs/rvc_studio_app.log`（2 MB × 3）

**设备监控**

- 运行时每 2.5 秒检查当前 Host API 下的输入/输出设备
- 输入、输出分别按设备能力检查
- 掉线后停止音频流、写日志、界面提示

**音质诊断**

- 输入/输出峰值、RMS、削波样本数/削波率
- 输入/输出静音比例
- F0 曲线、有效帧比例
- 诊断对话框：实时指标、F0 曲线、朗读文本、复制文本
- 录制输入/输出双声道 WAV，最长 60 秒
- 自动保存到 `diagnostics/`，同时生成 JSON
- 报告包含模型、F0 算法、Index Rate、Pitch、Formant、降噪开关

---

## 5. 音质诊断的使用

主界面 → 「音质诊断」。

```
1. 先在主界面点「开始音频转换」
2. 确认诊断窗口里「采集块数」在增加
3. 点「开始录制诊断音频」
4. 对着麦克风朗读窗口内 A/B/C 测试文本
5. 点「停止录制」
```

停止后自动保存：

```
diagnostics/diag_YYYYMMDD_HHMMSS_<f0>_nr<in><out>_idx<NNN>.wav
diagnostics/diag_YYYYMMDD_HHMMSS_<f0>_nr<in><out>_idx<NNN>.json
```

示例：

```
diag_20261127_025052_rmvpe_nr00_idx010.wav
```

WAV 为双声道：左 = 输入，右 = 输出。

### 实现方式

- 引擎在 `_process_audio` 末尾调用 `self.diagnostics.feed(raw_in, outdata, f0)`：
  - `raw_in` 为经过 `pre_shift` 与 `input_gain` 后的单声道信号（位于门限/降噪前，并非声卡裸输入）
  - `outdata` 为经过齿音抑制、DC 阻断、EQ、输出增益、语音模拟与限幅器后的最终输出（多声道时取第 0 声道统计与录制）
- F0 通过包装 `self.rvc.get_f0` 获取，不修改上游 `infer/rtrvc.py`
- RVC 不提供真实 F0 置信度，因此只记录「有效帧比例」，不伪造置信度
- 诊断只采集统计，不修改音频样本
- 点击「开始录制诊断音频」时 `start_recording()` 会先调用 `reset()` 清空前序统计，连续多段对照录制互不污染；JSON 报告已记录模型、索引、F0、Index Rate、Pitch、Formant、降噪、缓冲长度与增益/DSP 开关等完整实验上下文

---

## 6. 已做测试与发现

### 6.1 验证情况

已通过：

- Python 语法编译
- engine 导入、`stats_snapshot()` 计算
- 诊断模块独立喂数据（含多声道输出抽取、开始录制重置统计）、WAV 写入、JSON 导出
- 诊断对话框离屏创建与完整实验上下文导出
- 设置原子保存
- 诊断自动保存（WAV + JSON 成对生成）

可复跑的离屏回归命令（在 RVC 根目录执行）：

```bat
runtime\python.exe -m py_compile rvc_engine.py rvc_studio_qt.py audio_diagnostics.py
set QT_QPA_PLATFORM=offscreen && runtime\python.exe -c "import rvc_studio_qt, audio_diagnostics; print('import ok')"
```

尚未完成实机验证：

- 真实声卡下长时间运行稳定性
- 设备热插拔实际表现
- 诊断开启对 P99 / 溢出计数的影响
- F0 曲线在游戏语音场景下的实际观感

### 6.2 第一轮基线数据（2026-11-27）

3 段录音固定参数为：

```
RMVPE / Index 0.10 / Pitch +15 / Formant +0.25 / 输入输出降噪全关
模型 xinxin.pth / Host API MME / sr_device 48 kHz
block_time 0.30s / crossfade 0.01s / extra 2.50s / threhold -60
rms_mix_rate 0.32 / eq_high 3 dB / deess 0 / voice_sim 0 / 输入输出增益 1.0
```

> 注：当时配置中的 `index` 指向 `logs/shi_clean48/added_IVF256_Flat_nprobe_1_shi_clean48_v2.index`（与 `xinxin.pth` 为跨模型组合），第二轮对照实验前请先核对是否改为同模型索引或设为 `0` 对照。

| 文件 | 时长 | 输入峰值 | 输出峰值 | 削波 | F0 有效帧 |
|---|---|---|---|---|---|
| diag_…025052 | 27.0s | 0.257 | 0.324 | 0 | 28.3% |
| diag_…025121 | 20.1s | 0.257 | 0.324 | 0 | 35.2% |
| diag_…025136 | 11.6s | 0.257 | 0.324 | 0 | 37.5% |

观察：

- 输入电平偏低（峰值约 0.26，RMS 0.011–0.014），无削波
- 静音比例 66%–75%，F0 有效帧偏低，与停顿多有关
- A/B 输出频谱主峰约 270–282 Hz，符合输入约 120 Hz +15 半音的预期
- C（大小声交替）输出主峰约 167–168 Hz，与 F0 曲线中位数约 236 Hz 不一致，疑似音高不稳定或分析窗口问题，需要复测
- 3 段录音的输入/输出峰值完全一致（`0.257` / `0.324`），是因为旧版 `start_recording()` 未调用 `reset()` 导致峰值跨段累计；同时该批 JSON 的 RMS 显示为 `0.0`（旧版取最后一个静音块所致）。这两项均已在当前版本修复，旧 JSON 仅可参考部分字段，需按新版本重新采集对照数据

### 6.3 尚未完成的关键对照

```
F0 算法：RMVPE / FCPE / PM
降噪：输入开、输出开、都开、都关
Index Rate：0 / 0.25 / 0.5 / 0.75
```

没有这些对照，无法判断电音感来自 F0、降噪还是模型本身。

---

## 7. 已知问题与技术债

1. **音频回调仍然承担全部推理工作**
   `_process_audio()` 内仍然有 librosa、GPU 推理、重采样、SOLA。
   当前版本只增加了计时，没有做采集/推理/输出线程分离。

2. **统计锁在回调线程**
   审查版在 `audio_callback` 中用 `_lat_lock` 保护周期和计数。
   锁持有时间短，但严格实时性要求下应改成无锁 deque / 原子计数。

3. **设备恢复只做到停止 + 提示**
   没有自动重连状态机；设备恢复后需要用户手动重新开始。

4. **中文路径仍被拒绝**
   `set_values()` 仍会拒绝非 ASCII 的 `.pth` / `.index` 路径。

5. **没有自动化测试**
   当前只有手工/离屏验证，没有 `pytest` 或单元测试。

6. **Release 与 main 不同步**
   见第 2 节：`v2.7-beta` tag 停在 `17dd775`，后续修复需发 `v2.7-beta.1`。

7. **RVC 上游不提供 F0 置信度**
   诊断只能给有效帧比例，不能给出逐帧置信度。

---

## 8. 下一步计划

### 8.1 先做对照实验，不写新 DSP

```
1. 复测 C 语料两次，确认 167–168 Hz 是否稳定复现
2. F0 三算法对照：RMVPE / FCPE / PM
3. 降噪四组合：输入、输出分别开关
4. Index Rate 四档：0 / 0.25 / 0.5 / 0.75
```

每组参数固定模型、索引、Pitch、Formant、缓冲参数。

### 8.2 根据结果决定 v2.8

| 结果 | 方向 |
|---|---|
| 输入降噪明显影响自然度 | 优化输入降噪或更换方案 |
| 输出降噪影响自然度 | 优化输出降噪 |
| F0 曲线频繁跳变 / 半频 | F0 平滑、跳变抑制、置信度过滤 |
| F0 正常但仍有电音 | 查模型、Index Rate、训练质量 |
| 音量波动明显 | 输入 AGC |
| 停顿底噪明显 | 带 Attack/Release/Hold 的噪声门 |

### 8.3 暂不优先

- 完整输出 EQ、压缩器
- A/B 盲听
- 按进程自动切换
- 系统托盘/全局快捷键

这些不解决当前的电音感根因，等诊断结论出来再排期。

---

## 9. 同步、运行、回滚与发布

### 同步到 RVC 根目录

仅修改仓库不会改变实际运行版本，每次改完代码后需同步以下文件到 RVC 根目录：

- `rvc_engine.py`、`rvc_studio_qt.py`、`audio_diagnostics.py`
- `run.bat`、`go-rvc-studio-qt-safe.bat`
- `tools\cuda_graph.py`、`tools\file_io.py`、`tools\torchgate\`

PowerShell 同步示例（将 `$inst` 替换为你的 RVC 根目录）：

```powershell
$repo = 'F:\rvc\rvc-studio'
$inst = 'F:\rvc\RVC20260723Nvidia\RVC20260718Nvidia'
Copy-Item "$repo\rvc_engine.py","$repo\rvc_studio_qt.py","$repo\audio_diagnostics.py","$repo\run.bat","$repo\go-rvc-studio-qt-safe.bat" -Destination $inst -Force
Copy-Item "$repo\tools\*" -Destination "$inst\tools" -Recurse -Force
```

### 运行

```bat
runtime\pythonw.exe -X utf8 rvc_studio_qt.py
```

或双击 `run.bat`；遇到 CUDA Graph 疑似异常时双击 `go-rvc-studio-qt-safe.bat`（安全模式：`RVC_CUDA_GRAPH=0`）。

### 发布

仓库根目录含运行时生成的 `logs/`、`__pycache__/`，提交前先检查 `git status`，按实际改动显式暂存文件，避免裸用 `git add -A`：

```powershell
git status --short
git add rvc_engine.py rvc_studio_qt.py audio_diagnostics.py README.md .gitignore docs/ tools/ run.bat go-rvc-studio-qt-safe.bat
git commit -m "..."
git push origin main
```

补发包含最新修复的测试版 tag（GitHub Release 目前仅 tag + 说明，不带二进制资产）：

```powershell
git tag -a v2.7-beta.1 -m "v2.7 beta.1"
git push origin v2.7-beta.1
```

---

## 10. 接手快速检查清单

- [ ] 确认运行的是 RVC 安装目录里的文件，而不是仓库副本
- [ ] 确认 `audio_diagnostics.py`、`tools/` 与启动脚本已同步到 RVC 根目录
- [ ] 启动后标题显示 `v2.7 beta`
- [ ] 开始音频转换，确认状态卡片有 `P95/P99`
- [ ] 打开音质诊断，确认采集块数增加
- [ ] 录 5 秒，确认 `diagnostics/` 生成双声道 WAV + 完整 JSON
- [ ] 拔掉麦克风，确认停止并提示
- [ ] 关闭程序，确认 `rvc_studio_qt_settings.json` 有效且生成 `.bak`

---

## 11. 本机环境速查

- **本地安装目录**：`F:\rvc\RVC20260723Nvidia\RVC20260718Nvidia`
- **安装目录历史备份**：
  - `rvc_engine.py.pre-batch1.bak` / `rvc_studio_qt.py.pre-batch1.bak`（第一批改动前原始版本）
  - `rvc_engine.py.pre-merge.bak` / `rvc_studio_qt.py.pre-merge.bak`（合并审查补丁前版本）
- **回滚方式**：将安装目录中对应的 `.pre-batch1.bak` 或 `.pre-merge.bak` 覆盖回 `rvc_engine.py` / `rvc_studio_qt.py` 即可
- **常用测试模型**：`xinxin.pth`（注意核对 `rvc_studio_qt_settings.json` 中绑定的 `.index` 路径是否为预期文件）
