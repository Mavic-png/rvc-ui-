# RVC Studio 项目简介与交接文档

最后更新：2026-11-28
当前主线版本：v2.7 beta（界面标识）/ 最新提交 `2f9508d`

---

## 1. 项目是什么

RVC Studio 是 RVC 实时变声引擎的 PySide6 桌面客户端。

它不包含 RVC 本体，也不是独立 exe。使用方式是把本仓库的界面与引擎文件放进一个已经装好的 RVC 根目录，由 RVC 自带的 `runtime` 提供 Python、torch、CUDA、HuBERT、RMVPE 等依赖。

```
RVC 根目录/
├─ assets/weights/        .pth 声音模型
├─ configs/               RVC 配置
├─ infer/                 RVC 推理代码（上游）
├─ tools/                 RVC 工具（上游）
├─ runtime/               RVC Python 环境
├─ rvc_engine.py          本仓库：实时引擎
├─ rvc_studio_qt.py       本仓库：界面
├─ audio_diagnostics.py   本仓库：音质诊断采集
└─ diagnostics/           运行时生成的诊断录音与报告
```

### 包含

- PySide6 界面
- 实时音频回调、RVC 推理调用、SOLA 拼接
- 音高/共振峰/Index Rate/RMS/齿音/降噪等参数下发
- 音质诊断采集
- 配置持久化、设备监控、日志

### 不包含

- RVC 上游代码（`infer/`、`tools/`、`configs/`）
- HuBERT / RMVPE 预训练模型
- `.pth` 声音模型、`.index` 索引
- 独立安装包或 exe

---

## 2. 仓库与当前状态

```
GitHub   https://github.com/Mavic-png/rvc-ui-
分支     main
最新提交 2f9508d  fix(diagnostics): RMS 改为整段平均
Release  v2.7-beta（tag 指向 17dd775，落后 main 4 个提交）
```

### 提交历史（近期）

```
2f9508d  RMS 改为整段平均
a199b66  诊断录制自动保存到 diagnostics 文件夹
614b66a  录制前检查音频流，补充采集块数与录音时长
40c79ea  诊断对话框加入朗读测试文本与复制按钮
17dd775  v2.7 beta：音质诊断、设备监控与配置安全（Release tag）
2377a28  音质诊断采集、F0 曲线与诊断对话框
e8ebb77  回调/推理延迟统计、溢出计数与异常日志
9e45289  配置原子保存、设备热插拔监控、延迟统计显示与统一日志
44c5b02  v2.6（上一版基线）
```

注意：`v2.7-beta` 的 tag 没有跟着后续修复移动。要发布包含自动保存/RMS 修复的 beta，需要重新打 tag 或发 `v2.7-beta.1`。

### 本地安装目录

```
F:\rvc\RVC20260723Nvidia\RVC20260718Nvidia
```

仓库文件同步到这里才会生效。仅修改 `F:\rvc\rvc-studio` 不会改变实际运行版本。

### 备份文件（安装目录）

```
rvc_engine.py.pre-batch1.bak
rvc_studio_qt.py.pre-batch1.bak
rvc_engine.py.pre-merge.bak
rvc_studio_qt.py.pre-merge.bak
```

`pre-batch1` 是第一批改动前的原始版本；`pre-merge` 是合并审查补丁前的版本。

---

## 3. 文件职责

| 文件 | 职责 |
|---|---|
| `rvc_studio_qt.py` | 全部界面、控件、状态卡片、诊断对话框、设置加载/保存 |
| `rvc_engine.py` | 音频设备、音频回调、RVC 推理、SOLA、降噪、EQ、限幅、统计 |
| `audio_diagnostics.py` | 峰值/RMS/削波/静音/F0 统计、录制 WAV、导出 JSON |
| `docs/` | 崩溃排查、本交接文档 |
| `logs/` | 运行日志、崩溃日志（运行时生成） |
| `diagnostics/` | 诊断 WAV/JSON（运行时生成） |

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

- 引擎在音频回调末尾调用 `self.diagnostics.feed(raw_in, outdata, f0)`
- F0 通过包装 `self.rvc.get_f0` 获取，不修改上游 `infer/rtrvc.py`
- RVC 不提供真实 F0 置信度，因此只记录「有效帧比例」，不伪造置信度
- 诊断只采集统计，不修改音频样本

---

## 6. 已做测试与发现

### 6.1 验证情况

已通过：

- Python 语法编译
- engine 导入、`stats_snapshot()` 计算
- 诊断模块独立喂数据、WAV 写入、JSON 导出
- 诊断对话框离屏创建
- 设置原子保存
- 诊断自动保存（WAV + JSON 成对生成）

尚未完成实机验证：

- 真实声卡下长时间运行稳定性
- 设备热插拔实际表现
- 诊断开启对 P99 / 溢出计数的影响
- F0 曲线在游戏语音场景下的实际观感

### 6.2 第一轮基线数据（2026-11-27）

3 段录音，参数均为：

```
RMVPE / Index 0.10 / Pitch +15 / Formant 0.25 / 输入输出降噪全关
模型 xinxin.pth
```

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
- 该批 JSON 的 RMS 显示为 0.0，是旧版「取最后一个音频块」的 Bug，已在 `2f9508d` 修复；旧 JSON 的 RMS 不可用，其他字段可用

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
   `v2.7-beta` tag 停在 `17dd775`，后续修复没有重新发版。

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

## 9. 运行、回滚与发布

### 运行

```bat
runtime\pythonw.exe -X utf8 rvc_studio_qt.py
```

或使用仓库里的 `run.bat`。

### 回滚

把安装目录中的两份 `.pre-batch1.bak` 或 `.pre-merge.bak` 覆盖回对应文件即可。

### 发布

```powershell
git add -A
git commit -m "..."
git push origin main
```

GitHub Release 目前不带二进制资产，仅 tag + 说明。

---

## 10. 接手快速检查清单

```
[ ] 确认运行的是安装目录里的文件，不是仓库副本
[ ] 确认 audio_diagnostics.py 已同步到 RVC 根目录
[ ] 启动后标题显示 v2.7 beta
[ ] 开始音频转换，确认状态卡片有 P95/P99
[ ] 打开音质诊断，确认采集块数增加
[ ] 录 5 秒，确认 diagnostics 目录生成 WAV + JSON
[ ] 拔掉麦克风，确认停止并提示
[ ] 关闭程序，确认设置文件有效且生成 .bak
```

---

## 11. 联系与归属

- 仓库：https://github.com/Mavic-png/rvc-ui-
- 基线版本：v2.7 beta
- 上游：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- 本仓库只包含界面与引擎封装，不含模型权重
