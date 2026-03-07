# 画面无人脸框 + Redis objects 为空 — 原因分析与排查

## 现象

- 最终显示画面中**没有人脸检测框**（连 bbox 都没有）。
- Redis stream 中 `"objects": []` 恒为空。

说明：**NvDsObjectMeta（含 bbox）没有到达各 StreamBranch 的 nvosd，也没有被 nvmsgconv 正确序列化。**

---

## 数据流回顾

```
[推理链] … → tracker → tee(sink) [probe 挂在这里]
                    ↓
         ┌─────────┴─────────┐
         ↓                   ↓
    msg 分支            demux 分支
    queue(30)           queue(40) → nvstreamdemux
         ↓                   ↓
    msgconv              src_0, src_1, …
         ↓                   ↓
    broker           各 StreamBranch: queue → nvvidconv → nvosd → valve → rtsp
```

- 人脸框应由 **pgie/sgie** 产生的 `NvDsObjectMeta`（含 `rect_params`）经 **tracker → tee → demux** 传到各分支的 **nvosd** 绘制。
- 若画面上连框都没有，则要么：
  1. 进入 demux 的 buffer 上 **batch_meta 异常**（被改坏或缺失），  
  2. 要么 **nvstreamdemux 输出** 的 buffer 没有正确带上每帧的 frame_meta/obj_meta。

---

## 可能原因（按优先级）

### 1. 在 tee sink 的 probe 里改 batch_meta，导致 demux 无法正确拷贝

- Probe 在 **tee 的 sink** 上对**同一块** buffer 的 `batch_meta` 加锁、遍历、并调用 `pyds.nvds_add_custom_msg_blob_to_frame()`。
- 若 nvstreamdemux 在「按帧拆分并拷贝 metadata」时，**只支持部分类型的 meta**，或对 **frame_user_meta_list / 自定义 blob** 处理不当，就可能：
  - 在拷贝时出错或跳过整帧，或  
  - 只拷贝了「原始」frame/obj 结构，没有把 probe 加上的内容一起拷贝，但**若实现有 bug，也可能顺带破坏或清空 obj_meta_list**。
- 因此：**probe 在 tee 上对 batch_meta 的修改，有可能和 demux 的 metadata 拷贝逻辑不兼容**，导致 demux 输出没有 bbox。

### 2. Tee 把同一 buffer 的 ref 给两个分支，demux 与 msg 分支共享 batch_meta

- 同一 GstBuffer（同一 batch_meta）先被 msg 分支的 msgconv 读，再被 demux 分支的 demux 读（或顺序相反）。
- 若 **nvmsgconv** 在「msg2p_newapi + dummy_payload」下对 batch_meta 做了**只读以外的操作**（例如临时挂 user_meta、改指针等），理论上可能影响另一分支上 demux 看到的 meta（虽然更常见是只读）。
- 若 **nvstreamdemux** 在「输入 buffer 被多处引用」时，对 batch_meta 的拷贝有 bug，也可能导致输出 buffer 上没有正确的 obj_meta。

### 3. nvstreamdemux 与当前拓扑/上游元素组合的兼容性

- 官方 demo 是 **pgie → nvstreamdemux** 直连；当前是 **tracker → tee → queue → nvstreamdemux**，且上游还有 **nvdspreprocess**。
- 若 demux 对「经过 tee/queue 的 buffer」或对「带 preprocess 的 batch 结构」有未覆盖到的路径，可能导致输出 buffer 未正确挂上 frame_meta/obj_meta（即 bbox 丢失）。

---

## 建议的排查步骤（不改业务逻辑，只做最小验证）

### 步骤 1：确认是否与 tee 上的 probe 有关

- **临时**让 `_batch_buffer_probe` 只做「取到 batch_meta 后立刻 return Gst.PadProbeReturn.OK」：
  - 不调用 `nvds_acquire_meta_lock`、不遍历 frame/obj、不调用 `nvds_add_custom_msg_blob_to_frame`、不写 `display_text`。
- 再跑一次 pipeline：
  - 若此时**人脸框出现** → 说明问题与「在 tee sink 上对 batch_meta 的读写」强相关，需要把 probe 挪到 demux 之后（例如各分支 nvosd 的 sink）或改为不修改 batch_meta 的写法。
  - 若仍然**没有人脸框** → 说明问题更可能出在 demux 或更前面的拓扑（见步骤 2）。

### 步骤 2：确认 demux 入口是否已有 bbox

- 在 **nvstreamdemux 的 sink pad** 上加一个**只读** probe：
  - 用 `pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))` 取 batch_meta；
  - 遍历 `frame_meta_list` → `obj_meta_list`，只读 `rect_params` 或 `num_obj_meta`，打 log 或计数；
  - 不写、不挂任何 meta，直接 return OK。
- 若这里**已经能看到每帧的 obj_meta 和 bbox**，说明「进入 demux 的 buffer」是正常的，问题在 **demux 的输出**（或下游 nvosd）。
  - 可再在 **某个 StreamBranch 的 queue 的 sink pad**（即 demux 的 src_N 的下游）加只读 probe，看该 buffer 上是否还有 batch_meta 且含 obj_meta；若这里已经没了，则基本可认定是 **nvstreamdemux 在拆批时没有把 metadata 正确带到输出 buffer**。

### 步骤 3：确认 Redis 的 objects 与 msgconv 行为

- 在确认「是否 probe 导致 demux 丢 bbox」之后，再单独看 Redis：
  - 若步骤 1 中「去掉 probe 对 batch_meta 的修改」后 bbox 恢复，可再恢复 probe 但**只做 FAISS + display_text，暂不调用 nvds_add_custom_msg_blob_to_frame**，看 Redis 的 `objects` 是否仍为空。
  - 若仍为空，则更可能是 **payload_type / msg2p_newapi 与 NvDsObjectMeta 的序列化路径** 问题（你之前已分析过的 schema 问题），与「有没有 bbox」可分开查。

### 步骤 2 延伸：pgie/sgie src 均为 per_frame_objs=[0,0]

说明 **PGIE 就没有产出任何检测框**，问题在 PGIE 或其上游输入。

1. **看 pgie sink 的只读 probe 日志**（已加 `_pgie_sink_probe_readonly`）：
   - `pgie sink: batch_meta is None` → 进入 PGIE 的 buffer 没有 batch_meta，问题在 **preprocess 或 mux**。
   - `pgie sink: n_frames=2 frame_wh=[(2560, 1440), (2560, 1440)]` → 输入有 2 帧且尺寸正常，问题在 **PGIE 推理或配置**（见下）。

2. **PGIE 配置与运行环境**（以 SCRFD 为例）：
   - **batch-size**：配置文件里 `batch-size` 需与路数一致（如 2 路则设为 2）。当前 `dsapp_scrfd_pgie_config.txt` 里为 `batch-size=1`，若 nvinfer 在 state 变化时从文件重新加载并覆盖代码里 `update_batch_size(2)` 的设置，会导致只处理 1 帧或行为异常；建议把配置文件中的 `batch-size` 改为与最大路数一致（如 2）。
   - **preprocess**：`dsapp_preprocess_config.txt` 中 `target-unique-ids=1`（对应 PGIE 的 gie-unique-id）、`network-input-shape` 第一维 ≥ batch（如 `4;3;640;640`），`processing-width/height=640` 与 SCRFD 输入一致。
   - **custom 解析库**：`custom-lib-path` 指向的 `libnvdsinfer_custom_scrfd.so` 存在且导出 `NvDsInferParseCustomSCRFD`；否则 nvinfer 可能静默不挂 bbox。
   - **模型与 engine**：`onnx-file` / `model-engine-file` 路径正确；首次跑会生成 engine，若有报错会打在启动日志。

3. **启动日志**：看是否有 nvinfer 报错（如 "Could not parse bbox"、"Failed to load engine"、"custom parser not found" 等）。

---

## 小结

- **画面无人脸框**：根本原因是「画图用的 buffer」上**没有可用的 NvDsObjectMeta（bbox）**；这些 meta 应来自 pgie/sgie，经 tracker → tee → demux 传到各分支 nvosd。
- **最可疑**：在 **tee sink 上对同一 buffer 的 batch_meta 加锁、写 custom blob（以及 display_text）** 与 **nvstreamdemux 的 metadata 拷贝** 不兼容，导致 demux 输出没有正确带上 obj_meta。
- **建议先做**：用步骤 1 的「probe 临时空跑」验证是否与 probe 有关；再用步骤 2 的「demux 入口/出口只读 probe」确认 metadata 是在 demux 内丢失还是在更早/更晚阶段丢失。根据结果再决定是移动 probe、改 probe 写法，还是查 nvstreamdemux/配置。
