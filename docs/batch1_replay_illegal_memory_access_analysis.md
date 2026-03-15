# Batch=1 时 Replay 阶段 CUDA Illegal Memory Access 根因分析

## 现象
- **条件**：batch=1，在 reply/replay 图（decode 阶段使用 CUDA graph replay）时出现 **CUDA illegal memory access**。
- **要求**：先查明具体原因，暂不修改代码。

---

## 1. 相关代码路径概览

### 1.1 CUDA Graph 与 TBO（Two Batch Overlap）
- **cuda_graph_runner.py**：负责 decode 的 CUDA graph 的 capture 与 replay；当 `enable_two_batch_overlap` 时，会为「小 batch」同时 capture 两张图：`use_tbo=True` 与 `use_tbo=False`。
- **two_batch_overlap.py**：TBO 最小 batch 由 `_tbo_min_batch_size`（默认 128）控制；`compute_split_seq_index` 在 `batch_size < _tbo_min_batch_size` 时返回 `None`，即 batch=1 时不做 TBO 切分。
- **tbo_backend.py**：TboAttnBackend 在 replay 时若 `use_tbo=False`，只对 primary 做 `init_forward_metadata_replay_cuda_graph`，**不** init children，注释明确说明是为了避免「写共享 buffer 导致 illegal memory access」。

### 1.2 Batch=1 时的行为
- **can_run_tbo**：`ForwardBatch.can_run_tbo` 为 `tbo_split_seq_index is not None`。batch=1 时 scheduler 侧 `local_tbo_split_seq_index` 为 `None`，聚合后 `batch.tbo_split_seq_index = None`，故 **can_run_tbo = False**。
- **Replay 使用的图**：batch=1 时应使用 `(bs=1, use_tbo=False)` 的图，即只跑 primary、不跑 TBO 子 batch。
- **TboAttnBackend replay**：当 `use_tbo=False` 时只 init primary；当 `use_tbo=True` 时，会走 `_init_forward_metadata_cuda_graph_children`，但 bs=1 时 `compute_split_indices_for_cuda_graph_replay` 返回 `(0, 0)`，导致 `num_tokens_child_left == 0`，直接 return，**同样不会 init children**。

因此，从设计上看，batch=1 时不应去 init 或跑 TBO 的 children，只应跑 primary。

---

## 2. 可能根因方向

### 2.1 【高疑】Replay 时误用 (1, True) 图 或 图内仍跑 0-batch children
- 若因调度/状态错误导致 **can_run_tbo == True**（例如 `tbo_split_seq_index` 被错误设置），则会选到 `(1, True)` 的图进行 replay。
- **Capture (1, True) 时**：`compute_split_seq_index` 对 bs=1 返回 `None`，在 `capture_one_batch_size` 里被置为 0，然后 `prepare_raw` 会构造：
  - **child_a**：`start_token_index=0, end_token_index=0` → **0 tokens, 0 sequences**；
  - **child_b**：1 token, 1 sequence。
- 即 **(1, True) 图在 capture 时就已经包含「一个 batch_size=0 的 child」**。TboAttnBackend 对 `batch_size > 0` 才 `child.init_forward_metadata`，所以 **child_a 的 metadata 不会被 init**。
- Replay (1, True) 时，若仍不 init children（因为 split 为 (0,0) 会 return），则图内已录制的 children 相关 kernel 会带着 **未更新或为 0 的 metadata / 共享 buffer** 执行，极易产生 **illegal memory access**（访问未初始化或越界指针）。

**结论**：若实际 replay 的是 **(1, True)** 图，或图中逻辑仍执行 0-batch 的 child 路径，则非常符合「batch=1 + replay 时非法访存」的现象。

**建议排查**：在 replay 前打日志确认 `graph_key == (1, False)` 且 `forward_batch.can_run_tbo == False`；若出现 `(1, True)` 或 `can_run_tbo == True`，则需查 scheduler / ForwardBatch 上 `tbo_split_seq_index` 的赋值。

---

### 2.2 【中疑】Padding 到 bs=2 时的 page_table / batch_size 不一致（当前已禁用图）
- **cuda_graph_runner.py** 中已有保护逻辑（约 698–704 行）：当 `enable_two_batch_overlap` 且 `cuda_graph_bs < padded_bs` 且 `padded_bs <= 2` 时，将 **is_bs_supported = False**，即 **禁用 CUDA graph**。
- 注释说明：「raw_bs 被 pad 到 2 时，attention 侧 page_table 等可能仍为 1，导致 batch_size != batch_size_k，禁用图」。
- 因此，当 batch=1 被 pad 到 2 时，**不会走 graph replay**，而是走 **eager 路径**。若 illegal memory access 出现在「未用 graph」的 decode 路径上，则可能是：
  - **Eager decode** 下，batch=1 时某些 attention 实现（如 page_table / block_table / cu_seqlens）的 shape 或索引在实现上有 batch=1 的边界问题；
  - 或 MoE/其他层在 batch=1 时有未覆盖的索引或共享 buffer 使用错误。

---

### 2.3 【中疑】Primary 后端在 bs=1 时的 metadata 或 kernel
- 即使正确使用 **(1, False)** 图且只 init primary，若 **primary 的 decode attention 后端**（如 FlashInfer / FlashAttention 等）在 **bs=1** 时存在：
  - `init_forward_metadata_replay_cuda_graph` 中写入的 **cu_seqlens / page_table / block_table** 等与 kernel 假设不一致；或
  - 某些 kernel 对 **batch_size=1** 有未测试路径（如假设 `batch_size >= 2` 或对 `batch_size+1` 的边界处理有误），  
  也可能在 **replay** 时触发 illegal memory access。
- 需要针对当前使用的 attention 后端，单独检查 bs=1 的 metadata 构造与 kernel 调用。

---

### 2.4 【低疑】Capture (1, True) 时 0-batch child 已写入错误状态
- Capture 阶段若运行了 (1, True) 图，图中会执行「一个 0-batch child」的路径；若某些 kernel 对 batch_size=0 未做防护，可能在 **capture 时**就写入非法地址或污染共享 buffer。
- 后续任意一次 replay（包括 (1, False)）若复用同一块共享 buffer，也可能间接触发非法访存。该问题与 2.1 相关，可通过「不为 bs=1 capture (1, True) 图」或「在 capture 时跳过 0-batch child」来规避。

---

## 3. 建议的定位步骤（不改逻辑，只查原因）

1. **确认 replay 时使用的 graph_key 与 can_run_tbo**
   - 在 `cuda_graph_runner.replay()` 内、`self.graphs[graph_key].replay()` 前打日志：
     - `raw_bs`, `self.bs`, `self.use_tbo`, `graph_key`, `forward_batch.can_run_tbo`。
   - 确认：batch=1 时是否为 `graph_key=(1, False)` 且 `can_run_tbo=False`；若出现 `(1, True)` 或 `can_run_tbo=True`，则根因指向 2.1。

2. **确认是否真的在用 CUDA graph**
   - 若 `capture_bs` 不含 1，则 batch=1 会被 pad 到 2，且因 `padded_bs <= 2` 被禁用图，此时走的是 **eager decode**。
   - 在 `can_run()` 与 `replay_prepare` 处打日志，确认 batch=1 时 `can_run` 为 True 还是 False；若为 False，则非法访存发生在 **eager 路径**，需在非 graph 的 decode 路径上排查（attention / MoE 等）。

3. **用 CUDA 工具精确定位非法访问**
   - 使用 `CUDA_LAUNCH_BLOCKING=1` 让错误在真正出错的 kernel 处报出。
   - 使用 `compute-sanitizer --tool memcheck` 跑一次，定位首次 illegal memory access 的调用栈与 kernel，再结合上述 1/2 判断是 TBO 图、primary 后端还是 eager 路径。

4. **检查 tbo_backend 与 two_batch_overlap 的接口是否一致**
   - `tbo_backend._init_forward_metadata_cuda_graph_children` 中调用了  
     `two_batch_overlap.compute_split_indices_for_cuda_graph_replay(..., use_symmetric_split_for_small_batch=...)`。
   - 当前 `two_batch_overlap.compute_split_indices_for_cuda_graph_replay` 的签名中**没有** `use_symmetric_split_for_small_batch` 参数；若未通过 `**kwargs` 等形式接受，运行时会报 `TypeError`。若你方环境能正常跑到 replay，说明要么本地对该函数做了扩展，要么该分支未执行到；否则需确认两处接口一致，避免在 replay 前就因参数错误退出或走错路径。

---

## 4. 小结

| 方向 | 描述 | 建议 |
|------|------|------|
| **误用 (1, True) 图或 0-batch child** | batch=1 时若用了 TBO 图，图中会执行 0-batch child，易导致非法访存 | 用日志确认 `graph_key` 与 `can_run_tbo`，保证 batch=1 只用 (1, False) |
| **Padding 到 2 已禁用图** | 注释与代码已说明并禁用「pad 到 2」的图，避免 page_table 与 batch_size 不一致 | 若未用图，则问题在 eager decode，需查 attention/MoE 的 batch=1 路径 |
| **Primary 后端 bs=1** | 仅 primary、bs=1 时，metadata 或 kernel 仍可能有边界问题 | 用 sanitizer 定位到具体 kernel，再查对应后端的 bs=1 逻辑 |
| **接口不一致** | `compute_split_indices_for_cuda_graph_replay` 与 tbo_backend 的传参可能不一致 | 确认函数签名与调用处，避免隐藏的 TypeError 或错误分支 |

**最可能的原因**：batch=1 时错误地使用了 **(1, True)** 的 CUDA 图，或图中仍执行了 0-batch 的 TBO child 路径，导致基于未初始化/错误的 metadata 或共享 buffer 的访存。建议优先按第 3 节的步骤 1 和 3 做一次确认与定位。
