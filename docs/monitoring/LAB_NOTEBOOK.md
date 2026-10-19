# Lab notebook: Research A (activation-based monitoring)

What went wrong while the monitoring pipeline was built, and what changed. Nothing here is a
monitoring result. Entries M-1 … M-6 were recorded before any detector scored actor activations.

## 2026-10-19

### M-1. Assistant-only training of the stand-in actor stalls at the marginal answer distribution
* **Observed.** A 6-layer, d = 96 `Qwen2ForCausalLM` trained from scratch on planner targets with
  loss only on the 2–3 answer tokens reached loss ≈ 0.83 after 600 steps and stayed there. Its
  verb agreement with the planner was 0.29, *below* always answering "allocate" (≈ 0.43). The
  model had learned the marginal distribution over candidate sequences. Under sequence-level argmax
  that favours single-token answers (`review`) over allocations split across pools.
* **Fix (stand-in only).** A two-phase curriculum (1,000 steps on two-pool roots without queued
  tickets or logs, then the full distribution) plus a 0.2-weight next-token loss on prompt tokens.
  In the trial, verb agreement reached 0.78 within 200 full-distribution steps. In the pilot run it
  was 0.81 at step 1,500 of 3,000.
* **Scope.** The organism adapter (M2) is still trained with assistant-only loss (syllabus T2). The
  prompt-token loss is decision A-D12. It affects only how competent the stand-in is, not the
  monitoring comparison, which uses one fixed checkpoint for every detector.

### M-2. Full-dimensional Mahalanobis on response signatures dilutes a low-rank change
* **Observed (synthetic gate, before any actor data).** With the task-changing response fully
  removed for violators, the response detector reached only TPR 0.30 at 5 % FPR (d = 24). Its
  Euclidean residual norms separated cleanly (violators 21.4 vs normal 9.1), but the change lay in
  ~8 of 144 residual dimensions. Mahalanobis distance weighs all 144 equally in χ² terms, so the
  change was diluted.
* **Second issue.** Group z-scores were standardised with *in-sample* distances (the covariance
  was fit on the same residuals). Test-normal z-scores were therefore biased by about +2.
* **Fix.** (a) Reduce each signature group to its top 16 principal components, fit on normal-fitting
  signatures only. (b) Standardise with out-of-fold distances. TPR rose to 0.76 (d = 24) and 0.93
  (d = 48), with FPR at target. Both choices were frozen in the preregistration before the actor
  existed (decision A-D7).

### M-3. One gate expectation was wrong
* The first gate expected the response detector (without its base term) to stay at α under a pure
  base shift. It flagged 13–18 %. That is legitimate: the context model *predicts* the response
  from base activations, so a displaced base displaces the prediction, and a correct response then
  leaves a residual. The check now requires the response detector to stay far below pooled
  Mahalanobis on base shifts (0.13 vs 0.97).

### M-4. Shallow models map the declared sites onto duplicates
* With a 4-layer smoke model the relative-depth rule mapped blocks (6, 13, 20) to (1, 1, 2). The
  hook was registered twice on block 1 and stored two rows per root. `site_blocks` now
  deduplicates, and so does `ResidualRecorder`. For 28 blocks the rule still returns exactly (6, 13, 20).

### M-5. Chat-template tokenization changed type across `transformers` versions
* In `transformers` 5.x, `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`, which is
  not a `dict` subclass. `list(...)` of it yields its *key names*, so every Qwen prompt would have
  been the two tokens `input_ids attention_mask`. The offline Qwen-path test
  (`tests/monitor/test_qwen_path.py`, fake chat tokenizer + tiny random Qwen2) caught it. The wrapper
  now renders the template to text and tokenizes that, which is version-stable.

### M-6. `pkill -f` matched its own shell again
* Same failure as during the Research B pilots: a `pkill -f <script>` command line contains its own
  pattern, so it killed the invoking shell. Background jobs are now stopped by PID.
