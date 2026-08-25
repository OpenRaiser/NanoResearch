# SDPO Reproducibility Settings

SDPO updates the personalized Planner after each user-feedback round. The backbone model is shared and frozen, while each user has an independent LoRA adapter and feedback buffer.

## Training Recipe

| Setting | Configuration |
| --- | --- |
| Planner backbone | Qwen2.5-7B-Instruct |
| Hardware | One NVIDIA A100 GPU |
| Update frequency | One SDPO update after each user-feedback round |
| Data per update | All valid feedback triples accumulated for the current user up to that round |
| Training sample | `(x, y, feedback)`: Planner input, original Planner output, and subsequent user feedback |
| Student forward pass | Conditions on `(x, y)` and computes token-level log probabilities over `y` |
| Teacher forward pass | Conditions on `(x, feedback, y)` using the same model; gradients are stopped through this branch |
| Token-level advantage | Feedback-conditioned token log probability minus the original-policy token log probability |
| Trainable parameters | LoRA parameters only; the backbone remains frozen |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | 0 |
| Micro-batch size | 1 feedback triple |
| Gradient accumulation | 1 |
| Effective batch size | 1 |
| Update steps | 50 optimization steps per feedback round |
| Maximum sequence length | 2,048 tokens |
| Maximum optimized response length | 512 tokens |
| Truncation | Left truncation, preserving the feedback and response tail |
| Advantage clipping | `[-5, 5]` |
| Gradient clipping | Global norm 1.0 |
| Precision | bfloat16 |
| Random seed | 42 |
| Checkpointing | Save the user-specific LoRA adapter after each update and resume it in the next round |

## User Isolation

Each user uses a separate feedback directory and LoRA checkpoint. Feedback examples and trainable adapter parameters are not shared across users. Only the frozen Qwen2.5-7B-Instruct backbone is shared.

## Overfitting Safeguards

- Only LoRA parameters are updated.
- Each feedback-round update is limited to 50 optimization steps.
- Historical feedback is replayed together with the latest feedback.
- LoRA dropout is set to 0.05.
- Token-level advantages are clipped to `[-5, 5]`.
- Gradient global norm is clipped to 1.0.

## Reproduction Command

Use separate paths for every user:

```bash
nanoresearch ram-train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-root ~/.nanoresearch/users/<user_id>/ram_data \
  --output ~/.nanoresearch/users/<user_id>/ram-sdpo \
  --max-steps 50 \
  --learning-rate 1e-4 \
  --max-sequence-length 2048 \
  --max-trained-tokens 512 \
  --gradient-accumulation-steps 1 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --dtype bfloat16 \
  --device cuda
```
