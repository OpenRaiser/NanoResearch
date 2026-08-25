"""Self-Distillation Policy Optimization for NanoResearch RAM.

The implementation follows NanoResearch equations 14--15.  A feedback-
conditioned forward pass acts as a stop-gradient self-teacher, while the
unconditioned forward pass updates only trainable adapter parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Iterable

from .ram_data import RAMTriple
from .sdpo_adapter import SDPOAdapterManager

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SDPOConfig:
    learning_rate: float = 1e-4
    max_steps: int = 50
    max_sequence_length: int = 2048
    max_trained_tokens: int = 512
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    advantage_clip: float = 5.0
    max_grad_norm: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if self.max_trained_tokens <= 0:
            raise ValueError("max_trained_tokens must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True)
class SDPOExample:
    input_text: str
    output_text: str
    feedback: str
    triple_id: str = ""
    quality_signal: float = 0.0

    @classmethod
    def from_triple(cls, triple: RAMTriple) -> "SDPOExample":
        return cls(
            input_text=triple.x_context,
            output_text=triple.y_output,
            feedback=triple.o_feedback,
            triple_id=triple.triple_id,
            quality_signal=triple.o_quality_signal,
        )

    def validate(self) -> None:
        if not self.input_text.strip():
            raise ValueError("SDPO example input_text is empty")
        if not self.output_text.strip():
            raise ValueError("SDPO example output_text is empty")
        if not self.feedback.strip():
            raise ValueError("SDPO example feedback is empty")


def examples_from_triples(triples: Iterable[RAMTriple]) -> list[SDPOExample]:
    """Convert completed RAM interactions into validated SDPO examples."""
    examples: list[SDPOExample] = []
    for triple in triples:
        example = SDPOExample.from_triple(triple)
        try:
            example.validate()
        except ValueError:
            logger.warning("Skipping incomplete SDPO triple %s", triple.triple_id)
            continue
        examples.append(example)
    return examples


def _encode(tokenizer: Any, text: str, device: str) -> torch.Tensor:
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device)


def _feedback_prompt(input_text: str, feedback: str) -> str:
    return (
        input_text.rstrip()
        + "\n\n<hindsight_context>\n"
        + "The following is user feedback on the prior response. "
        + "Internalize it when predicting that response:\n"
        + feedback.strip()
        + "\n</hindsight_context>\n"
    )


def _model_inputs(
    *,
    tokenizer: Any,
    prompt_text: str,
    response_ids: torch.Tensor,
    max_sequence_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    prompt_ids = _encode(tokenizer, prompt_text, device)
    input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    if input_ids.size(1) > max_sequence_length:
        input_ids = input_ids[:, -max_sequence_length:]
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _tail_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    train_token_count: int,
) -> torch.Tensor:
    import torch

    if train_token_count <= 0 or input_ids.size(1) <= train_token_count:
        raise ValueError("Not enough context tokens for SDPO next-token prediction")
    start = input_ids.size(1) - train_token_count - 1
    end = input_ids.size(1) - 1
    prediction_logits = logits[:, start:end, :]
    labels = input_ids[:, -train_token_count:]
    log_probs = torch.log_softmax(prediction_logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def sdpo_loss(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    example: SDPOExample,
    config: SDPOConfig,
    device: str,
) -> torch.Tensor:
    """Compute one SDPO loss with gradients through the student only."""
    import torch

    config.validate()
    example.validate()

    response_ids = _encode(tokenizer, example.output_text, device)
    if response_ids.size(1) == 0:
        raise ValueError("SDPO response tokenization produced no tokens")

    student_ids, student_mask = _model_inputs(
        tokenizer=tokenizer,
        prompt_text=example.input_text,
        response_ids=response_ids,
        max_sequence_length=config.max_sequence_length,
        device=device,
    )
    teacher_ids, teacher_mask = _model_inputs(
        tokenizer=tokenizer,
        prompt_text=_feedback_prompt(example.input_text, example.feedback),
        response_ids=response_ids,
        max_sequence_length=config.max_sequence_length,
        device=device,
    )

    train_token_count = min(
        response_ids.size(1),
        config.max_trained_tokens,
        student_ids.size(1) - 1,
        teacher_ids.size(1) - 1,
    )
    if train_token_count <= 0:
        raise ValueError("SDPO example has no trainable response tokens")

    # Evaluation mode removes dropout noise while still allowing gradients.
    was_training = model.training
    model.eval()
    try:
        student_logits = model(
            input_ids=student_ids,
            attention_mask=student_mask,
        ).logits
        student_logp = _tail_token_logprobs(
            student_logits,
            student_ids,
            train_token_count,
        )

        with torch.no_grad():
            teacher_logits = model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
            ).logits
            teacher_logp = _tail_token_logprobs(
                teacher_logits,
                teacher_ids,
                train_token_count,
            )

        advantage = (teacher_logp - student_logp).detach()
        if config.advantage_clip > 0:
            advantage = advantage.clamp(
                -config.advantage_clip,
                config.advantage_clip,
            )
        return -(advantage * student_logp).mean()
    finally:
        if was_training:
            model.train()


class SDPOTrainer:
    """Optimize and persist a trainable RAM LoRA adapter."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        adapter_manager: SDPOAdapterManager,
        config: SDPOConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adapter_manager = adapter_manager
        self.config = config or SDPOConfig()

    def train(self, examples: list[SDPOExample]) -> dict[str, float]:
        import torch
        from torch.optim import AdamW

        self.config.validate()
        if not examples:
            raise ValueError("No completed RAM triples are available for SDPO training")
        for example in examples:
            example.validate()

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters; attach a trainable LoRA adapter first")

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        optimizer = AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer.zero_grad(set_to_none=True)
        self.model.eval()

        losses: list[float] = []
        optimizer_steps = 0
        for step in range(self.config.max_steps):
            example = examples[step % len(examples)]
            raw_loss = sdpo_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                example=example,
                config=self.config,
                device=self.device,
            )
            loss = raw_loss / self.config.gradient_accumulation_steps
            loss.backward()
            losses.append(float(raw_loss.detach().cpu()))

            should_step = (
                (step + 1) % self.config.gradient_accumulation_steps == 0
                or step + 1 == self.config.max_steps
            )
            if should_step:
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        self.model.eval()
        output_path = self.adapter_manager.save(self.model)
        summary = {
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "num_steps": float(self.config.max_steps),
            "optimizer_steps": float(optimizer_steps),
            "num_examples": float(len(examples)),
        }
        logger.info("SDPO training complete: %s (adapter=%s)", summary, output_path)
        return summary


def load_model_for_sdpo(
    *,
    model_name_or_path: str,
    adapter_manager: SDPOAdapterManager,
    device: str = "auto",
    dtype: str = "bfloat16",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> tuple[Any, Any, str]:
    """Load a causal LM and attach a trainable LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(dtype_map)}")
    torch_dtype = torch.float32 if resolved_device == "cpu" else dtype_map[dtype]
    if resolved_device == "mps" and torch_dtype == torch.bfloat16:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(resolved_device)
    base_model.config.use_cache = False

    lora_config = adapter_manager.make_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    model = adapter_manager.attach_trainable(base_model, config=lora_config)
    return model, tokenizer, resolved_device
