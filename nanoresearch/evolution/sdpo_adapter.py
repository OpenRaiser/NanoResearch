"""Trainable LoRA adapter lifecycle for RAM SDPO updates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import PreTrainedModel


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class SDPOAdapterManager:
    """Create, resume, and save the trainable RAM LoRA adapter."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir).expanduser()

    @property
    def exists(self) -> bool:
        return (self.output_dir / "adapter_config.json").is_file()

    @staticmethod
    def make_config(
        *,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    ) -> LoraConfig:
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            inference_mode=False,
        )

    def attach_trainable(
        self,
        base_model: PreTrainedModel,
        *,
        config: LoraConfig | None = None,
    ) -> Any:
        """Attach an existing adapter or create a new trainable one."""
        from peft import PeftModel, get_peft_model

        if self.exists:
            return PeftModel.from_pretrained(
                base_model,
                str(self.output_dir),
                is_trainable=True,
            )
        return get_peft_model(base_model, config or self.make_config())

    def save(self, model: Any) -> Path:
        """Persist adapter weights without merging them into the base model."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.output_dir), safe_serialization=True)
        return self.output_dir
