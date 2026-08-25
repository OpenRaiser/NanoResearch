from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanoresearch.evolution.ram_data import RAMTriple  # noqa: E402
from nanoresearch.evolution.sdpo import (  # noqa: E402
    SDPOConfig,
    SDPOExample,
    SDPOTrainer,
    examples_from_triples,
    sdpo_loss,
)
from nanoresearch.evolution.sdpo_adapter import SDPOAdapterManager  # noqa: E402


class TinyTokenizer:
    vocab_size = 128

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        token_ids = [1 + (ord(char) % (self.vocab_size - 1)) for char in text]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 24) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.recurrent = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.adapter_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        # The trainable head stands in for LoRA parameters in this offline test.
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False
        for parameter in self.recurrent.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return SimpleNamespace(logits=self.adapter_head(self.dropout(hidden)))


class RecordingAdapterManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.saved = False

    def save(self, model: torch.nn.Module) -> Path:
        del model
        self.root.mkdir(parents=True, exist_ok=True)
        self.saved = True
        return self.root


def _example() -> SDPOExample:
    return SDPOExample(
        input_text="Plan a small reproducible experiment.",
        output_text="Run a baseline, then evaluate the proposed method.",
        feedback="Use explicit ablations and keep the plan concise.",
        triple_id="ram-test",
    )


def test_sdpo_loss_is_deterministic_and_updates_trainable_parameters():
    torch.manual_seed(3)
    model = TinyCausalModel()
    tokenizer = TinyTokenizer()
    config = SDPOConfig(max_steps=1, max_sequence_length=256, max_trained_tokens=32)

    model.train()
    first = sdpo_loss(
        model=model,
        tokenizer=tokenizer,
        example=_example(),
        config=config,
        device="cpu",
    )
    second = sdpo_loss(
        model=model,
        tokenizer=tokenizer,
        example=_example(),
        config=config,
        device="cpu",
    )

    assert model.training is True
    assert first.ndim == 0
    assert torch.isfinite(first)
    assert torch.allclose(first, second)

    first.backward()
    gradient = model.adapter_head.weight.grad
    assert gradient is not None
    assert torch.any(gradient != 0)
    assert all(parameter.grad is None for parameter in model.embedding.parameters())


def test_sdpo_trainer_runs_gradient_accumulation_and_saves(tmp_path):
    torch.manual_seed(7)
    model = TinyCausalModel()
    tokenizer = TinyTokenizer()
    manager = RecordingAdapterManager(tmp_path / "adapter")
    before = model.adapter_head.weight.detach().clone()

    trainer = SDPOTrainer(
        model,
        tokenizer,
        device="cpu",
        adapter_manager=manager,
        config=SDPOConfig(
            learning_rate=5e-3,
            max_steps=3,
            max_sequence_length=256,
            max_trained_tokens=32,
            gradient_accumulation_steps=2,
        ),
    )
    summary = trainer.train([_example()])

    assert manager.saved is True
    assert summary["num_steps"] == 3
    assert summary["optimizer_steps"] == 2
    assert summary["num_examples"] == 1
    assert not torch.equal(before, model.adapter_head.weight.detach())


def test_sdpo_loss_handles_left_truncation():
    model = TinyCausalModel()
    loss = sdpo_loss(
        model=model,
        tokenizer=TinyTokenizer(),
        example=_example(),
        config=SDPOConfig(
            max_steps=1,
            max_sequence_length=24,
            max_trained_tokens=8,
        ),
        device="cpu",
    )

    assert torch.isfinite(loss)


def test_adapter_manager_saves_unmerged_adapter(tmp_path):
    class FakePeftModel:
        def save_pretrained(self, path: str, *, safe_serialization: bool) -> None:
            assert safe_serialization is True
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")

    manager = SDPOAdapterManager(tmp_path / "adapter")
    output = manager.save(FakePeftModel())

    assert output == tmp_path / "adapter"
    assert manager.exists is True


def test_examples_from_triples_uses_only_complete_interactions():
    base = {
        "timestamp": datetime.now(timezone.utc),
        "subsystem": "method_gen",
        "stage": "planning",
        "x_context": "input",
        "session_id": "session",
        "workspace_id": "workspace",
    }
    complete = RAMTriple(
        triple_id="complete",
        y_output="output",
        o_feedback="feedback",
        o_quality_signal=0.8,
        **base,
    )
    incomplete = RAMTriple(
        triple_id="incomplete",
        y_output="",
        o_feedback="feedback",
        **base,
    )

    examples = examples_from_triples([complete, incomplete])

    assert len(examples) == 1
    assert examples[0].triple_id == "complete"
    assert examples[0].quality_signal == 0.8


def test_sdpo_config_rejects_invalid_training_limits():
    with pytest.raises(ValueError, match="max_steps"):
        SDPOConfig(max_steps=0).validate()
