from lit_gpt.model import GPT
from lit_gpt.config import Config
from lit_gpt.tokenizer import Tokenizer
from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "Lit-GPT requires Lightning nightly (future lightning 2.1). Please run:\n"
        f" pip uninstall -y lightning; pip install -r requirements.txt\n{str(_LIGHTNING_AVAILABLE)}"
    )

try:
    from lit_gpt.fused_cross_entropy import FusedCrossEntropyLoss as _FusedCrossEntropyLoss
except Exception:  # pragma: no cover
    import torch

    class FusedCrossEntropyLoss(torch.nn.Module):
        def __init__(
            self,
            ignore_index: int = -100,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
            **_: object,
        ) -> None:
            super().__init__()
            self.ignore_index = ignore_index
            self.reduction = reduction
            self.label_smoothing = label_smoothing

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

else:
    FusedCrossEntropyLoss = _FusedCrossEntropyLoss


__all__ = ["GPT", "Config", "Tokenizer"]
