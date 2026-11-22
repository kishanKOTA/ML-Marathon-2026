"""Interface contract for models/dual_resnet18/model.py (fails until the TODOs there are implemented).

Every other file in models/dual_resnet18/ relies only on this contract:
    DualResNet18(n_age, n_antler, month_dim, month_hidden, pretrained)
    forward(image (B, 3, 224, 224), month (B, 2)) -> (age_logits (B, n_age), antler_logits (B, n_antler))

Run: uv run pytest tests/test_dual_resnet18.py
"""

import math  # month encodings for the wiring test

import pytest  # fixtures
import torch  # tensors
import torch.nn.functional as F  # cross_entropy for the gradient test

from models.dual_resnet18.model import DualResNet18  # model under test

B = 4


@pytest.fixture
def model() -> DualResNet18:
    """Untrained model, no weight download."""
    torch.manual_seed(0)
    return DualResNet18(n_age=2, n_antler=2, month_dim=2, month_hidden=16, pretrained=False)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random images and real month encodings."""
    torch.manual_seed(1)
    months = torch.randint(low=1, high=13, size=(B,)).float()
    angle = 2 * math.pi * months / 12
    return torch.randn(B, 3, 224, 224), torch.stack([angle.sin(), angle.cos()], dim=1)


def test_output_is_tuple_of_two_logit_tensors(model: DualResNet18, batch: tuple) -> None:
    """Forward returns (age_logits (B, 2), antler_logits (B, 2)) as float32."""
    out = model(*batch)
    assert isinstance(out, tuple) and len(out) == 2
    age_logits, antler_logits = out
    assert age_logits.shape == (B, 2) and antler_logits.shape == (B, 2)
    assert age_logits.dtype == torch.float32 and antler_logits.dtype == torch.float32


def test_outputs_are_raw_logits_not_probabilities(model: DualResNet18, batch: tuple) -> None:
    """No softmax inside the model: rows need not sum to 1 or stay in [0, 1]."""
    age_logits, antler_logits = model(*batch)
    for logits in (age_logits, antler_logits):
        is_prob = torch.allclose(logits.sum(dim=1), torch.ones(B), atol=1e-4) and (logits >= 0).all()
        assert not is_prob


def test_gradients_reach_backbone_and_month_mlp(model: DualResNet18, batch: tuple) -> None:
    """One backward on the summed loss updates both input branches."""
    age_logits, antler_logits = model(*batch)
    loss = F.cross_entropy(age_logits, torch.tensor([0, 1, 0, 1])) + F.cross_entropy(
        antler_logits, torch.tensor([1, 0, 0, 1])
    )
    loss.backward()
    backbone_grad = next(p.grad for p in model.backbone.parameters() if p.requires_grad)
    month_grad = next(p.grad for p in model.month_mlp.parameters() if p.requires_grad)
    assert backbone_grad is not None and backbone_grad.abs().sum() > 0
    assert month_grad is not None and month_grad.abs().sum() > 0


def test_month_input_changes_outputs(model: DualResNet18, batch: tuple) -> None:
    """Same images, January vs July: outputs must differ (month branch is wired in)."""
    model.eval()
    images, _ = batch
    jan = torch.tensor([[math.sin(2 * math.pi / 12), math.cos(2 * math.pi / 12)]]).expand(B, 2)
    jul = torch.tensor([[math.sin(2 * math.pi * 7 / 12), math.cos(2 * math.pi * 7 / 12)]]).expand(B, 2)
    with torch.no_grad():
        age_jan, ant_jan = model(images, jan)
        age_jul, ant_jul = model(images, jul)
    assert not torch.allclose(age_jan, age_jul) or not torch.allclose(ant_jan, ant_jul)
