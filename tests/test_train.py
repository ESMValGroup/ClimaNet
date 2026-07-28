import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import _run_one_batch


@pytest.fixture
def dummy_batch():
    # a dummy batch for testing
    return {
        "daily_patch": torch.rand(1, 1, 2, 31, 40, 40),
        "monthly_patch": torch.rand(1, 2, 40, 40),
        "daily_mask_patch": torch.rand(1, 1, 2, 31, 40, 40) > 0.5,  # boolean mask
        "land_mask_patch": torch.rand(1, 40, 40) > 0.5,  # boolean mask
        "daily_timef_patch": torch.rand(1, 2, 31, 3),
        "padded_days_mask": torch.rand(1, 2, 31) > 0.5,  # boolean mask
        "scale_feature_patch": torch.rand(1, 10),
        "geo_pos_embedding_patch": torch.rand(1, 96),
        "sh_embed_dim": torch.rand(1),
        "harmonic_order": torch.rand(1),
        "scale_f_dim": torch.rand(1),
        "coords": torch.rand(1, 2),
        "lat_patch": torch.rand(1, 40),
        "lon_patch": torch.rand(1, 40),
    }


def test_model_meta_device(dummy_batch):
    """Test that the model can run on a meta device and compute loss without errors.

    Device is set to 'meta' for fast model construction, shape propagation,
    and validating model architecture without executing ops.

    """
    model = SpatioTemporalModel(
        patch_size=(1, 4, 4),
        overlap=2,
        embed_dim=64,
        dropout=0.2,
        hidden=64,
        use_checkpoint=True,
    )
    device = "meta"

    model = model.to(device)
    dummy_batch = {k: v.to(device, non_blocking=False) for k, v in dummy_batch.items()}

    model.train()
    loss = _run_one_batch(model, dummy_batch)
    loss.backward()

    assert loss.device.type == "meta"


def test_model_fake_tensor(dummy_batch):
    """Test that the model can run with fake tensors and compute loss without errors.

    This test uses fake tensors to test operator dispatch, device placement, and
    graph correctness without executing real kernels.

    For this test, checkpointing is disabled to avoid issues with fake tensors and autograd.

    """
    model = SpatioTemporalModel(
        patch_size=(1, 4, 4),
        overlap=2,
        embed_dim=64,
        dropout=0.2,
        hidden=64,
        use_checkpoint=False,
    )
    device = "cpu"

    model = model.to(device)

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        dummy_batch = {k: mode.from_tensor(v) for k, v in dummy_batch.items()}
        loss = _run_one_batch(model, dummy_batch)
        loss.backward()

    assert loss.device.type == "cpu"
