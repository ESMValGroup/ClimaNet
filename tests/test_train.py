import pytest
import torch

from climanet.train import _run_one_batch
from climanet.st_encoder_decoder import SpatioTemporalModel
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture
def create_dummy_batch():
    # create dummy batch for testing
    batch = {
        "daily_patch": torch.rand(1, 1, 2, 31, 40, 40),
        "monthly_patch": torch.rand(1, 2, 40, 40),
        "daily_mask_patch": torch.rand(1, 1, 2, 31, 40, 40) > 0.5, # make it boolean mask
        "land_mask_patch": torch.rand(1, 40, 40) > 0.5, # make it boolean mask
        "daily_timef_patch": torch.rand(1, 2, 31, 2),
        "padded_days_mask": torch.rand(1, 2, 31) > 0.5, # make it boolean mask
        "scale_feature_patch": torch.rand(1, 10),
        "geo_pos_embedding_patch": torch.rand(1, 96),
        "sh_embed_dim": torch.rand(1),
        "harmonic_order": torch.rand(1),
        "scale_f_dim": torch.rand(1),
        "coords": torch.rand(1, 2),
        "lat_patch": torch.rand(1, 40),
        "lon_patch": torch.rand(1, 40),
    }
    return batch


def test_model_meta_device(create_dummy_batch):
    """ Test that the model can run on a meta device and compute loss without errors.

    Device is set to 'meta' for fast model construction, shape propagation,
    and validating model architecture without executing ops.

    """
    batch = create_dummy_batch
    model = SpatioTemporalModel(
        patch_size=(1, 4, 4),
        overlap=2,
        num_months=2,
        embed_dim=64,
        dropout=0.2,
        hidden=64,
        use_checkpoint=True
    )
    device= "meta"

    model = model.to(device)
    decoder = model.decoder
    mean, std = torch.rand(1), torch.rand(1)
    with torch.no_grad():
        decoder.bias.copy_(mean)
        decoder.scale.copy_(std + 1e-6)

    batch = {
    k: v.to(device, non_blocking=False)
    for k, v in batch.items()
    }

    model.train()
    loss = _run_one_batch(model, batch)
    loss.backward()

    assert loss.device.type == "meta"


def test_model_fake_tensor(create_dummy_batch):
    """ Test that the model can run with fake tensors and compute loss without errors.

    This test uses fake tensors to test operator dispatch, device placement, and
    graph correctness without executing real kernels.

    """
    batch = create_dummy_batch
    model = SpatioTemporalModel(
        patch_size=(1, 4, 4),
        overlap=2,
        num_months=2,
        embed_dim=64,
        dropout=0.2,
        hidden=64,
        use_checkpoint=False
    )
    device = "cpu"

    model = model.to(device)
    decoder = model.decoder
    mean, std = torch.rand(1), torch.rand(1)
    with torch.no_grad():
        decoder.bias.copy_(mean)
        decoder.scale.copy_(std + 1e-6)

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        batch = { k: mode.from_tensor(v) for k, v in batch.items() }
        loss = _run_one_batch(model, batch)
        loss.backward()
