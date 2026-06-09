import numpy as np
import pytest
import torch

from climanet.geo_embedding_utils import (
    real_sph_harm,
    compute_sh_on_grid,
    compute_area_weights,
    fit_weighted_sh_pca,
    apply_sh_pca_projection,
    calculate_sh_geo_pos_embeddings,
    compute_patch_geo_pos_embedding,
    compute_patch_scale_features,
)


# real_sph_harm
def test_real_sph_harm_scalar_output():
    y = real_sph_harm(
        order_l=2,
        mode_m=1,
        theta=np.pi / 4,
        phi=np.pi / 3,
    )

    assert np.isfinite(y)
    assert np.ndim(y) == 0


def test_real_sph_harm_array_shape():
    theta = np.linspace(0, np.pi, 5)[:, None]
    phi = np.linspace(0, 2 * np.pi, 7)[None, :]

    y = real_sph_harm(3, -2, theta, phi)

    assert y.shape == (5, 7)
    assert np.all(np.isfinite(y))


def test_real_sph_harm_m0_is_real():
    theta = np.pi / 4
    phi = np.pi / 5

    y = real_sph_harm(4, 0, theta, phi)

    assert np.isreal(y)


# compute_sh_on_grid
def test_compute_sh_on_grid_shape():
    lat = np.linspace(-90, 90, 8)
    lon = np.linspace(0, 360, 16, endpoint=False)

    L = 3

    sh = compute_sh_on_grid(lat, lon, L)

    expected_dim = (L + 1) ** 2

    assert sh.shape == (8, 16, expected_dim)


def test_compute_sh_on_grid_dtype():
    lat = np.linspace(-45, 45, 4)
    lon = np.linspace(0, 360, 6, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, 2, dtype=torch.float64)

    assert sh.dtype == torch.float64


def test_compute_sh_on_grid_finite():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, 5)

    assert torch.isfinite(sh).all()


# compute_area_weights
def test_compute_area_weights_shape():
    lat = np.array([-90, -45, 0, 45, 90])

    weights = compute_area_weights(lat)

    assert weights.shape == (5,)


def test_compute_area_weights_equator_maximum():
    lat = np.array([-90, 0, 90])

    weights = compute_area_weights(lat)

    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_compute_area_weights_nonnegative():
    lat = np.linspace(-90, 90, 50)

    weights = compute_area_weights(lat)

    assert torch.all(weights >= 0)


# fit_weighted_sh_pca
def test_fit_weighted_sh_pca_shapes():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=4)

    mean, components, scales = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=6,
    )

    D = (4 + 1) ** 2

    assert mean.shape == (D,)
    assert components.shape == (D, 6)
    assert scales.shape == (6,)


def test_fit_weighted_sh_pca_scales_positive():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=3)

    _, _, scales = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=5,
    )

    assert torch.all(scales > 0)


def test_fit_weighted_sh_pca_components_orthogonal():
    lat = np.linspace(-90, 90, 12)
    lon = np.linspace(0, 360, 24, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=4)

    _, components, _ = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=5,
    )

    gram = components.T @ components

    identity = torch.eye(5)

    assert torch.allclose(gram, identity, atol=1e-4)


# apply_sh_pca_projection
def test_apply_sh_pca_projection_shape():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=4)

    mean, components, _ = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=8,
    )

    emb = apply_sh_pca_projection(
        sh,
        mean,
        components,
    )

    assert emb.shape == (10, 20, 8)


def test_apply_sh_pca_projection_finite():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=4)

    mean, components, _ = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=8,
    )

    emb = apply_sh_pca_projection(
        sh,
        mean,
        components,
    )

    assert torch.isfinite(emb).all()


def test_apply_sh_pca_projection_reasonable_scale():
    lat = np.linspace(-90, 90, 12)
    lon = np.linspace(0, 360, 24, endpoint=False)

    sh = compute_sh_on_grid(lat, lon, L=5)

    mean, components, _ = fit_weighted_sh_pca(
        sh,
        lat,
        embed_dim=10,
    )

    emb = apply_sh_pca_projection(
        sh,
        mean,
        components,
        emb_scale=0.02,
    )

    assert emb.abs().mean() < 0.1


# calculate_sh_geo_pos_embeddings
def test_calculate_sh_geo_pos_embeddings_shape():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    emb = calculate_sh_geo_pos_embeddings(
        lat,
        lon,
        L=5,
        sh_embed_dim=12,
    )

    assert emb.shape == (10, 20, 12)


def test_calculate_sh_geo_pos_embeddings_finite():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    emb = calculate_sh_geo_pos_embeddings(
        lat,
        lon,
        L=5,
        sh_embed_dim=12,
    )

    assert torch.isfinite(emb).all()


def test_calculate_sh_geo_pos_embeddings_rank_error():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)

    with pytest.raises(ValueError):
        calculate_sh_geo_pos_embeddings(
            lat,
            lon,
            L=2,
            sh_embed_dim=20,
        )


# compute_patch_geo_pos_embedding
def test_compute_patch_geo_pos_embedding_shape():
    patch = torch.randn(4, 5, 12)

    lat_patch = np.linspace(-20, 20, 4)

    emb = compute_patch_geo_pos_embedding(
        patch,
        lat_patch,
    )

    assert emb.shape == (12,)


def test_compute_patch_geo_pos_embedding_finite():
    patch = torch.randn(6, 7, 10)

    lat_patch = np.linspace(-60, 60, 6)

    emb = compute_patch_geo_pos_embedding(
        patch,
        lat_patch,
    )

    assert torch.isfinite(emb).all()


def test_compute_patch_geo_pos_embedding_constant_field():
    const_val = 3.5

    patch = torch.full((4, 5, 8), const_val)

    lat_patch = np.linspace(-30, 30, 4)

    emb = compute_patch_geo_pos_embedding(
        patch,
        lat_patch,
    )

    expected = torch.full((8,), const_val)

    assert torch.allclose(emb, expected, atol=1e-5)


# compute_patch_scale_features
def test_compute_patch_scale_features_shape():
    lat_patch = np.linspace(-10, 10, 5)
    lon_patch = np.linspace(0, 20, 6)

    features = compute_patch_scale_features(
        lat_patch,
        lon_patch,
    )

    assert features.shape == (10,)


def test_compute_patch_scale_features_finite():
    lat_patch = np.linspace(-30, 30, 8)
    lon_patch = np.linspace(10, 40, 9)

    features = compute_patch_scale_features(
        lat_patch,
        lon_patch,
    )

    assert torch.isfinite(features).all()


def test_compute_patch_scale_features_changes_with_extent():
    lat_small = np.linspace(-5, 5, 5)
    lon_small = np.linspace(0, 10, 5)

    lat_large = np.linspace(-20, 20, 5)
    lon_large = np.linspace(0, 40, 5)

    feat_small = compute_patch_scale_features(
        lat_small,
        lon_small,
    )

    feat_large = compute_patch_scale_features(
        lat_large,
        lon_large,
    )

    # log(dx) and log(dy) should increase
    assert feat_large[0] > feat_small[0]
    assert feat_large[1] > feat_small[1]
