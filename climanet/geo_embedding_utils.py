import numpy as np
import torch

from scipy.special import sph_harm_y


def real_sph_harm(order_l, mode_m, theta, phi):
    """
    Real-valued spherical harmonics of order l, and mode m
    at a position (theta, phi) in polar cooridnates.

    Args:
        order_l: int; order of harmonic, l >= 0
        mode_m: int; mode of harmonic -l <= m <= l
        theta: float; colatitude [0, pi]
        phi: float; longitude [0, 2pi]

    Returns:
        ndarray
    """

    Y = sph_harm_y(order_l, abs(mode_m), theta, phi)

    if mode_m < 0:
        return np.sqrt(2) * (-1) ** mode_m * Y.imag
    elif m == 0:
        return Y.real
    else:
        return np.sqrt(2) * (-1) ** mode_m * Y.real


def compute_sh_on_grid(lat, lon, L, dtype=torch.float32):
    """
    Compute spherical harmonics on grid spanned by lat and lon to degree L.
    Note: In the context of climanet this is used in the process of pre-computing
    spehrical harmonic embeddings. It is up to the user to ensure that the grid
    corresponds to that which is used in the training or inference, i.e. that H
    and W are correct

    Args
        lat : (H,) array
        lon : (W,) array
        L   : max harmonic degree

    Returns
        sh : (H, W, D) torch tensor D = (L+1)^2)
    """

    H = len(lat)
    W = len(lon)

    theta = torch.deg2rad(90.0 - torch.tensor(lat, dtype=dtype))  # colatitude
    phi = torch.deg2rad(torch.tensor(lon, dtype=dtype))  # longitude

    D = (L + 1) ** 2

    sh = torch.zeros((H, W, D), dtype=dtype)

    idx = 0

    for order_l in range(L + 1):
        for mode_m in range(-1 * order_l, order_l + 1):
            Y = real_sph_harm(
                order_l,
                mode_m,
                theta[:, None],
                phi[None, :],
            )

            sh[:, :, idx] = torch.as_tensor(Y, dtype=dtype)
            idx += 1

    return sh


def compute_area_weights(lat):
    """
    Compute spherical area weights for latitude rows.
    Actual area of a "cell" defined by dlat dlon on a
    regular lat/lon grid varies with lat.
    Specifically dA ~ sin(theta) dtheta dphi = cos(lat) dlat dlon

    This function provides latitude area weights for use in
    subsequent PCA

    Args:
        lat : (H,) latitude in degrees

    Returns:
        weights : (H,)
    """

    lat_rad = torch.deg2rad(torch.tensor(lat, dtype=torch.float32))

    weights = torch.cos(lat_rad)

    # avoid tiny negatives from floating point
    weights = torch.clamp(weights, min=0.0)

    return weights


def fit_weighted_sh_pca(sh_grid, lat, embed_dim):
    """
    Perform area-weighted PCA on gridded spherical harmonics.
    Scales for (optional) normlaization by eigenvalues (whitening)
    can also be returned.

    Args
        sh_grid : (H,W,D) torch tensor with spherical
            harmonics to order L with D=(L+1)^2
        lat : (H,) latitudes
        embed_dim: int ; target embeddig dimension


    Returns
        mean : (D,)
        components : (D, embed_dim)
        scales (embed_dim,)
    """

    H, W, D = sh_grid.shape

    # reshape grid
    sh_grid = sh_grid.view(H * W, D)

    # calculate area weights
    row_weights = compute_area_weights(lat)  # (H,)
    weights = row_weights[:, None].repeat(1, W).reshape(-1)  # repeat weights arcos lon
    weights = weights / weights.sum()  # normalize

    # center; weighted mean
    wmeanval = (weights[:, None] * sh_grid).sum(dim=0)
    sh_grid_centered = sh_grid - wmeanval[None]

    # weighted covariance
    sh_grid_centered_weighted = sh_grid_centered * torch.sqrt(weights[:, None])

    # perform singular value decomposition SVD
    U, S, Vh = torch.linalg.svd(sh_grid_centered_weighted, full_matrices=False)

    components = Vh[:embed_dim].T.contiguous()

    # whitening scales
    eigenvalues = S[:embed_dim] ** 2  # SV already correspond to weighted covariance
    scales = torch.sqrt(eigenvalues)

    return wmeanval, components, scales


def apply_sh_pca_projection(
    sh_grid, mean, components, eps=1e-5, clip_scale=3.0, emb_scale=0.02
):
    """
    Apply pre-computed weighted PCA projection to grid of spherical harmonics.
    Per dimension regularization is applied as well as tanh based soft-clipping
    to stabalize high frequency modes with pathological fluctuations
    a fixed embeddinng scale factor is applied to avoid dominating embeddings
    (at intial stages)

    Args
        sh_grid : (H,W,D) torch tensor with spherical
            harmonics to order L with D=(L+1)^2
        mean : (D,) torch tensor; spatial mean value for each spherical harmonic
        components : (D, embed_dim) PCA components up to embedding dimension (embed_dim)
        eps : float ; epsilon for numerical stabalization of division by standard deviation
        clip_scale: float; (approximate) number of standard deviations after which to apply
            tanh soft-clipping (suppression of outliers)
        emd_scale: flaot ; scaling parameter for embeddings

    Returns
        sph_emb_grid : (H,W, embed_dim) Spherical harmonics projected to PCA
    """

    H, W, D = sh_grid.shape
    sh_grid = sh_grid.view(H * W, D)

    sh_grid_centered = sh_grid - mean[None]

    # PCA projection
    sph_emb_grid = sh_grid_centered @ components
    # Per dimension regularization
    sph_emb_grid = sph_emb_grid - sph_emb_grid.mean(dim=0, keepdim=True)
    sph_emb_grid = sph_emb_grid / (sph_emb_grid.std(dim=0, keepdim=True) + eps)
    sph_emb_grid = clip_scale * torch.tanh(
        sph_emb_grid / clip_scale
    )  # suppress pathological outliers
    sph_emb_grid = (
        emb_scale * sph_emb_grid
    )  # scale embdiings to addition friendly vlaues

    sph_emb_grid = sph_emb_grid.view(H, W, -1)

    return sph_emb_grid


def calculate_SH_geo_pos_embeddings(
    lat,
    lon,
    L,
    sh_embed_dim,
    eps=1e-5,
    clip_scale=3.0,
    emb_scale=1.0,
    dtype=torch.float32,
):
    """
    calculate SH based geo-positional encodings for an grid of lat/lon cooridnates
    using spherical harmonic functions up to order L and project to embedding
    dimension using area-weighted PCA. Embeddings are scaled to zero mean and
    unit variance, tanh-based softclipping at level 3-sigma is applied.
    Embeddings can be scaled before output using emb_scale
    NOTE:The following should hold: (L+1)**2 >= sh_embed_dim

    Args
        lat : (H,) array
        lon : (W,) array
        L   : max harmonic degree
        sh_embed_dim : target embedding diemnsion
        eps : numerical stablization scale (additive constant when
            dividing by potentially very small value)
        clip_scale : level at which tanh soft-clipping is applied
            value correspond to multiples of sigma (std)
        emd_scale : multiplicative scaling factor for embeddings
        dtype : output datatype

    Returns
        torch tensor (H,W,sh_embed_dim)
    """

    rankSH = (L + 1) ** 2
    if rankSH < sh_embed_dim:
        raise ValueError(
            f"(Rank of SH ({rankSH}= (L+1)**2) is less than requestd PCA dimension {sh_embed_dim}.)"
        )

    # calculate sperical harmonics on grid
    sh_grid = compute_sh_on_grid(lat, lon, L)
    # perform area-weighted PCA decomposition to specified embedding dimension
    mean_val, pca_components, _ = fit_weighted_sh_pca(sh_grid, lat, sh_embed_dim)
    # project grided harmonics to pca basis
    sh_grid_geo_pos_embeddings = apply_sh_pca_projection(
        sh_grid, mean_val, pca_components
    )

    return sh_grid_geo_pos_embeddings


def compute_patch_geo_pos_embedding(geo_pos_patch_grid, lat_patch):
    """
    Compute area weighted mean pca projected SH embedding for patch
    Latitudes are used to calculate area weights as w_i = cos(theta_i)
    with theta_i being the latitude.

    Args
        geo_pos_patch_grid: tensor (pH, pW, sh_embed_dim) ; grid of SH harmonic embeddings of the patch
        lat_patch: tensor (pH,): latitudes of patch

    Returns
        geo_pos_patch: tensor; (sh_embed_dim,); weighted mean patch embedding
    """
    weights = torch.from_numpy(
        np.clip(np.cos(np.deg2rad(lat_patch)), 1e-3, None)
    ).float()  # (pH,)
    weights_2d = weights[:, None, None]  # (pH,1,1)
    # weigthed sum over spatial dimensions
    weighted_sum = (geo_pos_patch_grid * weights_2d).sum(dim=(0, 1))
    # normalization
    normalization = weights.sum() * geo_pos_patch_grid.shape[1]
    geo_pos_patch = weighted_sum / normalization

    return geo_pos_patch


def compute_patch_scale_features(
    lat_patch, lon_patch, earth_radius=6371000.0, eps=1e-6
):
    """
    Compute physical patch geometry scale features.
    These features provide awareness of the physical scales
    represented in input data (i.e. the resolution). With input
    on a regular grid in lat/lon actual patch size, and

    Parameters
    ----------
    lat_patch : (ph,)
    lon_patch : (pw,)
    earth_radius : float ; Earth radius in [m]
    eps : float ; numerical stabliztion

    Returns
    -------
    features : (10,)
        [
            log_dx,
            log_dy,
            log_area,
            log_aspect_ratio,
            log_dx_pix,
            log_dy_pix,
            log_isotropic_scale_pix,
            log_aspect_anisotropy_pix,
            log_spectral_cutoff_x,
            log_spectral_cutoff_y,
        ]
    """

    lat_ext = len(lat_patch)
    lon_ext = len(lon_patch)

    lat_min = np.deg2rad(lat_patch.min())
    lat_max = np.deg2rad(lat_patch.max())

    lon_min = np.deg2rad(lon_patch.min())
    lon_max = np.deg2rad(lon_patch.max())

    lat_c = 0.5 * (lat_min + lat_max)
    cos_lat_c = np.clip(np.cos(lat_c), 1e-3, None)

    dlat = abs(lat_max - lat_min)
    dlon = abs(lon_max - lon_min)

    # north-south extent
    dy = earth_radius * dlat
    dy_pix = dy / max(lat_ext - 1, 1)

    # east-west extent
    dx = earth_radius * cos_lat_c * dlon
    dx_pix = dx / max(lon_ext - 1, 1)

    # stabilize

    area = earth_radius**2 * (np.sin(lat_max) - np.sin(lat_min)) * dlon
    aspect = dx / (dy + eps)

    isotropic_scale_pix = np.sqrt(dx_pix * dy_pix)
    aspect_anisotropy_pix = dx_pix / (dy_pix + eps)
    spectral_cutoff_x = np.pi * earth_radius / (dx + eps)
    spectral_cutoff_y = np.pi * earth_radius / (dy + eps)

    features = np.array(
        [
            np.log(dx + eps),
            np.log(dy + eps),
            np.log(area + eps),
            np.log(aspect + eps),
            np.log(dx_pix + eps),
            np.log(dy_pix + eps),
            np.log(isotropic_scale_pix + eps),
            np.log(aspect_anisotropy_pix + eps),
            np.log(spectral_cutoff_x + eps),
            np.log(spectral_cutoff_y + eps),
        ],
        dtype=np.float32,
    )

    return torch.from_numpy(features).float()
