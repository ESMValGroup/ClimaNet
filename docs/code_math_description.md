# Spatio Temporal Model (class `SpatioTemporalModel`)

**Summary:**

The model takes daily/hourly SST (or residuals) data in video format: `x ∈ ℝ^{B
× 1 × T × H × W}` and a `mask` indicating missing pixels in time and space. It
also takes `land_mask_patch` indicating land regions in the output.

```bash
# 1. Encode spatio-temporal patches
X (VideoEncoder)---------> X_patch

# 2. Spatial mixing
X_patch (MaskedSpatialMix)---------> X_spatial_mixed

# 3. Temporal aggregation:
X_spatial_mixed (TemporalAttentionAggregator)---------> x_averaged

# 4. Add explicit statistics per patch
x_averaged + stat_embed (PatchStats)---------> x

# 5. Decode to original resolution:
x (MonthlyConvDecoder)---------> Output
```

## Process Overview

**Data processing:**

The goal is to learn a correction that turns the average of gappy hourly SST
into the true monthly mean. For each month, the hourly data is arranged into a
fixed grid of 31 days × 24 hours (744 slots), with the unused slots of shorter
months marked as padding. Missing observations are recorded in a mask and
filled with zero, and a separate padding mask and cyclic time features are
stored.

The target is the residual between the monthly product and the plain average of
the available hourly values, so the model learns only the difference caused by
sampling gaps and product differences. Everything is saved per year in chunked
zarr files. A land mask marks pixels that are never predicted, and the target is
divided by the training-set standard deviation.

**Dataset:**

The dataset serves training samples as 40×40-pixel crops of a single month. Each
sample contains the hourly data, the missing-data mask (land is never counted as
missing), the monthly residual target, the land mask, the time features and the
padding mask. Crops don’t overlap, and the data is read lazily from disk, when
needed.

**Data augmentation:**

Augmentation applies only to training data. With 50% probability (you can change
this), additional observations are hidden using realistic gap patterns, such as
scattered hours, spatially coherent multi-day outages or dropped hours of the
day, at a random severity. Because the target is defined relative to the mean of
the available hours, it is recomputed exactly for the new gap pattern. Pixels
left with too few observations are excluded from the loss. This produces many
different gap patterns for the same month, so the model can’t memorize the gaps
and must learn how the sampling pattern relates to the error.

**Model architecture:**

Each month is processed independently. The hourly values are converted to
anomalies relative to each pixel’s own observed mean, which removes the absolute
temperature level and leaves the diurnal and weather-driven variation.

Two views of the data are built. The first is a spatial gap-filling step, in
which missing hours at each pixel are estimated from what neighboring pixels
observed at that same hour, along with a confidence value. The second is a
video-style encoder that turns every hour and 4×4 pixel patch (as an example)
into a token, after which a masked spatial mixing layer lets tokens exchange
information with observed neighbors from the same hour.

Then a temporal attention module learns which hours matter most, using the hour
of day and the relative position within the month, and collapses the hourly
tokens of each patch into one monthly token.

Later, summary statistics of the sampling pattern are added to these tokens.

The decoder upsamples the monthly tokens smoothly to full resolution and
combines them with a separate pixel-level path. This path carries per-pixel
statistics (observed fraction, diurnal sampling and anomaly profile, and the
neighbor-based estimate of how much the missing hours would shift the mean, with
its confidence). A learned per-pixel gate decides how far to trust that
estimate, and convolution blocks produce the final correction map. This removes
the blocky patch artifacts and keeps fine-scale detail, while the land is set to
zero.

**Training:**

The model is trained on mini-batches of crops (due to memory limit) with a
masked mean squared error (MSE) computed only over ocean pixels that have valid
targets.

The optimizer is `AdamW` with a lower learning rate (it can be tunned) when
progress stalls. Activation checkpointing reduces the memory cost of the hourly
tokens. After every epoch the model is evaluated on the validation year, and the
checkpoint with the best validation loss is kept, with early stopping as a
safeguard.

Note that the training loss is higher than the validation loss because it
includes the augmented, harder samples and dropout, and the validation data is
evaluated clean.

**Prediction:**

At inference the model runs in evaluation mode on non-overlapping crops covering
the full grid, without augmentation. It outputs a normalized residual, which is
multiplied by the training standard deviation and added back to the average of
the available hourly data to give the final monthly estimate. The results are
written to netCDF, and the same masked loss is reported for the validation and
test years.

## Details on model architecture

The classes that are initialized in the main model `SpatioTemporalModel`:

- `self.encoder = VideoEncoder(...)`: it reads the raw hourly maps and
  translates them into a compact internal description that the rest of the model
  can work with, one summary for each small area at each hour. It also keeps
  track of which values were actually observed and which were missing.

  A 3D convolution splits each hourly map into a grid of non-overlapping pixel
  blocks and embeds each block as one token, using the validity mask and the
  neighbor-fill channels as extra inputs. The result is a sequence of 744 hourly
  tokens per patch.

- `self.temporal = TemporalAttentionAggregator(...)`: it decides which hours of
  the month matter most for each area and combines them into one monthly
  summary, like a weighted average where the model learns the weights (for
  example, trusting hours that are typical of the day over odd ones).

  An MLP refines each token using the hour-of-day and relative position in the
  month. A scorer then gives each hour a weight, padded hours are masked out,
  and the tokens are averaged with a softmax over those weights.

- `self.fill = NeighborFill(...)`: it gives the model a first guess of what a
  pixel would have seen during the hours it missed, by borrowing from nearby
  pixels that were observed at that time, and it says how much that guess can be
  trusted.

  For every hour, it blurs the observed anomalies of nearby ocean pixels with
  Gaussian kernels, divides by the blurred observation mask, and also returns
  that mask as a confidence channel. It has no learned parameters.

- `self.spatial_mix = MaskedSpatialMix(...)`: it lets each area "look at its
  neighbors" at the same hour, so a gap in one place can be informed by what was
  seen right next to it.

  A depthwise convolution runs over the patch grid, weighted by how much of each
  token was actually observed, and is added back as a residual. Missing tokens
  receive information from observed neighbors, but contribute nothing
  themselves.

- `self.pix_feat = PixelFeatures(...)`: it keeps the fine, pixel-level detail
  about the gaps that would otherwise be lost when pixels are grouped into
  patches. It also provides a direct estimate of how much the missing hours
  would change the monthly average.

  It builds full-resolution per-pixel inputs for the decoder, with no learned
  parameters. These include the observed fraction, the diurnal profile of
  sampling and anomalies by hour of day, and an estimate of how much the missing
  hours would shift the monthly mean, based on `NeighborFill`, together with its
  confidence.

- `self.decoder = MonthlyConvDecoder(...)`: it turns the coarse monthly
  summaries back into a detailed map. It combines the broad picture from the
  tokens with the pixel-level detail, and decides, pixel by pixel, how much to
  rely on the neighbor-based guess.

  It turns monthly tokens into a full-resolution map. It upsamples the tokens
  bilinearly and fuses them with the per-pixel features through convolution
  blocks. A learned per-pixel gate decides how much to trust the neighbor-based
  estimate, which is added to the output, and land is set to zero.

- `self.stats = PatchStats(...)`: it tells the model how the gaps look in each
  area, such as how many observations there were and at which times of day,
  because the sampling pattern alone says a lot about how biased the plain
  average is.

  It computes the same per-pixel diurnal statistics, passes them through a small
  MLP, and averages them over the ocean pixels of each patch. The result is
  added to the monthly tokens before decoding.

## References

- [Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762)
- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://doi.org/10.48550/arXiv.2203.12602)
- [Masked Autoencoders As Spatiotemporal Learners](
https://doi.org/10.48550/arXiv.2205.09113)
- [MAESSTRO: Masked Autoencoders for Sea Surface Temperature Reconstruction under Occlusion- 2024](https://doi.org/10.5194/os-20-1309-2024)
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://doi.org/10.48550/arXiv.1804.07723)
