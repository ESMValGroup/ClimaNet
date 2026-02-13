# Spatio Temporal Model (class `SpatioTemporalModel`)

**Summary:**
- Combines video encoder, temporal attention, spatial transformer, and decoder
- Encodes video into spatio-temporal patches
- Aggregates temporal information per spatial patch
- Mixes spatial features across patches
- Decodes back to original spatial resolution

**Detailed process:**

The model takes daily SST (or similar) data in video format: `x ∈ ℝ^{B × 1 × T ×
H × W}` and a `daily_mask` indicating missing pixels. It also takes
`land_mask_patch` indicating land regions in the output.

```bash
# 1. Patch embedding:
X (VideoEncoder)---------> X_patch

# 2. Add temporal encoding +
# 3. Temporal aggregation:
X_patch + PE (TemporalAttentionAggregator)---------> X_temp_agg

# 4. Add spatial encoding +
# 5. Spatial transformer:
X_temp_agg + PE (SpatialTransformer) ---------> X_mixed

# 6. Decode to original resolution:
X_mixed (MonthlyConvDecoder)---------> Output
```

## 1. Video Encoder (class `VideoEncoder`)

**Summary:**

- Masked pixels are removed but their locations are preserved
- Video is split into 3D patches (time × height × width)
- Each patch becomes a vector (embedding)
- Output is a sequence suitable for Transformer-based video models (e.g. VideoMAE)

**Detailed process:**

We start with a video: Input video `x ∈ R^{B × 1 × T × H × W}` (batch, channel,
time, height, width) and mask ∈ `{0,1}^{B × 1 × T × H × W}` where 1 (True) means
missing / masked at ocean pixels. We define a validity indicator: `valid = 1 −
mask`. So: valid = 1 → observed pixel, valid = 0 → missing pixel. We zero out
missing values: `x_valid = x ⊙ valid`.

We then concatenate the validity mask as a second channel: `x_cat =
concat(x_valid, valid)`. Now the input has 2 channels: `x_cat ∈ R^{B × 2 × T × H
× W}`. This allows the model to know which values were observed and which were
missing.

We split the video into non-overlapping spatio-temporal patches using a 3D
convolution, see
[torch.nn.Conv3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html).
Let the patch size be: `(Pt, Ph, Pw)`. The convolution uses: `kernel size = (Pt,
Ph, Pw)`, `stride = (Pt, Ph, Pw)`. This means each convolution output
corresponds to one patch and patches do not overlap. Resulting shape: `z ∈ R^{B
× D × T' × H' × W'}` where: `D = embed_dim`, `T' = T / Pt, H' = H / Ph, W' = W /
Pw`. Each `(t', h', w')` location is a patch embedding vector of length D.

We flatten the 3D grid of patches into a sequence: `N = T' × H' × W'`. So each
video becomes a sequence of patch embeddings, just like tokens in a Transformer.

For each patch embedding, Layer normalization is done. This stabilizes training
by normalizing across the embedding dimension, see
[torch.nn.LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

Randomly drops elements for regularization is done with Dropout. This helps
prevent overfitting during training, see
[torch.nn.Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

The final output is: `{B × N_patches × embed_dim}`. Each element
represents a spatio-temporal video patch, enriched with: visual information,
knowledge of which pixels were valid or missing.

## 2. Temporal positional encoding (class `TemporalPositionalEncoding`)

**Summary:**

- Each time step gets a unique vector
- Encodings are deterministic and fixed
- No learnable parameters
- Based on the Transformer positional encoding design

**Detailed process:**

The purpose of temporal positional encoding is to generate fixed temporal
position vectors so that a model can know at which time index a feature occurs.
The encoding depends only on time index, not on the data.

Assume a temporal sequence of length: `T = 0, 1, 2, ..., T−1`. Each time index `t`
is assigned a vector of length embedding dim. For time index `t` and
embedding dimension index `i`: Even dimensions use sine, odd dimensions use
cosine. This produces a unique, smooth encoding for each time step.

For a maximum supported temporal length `max_len`, the class precomputes `pe`
where row `t` contains the encoding for time index `t`. This matrix is fixed,
not trainable and stored as a buffer. Later, in `forward` method, given a
requested temporal length `T`, we have `output = PE[0:T]` and resulting shape is
`(T, embed_dim)`. No parameters are learned and no computation depends on input
data.

## 3. Temporal Attention Aggregator (class `TemporalAttentionAggregator`)

This is a temporal attention pooling over sequences of tokens.

**Summary:**

For each spatial patch (h, w):

- Collect its T temporal tokens: each spatial patch (h, w) has T temporal tokens
- Add temporal positional encoding
- Compute a learned scalar score per time step: the model learns which time
  steps are important for this patch
- Apply softmax over time: it ensures the weights form a probability
  distribution over time.
- Compute a weighted temporal sum: output is a temporal summary vector for each
  patch, suitable for downstream tasks.

**Detailed process:**

We start with: `x ∈ ℝ^{B × (T·H·W) × C}`, where `B` = batch size, `T` = number
of temporal tokens per spatial patch, `H`, `W` = number of spatial patches along
height and width, `C` = embedding dimension. We can reshape it to group by
spatial patch where each spatial patch has its temporal sequence of length `T`.

Then we add temporal positional encoding `pe` from `TemporalPositionalEncoding`.
Add it to each temporal token `seq = seq + pe`. This injects time information
into each patch’s token sequence.

Then we compute temporal attention weights by applying `nn.Sequential` to get a
scalar score, see
[torch.nn.Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
Here the explanation over each module in the sequential is as follows:

- `LayerNorm` normalizes the features across the embedding dimension, which
  helps stabilize training. This is a common practice before attention
  computation.
- `Linear(embed_dim, embed_dim)` learns which features are important for temporal weighting.
- `GELU` allows learning non-linear relationships.
- `Linear(embed_dim, 1)` projects to a single scalar score.

see [torch.nn](https://docs.pytorch.org/docs/stable/nn.html) for more details on
each module.

Then, we convert scores into attention weights using softmax over time that
represents importance of each temporal token for this patch.

Then we aggregate temporal tokens by weighted sum over the temporal dimension.
Result is one token per spatial patch.

## 4. Spatial Positional Encoding (class `SpatialPositionalEncoding2D`)

**Summary:**

- Generate fixed sinusoidal positional encodings for 2D spatial grid
- Encodings are not learnable
- Intended to be added to spatial tokens
- Sine and cosine functions of different frequencies allow the model to
  distinguish positions along height and width.

**Detailed process:**

The module generates fixed 2D positional encodings for a grid of size (H, W) and
embedding dimension embed_dim. Each spatial location (h, w) gets a unique vector
of length embed_dim. Encodings are deterministic (not learned) and based on
sine/cosine functions, similar to the `Temporal positional encoding`. The
encoding for each spatial location is a combination of sine and cosine functions
of different frequencies along height and width. This allows the model to know
the spatial position of each token when added to the spatial tokens.

## 5. Spatial Transformer (class `SpatialTransformer`)

It mixes spatial patch tokens using multi-head self-attention.

**Summary:**

- Applies a standard Transformer encoder to spatial tokens
- Mixes information across spatial locations: each patch token can attend to all
  other patches, allowing global spatial context
- Output is a sequence of spatially mixed tokens of the same shape as input

**Detailed process:**

We apply a single Transformer encoder block using `nn.TransformerEncoderLayer`
that performs self-attention and feedforward processing. In Multi-Head
Self-Attention, Every token looks at every other token and with `num_heads=4`,
this happens in 4 parallel "perspectives." In Feedforward Network (MLP), each
token is processed independently through a small neural network to allow complex
feature interactions. This is useful for spatial data and allows:

- Global context: Every patch can "see" every other patch
- Spatial mixing: Information flows across the entire image
- Learning relationships: Model learns which patches are relevant to each other

Then, `nn.TransformerEncoder` stacks multiple encoder layers sequentially; it's
a container that repeats the same transformer block `depth` times.

## 6. Monthly Convolution Decoder (class `MonthlyConvDecoder`)

**Summary:**

- Reshape latent tokens
- Apply 1x1 convolution to mix features
- Use transposed convolution to upsample to original spatial size
- Apply convolutional head for final output
- Optionally mask out land regions
- Add scale and bias to output

**Detailed process:**

We transforms the embedding dimension (default embed_dim=128) to the hidden
dimension (default hidden=128) at each spatial location independently by a 1×1
convolution (also called a "pointwise convolution") using `nn.Conv2d(...,
kernel_size=1)`. Even though both input and output dimensions are 128 by
default, this layer learns a linear transformation to mix and re-weight the
channel features.

Then we use a deconvolution (`ConvTranspose2d`) to map each patch to its
original pixel grid. It converts the low-resolution patch grid back to the
original image resolution. The padding is set to the overlap, which allows for
smooth upsampling and some overlap between patches.

Finally, using `nn.Sequential`, we convert the upsampled features into a single-channel output. This processing block contains:

- `Conv2d`: 3×3 Convolution with padding=1 to refines spatial details after upsampling.
- `BatchNorm2d`: Normalizes the features to stabilize training.
- `GELU`: Non-linear activation to allow complex feature interactions.
- `Conv2d(1×1)`: Final 1×1 convolution to reduce channels to 1, producing the final output map.

A single `Conv2d(64, 1, kernel_size=1)` could work, but the extra layers provide:

- Spatial refinement after deconvolution (which can produce artifacts)
- Non-linearity for more expressive power
- Better gradient flow during training

Then we apply scale, bias, and (optional) land mask to get the final output (he
reconstructed 2D map (e.g., SST) for each batch).

## References

- [Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762)
- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://doi.org/10.48550/arXiv.2203.12602)
- [Masked Autoencoders As Spatiotemporal Learners](
https://doi.org/10.48550/arXiv.2205.09113)
- [MAESSTRO: Masked Autoencoders for Sea Surface Temperature Reconstruction under Occlusion- 2024](https://doi.org/10.5194/os-20-1309-2024
)