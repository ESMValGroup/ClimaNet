#

## Prediction of Monthly SST

We provide an example of how to use the `st_encoder_decoder` module to predict
monthly sea surface temperature (SST) using a spatio-temporal encoder-decoder
architecture. The example is implemented in a Jupyter notebook, which can be
found in the `notebooks` directory.

### Setting patch sizes

In the example notebook, there are two patch sizes that need to be set: the
spatial patch size in `SpatioTemporalModel` and the spatial patch size in
`STDataset`. These are two different patch sizes with different purposes. Here
we called them `model_patch_size` and `dataset_patch_size` and explained what to
consider when setting these two patch sizes.

- model_patch_size: this is used in `VideoEncoder` where a video is split into
  non-overlapping spatio-temporal patches using a 3D convolution. The patch size
  should be set based on the spatial scale of the input data and the desired
  level of spatial abstraction. A larger patch size will capture more spatial
  context and fits in memory but may also lead to a loss of fine-grained
  details. A smaller patch size will capture finer details with higher costs but
  may also lead to a loss of broader context.

- dataset_patch_size: this is used in `STDataset` where the input data is split
  into (non-overlapping) spatial patches to manage memory in training and
  inference. The overlapping patches is disable because the loss function is
  computed on each patch icluding gaps. Otherwise, some pixels will be included
  in loss computation multiple times. The patch size should be set based on the
  available computational resources and spatial variability of the input data.
  The `dataset_patch_size` should be divisible by `model_patch_size`. We have to
  make sure that data is represntive enough for training purposes. Note that
  large `dataset_patch_size` might lead to memory issues during training and
  inference. During inference on a larger spatial extent, you'll need to handle
  stitching the patches.

- spatial extent of input data: the input data might be used for train_test
  split or for inference. If a model is trained on a specific
  `dataset_patch_size`, the same `dataset_patch_size` should be used for
  inference. If the spatial extent of the input data is larger than the
  `dataset_patch_size`, the dataloader can be used to split the input data into
  patches in inference and the results can be stitched together to get the final
  prediction.Also, you may need to preprocess the predictions for
  edge effect because of non-overlapping patches.

! Note: It is better to have larger `dataset_patch_size` and smaller `model_patch_size`.
