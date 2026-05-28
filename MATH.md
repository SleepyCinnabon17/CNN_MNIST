# Backpropagation Notes

## Conv2D

The convolution layer performs cross-correlation. For each output position, the
input patch is flattened into a row of an `im2col` matrix. The forward pass is:

```text
Y_col = X_col @ W_flat.T + b
```

Given upstream gradient `dY`, the weight and bias gradients are:

```text
dW_flat = dY_col.T @ X_col
db = sum(dY_col, axis=0)
```

The input gradient is first computed in column form:

```text
dX_col = dY_col @ W_flat
```

`col2im` then scatters each patch gradient back into its original spatial
location. Overlapping windows accumulate by addition. If padding was applied,
the padded border is removed before returning `dX`.

## MaxPool2D

The forward pass stores the argmax location inside every pooling window. During
backward propagation, each upstream scalar is routed only to the cached argmax
position; every other element in that window receives zero gradient.

## Softmax Cross-Entropy

Training uses the combined stable softmax-cross-entropy gradient. Logits are
shifted by the row maximum before exponentiation. For a batch of size `N`:

```text
dZ = (softmax(Z) - one_hot(y)) / N
```

This avoids explicitly forming the softmax Jacobian during training.

The standalone `Softmax` layer remains implemented for API completeness and
inference-style probability flows. Its backward pass computes the per-sample
Jacobian-vector product, but the default training architectures intentionally
end with logits and rely on `SoftmaxCrossEntropyLoss` instead.

The loss is computed from stable log-softmax algebra rather than
`log(prob + eps)`, so the backward expression above is exactly consistent with
the forward objective for finite logits.
