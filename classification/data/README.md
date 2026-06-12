# ImageNet-1K preparation

Obtain ImageNet-1K from the official provider under its access policy. The validation directory must use class folders:

```
/path/to/imagenet/val/n01440764/*.JPEG
...
```

The paper uses `Resize(256)`, `CenterCrop(224)`, and either `[0.5,0.5,0.5]` normalization for the ViT checkpoints or standard ImageNet normalization for DeiT, selected with `--preprocess inception|imagenet`.
