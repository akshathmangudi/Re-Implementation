# Re-Implementation
Currently going refactoring changes right now. Pointers to tackle: 
- [ ] Fix modularity: (0 / 3)
    1. Implement a base class for models and loss. 
    2. Implement the other classes using the base class. 
    3. Add docstrings and improve readability. 
- [ ] Fix implementation: (0 / 4)
    1. Switch to vectorized implementations of the model. 
    2. Fix any bottlenecks and add testing for each model and loss functions. 
    3. Implement logging onto TensorBoard or Wandb.
    4. Implement a training and testing pipeline to handle available models.
- [ ] Implement: (1 / 4)
    1. ~~SimCLR~~
    2. DINO
    3. Masked Autoencoder
    4. MSN

Current re-implementations are:

| Paper                                                                                                          | Code                 | Status    |
|----------------------------------------------------------------------------------------------------------------|----------------------|-----------|
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | [ViT](./vit)     | Completed |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)                               | [ResNet](./resnet) | Completed |
| [Autoencoders](https://arxiv.org/abs/2003.05991) | [Autoencoders](./autoencoders) | Completed |
| [Swin Transformer: Heirarchial Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | [Swin](./swin) | Completed |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [Transformers](./attention) | Completed | 
| [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545) | [ConvNeXt](./convnext/) | Completed | 
| [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) | [SRGAN](./srgan) | Completed |
| [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | [SimCLR](./nbs/simclr.ipynb) | In Progress

## Contributions:

These are my personal implementations in order to educate myself. That being said, if there are any issues with the
code, such as incorrect math,
not enough comments or documentation, or poor modularity, please create an issue so I can review and make changes. Pull
requests must be the last resort.

## Licensing:

This repository is under the MIT License. See the LICENSE file for more details.