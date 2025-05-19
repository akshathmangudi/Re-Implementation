# Re-Implementation
Currently going refactoring changes right now. Pointers to tackle: 
- [X] Fix modularity: (3 / 3)
    1. ~~Implement a base class for models and loss.~~ 
    2. ~~Implement the other classes using the base class.~~
    3. ~~Add docstrings and improve readability.~~
- [X] Fix implementation: (4 / 4)
    1. ~~Switch to vectorized implementations of the model.~~
    2. ~~Convert SimCLR implementation to follow the base class format.~~
    3. ~~Fix any bottlenecks and add testing for each model and loss functions.~~
    4. ~~Implement a training and testing pipeline to handle available models.~~
- [ ] Implement: (1 / 4)
    1. ~~SimCLR~~
    2. DINO
    3. Masked Autoencoder
    4. MSN
    5. LNN
- [ ] Convert the project into a library. 

Current re-implementations are:

| Paper                                                                                                          | Status    |
|----------------------------------------------------------------------------------------------------------------|-----------|
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | Completed |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)                               | Completed |
| [Autoencoders](https://arxiv.org/abs/2003.05991) | Completed |
| [Swin Transformer: Heirarchial Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | Completed |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Completed | 
| [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545) | Completed | 
| [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) | Completed |
| [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | Completed |
| [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) | Formulation |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | Formulation |
| [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) | Formulation |
| [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) | Formulation |

## Contributions:

These are my personal implementations in order to educate myself. That being said, if there are any issues with the
code, such as incorrect math,
not enough comments or documentation, or poor modularity, please create an issue so I can review and make changes. Pull
requests must be the last resort.

## Licensing:

This repository is under the MIT License. See the LICENSE file for more details.