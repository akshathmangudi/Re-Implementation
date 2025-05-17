import torch
import numpy as np

def patchify(images, n_patches):
    n, c, h, w = images.shape
    assert h == w, "Only square images supported"
    patch_size = h // n_patches
    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** (j / d)))
    return result