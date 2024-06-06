import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)


def patchify(images, n_patches):
    """
    In order to "sequentially" pass in the images, we can break down the main image into multiple sub-images
    and map them to a vector. This is exactly what this function does.

    Arguments:
    images: The image passed into this function
    n_patches: The number of patches to split the image into.

    Returns our patches aka the sub-images.
    """
    n, c, h, w = images.shape

    assert h == w, "Only for square images"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def positional_embeddings(sequence_length, d):
    """
    In order for the model to know where to place each image, one can use positional embeddings where high freq values
    are classified into the first few dimensions while low frequency values are added on to the latter dimensions. This
    function performs exactly that. It has two parameters.

    Arguments:
    sequence_length: The number of tokens for the dataset.
    d: The dimensionality for each token.

    Returns a matrix where each (i,j) is added as token i in dimension j.
    """
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** (j / d)))
    return result


class MSA(nn.Module):
    """
    This is the template implementation of the "Multi-Scale Attention" Layer. 

    The query, key and value mapping are matrix-multipled against each other in order to 
    find the attention, or, the relation of a word and its interaction with surrounding words. 
    """
    def __init__(self, d, n_heads=4):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0  # Shouldn't divide dimension (d) into n_heads

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class Residual(nn.Module):
    """
    This is how a Residual Layer is built. The MSA that we have written will be a part 
    of this residual block right here. 
    """

    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(Residual, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    """
    The workflow will be as follows. 
        1. Find the linear mapping of the input
        2. Embed them using the function that we have written
        3. Use 'n' MSA blocks and add a linear and a softmax layer at the end
    """

    def __init__(self, chw, n_patches=16, n_blocks=2, hidden_d=8, n_heads=4, out_d=10):
        super(ViT, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patch sizes
        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapping
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Classification token
        self.v_class = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.register_buffer('positional_embeddings', positional_embeddings(n_patches ** 2 + 1, hidden_d),
                             persistent=False)

        # Encoder blocks
        self.blocks = nn.ModuleList([MSA(hidden_d, n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # running tokenization
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.v_class.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]
        return self.mlp(out)
