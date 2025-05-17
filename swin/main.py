import torch
from models.swint import SwinTransformer

device = 'cuda' if torch.cuda.is_available else 'cpu'

def main():
    x = torch.randn((1, 3, 224, 224)).to(device)
    model = SwinTransformer().to(device)
    print(model(x).shape)

if __name__ == "__main__":
    main() # This will return the tuple (1, 49, 768), completing our computation whcih could be used for various datasets. 