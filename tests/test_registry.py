from refrakt.registry.model_registry import get_model
from refrakt.models import *

def test_model_registry():
    class DummyEncoder:
        output_dim = 2048
        def __call__(self, x): return x

    model = get_model("simclr", encoder=DummyEncoder(), projection_dim=128)
    assert model is not None
