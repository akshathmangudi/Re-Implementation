import torch
import tempfile
from models.srgan import SRGAN


def test_generator_output_shape():
    model = SRGAN(scale_factor=4)
    model.generator.eval()
    input_tensor = torch.randn(1, 3, 48, 48)  # 3-channel input (RGB)
    with torch.no_grad():
        output = model.generator(input_tensor)
    expected_shape = (1, 3, 192, 192)  # 48 * 4 = 192
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_discriminator_output_shape():
    model = SRGAN(scale_factor=4)
    model.discriminator.eval()
    input_tensor = torch.randn(1, 3, 192, 192)
    with torch.no_grad():
        output = model.discriminator(input_tensor)
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"


def test_generate_function():
    model = SRGAN(scale_factor=4)
    input_tensor = torch.randn(1, 3, 48, 48)
    output = model.generate(input_tensor)
    assert output.shape == (1, 3, 192, 192)
    assert (0 <= output).all() and (output <= 1).all(), "Output should be in [0, 1] range"


def test_discriminate_function():
    model = SRGAN(scale_factor=4)
    input_tensor = torch.randn(1, 3, 192, 192)
    output = model.discriminate(input_tensor)
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"
    assert (0 <= output).all() and (output <= 1).all(), "Discriminator output should be in [0, 1] range"


def test_model_save_and_load():
    model = SRGAN(scale_factor=4)
    input_tensor = torch.randn(1, 3, 48, 48)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        path = tmp.name
        model.save_model(path)
        new_model = SRGAN(scale_factor=4)  # use different init
        new_model.load_model(path)
        output = new_model.generate(input_tensor)
        assert output.shape == (1, 3, 192, 192)
