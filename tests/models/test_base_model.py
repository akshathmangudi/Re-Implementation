import pytest
import torch
import torch.nn as nn

from refrakt_core.models.templates.base import BaseModel


class TestModel(BaseModel):
    def __init__(self):
        super(TestModel, self).__init__(model_name="test_model", model_type="test")
        self.layer = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.layer(x)

class TestBaseModel:
    @pytest.fixture
    def model(self):
        return TestModel()
    
    def test_init(self, model):
        """Test that the model initializes correctly."""
        assert model.model_name == "test_model"
        assert model.model_type == "test"
        assert model.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_to_device(self, model):
        """Test that to_device method works."""
        cpu_device = torch.device("cpu")
        model = model.to_device(cpu_device)
        assert model.device == cpu_device
        
        for param in model.parameters():
            assert param.device == cpu_device
    
    def test_predict(self, model):
        """Test that predict method works."""
        x = torch.randn(2, 10)
        output = model.predict(x)
        assert output.shape == (2, 5)
    
    def test_save_load(self, model, tmp_path):
        """Test save and load functionality."""
        save_path = tmp_path / "model.pt"
        
        model.save_model(str(save_path))
        assert save_path.exists()
        
        new_model = TestModel()
        new_model.load_model(str(save_path))
        
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.all(torch.eq(p1, p2))
        
        assert new_model.model_name == model.model_name
        assert new_model.model_type == model.model_type
    
    def test_summary(self, model):
        """Test summary method."""
        summary = model.summary()
        
        # Verify the contents of the summary
        assert summary["model_name"] == "test_model"
        assert summary["model_type"] == "test"
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert summary["device"] == torch.device("cuda" if torch.cuda.is_available() else "cpu")