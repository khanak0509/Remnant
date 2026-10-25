import pytest
import torch

from config import WINDOW
from models import MODELS


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("out", [1, 2])
def test_forward_shape(name, out):
    model = MODELS[name](14, out=out)
    assert model(torch.randn(8, WINDOW, 14)).shape == (8, out)
