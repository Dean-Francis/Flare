import pickle

import pytest
import torch

from server.aggregator import _fedavg
from server.database import WeightUpdateData


class TinyModel(torch.nn.Module):
    """Small stand-in for DistilBERT — same state_dict interface, far cheaper to construct."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.zeros_like(self.linear.weight))


def make_update(delta: dict, num_samples: int) -> WeightUpdateData:
    return WeightUpdateData(num_samples=num_samples, weights=pickle.dumps(delta))


def test_single_update_applies_full_delta():
    model = TinyModel()
    delta = {"linear.weight": torch.ones_like(model.linear.weight)}
    updates = [make_update(delta, num_samples=10)]

    result = _fedavg(updates, model)
    assert torch.allclose(result.state_dict()["linear.weight"], torch.ones_like(model.linear.weight))


def test_equal_weighting_averages_two_deltas():
    model = TinyModel()
    ones  = {"linear.weight": torch.ones_like(model.linear.weight)}
    zeros = {"linear.weight": torch.zeros_like(model.linear.weight)}
    updates = [make_update(ones, 5), make_update(zeros, 5)]

    result = _fedavg(updates, model)
    expected = torch.full_like(model.linear.weight, 0.5)
    assert torch.allclose(result.state_dict()["linear.weight"], expected)


def test_weighting_is_proportional_to_num_samples():
    model = TinyModel()
    delta_a = {"linear.weight": torch.ones_like(model.linear.weight)}        # value 1
    delta_b = {"linear.weight": torch.full_like(model.linear.weight, 5.0)}    # value 5
    updates = [make_update(delta_a, 30), make_update(delta_b, 10)]            # 75/25 split

    result = _fedavg(updates, model)
    # Expected: 0.75*1 + 0.25*5 = 2.0
    expected = torch.full_like(model.linear.weight, 2.0)
    assert torch.allclose(result.state_dict()["linear.weight"], expected)


def test_mismatched_keys_are_dropped_but_others_aggregate():
    model = TinyModel()
    good = {"linear.weight": torch.ones_like(model.linear.weight)}
    bogus = {"some.other.key": torch.ones_like(model.linear.weight)}
    updates = [make_update(good, 10), make_update(bogus, 10)]

    result = _fedavg(updates, model)
    # Only the good update should count; weight should equal the good delta
    assert torch.allclose(result.state_dict()["linear.weight"], torch.ones_like(model.linear.weight))


def test_all_mismatched_keys_raises_value_error():
    model = TinyModel()
    bogus = {"some.other.key": torch.ones_like(model.linear.weight)}
    updates = [make_update(bogus, 10), make_update(bogus, 5)]

    with pytest.raises(ValueError):
        _fedavg(updates, model)
