"""Tests for scripts/model.py. Run from the repo root with: python -m pytest -v"""

import pytest

tf = pytest.importorskip("tensorflow")

from scripts import model as m


@pytest.fixture(scope="module")
def backbone():
    """Build once for all tests. weights=None gives the same architecture without downloading."""
    return m.build_backbone(weights=None)


def test_output_shape_is_7x7x1024(backbone):
    """A 224x224 batch should produce a 7x7x1024 feature map for each image."""
    assert m.check_output_shape(backbone, batch_size=2) == (2, 7, 7, 1024)


def test_fully_frozen_has_no_trainable_parameters(backbone):
    """With 0 unfrozen blocks (the paper's setup), nothing in the backbone should train."""
    m.set_backbone_trainable(backbone, 0)
    assert m.count_parameters(backbone)["trainable"] == 0


def test_unfreezing_one_block_only_touches_conv5(backbone):
    """Unfreezing 1 block should make only conv5 layers trainable."""
    m.set_backbone_trainable(backbone, 1)
    trainable = [layer.name for layer in backbone.layers if layer.trainable and layer.weights]
    assert trainable, "expected some trainable layers"
    assert all(name.startswith("conv5_") for name in trainable)


def test_more_unfrozen_blocks_means_more_trainable_parameters(backbone):
    """Trainable parameters should increase with every additional unfrozen block."""
    counts = []
    for n in range(5):
        m.set_backbone_trainable(backbone, n)
        counts.append(m.count_parameters(backbone)["trainable"])
    assert counts[0] == 0
    assert all(a < b for a, b in zip(counts, counts[1:]))


def test_total_parameters_never_change(backbone):
    """Freezing changes what trains, not how many parameters exist."""
    m.set_backbone_trainable(backbone, 0)
    frozen_total = m.count_parameters(backbone)["total"]
    m.set_backbone_trainable(backbone, 4)
    assert m.count_parameters(backbone)["total"] == frozen_total


def test_batchnorm_stays_frozen_when_unfreezing(backbone):
    """BatchNorm layers should stay frozen even with every block unfrozen."""
    m.set_backbone_trainable(backbone, 4)
    bn_layers = [l for l in backbone.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert bn_layers and not any(l.trainable for l in bn_layers)


def test_refreezing_works(backbone):
    """Setting 0 after unfreezing should freeze everything again."""
    m.set_backbone_trainable(backbone, 3)
    m.set_backbone_trainable(backbone, 0)
    assert m.count_parameters(backbone)["trainable"] == 0


@pytest.mark.parametrize("bad_n", [-1, 5])
def test_invalid_unfreeze_value_raises(backbone, bad_n):
    """Values outside 0-4 should raise a clear error."""
    with pytest.raises(ValueError):
        m.set_backbone_trainable(backbone, bad_n)


def test_unknown_backbone_raises():
    """An unsupported backbone name should raise a clear error."""
    with pytest.raises(ValueError):
        m.build_backbone(name="not_a_model", weights=None)