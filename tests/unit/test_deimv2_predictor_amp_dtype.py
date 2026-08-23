"""The predictor's autocast outputs must survive ``.numpy()``.

Regression for gen003 (slurm job 490). Training ran all 24 epochs cleanly,
then the post-training scoring pass died immediately:

    File "kwcoco_detector_kit/trainers/deimv2.py", line 880, in predict_image
      s = scores[0].cpu().numpy()
    TypeError: Got unsupported ScalarType BFloat16

numpy has a float16 dtype but no bfloat16, so every ``.numpy()`` downstream of
the eval autocast worked for as long as the autocast was fp16 and broke the
moment it became bf16. The fix casts floating outputs back to float32 inside
``_forward`` -- the single boundary both predict_image and predict_batch pass
through -- while leaving int64 labels alone.
"""
import pytest

torch = pytest.importorskip("torch")

from kwcoco_detector_kit.trainers.deimv2 import DEIMv2Predictor


class _StubModel:
    """Stands in for DEIM: returns (labels, boxes, scores) in a given dtype."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, im, sz):
        labels = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        boxes = torch.tensor([[[0., 0., 4., 4.]] * 3], dtype=self.dtype)
        scores = torch.tensor([[0.9, 0.5, 0.1]], dtype=self.dtype)
        return labels, boxes, scores


def _predictor(dtype, *, use_amp=False):
    """A predictor with just enough state for _forward, no checkpoint needed."""
    obj = object.__new__(DEIMv2Predictor)
    obj._model = _StubModel(dtype)
    obj._use_amp = use_amp          # False -> nullcontext, so this runs on CPU
    obj._amp_dtype_name = "bfloat16"
    obj._device = "cpu"
    return obj


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_forward_outputs_are_numpy_convertible(dtype):
    pred = _predictor(dtype)
    labels, boxes, scores = pred._forward(None, None)
    # The actual failure mode: this raised TypeError for bfloat16.
    scores[0].cpu().numpy()
    boxes[0].cpu().numpy()
    labels[0].cpu().numpy()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_upcasts_floating_outputs_to_float32(dtype):
    pred = _predictor(dtype)
    _, boxes, scores = pred._forward(None, None)
    assert boxes.dtype == torch.float32
    assert scores.dtype == torch.float32


def test_forward_leaves_integer_labels_alone():
    """A blanket .float() would turn class indices into floats."""
    pred = _predictor(torch.bfloat16)
    labels, _, _ = pred._forward(None, None)
    assert labels.dtype == torch.int64
    assert labels[0].tolist() == [1, 2, 3]


def test_forward_preserves_values_through_the_cast():
    pred = _predictor(torch.float32)
    _, boxes, scores = pred._forward(None, None)
    assert scores[0].tolist() == pytest.approx([0.9, 0.5, 0.1])
    assert boxes[0][0].tolist() == pytest.approx([0.0, 0.0, 4.0, 4.0])
