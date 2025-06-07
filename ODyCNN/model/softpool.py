import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

__all__ = [
    "SoftPool2d",
    "soft_pool2d",
]

def _pair(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Utility: ensure value is 2‑tuple."""
    if isinstance(v, tuple):
        assert len(v) == 2, "kernel_size/stride must be int or tuple of length 2"
        return v
    return (int(v), int(v))


def soft_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Union[int, Tuple[int, int], None] = None,
    padding: int = 0,
) -> torch.Tensor:
    """Functional interface of *SoftPool* (2‑D).

    Args:
        input: *(B, C, H, W)* tensor.
        kernel_size: pooling window size.
        stride: output stride. If ``None``, defaults to ``kernel_size``.
        padding: zero‑padding added on both sides of the input.

    Returns:
        Down‑sampled tensor of shape *(B, C, H_out, W_out)*.
    """
    kh, kw = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    sh, sw = _pair(stride)

    # Unfold input into patches of shape (B, C*kh*kw, L),
    # where L = H_out * W_out.
    patches = F.unfold(input, kernel_size=(kh, kw), padding=padding, stride=(sh, sw))

    # Reshape to (B, C, kh*kw, L) so we can softmax over the patch dimension.
    B, _, L = patches.shape
    patches = patches.view(B, input.size(1), kh * kw, L)

    # Compute soft weights and weighted sum.
    weights = torch.softmax(patches, dim=2)
    pooled = (patches * weights).sum(dim=2)

    # Recover spatial dims.
    H_out = (input.size(2) + 2 * padding - kh) // sh + 1
    W_out = (input.size(3) + 2 * padding - kw) // sw + 1
    return pooled.view(B, input.size(1), H_out, W_out)


class SoftPool2d(nn.Module):
    """SoftPool down‑sampling layer (2‑D).

    This layer retains all activations in the window, weighting them via
    a softmax so that larger activations contribute more to the result.

    Example:
        >>> pool = SoftPool2d(kernel_size=2)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> y = pool(x)  # (4, 64, 16, 16)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Union[int, Tuple[int, int], None] = None,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return soft_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


if __name__ == "__main__":
    # Minimal sanity check
    x = torch.randn(1, 3, 8, 8)
    sp = SoftPool2d(kernel_size=2)
    y = sp(x)
    print("input:", x.shape, "output:", y.shape)
