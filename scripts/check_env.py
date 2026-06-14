"""Verify the local GPU is visible to PyTorch. Run once after setup."""
from __future__ import annotations


def main() -> None:
    import torch

    available = torch.cuda.is_available()
    print(f"torch={torch.__version__} cuda_available={available}")
    if available:
        print(f"gpu={torch.cuda.get_device_name(0)}")
    assert available, "CUDA not visible to PyTorch. Training will be slow or fail."


if __name__ == "__main__":
    main()
