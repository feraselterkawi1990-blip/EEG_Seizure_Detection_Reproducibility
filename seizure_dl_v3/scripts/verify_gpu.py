"""
Verify CUDA + PyTorch + RTX 4070 are working.
Run this first before anything else.
"""
import sys

print("=" * 60)
print("GPU / PyTorch verification")
print("=" * 60)

try:
    import torch
    print(f"PyTorch version : {torch.__version__}")
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

print(f"CUDA available  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — PyTorch is using CPU only.")
    print("Check that:")
    print("  1. NVIDIA driver is up to date")
    print("  2. CUDA-enabled PyTorch is installed (cu121 or cu124)")
    sys.exit(1)

print(f"CUDA version    : {torch.version.cuda}")
print(f"cuDNN version   : {torch.backends.cudnn.version()}")
print(f"GPU count       : {torch.cuda.device_count()}")
print(f"GPU name        : {torch.cuda.get_device_name(0)}")
print(f"VRAM total      : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Quick benchmark — multiply two 4096×4096 matrices
print()
print("GPU smoke test (matmul 4096×4096):")
torch.cuda.empty_cache()
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()

import time
t0 = time.time()
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  5 multiplications: {elapsed*1000:.1f} ms ({elapsed*200:.1f} ms each)")

print()
print("All checks passed. GPU is ready for training.")
