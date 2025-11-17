# test_mps.py
import torch
print("Python:", __import__("platform").python_version())
print("Torch version:", torch.__version__)
try:
    mps_avail = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
except Exception:
    mps_avail = False
    mps_built = False
print("MPS available:", mps_avail)
print("MPS built:", mps_built)