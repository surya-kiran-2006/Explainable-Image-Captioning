import platform
import sys
import torch

print("Operating system:", platform.platform())
print("Architecture:", platform.machine())
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("MPS built:", getattr(torch.backends, "mps", None).is_built() if hasattr(torch.backends, "mps") else False)
print("MPS available:", getattr(torch.backends, "mps", None).is_available() if hasattr(torch.backends, "mps") else False)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    x = torch.ones((2, 2), device="mps")
    print("MPS test tensor:", x)
    print("SUCCESS: Apple GPU acceleration is available.")
else:
    print("MPS is unavailable. The app can still run on CPU, but more slowly.")
    print("Check that this is an Apple Silicon Mac and that macOS/PyTorch versions support MPS.")
