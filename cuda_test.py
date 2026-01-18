import torch
import time

if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")

print("Using device: ", device)

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

t0 = time.time()
z = x @ y
if device == "cuda":
  torch.cuda.synchronize()
print("time:", time.time() - t0)