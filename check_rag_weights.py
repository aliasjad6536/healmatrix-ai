import torch

before = torch.load("checkpoints/bge_weights_before.pt", map_location="cpu")
after = torch.load("checkpoints/bge_weights_after.pt", map_location="cpu")

print("Before shape:", before.shape)
print("After shape :", after.shape)
print("Same weights?", torch.equal(before, after))
print("Average difference:", torch.abs(after - before).mean().item())
