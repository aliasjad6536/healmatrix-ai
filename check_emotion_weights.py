import torch

before = torch.load("checkpoints/emotion_weights_BEFORE.pt", map_location="cpu")
after = torch.load("checkpoints/emotion_weights_AFTER.pt", map_location="cpu")

print("Before shape:", before.shape)
print("After shape :", after.shape)
print("Same weights?", torch.equal(before, after))
print("Average difference:", torch.abs(after - before).mean().item())

print("\nBefore first 4 values:", before[0][:4].tolist())
print("After  first 4 values:", after[0][:4].tolist())