import torch

# Just load and examine the model structure directly
checkpoint = torch.load("/scratch/vchaurasia/simple_unet_models/best_model.pth", map_location='cpu')

print("Keys in checkpoint:", list(checkpoint.keys()))
print("\nModel structure:")

state_dict = checkpoint['model_state_dict']
for key, tensor in list(state_dict.items())[:10]:  # First 10 layers
    print(f"{key}: {tensor.shape}")