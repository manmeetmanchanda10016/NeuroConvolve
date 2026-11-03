from model import load_model

model = load_model()

print("NeuroConvolve MLP loaded successfully!")
print("Device:", next(model.parameters()).device)