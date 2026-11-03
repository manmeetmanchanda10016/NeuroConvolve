import torch
import torch.nn as nn


class ConvolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@torch.no_grad()
def load_model(model_path="models/neuroconvolve_mlp.pth"):

    model = ConvolutionNet().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict(model, signal1, signal2):

    input_data = torch.tensor(
        list(signal1) + list(signal2),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_data)

    return prediction.squeeze(0).cpu().numpy()