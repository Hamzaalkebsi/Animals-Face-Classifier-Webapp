from model import ImprovedNet
import torch
from torchvision import transforms
from PIL import Image
import joblib


device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = ImprovedNet().to(device)

# Load just the weights
state_dict = torch.load("model_weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Image transformation
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ]
)


def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def get_prediction(image_path):
    image_tensor = transform_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        pred_index = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred_index])[0]
