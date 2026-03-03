import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

from model import SimpleCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SimpleCNN().to(device)
checkpoint = torch.load("best_model_cnn_cifar.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# CIFAR-10 classes
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435,  0.2616))
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    return {
        classes[i]: float(probs[0][i])
        for i in range(len(classes))
    }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload an image. Model trained from scratch on CIFAR-10."
)

if __name__ == "__main__":
    interface.launch()

