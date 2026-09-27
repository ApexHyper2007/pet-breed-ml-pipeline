import sys

import torch
from PIL import Image
from torchvision import transforms

from model import PetResNet


# Class names
class_names = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier"
]


# Check image filename
if len(sys.argv) < 2:
    print("Please provide an image filename.")
    print("Example: python predict.py cat.jpg")
    exit()


image_path = sys.argv[1]


# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Load model
model = PetResNet().to(device)

model.load_state_dict(
    torch.load(
        "best_pet_resnet18.pth",
        map_location=device
    )
)

model.eval()


# Load image
image = Image.open(image_path).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

image = image.to(device)


# Prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    top_probabilities, top_classes = torch.topk(
        probabilities, 5
    )


print("\nTop 5 predictions:")

for i in range(5):

    breed = class_names[top_classes[0][i].item()]

    confidence = (
        top_probabilities[0][i].item() * 100
    )

    print(
        f"{i + 1}. {breed}: {confidence:.2f}%"
    )