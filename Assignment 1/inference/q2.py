import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description="Ques 2")

parser.add_argument("--img1", required=True)
parser.add_argument("--img2", required=True)

args = parser.parse_args()
img1_path = args.img1
img2_path = args.img2
WTS_PATH = "matcher_wts.pth"


data_transforms = transforms.Compose(
    [
        transforms.Resize((110, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img1 = data_transforms(Image.open(img1_path))
img2 = data_transforms(Image.open(img2_path))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)

img1 = img1.to(device)
img2 = img2.to(device)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 512)
model.to(device)

saved_wts = torch.load(WTS_PATH, map_location=device)
model.load_state_dict(saved_wts)
model.eval()
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
e1 = model(img1)[0]
e2 = model(img2)[0]

sim = cos(e1, e2)
print("Similarity: ", sim.item())
