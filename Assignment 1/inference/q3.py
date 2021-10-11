import numpy as np
import cv2
from PIL import Image
import argparse

import torch
import torchvision
from torchvision import transforms, models

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


parser = argparse.ArgumentParser(description="Ques 3")

parser.add_argument("--img1", required=True)
parser.add_argument("--img2", required=True)

args = parser.parse_args()
img1_path = args.img1
img2_path = args.img2

WTS_PATH1 = "head_detector_wts.pth"
WTS_PATH2 = "matcher_wts.pth"

object_transforms = torchvision.transforms.ToTensor()

img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32)
img1 /= 255.0
img1 = object_transforms(img1)

img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32)
img2 /= 255.0
img2 = object_transforms(img2)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (head) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

saved_wts = torch.load(WTS_PATH1)
model.load_state_dict(saved_wts["best_model_wts"])


model.eval()


def get_bbox(image):
    image = image.to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    boxes = outputs[0]["boxes"].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]["scores"].data.cpu().numpy().astype(np.int32)
    boxes_sorted_idx = np.argsort(scores)[::-1]
    boxes_sorted = boxes[boxes_sorted_idx]
    box = boxes_sorted[0]
    return box


bbox1 = get_bbox(img1)
bbox2 = get_bbox(img2)


data_transforms = transforms.Compose(
    [
        transforms.Resize((110, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

img1 = img1.crop((bbox1[0], bbox1[1], bbox1[2], bbox1[3]))
img2 = img2.crop((bbox2[0], bbox2[1], bbox2[2], bbox2[3]))

img1 = data_transforms(img1)
img2 = data_transforms(img2)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)

img1 = img1.to(device)
img2 = img2.to(device)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 512)
model.to(device)

saved_wts = torch.load(WTS_PATH2, map_location=device)
model.load_state_dict(saved_wts)
model.eval()

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
e1 = model(img1)[0]
e2 = model(img2)[0]
sim = cos(e1, e2)
print("Similarity: ", sim.item())
