import numpy as np
import cv2
import os
from PIL import Image
import argparse

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description="Ques 1")

parser.add_argument("--dir_path", default="test/images", help="Path to image dir")
args = parser.parse_args()

DIR_INPUT = args.dir_path
DIR_ANNO = "b18153_b18042_Q1_predictions"
os.mkdir(DIR_ANNO)

WTS_PATH = "head_detector_wts.pth"

image_paths = []

for d in os.listdir(DIR_INPUT):
    for i in os.listdir(os.path.join(DIR_INPUT, d)):
        image_paths.append([d, i])


class DetectionDataset(Dataset):
    def __init__(self, image_paths):
        super().__init__()

        self.image_paths = image_paths
        self.transforms = torchvision.transforms.ToTensor()

    def __getitem__(self, index: int):

        image_id = self.image_paths[index]
        image = cv2.imread(
            os.path.join(DIR_INPUT, image_id[0], image_id[1]), cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = self.transforms(image)
        return image, image_id

    def __len__(self) -> int:
        return len(self.image_paths)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (head) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

saved_wts = torch.load(WTS_PATH)
model.load_state_dict(saved_wts["best_model_wts"])
model.eval()

valid_dataset = DetectionDataset(image_paths)
for i in range(len(valid_dataset)):
    image, image_path = valid_dataset[i]
    image = image.to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    boxes = outputs[0]["boxes"].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]["scores"].data.cpu().numpy().astype(np.int32)
    boxes_sorted_idx = np.argsort(scores)[::-1]
    boxes_sorted = boxes[boxes_sorted_idx]
    box = boxes_sorted[0]
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    w = int(box[2] - box[0])
    h = int(box[3] - box[1])

    image_name = image_path[1].split(".")[0]
    if os.path.exists(os.path.join(DIR_ANNO, image_path[0])) == False:
        os.mkdir(os.path.join(DIR_ANNO, image_path[0]))
    f = open(os.path.join(DIR_ANNO, image_path[0], image_name + ".txt"), "w")
    f.write(f"{cx} {cy} {w} {h}")
    f.close()

print("DONE, see the b18153_b18042_Q1_predictions directory")
