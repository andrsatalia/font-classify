import albumentations as A
import argparse
import numpy as np
import os
import timm
import torch
from pathlib import Path
import json

from albumentations.pytorch import ToTensorV2
from train import CutMax, ResizeWithPad
from PIL import Image


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Inference script")

    # Add arguments
    parser.add_argument(
        "--model_folder",
        type=str,
        default="model_output6",
        help="Path where the trained model was saved",
    )
    parser.add_argument(
        "--data_folder",
        type=Path,
        default="output6",
        help="Path to images to run inference on",
    )
    parser.add_argument(
        "-net",
        "--network_type",
        type=str,
        default="resnet50",
        help="Type of network architecture",
    )
    args = parser.parse_args()

    return args


def main(args):
    with open(os.path.join(args.model_folder, "class_names.txt"), "r") as f:
        class_names = f.read().splitlines()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        args.network_type, pretrained=False, num_classes=len(class_names)
    )
    model.to(device)

    model_path = os.path.join(args.model_folder, "best_model_params.pt")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()

    transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad((320, 320))),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    results = {}
    for i, class_name in enumerate(class_names):
        for image_file in os.listdir(args.data_folder/class_name):
            image_path = os.path.join(args.data_folder/class_name, image_file)
            image = np.array(Image.open(image_path).convert("RGB"))
            image = transform(image=image)["image"].unsqueeze(0).to(device)
            probs = model(image)
            _, prediction = torch.max(probs, 1)
            print(image_file, class_names[prediction])
            results[str(image_path)] = {
                "prediction": {"name": class_names[prediction], "i": int(prediction)},
                "ground_truth": {"name": class_name, "i":i}
            }
    with open("inference_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
