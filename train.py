import albumentations as A
import argparse
import cv2
import numpy as np
import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.optim import lr_scheduler
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Tuple

# Set device
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Training script")

    # Add arguments
    parser.add_argument(
        "--image_folder",
        type=str,
        default="output9",
        help="Path to the folder containing the images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="model_output9_no_transforms",
        help="Path to the folder where the trained model will be saved",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of the dataset to be used for testing",
    )
    parser.add_argument(
        "-net",
        "--network_type",
        type=str,
        default='resnet50',
        help="Type of network architecture",
    )#swin_base_patch4_window12_384 "resnet50"
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )#5e-5 0.0001
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=200, help="Number of epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, **kwargs):
        super(CustomImageFolder, self).__init__(root, **kwargs)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = np.array(sample)  # Convert PIL image to numpy array
            transformed = self.transform(image=sample)  # Apply Albumentations transform
            sample = transformed["image"]  # Extract transformed image

        return sample, target


class CustomImageFolderSWIN(ImageFolder):

    def __init__(self, root, transform, **kwargs):
        super(CustomImageFolderSWIN, self).__init__(root, **kwargs)
        self.feature_extractor = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        inputs = self.feature_extractor(sample)
        return inputs, target
    

class ResizeWithPad:

    def __init__(
        self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
    ) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )
        return image


class CutMax:
    """Cuts the image to the maximum size"""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Cuts the image to the maximum size"""
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    # Define the  model
    network_type = args.network_type
    model = timm.create_model(
        network_type, pretrained=True, num_classes=5
    )
    model.to(device)
    
    data_config = timm.data.resolve_model_data_config(model)
    transform_swin = timm.data.create_transform(**data_config, is_training=False)
    

    # Define a custom transform function to preprocess the images using Albumentations
    image_transforms = [
        # A.Lambda(image=CutMax(1024)),
        A.Lambda(image=ResizeWithPad((512, 512))),  # Custom SquarePad
        A.RandomCrop(320, 320)
        # A.ShiftScaleRotate(
        #     shift_limit=0.1,
        #     scale_limit=(-0.2, 0.2),
        #     rotate_limit=15,
        #     interpolation=1,
        #     p=0.7,
        # ),
        # # A.RandomBrightnessContrast(p=0.2),
        # A.ColorJitter(p=0.2),
        # A.ISONoise(p=0.2),
        # A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2)
    ]

    transform = A.Compose(
        image_transforms +
        [
            # A.CenterCrop(320, 320),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    check_transform = A.Compose(
        image_transforms
    )

    # Access the arguments
    image_folder = args.image_folder
    # label_file = args.label_file
    best_model_params_path = os.path.join(args.output_folder, "best_model_params.pt")

    # Create an instance of the custom dataset
    dataset = CustomImageFolder(image_folder, transform=transform)
    # dataset = CustomImageFolderSWIN(image_folder, transform_swin)
    n = len(dataset)  # total number of examples
    n_test = int(args.test_split * n)  # take ~10% for test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n - n_test, n_test]
    )

    check_dataset = CustomImageFolder(image_folder, transform=check_transform)
    Path(os.path.join(args.output_folder, "check")).mkdir(parents=True, exist_ok=True)
    for i, data in zip(range(100), check_dataset):
        img = data[0]
        Image.fromarray(img).save(os.path.join(args.output_folder, "check", f"{i}.png"))

    # Save classnames to a txt file
    class_names = dataset.classes
    with open(os.path.join(args.output_folder, "class_names.txt"), "w") as f:
        for item in class_names:
            f.write(f"{item}\n")
    print(f"Found {len(class_names)} classes.")

    # test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
    # train_set = torch.utils.data.Subset(dataset, range(n_test, n))  # take the rest
    dataset_sizes = {"train": len(train_dataset), "val": len(test_dataset)}

    # Create a dataloader for the dataset
    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
    )
    dataloaders = {"train": train_dataloader, "val": test_dataloader}

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    total_steps = len(train_dataloader) * args.num_epochs  # Number of iterations in total
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate*10, total_steps=total_steps, pct_start=0.1)

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # num_training_steps = len(train_dataloader) * args.num_epochs // 1
    # warmup_steps = int(0.1 * num_training_steps)
    # scheduler = get_linear_schedule_with_warmup(
    # optimizer,
    # num_warmup_steps=warmup_steps,
    # num_training_steps=num_training_steps
    # )


    # Create a TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}/{args.num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # ⭐️ ⭐️ Autocasting
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0],epoch)
                        scheduler.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == "train":
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {scheduler.get_last_lr()}")

            # Write the loss to TensorBoard
            writer.add_scalar(f"{phase}_Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase}_Accuracy", epoch_acc, epoch)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        print()

    # Save the trained model
    torch.save(
        model.state_dict(), os.path.join(args.output_folder, "trained_model.pth")
    )

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    args = parse_args()

    main(args)
