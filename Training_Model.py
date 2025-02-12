import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import albumentations as A
import cv2
from utils import load_data


class KidneyStoneDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        image = torch.tensor(image).permute(2, 0, 1).float()
        return image, label


def train_model(train_path='./Dataset/train', test_path='./Dataset/test'):

    train_transform = A.Compose([
        A.Resize(416, 416),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = A.Compose([
        A.Resize(416, 416),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_images, train_labels = load_data(train_path)
    test_images, test_labels = load_data(test_path)

    print(f"Found {len(train_images)} normal images and {len(test_images)} Test images images.")
    print(f"Found {len(train_labels)} normal labels and {len(test_labels)} Test images labels.")

    X_train, X_temp, y_train, y_temp = train_test_split(train_images, train_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    train_dataset = KidneyStoneDataset(X_train, y_train, transform=train_transform)
    val_dataset = KidneyStoneDataset(X_val, y_val, transform=val_transform)
    test_dataset = KidneyStoneDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    print("train_loader>>>", train_loader)
    print("val_loader>>>", val_loader)
    print("test_loader>>>", test_loader)

    os.environ['WANDB_MODE'] = 'disabled'

    model = YOLO('yolov9m.pt')
    model.train(
            data='/Users/7usamabuasbeh/PycharmProjects/Yolo9/kidney_stone_dataset.yaml',
            epochs=40,
            batch=16,
            imgsz=416,
            verbose=True,
            workers=2,
            amp=True
    )

    # model_dir = '/Users/7usamabuasbeh/PycharmProjects/Yolo9/models'
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, 'kidney_stone_yolov9.pt')
    # model.save(model_path)
    # print(f"Model saved at: {model_path}")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_model()
