import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import albumentations as A
import matplotlib
import tensorflow as tf
import cv2
matplotlib.use('Agg')


def load_and_preprocess_image(image_path, label, transform=None):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform:
        augmented = transform(image=image)
        image = augmented['image']

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label



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
train_normal_images_path = './Dataset/Kidney_stone_train/Normal'
train_stone_images_path = './Dataset/Kidney_stone_train/Kidney_stone'
test_normal_images_path = './Dataset/Kidney_stone_test/Normal'
test_stone_images_path = './Dataset/Kidney_stone_test/Kidney_stone'

normal_images = [os.path.join(normal_images_path, img) for img in os.listdir(normal_images_path)]
stone_images = [os.path.join(stone_images_path, img) for img in os.listdir(stone_images_path)]

print(f"Found {len(normal_images)} normal images and {len(stone_images)} kidney stone images.")

all_images = normal_images + stone_images
labels = [0] * len(normal_images) + [1] * len(stone_images)

X_train, X_temp, y_train, y_temp = train_test_split(all_images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

train_normal = sum(1 for label in y_train if label == 0)
train_stone = sum(1 for label in y_train if label == 1)

print(f"Training on {len(X_train)} images: {train_normal} normal and {train_stone} kidney stone.")

train_dataset = load_and_preprocess_image(X_train, y_train, transform=train_transform)
val_dataset = load_and_preprocess_image(X_val, y_val, transform=val_transform)
test_dataset = load_and_preprocess_image(X_test, y_test, transform=val_transform)

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

model_dir = '/Users/7usamabuasbeh/PycharmProjects/Yolo9/models'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'kidney_stone_yolov9.pt')
model.save(model_path)
print(f"Model saved at: {model_path}")

torch.cuda.empty_cache()
