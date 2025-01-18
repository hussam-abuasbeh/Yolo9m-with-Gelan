import os
from PIL import Image
from ultralytics import YOLO


model = YOLO('runs/detect/train2/weights/best.pt')

image_path = '/Users/7usamabuasbeh/PycharmProjects/Yolo9/Dataset/Kidney_stone_test/Kidney_stone/1.3.46.670589.33.1.63740863580178781200001.5086422129430942692.png'

results = model.predict(source=image_path, save=True, imgsz=640, conf=0.5)

save_dir = results[0].save_dir

image_name = os.path.basename(image_path).replace('.png', '.jpg')
predicted_image_path = os.path.join(save_dir, image_name)

predicted_image = Image.open(predicted_image_path).convert("RGB")
predicted_image.show()

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        class_name = 'Kidney Stone' if cls == 1 else 'Normal'
        print(f"Predicted Class: {class_name}, Confidence: {conf:.2f}")



