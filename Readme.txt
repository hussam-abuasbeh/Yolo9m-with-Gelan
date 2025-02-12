
Kidney Stone Detection Project

Overview
--------
This project uses YOLOv9 and PyTorch to detect kidney stones in medical images. The workflow includes preparing the dataset, applying transformations, training the model, and testing it with real data.

The model is trained to classify images into two categories:
1. Kidney Stone (Class 1)
2. Normal (Class 0)

Project Structure
-----------------
.
├── Dataset/
│   ├── Kidney_stone_train/
│   │   ├── Normal/
│   │   └── Kidney_stone/
│   ├── Kidney_stone_test/
│   │   ├── Normal/
│   │   └── Kidney_stone/
├── models/                     # Directory to save the trained model
├── train/                      # Stores generated training images and labels
│   ├── images/
│   └── labels/
└── test/                       # Stores generated test images and labels
    ├── images/
    └── labels/



How to Run the Project
----------------------
Step 1: Prepare the Dataset
Use the prepare_dataset() function to copy images and generate labels.

    prepare_dataset('./Dataset', './Dataset/Kidney_stone_train', './Dataset/Kidney_stone_test')

Step 2: Train the YOLOv9 Model
The model is trained using YOLOv9:

    model = YOLO('yolov9m.pt')
    model.train(
        data='/path/to/kidney_stone_dataset.yaml',
        epochs=40,     #Enable to increase
        batch=16,
        imgsz=416,
        verbose=True,
        workers=2,
        amp=True
    )

Step 3: Test and Predict
Use the trained model to predict kidney stones in test images.

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO('models/kidney_stone_yolov9.pt')
    results = model.predict(source=image_path, save=True, imgsz=640, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            class_name = 'Kidney Stone' if cls == 1 else 'Normal'
            print(f"Predicted Class: {class_name}, Confidence: {conf:.2f}")

Code Explanation
----------------
1. Dataset Preparation
- prepare_dataset() copies images and generates YOLO-compatible label files.

2. YOLO Training
- Trains the YOLOv9 model using kidney stone images.

3. Prediction
- Predicts the class and confidence score for new images.

Dependencies
------------
- Python 3.x
- PyTorch
- Albumentations
- OpenCV
- PIL
- ultralytics (YOLOv9)

