import os
import io
from PIL import Image
from ultralytics import YOLO
import PySimpleGUI as sg

model = YOLO('runs/detect/train20/weights/best.pt')

# image_path = '/Users/7usamabuasbeh/PycharmProjects/Yolo9/Dataset/Kidney_stone_test/Kidney_stone/1.3.46.670589.33.1.63740863580178781200001.5086422129430942692.png'
#
# results = model.predict(source=image_path, save=True, imgsz=640, conf=0.5)
#
# save_dir = results[0].save_dir
#
# image_name = os.path.basename(image_path).replace('.png', '.jpg')
# predicted_image_path = os.path.join(save_dir, image_name)
#
# predicted_image = Image.open(predicted_image_path).convert("RGB")
# predicted_image.show()
#
# for result in results:
#     for box in result.boxes:
#         cls = int(box.cls[0])
#         conf = box.conf[0]
#         class_name = 'Kidney Stone' if cls == 1 else 'Normal'
#         print(f"Predicted Class: {class_name}, Confidence: {conf:.2f}")

def predict_image(image_path):
    results = model.predict(source=image_path, save=True, imgsz=640, conf=0.5)
    save_dir = results[0].save_dir
    image_name = os.path.basename(image_path).replace('.png', '.jpg')
    predicted_image_path = os.path.join(save_dir, image_name)

    details = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            class_name = 'Kidney Stone' if cls == 1 else 'Normal'
            details.append(f"Class: {class_name}, Confidence: {conf:.2f}")

    return predicted_image_path, details

layout = [
    [sg.Text("Select an Image for Kidney Stone Detection:")],
    [sg.Input(key="-FILE-", enable_events=True, visible=False), sg.FileBrowse("Browse", file_types=(("PNG Files", "*.png")))],
    [sg.Image(key="-IMAGE-", size=(400, 400))],
    [sg.Text("Prediction Results:", key="-RESULTS-", size=(40, 5))],
    [sg.Button("Exit")]
]

window = sg.Window("Kidney Stone Detection", layout, resizable=True)

while True:
    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

    if event == "-FILE-":
        image_path = values["-FILE-"]
        print("Selected File Path:", image_path)
        if image_path:
            try:
                predicted_image_path, predictions = predict_image(image_path)

                predicted_image = Image.open(predicted_image_path).convert("RGB")
                predicted_image.thumbnail((400, 400))
                bio = io.BytesIO()
                predicted_image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
                window["-RESULTS-"].update("\n".join(predictions))
            except Exception as e:
                print(f"Error: {e}")

window.close()

