import os


def load_data(path):
    images_path = os.path.join(path, 'images')
    labels_path = os.path.join(path, 'labels')

    images = [os.path.join(images_path, img) for img in os.listdir(images_path)]
    labels = [os.path.join(labels_path, lbl) for lbl in os.listdir(labels_path)]

    images.sort()
    labels.sort()

    return images, labels
