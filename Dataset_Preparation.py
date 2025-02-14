import os
import shutil

def prepare_dataset(base_path, train_source, test_source):
    print("Preparing dataset...")
    os.makedirs(f'{base_path}/train/images', exist_ok=True)
    os.makedirs(f'{base_path}/train/labels', exist_ok=True)
    os.makedirs(f'{base_path}/test/images', exist_ok=True)
    os.makedirs(f'{base_path}/test/labels', exist_ok=True)

    for category in ['Kidney_stone', 'Normal']:
        category_path_train = os.path.join(train_source, category)
        category_path_test = os.path.join(test_source, category)
        label_value = 1 if category == 'Kidney_stone' else 0

        for img_file in os.listdir(category_path_train):
            src_path = os.path.join(category_path_train, img_file)
            dest_path = os.path.join(f'{base_path}/train/images', img_file)
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_file} to train as {dest_path}.")
            generate_label(dest_path, f'{base_path}/train/labels', label_value)

        for img_file in os.listdir(category_path_test):
            src_path = os.path.join(category_path_test, img_file)
            dest_path = os.path.join(f'{base_path}/test/images', img_file)
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_file} to test/images as {dest_path}.")
            generate_label(dest_path, f'{base_path}/test/labels', label_value)

    print("Dataset preparation completed.")


def generate_label(image_path, labels_folder, label_value):
    os.makedirs(labels_folder, exist_ok=True)
    img_file = os.path.basename(image_path)
    print("img_file", img_file)
    label_file = os.path.splitext(img_file)[0] + '.txt'
    print("label_file", label_file)
    label_path = os.path.join(labels_folder, label_file)
    print("label_path", label_path)

    try:
        with open(label_path, 'w') as f:
            f.write(f'{label_value} 0.5 0.5 0.5 0.5\n')
            print(f"Labeled {img_file} as Class {label_value}.")
    except Exception as e:
        print(f"Error writing to {label_path}: {e}")
