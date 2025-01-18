import os
import shutil


def prepare_dataset(base_path, train_source, test_source):
    print("Preparing dataset...")
    print("basepath>>>>", base_path)
    os.makedirs(f'{base_path}/train/images', exist_ok=True)
    os.makedirs(f'{base_path}/train/labels', exist_ok=True)
    os.makedirs(f'{base_path}/test/images', exist_ok=True)
    os.makedirs(f'{base_path}/test/labels', exist_ok=True)

    for category in ['Kidney_stone', 'Normal']:
        category_path = os.path.join(train_source, category)
        for img_file in os.listdir(category_path):
            src_path = os.path.join(category_path, img_file)
            dest_path = os.path.join(f'{base_path}/train/images', img_file)
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_file} to train.")

    for category in ['Kidney_stone', 'Normal']:
        category_path = os.path.join(test_source, category)
        for img_file in os.listdir(category_path):
            src_path = os.path.join(category_path, img_file)
            dest_path = os.path.join(f'{base_path}/test/images', img_file)
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_file} to test/images.")

    print("Generating labels for training data...")
    generate_labels(f'{base_path}//train', f'{base_path}/train/labels')
    print("Generating labels for test data...")
    generate_labels(f'{base_path}/test/images', f'{base_path}/test/labels')

    print("Dataset preparation completed.")


def generate_labels(image_folder, labels_folder):
    os.makedirs(labels_folder, exist_ok=True)

    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)

            try:
                with open(label_path, 'w') as f:
                    if 'Normal' in image_folder:
                        f.write(labels_folder+'/0 0.5 0.5 0.5 0.5\n')
                        print(f"Labeled {img_file} as Normal (Class 0).")
                    elif 'Kidney_stone' in image_folder:
                        f.write(labels_folder+'/1 0.5 0.5 0.5 0.5\n')
                        print(f"Labeled {img_file} as Kidney Stone (Class 1).")
                    else:
                        print(f"Warning: {img_file} did not match any known class.")
            except IOError as e:
                print(f"Error writing to {label_path}: {e}")

