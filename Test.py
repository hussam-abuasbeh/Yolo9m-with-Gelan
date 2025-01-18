import os

label_dir = '/Users/7usamabuasbeh/PycharmProjects/Yolo9/Dataset/train/labels'
class_1_count = 0

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        content = f.readlines()
        for line in content:
            if line.startswith('0'):
                class_1_count += 1

def size_files(folder_path):
    image_counts = {}
    valid_extensions = (".jbg", ".jpeg", ".png", ".gif", ".bmp", "tiff", "webp")
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            print("dir_name", dir_name)
            subfolder = os.path.join(root, dir_name)
            print("subfolder", subfolder)
            images = [
                file for file in os.listdir(subfolder)
                if file.lower().endswith(valid_extensions)
            ]
            image_counts[dir_name] = len(images)
    return image_counts


if __name__ ==  "__main__":
    folder_path = "Dataset/Kidney_stone_train"
    image_counts = size_files(folder_path)

    # Print the results
    for folder, count in image_counts.items():
        print(f"Folder '{folder}' contains {count} image(s).")



