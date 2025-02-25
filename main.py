from Dataset_Preparation import prepare_dataset
from Training_Model import train_model
from Prediction import predict_image

def main():
    base_path = './Dataset'
    prepare_dataset(
        base_path=base_path,
        train_source=f'{base_path}/Kidney_stone_train',
        test_source=f'{base_path}/Kidney_stone_test'
    )

    train_model()
    sample_image = './Dataset/Kidney_stone_test/Kidney_stone/1.3.46.670589.33.1.63713387527670842200001.5657070162345407644.png'
    predict_image(sample_image)

if __name__ == "__main__":
    main()

