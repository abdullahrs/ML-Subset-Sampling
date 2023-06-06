from resnet_classifer import ImageClassifier
from time import time


def main():
    image_classifier = ImageClassifier(
        batch_size=256,
        num_epochs=16,
        learning_rate=0.01,
        dataset_paths={
            'train_path': 'D:/train',  # 'E:/İndirmeler/temp/ILSVRC/Data/CLS-LOC/train',
        },
        start_num_samples=1024,
        sample_multiplier=4,
        normal_model_epoch=64,
        # load_subset_model=True,
    )
    start_time = time()
    image_classifier.run_test()
    end_time = time()
    print(f"Total time: {end_time - start_time} ".ljust(100, '='))


if __name__ == '__main__':
    main()
