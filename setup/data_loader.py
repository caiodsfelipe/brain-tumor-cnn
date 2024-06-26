import cv2
from tqdm import tqdm
import os
from sklearn.utils import shuffle
import numpy as np
from setup.parameters import image_size

def preprocessing(img):
    # Resize image
    img = cv2.resize(img, (image_size, image_size))

    # create a CLAHE object
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Revert color to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

# Reads and preprocess the images in a folder
def process_images(folder_path):
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = preprocessing(img)
        images.append(img)
    return images

def load_data(class_names, train_ratio=0.8):
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []

    for class_name in class_names:
        if class_name != 'glioma':
            for data_type in ['Training', 'Testing']:
                folder_path = os.path.join('..\data\mri_dataset', data_type, class_name)
                X, Y = (training_images, training_labels) if data_type == 'Training' else (test_images, test_labels)
                images = process_images(folder_path)
                X.extend(images)
                Y.extend([class_name] * len(images))

    glioma_folder_path = os.path.join('..\data\glioma_dataset\\new\\new\Contrast_Stretching', 'glioma')
    glioma_images = process_images(glioma_folder_path)
    glioma_labels = ['glioma'] * len(glioma_images)
    glioma_images, glioma_labels = shuffle(glioma_images, glioma_labels)

    split_index = int(len(glioma_images) * train_ratio)
    training_images.extend(glioma_images[:split_index])
    training_labels.extend(glioma_labels[:split_index])
    test_images.extend(glioma_images[split_index:])
    test_labels.extend(glioma_labels[split_index:])

    training_images = np.array(training_images) / 255.0
    test_images = np.array(test_images) / 255.0

    return training_images, training_labels, test_images, test_labels
