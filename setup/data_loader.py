import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from setup.parameters import image_size

def preprocess_image(img):
    """Preprocess a single image."""
    img_resized = cv2.resize(img, (image_size, image_size))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = clahe.apply(img_gray)
    
    img_rgb = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)
    return img_rgb

def load_images_from_folder(folder_path):
    """Load and preprocess all images from a given folder."""
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        preprocessed_img = preprocess_image(img)
        images.append(preprocessed_img)
    return images

def load_data(class_names, train_ratio=0.8):
    """Load and process data for all classes."""
    training_data = {'images': [], 'labels': []}
    test_data = {'images': [], 'labels': []}
    
    for class_name in class_names:
        if class_name != 'glioma':
            for data_type in ['Training', 'Testing']:
                folder_path = os.path.join('..', 'data', 'mri_dataset', data_type, class_name)
                data = training_data if data_type == 'Training' else test_data
                images = load_images_from_folder(folder_path)
                data['images'].extend(images)
                data['labels'].extend([class_name] * len(images))
    
    # Handle 'glioma' class separately
    glioma_folder_path = os.path.join('..', 'data', 'glioma_dataset', 'new', 'new', 'Contrast_Stretching', 'glioma')
    glioma_images = load_images_from_folder(glioma_folder_path)
    glioma_labels = ['glioma'] * len(glioma_images)
    
    glioma_images, glioma_labels = shuffle(glioma_images, glioma_labels)
    split_index = int(len(glioma_images) * train_ratio)
    
    training_data['images'].extend(glioma_images[:split_index])
    training_data['labels'].extend(glioma_labels[:split_index])
    test_data['images'].extend(glioma_images[split_index:])
    test_data['labels'].extend(glioma_labels[split_index:])
    
    # Normalize pixel values
    training_data['images'] = np.array(training_data['images']) / 255.0
    test_data['images'] = np.array(test_data['images']) / 255.0
    
    return (training_data['images'], training_data['labels'], 
            test_data['images'], test_data['labels'])
