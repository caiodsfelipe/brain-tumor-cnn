# Set main parameters for model training and dataset configuration

# List of class names for the classification task
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Size to which images will be resized (width and height)
image_size = 64

# Number of epochs for which the model will be trained
epochs = 60

# Number of samples per batch during training
batch_size = 32

# Number of folds for cross-validation
n_folds = 5

# Number of classes in the classification task
n_classes = 4
