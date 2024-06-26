# An Approach in Brain Tumor Classification: The Development of a New Convolutional Neural Network Model

This repository contains the code used in the research article titled "An Approach in Brain Tumor Classification: The Development of a New Convolutional Neural Network Model" This study presents a CNN model designed to identify and classify brain tumors from MRI. We used GRAD-CAM to validate our results. For more information, here is the article: https://doi.org/10.5753/eniac.2023.233530 

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/caiodsfelipe/brain-tumor-cnn.git
    cd brain_tumor_cnn
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv/Scripts/activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the environment (e.g., configure Kaggle API key):
    ```bash
    mkdir -p ~/.kaggle
    cp /path/to/your/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

5. Download and unzip the datasets (assuming the commands in the notebook):
    ```bash
    kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/mri_dataset
    kaggle datasets download -d rahimanshu/figshare-brain-tumor-classification -p data/glioma_dataset
    kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/validation_dataset
    unzip data/mri_dataset/brain-tumor-mri-dataset.zip -d data/mri_dataset
    unzip data/glioma_dataset/figshare-brain-tumor-classification.zip -d data/glioma_dataset
    unzip data/validation_dataset/brain-tumor-classification-mri.zip -d data/validation_dataset
    ```

Thats it, now you can use the **notebooks/brain_tumor_cnn** notebook to run the code!
