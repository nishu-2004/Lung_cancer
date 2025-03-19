# Lung_cancer

Got it! If you haven't uploaded the dataset to GitHub, we can structure the README file to focus on the code, methodology, and results without directly referencing the dataset file. Here's an updated **README.md** that avoids mentioning the dataset file explicitly:

---

# **Lung Cancer Prediction Project**

## **Overview**
This project focuses on predicting lung cancer survival based on patient attributes such as hypertension, asthma, cirrhosis, BMI, cholesterol levels, and temporal features related to diagnosis and treatment. The goal is to develop a robust machine learning model that can accurately classify whether a patient will survive based on these features.

The project involves:
- **Data Preprocessing**: Handling missing values, feature engineering, and addressing class imbalance.
- **Model Training**: Building and training an Artificial Neural Network (ANN) using TensorFlow/Keras.
- **Evaluation**: Assessing model performance using accuracy and loss metrics.

---

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Methodology](#methodology)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Future Enhancements](#future-enhancements)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Project Structure**
```
Lung_cancer/
├── plots/                   # Folder for saving plots
│   ├── mutual_info.jpg      # Mutual information scores plot
│   ├── loss_curve.jpg       # Training and validation loss curves
│   └── correlation_heatmap.jpg  # Feature correlation heatmap
├── model1.h5            # Trained ANN model
├── train.py             # Script for training the model
├── README.md                # Project README file
└── requirements.txt         # List of dependencies
```

---

## **Methodology**
### **Data Preprocessing**
1. **Feature Engineering**:
   - Extracted day, month, and year from diagnosis and treatment dates to create temporal features.
   - Computed mutual information scores to identify the most important features for prediction.
2. **Handling Class Imbalance**:
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) and Random UnderSampling to balance the dataset.
   - Adjusted the class distribution from 80:20 to 40:60.
3. **Feature Scaling**:
   - Standardized numerical features using `StandardScaler` to ensure consistent scaling.

### **Model Training**
- **Model Architecture**:
  - An Artificial Neural Network (ANN) with multiple dense layers, batch normalization, and dropout for regularization.
  - Optimized using the Adam optimizer with a learning rate of 1e-4.
  - Trained using binary cross-entropy loss for binary classification.
- **Callbacks**:
  - Model checkpointing to save the best model weights.
  - Early stopping to prevent overfitting.
  - Learning rate reduction on plateau for better convergence.

### **Evaluation**
- The model's performance was evaluated using training and validation accuracy, as well as loss curves.
- Key metrics:
  - **Validation Accuracy**: Approximately 70%.
  - **Training Accuracy**: Similar to validation accuracy, indicating no significant overfitting.

---

## **Installation**
To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishu-2004/Lung_cancer.git
   cd Lung_cancer
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
To train the model and generate results, run the following script:

```bash
python scripts/train.py
```

### **Script Details**
- **Input**: The script expects a dataset with features such as hypertension, asthma, cirrhosis, BMI, cholesterol levels, and temporal features.
- **Output**:
  - Trained model saved in `models/model1.h5`.
  - Plots saved in the `plots/` folder:
    - `mutual_info.jpg`: Mutual information scores for feature selection.
    - `loss_curve.jpg`: Training and validation loss curves.
    - `correlation_heatmap.jpg`: Feature correlation heatmap.

---

## **Results**
### **Model Performance**
- **Validation Accuracy**: Approximately 70%.
- **Training Accuracy**: Similar to validation accuracy, indicating no significant overfitting.
- **Loss Curves**: The training and validation loss curves show a steady decrease, indicating effective learning.

### **Key Observations**
- The model achieved moderate performance, with room for improvement through advanced techniques like transfer learning or hyperparameter tuning.
- The use of SMOTE and Random UnderSampling helped address class imbalance, improving the model's ability to generalize.

---

## **Future Enhancements**
- **Transfer Learning**: Leveraging pre-trained models for better feature extraction.
- **Hyperparameter Tuning**: Optimizing batch size, learning rate, and network depth.
- **Advanced Data Augmentation**: Adding more synthetic data generation techniques to further balance the dataset.

---

## **Contributing**
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
For any questions or feedback, feel free to reach out:
- **Name**: Nishanth P Kashyap
- **GitHub**: [nishu-2004](https://github.com/nishu-2004)
