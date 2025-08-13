# Logistic_Regression

# 🧠 Breast Cancer Classification using Logistic Regression

This project demonstrates a complete machine learning pipeline using **Logistic Regression** to classify tumors as **malignant** or **benign** based on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

## 📁 Dataset

The dataset used is the **WDBC dataset** in CSV format, located at:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


- Each row represents a tumor sample.
- Features are real-valued measurements computed from digitized images of fine needle aspirates (FNAs) of breast masses.
- The first column is the ID (removed in preprocessing).
- The second column is the diagnosis label:
  - `M` = Malignant (encoded as `1`)
  - `B` = Benign (encoded as `0`)
- The remaining columns are numerical features extracted from images.

---

## 📊 Workflow

### 1. **Data Preprocessing**
- Removed the ID column.
- Encoded diagnosis labels (`M` / `B`) into integers (`1` / `0`).
- Converted features to float for model compatibility.
- Applied **StandardScaler** to normalize the features.

### 2. **Data Splitting**
- The dataset is split into:
  - **Training set**: 70%
  - **Validation set**: 10%
  - **Test set**: 20%
- Stratified sampling is used to maintain class distribution.

### 3. **Model Training**
- A **Logistic Regression** model is trained using the training data.

### 4. **Evaluation**
The model is evaluated on the test set using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## 📌 Example Output

```text
Encoded first column labels are:
[1 1 1 ... 0 0 0]

Training data size: X = 398 , y= 398  
Testing data size: X = 114 , y= 114  
Validation data size: X = 57 , y= 57  

Evaluation metrics of Logistic Regression model 
Accuracy: 0.965
Precision: 0.958
Recall: 0.958
F-score : 0.958
```

## 📦 Requirements
Install the required Python packages using:
pip install pandas matplotlib scikit-learn

## 📈 Future Improvements

- **Add comparisons with other models**: 
  - Include additional models like **Support Vector Machine (SVM)** and **Multi-layer Perceptron Classifier (MLPClassifier)** for a more comprehensive comparison of performance.
  
- **Visualize confusion matrix and ROC curves**: 
  - Implement confusion matrix to visualize the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
  - Plot **ROC (Receiver Operating Characteristic) curves** to evaluate the trade-off between true positive rate and false positive rate at various thresholds.

- **Perform hyperparameter tuning using the validation set**: 
  - Fine-tune the model's hyperparameters (e.g., regularization strength for logistic regression) using techniques like **GridSearchCV** or **RandomizedSearchCV** on the validation set for improved performance.

## 📊 Discussion of Logistic Regression Results

The bar chart shows the performance of the logistic regression model on the **breast cancer dataset**, evaluated using four common metrics:

- **Accuracy**: Represents the overall correctness of the model. A high accuracy (e.g., ~95%+) suggests the model performs well in general.
- **Precision**: Indicates how many of the predicted positive cases (malignant tumors) were actually positive. High precision implies few false positives.
- **Recall**: Measures how many of the actual positive cases were correctly identified. High recall means fewer false negatives.
- **F1-Score**: Harmonic mean of precision and recall. It balances the trade-off between the two, especially when the classes are imbalanced.

If the values of precision and recall are both high and close to the accuracy, it indicates the model is not only accurate but also correctly identifies both **benign** and **malignant** cases with minimal misclassification. This is critical in **medical diagnostics**, where both false positives and false negatives have serious consequences.

<img width="1426" height="1056" alt="Screenshot 2025-08-13 at 11 25 11 AM" src="https://github.com/user-attachments/assets/2ddc3d78-499d-4192-97dc-cc2b10e47e66" />
