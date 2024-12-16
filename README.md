# DIseases


Here is the README for your GitHub repository in English:

Heart Disease Prediction Model
This project aims to develop a machine learning model that predicts the presence of heart disease in individuals based on health data. We use techniques like Histogram-based Gradient Boosting and data balancing with the SMOTE technique to improve the model's performance.

Contents
Project Description
Prerequisites
Running Instructions
Results and Evaluation
Dataset Information
Project Description
This project focuses on using health data to predict the presence of heart disease. The model utilizes several preprocessing steps like handling missing data, encoding categorical features, and balancing class distribution with SMOTE. The model is then evaluated using cross-validation, F1-score, and AUC-ROC.

The model used in this project is the Histogram-based Gradient Boosting Classifier, which is a powerful ensemble learning technique suitable for high-dimensional datasets.

Prerequisites
Before running the code, ensure you have the following libraries installed:

pandas
numpy
seaborn
matplotlib
sklearn
imblearn
You can install these dependencies using pip:

bash
Αντιγραφή κώδικα
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
Running Instructions
Download the dataset: The dataset used in this project is the "heart disease UCI dataset." You can download the dataset from UCI Heart Disease Dataset.

Set the file path: Ensure the file path in the code points to the correct location of the dataset on your system. The default file path is set to:

python
Αντιγραφή κώδικα
file_path = '/content/drive/My Drive/Python Projects/Diseases/cleaned_heart_disease_uci.csv'
Run the script: Once the dependencies are installed and the dataset is in place, you can run the script in your environment (e.g., Jupyter Notebook, Google Colab).

Model Training and Evaluation: The script will perform the following steps:

Data preprocessing: handling missing values and encoding categorical variables.
Data balancing using SMOTE to handle class imbalance.
Training the model using Histogram-based Gradient Boosting.
Evaluating the model with cross-validation, F1-score, and AUC-ROC.
Generating and displaying performance metrics.
Results
The script will output several key evaluation metrics:

Accuracy: Measures the percentage of correctly predicted instances.
F1-Score: A weighted harmonic mean of precision and recall.
AUC-ROC: The area under the Receiver Operating Characteristic curve, useful for evaluating classification performance.
Example output:

sql
Αντιγραφή κώδικα
Unique values in the target (y): [0. 2. 1. 3. 4.]
Average Accuracy with Cross-Validation: 0.84
Model Accuracy: 0.88
F1-Score: 0.88
AUC-ROC: 0.98

Classification Metrics:
              precision    recall  f1-score   support

         0.0       0.87      0.88      0.87        75
         1.0       0.86      0.81      0.83        74
         2.0       0.82      0.77      0.80        61
         3.0       0.90      0.94      0.92        81
         4.0       0.94      0.99      0.96        75

    accuracy                           0.88       366
   macro avg       0.88      0.88      0.88       366
weighted avg       0.88      0.88      0.88       366
Additionally, a scatter plot of the real vs predicted values will be displayed.

Dataset Information
The dataset used in this project is the "Heart Disease UCI" dataset, which contains various health features like age, sex, blood pressure, cholesterol levels, and more. The target variable (num) indicates whether the individual has heart disease (1) or not (0).

The dataset includes features such as:

age: Age of the patient
sex: Gender of the patient
cp: Chest pain type
trestbps: Resting blood pressure
chol: Serum cholesterol level
fbs: Fasting blood sugar
restecg: Resting electrocardiographic results
thalach: Maximum heart rate achieved
exang: Exercise induced angina
oldpeak: Depression induced by exercise relative to rest
slope: Slope of the peak exercise ST segment
ca: Number of major vessels colored by fluoroscopy
thal: Thalassemia
num: Target variable indicating the presence of heart disease
