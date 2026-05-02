# 🏥 Diabetes Prediction System using SVM

A machine learning web application that predicts diabetes 
based on patient medical data using a Support Vector Machine (SVM) classifier.

## 🔗 Live Demo
[Click here to try the app](https://YOUR_USERNAME-diabetes-svm-app.streamlit.app)

## 📌 Project Overview
This project aims to develop a classification model using 
Support Vector Machine (SVM) to predict diabetes based on medical data.

## ✅ Tasks Completed
- **Preprocessing:** Handled missing values, normalized features, removed outliers
- **Feature Selection:** Identified key biomarkers using ANOVA F-Score
- **Model Training:** Trained SVM classifier with RBF kernel
- **Model Evaluation:** Tested using Accuracy, Precision, Recall and F1 Score
- **Deployment:** Created a user-friendly web interface for doctors

## 📊 Model Performance
| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~97%   |
| Precision | ~95%   |
| Recall    | ~93%   |
| F1 Score  | ~94%   |

## 🔑 Key Features (Biomarkers)
1. Blood Glucose Level
2. HbA1c Level
3. BMI
4. Age
5. Hypertension
6. Heart Disease

## 🛠️ Technologies Used
- Python
- Scikit-learn (SVM)
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn

## 📁 Dataset
- 10,000 patient records
- 9 features including gender, age, BMI, HbA1c level, blood glucose level
- Binary classification: Diabetes / No Diabetes

## 🚀 How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
Your Name
