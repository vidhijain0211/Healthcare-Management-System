#  HealthCare Prediction System using Machine Learning

This project is a Machine Learning-based healthcare management system built with **Python**, **Streamlit**, **SQLite3**, and **Scikit-learn**. It allows users to predict the chances of suffering from multiple diseases like **Diabetes**, **Heart Disease**, **Parkinson’s**, **Lung Cancer**, and **Thyroid** based on input symptoms.

---

##  Features

-  User Login & Registration System
- Multi-Disease Prediction
- Trained on 5 different ML models per disease and selected the best-performing one
- Lightweight and fast with local storage using SQLite3
- Clean and responsive **Streamlit UI**
- Easy deployment on local or cloud (Streamlit Share, Heroku, etc.)

---

## 🛠Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn
- **Database**: SQLite3
- **ML Models**: Logistic Regression, Random Forest, Decision Tree, KNN, SVM
- **Other Tools**: Pandas, NumPy, joblib

---

## Supported Diseases & Models

| Disease         | Models Trained                   | Best Model Selected |
|----------------|----------------------------------|---------------------|
| Diabetes        | LR, RF, DT, KNN, SVM             | Random Forest       |
| Heart Disease   | LR, RF, DT, KNN, SVM             | Logistic Regression |
| Parkinson's     | LR, RF, DT, KNN, SVM             | SVM                 |
| Lung Cancer     | LR, RF, DT, KNN, SVM             | Random Forest       |
| Thyroid         | LR, RF, DT, KNN, SVM             | Decision Tree       |

---

## Folder Structure
├── app.py
├── requirements.txt
├── models/
│ ├── diabetes_model.sav
│ ├── heart_model.sav
│ └── ...
├── data/
│ └── sample_datasets.csv (optional)
├── user_db/
│ └── users.db
├── pages/
│ └── individual_disease_pages.py

