# 🩺 Diabetes Prediction using Machine Learning

This project focuses on predicting whether a patient is diabetic or not using various machine learning models. The dataset is preprocessed, visualized, and analyzed using models like Random Forest, Logistic Regression, SVM, and XGBoost, with hyperparameter tuning applied to improve performance.

---

## 📂 Dataset

The dataset used is **`diabetes.csv`**, which contains the following features:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome *(Target: 0 = Non-diabetic, 1 = Diabetic)*

---

## 📊 Exploratory Data Analysis

- **Missing values** checked using `.isnull().sum()`
- **Correlation heatmap** plotted using Seaborn
- **Class distribution** visualized with a count plot

---

## 🧠 Models Used

1. **Random Forest**
2. **XGBoost**
3. **Support Vector Machine (SVM)**
4. **Logistic Regression**
5. **Tuned Random Forest** using `GridSearchCV`

---

## 🧪 Model Evaluation Metrics

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **ROC Curve** and **AUC Score**

---

## 🛠️ GridSearchCV for Tuning

Hyperparameter tuning for Random Forest:

- 3-fold cross-validation
- Optimized using **accuracy score**

---

## 📈 Visualizations

- Heatmaps of correlations and confusion matrices
- ROC Curves for probabilistic models
- Bar plots comparing:
  - Accuracy of all models
  - ROC AUC Scores

---

## ✅ How to Run

### 1. Clone the repo or copy the code  

```bash
git clone <repo-url>
cd <project-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python diabetes_prediction.py
```

---

## 📦 Requirements

```txt
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

> You can install all at once using:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
```

---

## 📌 Key Insights

- **XGBoost** and **Tuned Random Forest** performed the best in terms of accuracy and AUC.
- Data is slightly imbalanced; hence ROC-AUC was an important metric.
- Visualizations made it easy to compare models effectively.

---

## 🙌 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).
