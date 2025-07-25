# 🍕 Food Delivery Time Prediction

This project aims to predict whether a food delivery will be **Fast** or **Delayed** using various features such as location, weather, traffic conditions, and order details. We explore both a traditional machine learning model (Logistic Regression) and a deep learning model (CNN with synthetic image-based inputs).

---

## 📌 Objective

To classify food deliveries as *Fast* or *Delayed* using:
- Logistic Regression (baseline model)
- Convolutional Neural Network (CNN) using synthetic location image grids

---

## 🧠 Methodology

### 1. **Data Preprocessing**
- Handled missing values (drop/impute)
- One-Hot Encoding for categorical variables
- Standardization for numerical variables
- Created new features like:
  - Haversine distance between customer and restaurant
  - Rush hour indicator
- Target binarized based on median delivery time

### 2. **CNN Modeling**
- Created synthetic 32x32 grayscale images with customer and restaurant plotted
- CNN Architecture:
  - Conv2D → MaxPooling
  - Conv2D → MaxPooling
  - Flatten → Dense(32) → Dense(1, sigmoid)
- Input: Image grid (1 channel)
- Output: Binary class (0: Delayed, 1: Fast)

### 3. **Logistic Regression**
- Trained on all tabular features (excluding images)
- Standard ML pipeline

---

## 🧪 Evaluation

### CNN Results:
- **Accuracy**: 0.55
- **F1-score**: 0.51
- **Confusion Matrix**:
[[12 8]
[10 10]]

### Logistic Regression Results:
- **Accuracy**: 0.42
- **F1-score**: 0.42

### Cross-Validation (CNN):
- Mean Accuracy: 0.545
- Mean F1: 0.513

---

## 📊 Comparison

| Metric        | CNN    | Logistic Regression |
|---------------|--------|---------------------|
| Accuracy      | 0.545  | 0.42                |
| Precision     | 0.549  | 0.42                |
| Recall        | 0.520  | 0.43                |
| F1-score      | 0.513  | 0.42                |

---

## 🔍 Key Takeaways

- CNN performed better than logistic regression by ~12% in F1-score.
- Even with synthetic image inputs, spatial patterns helped the CNN model.
- Logistic regression was not able to capture complex non-linear/spatial patterns.

---

## ⚠️ Limitations

- Synthetic grid images used instead of actual GPS/map route data.
- Small dataset (only 40 samples in validation).
- No ensemble or hyperparameter tuning applied yet.

---

## 📈 Future Work

- Use real route images (e.g., from Google Maps API)
- Try stronger deep models (LSTM + CNN, or ViT)
- Hyperparameter tuning & ensembling
- Increase dataset size for better generalization

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib / Seaborn (for EDA)

---

## 👤 Author

Prepared by: **Tushar walia**  


---

