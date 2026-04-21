# 🚢 Titanic Survival Predictor

A machine learning web app built with **Streamlit** that predicts whether a Titanic passenger would have survived, based on their personal details. Powered by a Logistic Regression model trained on the classic Titanic dataset.

---

## 📸 Features

- Interactive passenger input form (class, sex, age, fare, embarkation, family size)
- Instant survival prediction with probability score
- Color-coded result cards (green = survived, red = did not survive)
- Probability bar chart for visual breakdown
- Clean, dark-themed nautical UI

---

## 🗂️ Project Structure

```
├── app.py                  # Streamlit application
├── tested.csv              # Dataset used for training
├── model_training.py       # Model training script
├── Logistic.pkl            # Saved Logistic Regression model
├── transformer.pkl         # Saved ColumnTransformer (OneHotEncoder)
├── columns.pkl             # Saved input column order
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Muhammad-Khan-Khichi/Titanic-Survival-Predictor
cd titanic-survival-predictor
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 3. Train the model (if `.pkl` files are missing)

```bash
python model_training.py
```

This generates `Logistic.pkl`, `transformer.pkl`, and `columns.pkl`.

### 4. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Details

| Component        | Detail                                      |
|------------------|---------------------------------------------|
| Algorithm        | Logistic Regression                         |
| Library          | scikit-learn                                |
| Train/Test Split | 80% / 20% (`random_state=42`)               |
| Encoding         | OneHotEncoder on `Sex` and `Embarked`       |
| Dropped Columns  | `PassengerId`, `Name`, `Cabin`, `Ticket`    |
| Target           | `Survived` (0 = No, 1 = Yes)                |

### Models evaluated during development

| Model               | Notes                        |
|---------------------|------------------------------|
| Logistic Regression | ✅ Selected & saved           |
| K-Nearest Neighbors | Evaluated                    |
| Naive Bayes         | Evaluated                    |
| Decision Tree       | Evaluated                    |
| SVM                 | Evaluated                    |

---

## 📥 Input Features

| Feature    | Description                              | Type        |
|------------|------------------------------------------|-------------|
| `Pclass`   | Passenger class (1 = Upper, 3 = Lower)   | Integer     |
| `Sex`      | Gender of the passenger                  | Categorical |
| `Age`      | Age in years                             | Float       |
| `SibSp`    | Number of siblings/spouses aboard        | Integer     |
| `Parch`    | Number of parents/children aboard        | Integer     |
| `Fare`     | Ticket fare paid (£)                     | Float       |
| `Embarked` | Port of embarkation (S / C / Q)          | Categorical |

---

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
joblib
```

---

## 📄 License

This project is open-source.

---

> Built for educational purposes using the public Titanic dataset.
