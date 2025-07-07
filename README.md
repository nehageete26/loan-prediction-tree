# 📊 Loan Approval Prediction using Decision Tree

This project uses a Decision Tree Classifier to predict loan approval status based on applicant details. The dataset used is from [Kaggle - Loan Prediction Problem](https://www.kaggle.com/datasets/krishnaraj30/finance-loan-approval-prediction-data).

---

## 🧠 Objective

To build a Machine Learning model that predicts whether a loan application will be approved or not based on several features such as income, employment status, property area, etc.

---

## 📁 Dataset

The dataset used is `train.csv` from the Loan Prediction competition on Kaggle.

**Columns:**

- `Loan_ID`: Unique Loan ID *(Dropped for modeling)*
- `Gender`: Male/Female
- `Married`: Applicant's marital status
- `Dependents`: Number of dependents *(Not used in current model)*
- `Education`: Graduate/Not Graduate
- `Self_Employed`: Employment status
- `ApplicantIncome`: Income of the applicant
- `CoapplicantIncome`: Income of the co-applicant
- `LoanAmount`: Loan amount requested
- `Loan_Amount_Term`: Term of the loan in months
- `Credit_History`: Applicant’s credit history (1 = good, 0 = bad)
- `Property_Area`: Urban, Semiurban, or Rural
- `Loan_Status`: (Target) Approved (Y) / Not Approved (N)

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install pandas scikit-learn matplotlib
```

---

## 🚀 How it Works

### ✅ Step-by-Step Flow

1. **Data Loading**  
   Load the dataset using `pandas`.

2. **Preprocessing**
   - Drop `Loan_ID` as it's not useful for model training.
   - Drop rows with missing values.
   - Encode categorical variables manually using mapping.

3. **Feature Selection**
   - Define features `x` (excluding `Loan_Status`).
   - Define target `y` as the `Loan_Status`.

4. **Train/Test Split**
   - Split data into training and testing sets (80%/20%).

5. **Model Training**
   - Train a `DecisionTreeClassifier` on the training data.

6. **Prediction**
   - Predict loan approval on the test data.

7. **Evaluation**
   - Use `accuracy_score` to evaluate model performance.

8. **Visualization**
   - Visualize the trained decision tree using `plot_tree()`.

---

## 🧪 Accuracy

After training the model, the accuracy is printed:

```
accuracy: 73.98%  # (example output)
```

---

## 🌳 Decision Tree Visualization

The decision tree is visualized using `matplotlib`. This helps in understanding the decision-making process of the model.

```python
plt.figure(figsize=(20,20))
plot_tree(
    model,
    filled=True,
    feature_names=x.columns,
    class_names=["No", "Yes"]
)
plt.title("Decision Tree for Loan Approval", fontsize=16)
plt.show()
```

---

## 🧠 Possible Improvements

- Use imputation (mean/median) for handling missing values instead of dropping rows.
- Try encoding using `LabelEncoder` or `OneHotEncoder`.
- Evaluate using confusion matrix, precision, recall, and F1-score.
- Use more robust models like `RandomForestClassifier`.
- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

---

## 📌 Author

**Neha Geete**

If you found this helpful, feel free to ⭐ the repo or connect!

---

## 📂 File Structure

```
.
├── train.csv               # Kaggle dataset
├── loan_approval.py        # Python script for training & prediction
├── README.md               # Project overview
```

---

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).
