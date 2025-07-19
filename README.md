# SkillCraft Internship - Task 03  
## Decision Tree Classifier - Bank Marketing Dataset  

This project is part of my **SkillCraft Data Science Internship**.  
The task focuses on building a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit based on various features from the **Bank Marketing dataset**.

---

## 🔥 Key Features  
✅ Data cleaning and preprocessing of the Bank Marketing dataset  
✅ Encoding of categorical variables for model training  
✅ Splitting data into training & testing sets  
✅ Building and evaluating a **Decision Tree Classifier**  
✅ Accuracy score and confusion matrix for performance evaluation  
✅ Visualization of the Decision Tree  

---

## 🛠 Technologies Used  
- Python  
- Pandas  
- Scikit-learn (sklearn)  
- Matplotlib  
- Seaborn  

---

## 📂 Files in this Repository  
- `3rdtask.py` → Python script to build and evaluate the Decision Tree model  
- `bank.csv` → Dataset used for training (Bank Marketing dataset)  
- `decision_tree_visualization.png` → Visualization of the decision tree  

---

## ✅ Code Snippet  

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("bank.csv")
print("✅ Data Loaded Successfully!")
print(df.head())

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split data into features & target
X = df.drop('y_yes', axis=1)  # target column after encoding ('y_yes' indicates subscription)
y = df['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree model
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = model.predict(X_test)
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_visualization.png")
plt.show()# skillcraft-third-task
"This repository contains my SkillCraft Internship Task 03 project, where I built a Decision Tree Classifier on the Bank Marketing dataset. The project includes data preprocessing, model training, evaluation, and visualizations such as feature importance, confusion matrix, and decision tree plot."

Author
Kumar Akash Deep
Skillcraft Data Science Intern
