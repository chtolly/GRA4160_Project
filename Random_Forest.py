import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# load dataset
df = pd.read_csv("mushroom.csv")

print(df.head())
print(df.shape)

print(df["class"].value_counts())

# features and target
X = df.drop("class", axis=1)
y = df["class"]

# encode categorical features
X = pd.get_dummies(X)

print(X.head())
print(X.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


# -------------------------
# RANDOM FOREST MODEL
# -------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# train the model
rf_model.fit(X_train, y_train)

# make predictions
rf_pred = rf_model.predict(X_test)

# accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)
print("Random Forest Confusion Matrix:\n", rf_cm)

sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()


# feature importance
rf_importance = rf_model.feature_importances_

feature_names = X.columns

indices = np.argsort(rf_importance)[-10:]

plt.barh(range(len(indices)), rf_importance[indices])
plt.yticks(range(len(indices)), feature_names[indices])
plt.title("Random Forest Top 10 Important Features")
plt.show()