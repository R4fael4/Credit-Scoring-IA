import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("clients.csv")

encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == "object" and column != "score_credito":
        data[column] = encoder.fit_transform(data[column])

X = data.drop(["score_credito", "id_cliente"], axis=1)
y = data["score_credito"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()

rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

score_counts = data["score_credito"].value_counts()
print(score_counts['Standard'] / sum(score_counts))

rf_prediction = rf_model.predict(X_test)
knn_prediction = knn_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_prediction))
print("KNN Accuracy:", accuracy_score(y_test, knn_prediction))

importance = pd.DataFrame(
    index=X.columns, 
    data=rf_model.feature_importances_, 
    columns=["Importance (%)"]
)
importance["Importance (%)"] *= 100
importance = importance.sort_values(by="Importance (%)", ascending=False)
print(importance.round(2))

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance (%)", y=importance.index, data=importance, palette="viridis")

plt.title("Feature Importance in the Random Forest Model", fontsize=14)
plt.xlabel("Importance (%)", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()

plt.show()
