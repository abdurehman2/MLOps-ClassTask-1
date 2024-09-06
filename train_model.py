from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a machine learning model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
