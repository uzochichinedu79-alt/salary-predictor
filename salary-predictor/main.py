# ==============================
# Salary Predictor (ML Project)
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():

    # Load dataset
    df = pd.read_csv("salary.csv")

    print("\n📊 First 5 rows of dataset:")
    print(df.head())

    # Features and target
    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Results table
    comparison = pd.DataFrame({
        "Actual Salary": y_test.values,
        "Predicted Salary": y_pred
    })

    print("\n📌 Prediction Results:")
    print(comparison.head())

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📉 Mean Absolute Error: {mae:.2f}")

    # Visualization
    plt.scatter(X_test, y_test, label="Actual Salary")
    plt.plot(X_test, y_pred, color="red", label="Prediction Line")

    plt.title("Salary Prediction Model")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()