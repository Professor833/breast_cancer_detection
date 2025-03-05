import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score,
)


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the breast cancer dataset.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Load the data
    dataset = pd.read_csv(filepath)

    # Check for missing values
    print(f"Missing values per column:\n{dataset.isnull().sum()}")

    # Basic data exploration
    print(f"\nData shape: {dataset.shape}")
    print(f"\nClass distribution:\n{dataset.iloc[:, -1].value_counts()}")

    # Extract features and target
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, C=1.0):
    """
    Train the logistic regression model.

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        C (float): Inverse regularization parameter

    Returns:
        LogisticRegression: Trained classifier
    """
    classifier = LogisticRegression(
        random_state=0,
        C=C,  # Regularization strength (smaller values = stronger regularization)
        solver="liblinear",  # Algorithm to use (works well for small datasets)
        max_iter=1000,  # Increase max iterations for convergence
    )
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(classifier, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model using various metrics.

    Args:
        classifier (LogisticRegression): Trained classifier
        X_train, X_test, y_train, y_test: Data splits

    Returns:
        tuple: (accuracy, confusion matrix, classification report, accuracies from cross validation)
    """
    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report:\n{report}")

    # Cross-validation
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print(
        f"\nCross-Validation Accuracy: {accuracies.mean():.4f} ± {accuracies.std():.4f}"
    )
    print(f"Accuracy: {accuracies.mean() * 100:.2f} %")
    print(f"Standard Deviation: {accuracies.std() * 100:.2f} %")

    # ROC Curve and AUC
    y_prob = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC: {auc:.4f}")

    return accuracy, cm, report, accuracies, (fpr, tpr, auc)


def visualize_results(cm, roc_data):
    """
    Visualize the model results with plots.

    Args:
        cm (numpy.ndarray): Confusion matrix
        roc_data (tuple): ROC curve data (fpr, tpr, auc)
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Plot ROC curve
    fpr, tpr, auc = roc_data
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("./results/logistic_regression_results.png")
    plt.show()


def main():
    """Main function to run the breast cancer detection workflow."""
    print("Starting breast cancer detection using Logistic Regression...")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "./dataset/breast_cancer.csv"
    )

    # Train the model
    classifier = train_model(X_train, y_train)

    # Evaluate the model
    _, cm, _, _, roc_data = evaluate_model(classifier, X_train, X_test, y_train, y_test)

    # Visualize results
    visualize_results(cm, roc_data)

    print("Breast cancer detection complete!")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os

    os.makedirs("./results", exist_ok=True)

    main()
