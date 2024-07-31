import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE

class LogisticRegressionModel:
    def __init__(self, data_path, target_column, test_size=0.2, random_state=0, rfe_features=5):
        """
        Initialize the logistic regression model.

        Parameters:
        - data_path: Path to the CSV file containing the dataset.
        - target_column: The name of the target column.
        - test_size: The proportion of the dataset to include in the test split.
        - random_state: Random state for reproducibility.
        - rfe_features: Number of top features to select using Recursive Feature Elimination (RFE).
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.rfe_features = rfe_features
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.scaler = StandardScaler()
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        self.y_pred = None
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None

    def load_data(self):
        """Load the data from the CSV file."""
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully with shape: {self.data.shape}")

    def check_vif(self, X):
        """
        Calculate the Variance Inflation Factor (VIF) for each feature.

        Parameters:
        - X: The features dataframe.

        Returns:
        - vif_data: DataFrame containing features and their corresponding VIF values.
        """
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def preprocess_data(self):
        """Preprocess the data by checking VIF, splitting, and scaling."""
        # Split the dataset into features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Calculate VIF before scaling
        vif_before_scaling = self.check_vif(X)
        print("\nVIF before scaling:\n", vif_before_scaling)

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Calculate VIF after scaling
        X_scaled = pd.DataFrame(self.X_train, columns=X.columns)
        vif_after_scaling = self.check_vif(X_scaled)
        print("\nVIF after scaling:\n", vif_after_scaling)

        # Remove features with VIF > 10 (commonly used threshold)
        features_to_remove = vif_after_scaling[vif_after_scaling['VIF'] > 10]['feature']
        if not features_to_remove.empty:
            print(f"\nRemoving features with high VIF: {features_to_remove.tolist()}")
            X_scaled.drop(columns=features_to_remove, inplace=True)
            self.X_train, self.X_test = train_test_split(
                X_scaled, test_size=self.test_size, random_state=self.random_state
            )
        else:
            print("\nNo features removed based on VIF.")

        # Recursive Feature Elimination (RFE)
        rfe = RFE(self.model, n_features_to_select=self.rfe_features)
        rfe.fit(self.X_train, self.y_train)

        selected_features = [f for f, s in zip(X.columns, rfe.support_) if s]
        print(f"\nSelected features after RFE: {selected_features}")

        # Update X_train and X_test with selected features
        self.X_train = rfe.transform(self.X_train)
        self.X_test = rfe.transform(self.X_test)

        print("\nData preprocessed, scaled, and feature-selected successfully.")

    def train_model(self):
        """Train the logistic regression model."""
        self.model.fit(self.X_train, self.y_train)
        print("\nModel trained successfully.")

    def predict(self):
        """Make predictions using the logistic regression model."""
        self.y_pred = self.model.predict(self.X_test)
        print("\nPredictions made successfully.")

    def evaluate_model(self):
        """Evaluate the model and print the evaluation metrics."""
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)

        print(f"\nAccuracy: {self.accuracy}")
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)
        print("\nClassification Report:")
        print(self.classification_report)

    def plot_confusion_matrix(self):
        """Plot the confusion matrix for the logistic regression model."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

# Example usage
def main():
    # Path to your dataset
    data_path = 'c:\\Users\\rahul\\Downloads\\spam.csv'  # Update this path to your dataset
    target_column = 'is_spam'  # Update this with the correct target column

    # Initialize the logistic regression model
    logistic_model = LogisticRegressionModel(data_path=data_path, target_column=target_column, rfe_features=5)

    # Load and preprocess the data
    logistic_model.load_data()
    logistic_model.preprocess_data()

    # Train, predict, and evaluate the model
    logistic_model.train_model()
    logistic_model.predict()
    logistic_model.evaluate_model()
    logistic_model.plot_confusion_matrix()

if __name__ == "__main__":
    main()
