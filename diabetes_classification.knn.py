import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import warnings
import joblib  # Import joblib for saving and loading the model

sns.set()
warnings.filterwarnings('ignore')
# %matplotlib inline  # Uncomment if running in Jupyter Notebook

class DiabetesClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_params = None
        self.best_score = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        print(self.data.head())  # Display the first few rows of the dataset to verify columns

        # Display dataset information
        print("Dataset Information:")
        self.data.info()

        # Check for any unnamed columns and drop unnecessary ones
        if 'Unnamed: 0' in self.data.columns:
            self.data.drop(['Unnamed: 0'], axis=1, inplace=True)
        if 'Id' in self.data.columns:
            self.data.drop(['Id'], axis=1, inplace=True)

        print(f"Columns after dropping: {self.data.columns.tolist()}")

        # Handle missing values
        data_copy = self.data.copy(deep=True)
        data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
            data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

        print("\nMissing Values Before Imputation:")
        print(data_copy.isnull().sum())

        # Fill missing values
        data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
        data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
        data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
        data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
        data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)

        # Standardize features
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = pd.DataFrame(self.scaler.fit_transform(data_copy[features]), columns=features)

        y = data_copy['Outcome']

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=42, stratify=y)

    def train_knn(self, neighbors_range=(1, 15)):
        # Train and evaluate KNN model for different numbers of neighbors
        test_scores = []
        train_scores = []

        for i in range(neighbors_range[0], neighbors_range[1]):
            knn = KNeighborsClassifier(i)
            knn.fit(self.X_train, self.y_train)

            train_scores.append(knn.score(self.X_train, self.y_train))
            test_scores.append(knn.score(self.X_test, self.y_test))

        max_train_score = max(train_scores)
        train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
        max_test_score = max(test_scores)
        test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

        print('Max train score {} % and k = {}'.format(max_train_score * 100,
                                                       list(map(lambda x: x + 1, train_scores_ind))))
        print('Max test score {} % and k = {}'.format(max_test_score * 100,
                                                      list(map(lambda x: x + 1, test_scores_ind))))

        # Plot train and test scores
        plt.figure(figsize=(12, 5))
        sns.lineplot(x=range(1, neighbors_range[1]), y=train_scores, marker='*', label='Train Score')
        sns.lineplot(x=range(1, neighbors_range[1]), y=test_scores, marker='o', label='Test Score')
        plt.title("Train vs. Test Scores")
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

        # Train the best model
        optimal_k = test_scores_ind[0] + 1  # Using the first optimal k for simplicity
        self.model = KNeighborsClassifier(optimal_k)
        self.model.fit(self.X_train, self.y_train)
        print(f'Trained KNN with k={optimal_k}.')

    def save_model(self, filename='knn_model.joblib'):
        # Save the model to a file
        if self.model is not None:
            joblib.dump((self.model, self.scaler), filename)
            print(f'Model saved to {filename}')
        else:
            print("No model to save!")

    def load_model(self, filename='knn_model.joblib'):
        # Load the model from a file
        self.model, self.scaler = joblib.load(filename)
        print(f'Model loaded from {filename}')

    def evaluate_model(self):
        # Evaluate the trained model
        if self.model is None:
            raise ValueError("Model is not trained yet!")

        y_pred = self.model.predict(self.X_test)
        cnf_matrix = confusion_matrix(self.y_test, y_pred)

        # Create a grid of plots for confusion matrix, ROC curve, etc.
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot confusion matrix
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual Label')
        axes[0].set_xlabel('Predicted Label')

        # Plot ROC curve
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].plot(fpr, tpr, label='KNN')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')

        # Calculate and display classification report
        cls_report = classification_report(self.y_test, y_pred, output_dict=True)
        sns.heatmap(pd.DataFrame(cls_report).iloc[:-1, :].T, annot=True, cmap="Blues", ax=axes[2])
        axes[2].set_title('Classification Report')

        plt.tight_layout()
        plt.show()

        # Calculate AUC
        auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f'ROC AUC Score: {auc:.4f}')

    def perform_grid_search(self):
        # Perform GridSearchCV for finding the best hyperparameters
        param_grid = {'n_neighbors': np.arange(1, 50)}
        knn = KNeighborsClassifier()
        knn_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_cv.fit(self.X_train, self.y_train)

        self.best_score = knn_cv.best_score_
        self.best_params = knn_cv.best_params_
        print(f"Best Score: {self.best_score}")
        print(f"Best Parameters: {self.best_params}")

    def predict_new(self, new_data):
        # Predict on new data
        if self.model is None:
            raise ValueError("Model is not trained yet!")

        # Scale the new data using the stored scaler
        new_data_scaled = pd.DataFrame(self.scaler.transform(new_data), columns=new_data.columns)
        prediction = self.model.predict(new_data_scaled)
        return prediction

# Example usage:
if __name__ == "__main__":
    classifier = DiabetesClassifier(data_path='c:\\Users\\rahul\\Downloads\\diabetes.csv')
    classifier.load_and_preprocess_data()
    classifier.train_knn()
    classifier.evaluate_model()
    classifier.perform_grid_search()

    # Save the trained model
    classifier.save_model('knn_diabetes_model.joblib')

    # Load the trained model
    # classifier.load_model('knn_diabetes_model.joblib')
    # Make a prediction on