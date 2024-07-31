import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define the DataPreprocessor class
class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.cleaned_data = None

    def load_data(self):
        """Load the data from the CSV file."""
        self.data = pd.read_csv(self.filepath, encoding='latin-1')
        self.data = self.data.iloc[:, :2]
        self.data.columns = ['label', 'text']
        return self.data

    def clean_data(self):
        """Clean the text data by removing stopwords and punctuation."""
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        # Clean and preprocess text data
        self.data['text'] = self.data['text'].str.lower()
        self.data['text'] = self.data['text'].str.replace(r'\W', ' ')
        self.data['text'] = self.data['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
        self.cleaned_data = self.data
        return self.cleaned_data

    def get_features_and_labels(self):
        """Extract features and labels from the cleaned data."""
        vectorizer = CountVectorizer(max_features=1500)
        X = vectorizer.fit_transform(self.cleaned_data['text']).toarray()
        y = pd.get_dummies(self.cleaned_data['label'], drop_first=True).values.flatten()
        return train_test_split(X, y, test_size=0.2, random_state=0)

# Define the SpamClassifier class
class SpamClassifier:
    def __init__(self, model_type='knn', n_neighbors=5, criterion='gini'):
        """
        Initialize the classifier with a choice of KNN or Decision Tree.

        Parameters:
        - model_type: 'knn' for K-Nearest Neighbors, 'decision_tree' for Decision Tree
        - n_neighbors: Number of neighbors for KNN (used only if model_type is 'knn')
        - criterion: Criterion for Decision Tree ('gini' or 'entropy') (used only if model_type is 'decision_tree')
        """
        self.model_type = model_type
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(criterion=criterion, random_state=0)
        else:
            raise ValueError("model_type must be 'knn' or 'decision_tree'")

    def train(self, X_train, y_train):
        """Train the selected model on the training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """Evaluate the model and return metrics."""
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, cm, report

# Define the ModelEvaluator class
class ModelEvaluator:
    def __init__(self, acc, cm, report):
        self.accuracy = acc
        self.confusion_matrix = cm
        self.classification_report = report

    def plot_confusion_matrix(self):
        """Plot the confusion matrix for model evaluation."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def print_evaluation_report(self):
        """Print the evaluation metrics for the model."""
        print(f"Accuracy: {self.accuracy}")
        print("Classification Report:")
        print(self.classification_report)

# Integrate the classes for complete functionality
def main():
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor('c:\\Users\\rahul\\Downloads\\spam.csv')
    preprocessor.load_data()
    preprocessor.clean_data()
    X_train, X_test, y_train, y_test = preprocessor.get_features_and_labels()

    # Choose the classifier ('knn' or 'decision_tree')
    classifier_choice = 'knn'  # Change this to 'decision_tree' for Decision Tree model

    # Initialize and train the spam classifier
    classifier = SpamClassifier(model_type=classifier_choice, n_neighbors=5, criterion='entropy')
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    acc, cm, report = classifier.evaluate(y_test, y_pred)
    evaluator = ModelEvaluator(acc, cm, report)
    evaluator.print_evaluation_report()
    evaluator.plot_confusion_matrix()

if __name__ == "__main__":
    main()
