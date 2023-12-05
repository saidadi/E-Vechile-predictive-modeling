import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class PredictiveModeling:
    """
    The PredictiveModeling class enables the development, training, and evaluation of a predictive model using a RandomForestClassifier. Upon initialization, it takes a pandas DataFrame as input. The class provides methods for preprocessing the data, training the model, and visualizing model performance.

    Methods:
    1. preprocess_data(): Encodes object-type (categorical) columns using LabelEncoder and imputes missing values using SimpleImputer with a specified strategy (default is mean).

    2. train_model(target_column, positive_class_label): Splits the data into training and testing sets, converts the target column into a binary classification problem, trains the RandomForestClassifier, and returns the accuracy, confusion matrix, predicted probabilities, and true labels for the test set.

    3. plot_confusion_matrix(confusion_mat): Plots a confusion matrix for the binary classification problem.

    4. plot_roc_curve(y_test, probas): Plots the Receiver Operating Characteristic (ROC) curve using the true labels and predicted probabilities.

    Attributes:
    - data (pd.DataFrame): The input DataFrame for predictive modeling.
    - model (RandomForestClassifier): The RandomForestClassifier used for prediction.
    - label_encoder (LabelEncoder): The LabelEncoder used for encoding categorical columns.
    """

    def __init__(self, data):
        """
        Initializes the PredictiveModeling object with the provided pandas DataFrame.

        Parameters:
        - data (pd.DataFrame): The input DataFrame for predictive modeling.
        """
        self.data = data
        self.model = RandomForestClassifier()
        self.label_encoder = LabelEncoder()

    def preprocess_data(self):
        """
        Encodes object-type (categorical) columns using LabelEncoder and imputes missing values using SimpleImputer with a specified strategy (default is mean).

        Returns:
        - None
        """
        object_cols = self.data.select_dtypes(include='object').columns
        for col in object_cols:
            self.data[col] = self.label_encoder.fit_transform(self.data[col])

        imputer = SimpleImputer(strategy='mean')
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def train_model(self, target_column, positive_class_label):
        """
        Splits the data into training and testing sets, converts the target column into a binary classification problem,
        trains the RandomForestClassifier, and returns the accuracy, confusion matrix, predicted probabilities, and true labels for the test set.

        Parameters:
        - target_column (str): The target column for prediction.
        - positive_class_label: The label indicating the positive class.

        Returns:
        - accuracy (float): The accuracy of the model on the test set.
        - confusion_mat (array): The confusion matrix for the binary classification problem.
        - probas (array): Predicted probabilities for the positive class.
        - y_test (array): True labels for the test set.
        """
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]

        y_binary = (y == positive_class_label).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        confusion_mat = confusion_matrix(y_test, predictions)

        return accuracy, confusion_mat, probas, y_test

    def plot_confusion_matrix(self, confusion_mat):
        """
        Plots a confusion matrix for the binary classification problem.

        Parameters:
        - confusion_mat (array): The confusion matrix.

        Returns:
        - None
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = ['Class 0', 'Class 1']
        tick_marks = range(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_roc_curve(self, y_test, probas):
        """
        Plots the Receiver Operating Characteristic (ROC) curve using the true labels and predicted probabilities.

        Parameters:
        - y_test (array): True labels for the test set.
        - probas (array): Predicted probabilities for the positive class.

        Returns:
        - None
        """
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
