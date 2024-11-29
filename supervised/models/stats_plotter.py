import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class EDAAnalyzer:
    def __init__(self, df, target_column):
        """
        Initialize the EDAAnalyzer with the dataset and target column.

        Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.
        """
        self.df = df
        self.target_column = target_column

    def combine_columns(self, category):
        """
        Combine one-hot encoded columns back into their original categorical column.

        Parameters:
        category (str): The prefix of the category columns to combine.
        """
        category_columns = [col for col in self.df.columns if col.startswith(f'{category}=')]
        self.df[category] = self.df[category_columns].idxmax(axis=1).apply(lambda x: x.split('=')[1])
        self.df.drop(columns=category_columns, inplace=True)

    def plot_categorical_feature(self, feature):
        """
        Plot the count of anomaly for each value of a categorical feature.

        Parameters:
        feature (str): The name of the categorical feature to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=self.target_column, data=self.df)
        plt.title(f'Count of {self.target_column} for Each {feature} Value')
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend(title=self.target_column, loc='upper right')
        plt.show()

    def plot_all_categorical_features(self):
        """
        Plot the count of anomaly for all categorical features in the dataset.
        """
        categorical_features = self.df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            if feature != self.target_column:
                self.plot_categorical_feature(feature)

    def scatter_plot(self, x_feature, y_feature, labels):
        """
        Plot a scatter plot for two features with points colored by the target variable.

        Parameters:
        x_feature (str): The name of the feature for the x-axis.
        y_feature (str): The name of the feature for the y-axis.
        labels (list): Labels for the classes.
        """
        plt.figure(figsize=(10, 6))
        for label in np.unique(self.df[self.target_column]):
            subset = self.df[self.df[self.target_column] == label]
            plt.scatter(subset[x_feature], subset[y_feature], label=labels[label], alpha=0.7)
        plt.title(f'Scatter Plot of {x_feature} vs {y_feature}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, model, labels):
        """
        Plot a confusion matrix for a given model.

        Parameters:
        model: The trained model to evaluate.
        labels (list): Labels for the classes.
        """
        y_pred = model.predict(self.df.drop(columns=self.target_column))
        cm = confusion_matrix(self.df[self.target_column], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='viridis')
        plt.title('Confusion Matrix')
        plt.show()

    def feature_importance(self, model):
        """
        Plot the feature importance for a tree-based model.

        Parameters:
        model: The trained tree-based model.
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = self.df.drop(columns=self.target_column).columns
            sorted_indices = np.argsort(importance)[::-1]

            plt.figure(figsize=(12, 8))
            sns.barplot(x=importance[sorted_indices], y=features[sorted_indices], orient='h')
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.show()
        else:
            print("The provided model does not have feature importances.")
