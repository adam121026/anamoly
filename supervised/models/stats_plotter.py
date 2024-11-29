import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

class StatsPlotter:
    """
    A class to encapsulate plotting and metrics functionalities for model evaluation.
    """

    def __init__(self, X_test, y_test, y_pred):
        """
        Initializes the StatsPlotter with the test set and predictions.
        
        Parameters:
        - X_test: pd.DataFrame or np.ndarray, test features.
        - y_test: array-like, true labels for the test set.
        - y_pred: array-like, predicted labels for the test set.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_scatter(self, feature1_index, feature2_index, title="Scatter Plot of Anomalies vs Normals"):
        """
        Plots a scatter plot of anomalies (class 1) and normals (class 0).

        Parameters:
        - feature1_index: int, column index or name for the x-axis feature.
        - feature2_index: int, column index or name for the y-axis feature.
        - title: str, title of the plot.
        """
        feature1 = self.X_test.iloc[:, feature1_index] if hasattr(self.X_test, 'iloc') else self.X_test[:, feature1_index]
        feature2 = self.X_test.iloc[:, feature2_index] if hasattr(self.X_test, 'iloc') else self.X_test[:, feature2_index]

        anomalies = self.y_pred == 1
        normals = self.y_pred == 0

        plt.figure(figsize=(10, 6))
        plt.scatter(feature1[normals], feature2[normals], c='blue', label='Normal', alpha=0.6)
        plt.scatter(feature1[anomalies], feature2[anomalies], c='red', label='Anomaly', alpha=0.6)
        plt.xlabel(f'Feature {feature1_index}')
        plt.ylabel(f'Feature {feature2_index}')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, labels=["Normal", "Anomaly"], title="Confusion Matrix"):
        """
        Plots the confusion matrix.

        Parameters:
        - labels: list of str, names of the classes.
        - title: str, title of the confusion matrix plot.
        """
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        disp.plot(cmap='viridis')
        plt.title(title)
        plt.show()

    def print_classification_report(self):
        """
        Prints the classification report.
        """
        report = classification_report(self.y_test, self.y_pred, target_names=["Normal", "Anomaly"])
        print("Classification Report:\n", report)

##usage##

# from stats_plotter import StatsPlotter

# # After getting predictions
# y_pred_recall = best_model_recall.predict(X_test)

# # Initialize the StatsPlotter with test data and predictions
# plotter = StatsPlotter(X_test=X_test, y_test=y_test, y_pred=y_pred_recall)

# # Generate scatter plot
# print("\nGenerating scatter plot for anomalies vs normals...")
# plotter.plot_scatter(
#     feature1_index=0,  # Replace with your feature1 index or column name
#     feature2_index=1,  # Replace with your feature2 index or column name
#     title="Scatter Plot of Anomalies vs Normals"
# )

# # Generate confusion matrix
# print("\nGenerating confusion matrix for the best recall model...")
# plotter.plot_confusion_matrix(labels=['Normal', 'Anomaly'], title="Confusion Matrix")

# # Print classification report
# print("\nClassification report:")
# plotter.print_classification_report()
