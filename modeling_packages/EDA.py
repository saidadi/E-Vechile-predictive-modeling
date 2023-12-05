import pandas as pd
import matplotlib.pyplot as plt

class ExploratoryDataAnalysis:
    """
    The ExploratoryDataAnalysis class facilitates exploratory data analysis (EDA) by providing visualizations and insights into a given dataset. Upon initialization, it takes a pandas DataFrame as input. The class offers four main methods for visualizing data:

    1. distribution_plot(column): Generates a histogram to visualize the distribution of a specified column in the DataFrame, helping to understand the data's central tendency and spread.

    2. scatter_plot(x_column, y_column): Creates a scatter plot to explore the relationship between two specified columns, aiding in identifying potential patterns or correlations.

    3. geographical_distribution(): Extracts latitude and longitude information from a 'Vehicle Location' column, if available, and produces a scatter plot to visualize the geographical distribution of data points. This is particularly useful for datasets involving spatial information.

    4. correlation_heatmap(): Generates a heatmap of the correlation matrix for numerical columns in the DataFrame, providing an overview of the pairwise correlations between variables.

    These visualization methods can be employed to gain insights into the data's structure, relationships, and potential patterns, aiding in the initial stages of data exploration and hypothesis generation.
    """

    def __init__(self, data):
        """
        Initializes the ExploratoryDataAnalysis object with the provided pandas DataFrame.

        Parameters:
        - data (pd.DataFrame): The input DataFrame for which exploratory data analysis will be conducted.
        """
        self.data = data

    def distribution_plot(self, column):
        """
        Generates a histogram to visualize the distribution of a specified column in the DataFrame.

        Parameters:
        - column (str): The column for which the distribution will be visualized.

        Returns:
        - None
        """
        plt.figure(figsize=(10, 6))
        self.data[column].hist(bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def scatter_plot(self, x_column, y_column):
        """
        Creates a scatter plot to explore the relationship between two specified columns.

        Parameters:
        - x_column (str): The column for the x-axis.
        - y_column (str): The column for the y-axis.

        Returns:
        - None
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[x_column], self.data[y_column], alpha=0.5)
        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def geographical_distribution(self):
        """
        Extracts latitude and longitude information from a 'Vehicle Location' column, if available, and produces a scatter plot to visualize the geographical distribution of data points.

        Returns:
        - None
        """
        self.data[['Latitude', 'Longitude']] = self.data['Vehicle Location'].str.extract(r'\((-?\d+\.\d+) (-?\d+\.\d+)\)').astype(float)
        plt.scatter(self.data['Longitude'], self.data['Latitude'], alpha=0.5)
        plt.title('Geographical Distribution of Electric Vehicles')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def correlation_heatmap(self):
        """
        Generates a heatmap of the correlation matrix for numerical columns in the DataFrame, providing an overview of the pairwise correlations between variables.

        Returns:
        - None
        """
        plt.figure(figsize=(12, 8))
        corr_matrix = self.data.corr()
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title('Correlation Heatmap')
        plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
        plt.show()
