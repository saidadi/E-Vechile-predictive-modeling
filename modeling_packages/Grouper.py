"""The purpose of this code is to group the Electric Vehicle Population Dataset by 'Model Year' and 'Electric Vehicle Type' and calculate the count. 
The code provides a method (plot_grouped_data) to visualize the grouped data using Seaborn and Matplotlib in the form of a bar chart, displaying the count by 'Model Year' with different colors for each 'Electric Vehicle Type'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Grouper:
    """
    Grouper class for handling Electric Vehicle Population Data grouping and visualization.

    Attributes:
    - df (pd.DataFrame): DataFrame containing Electric Vehicle Population Data.

    Methods:
    - __init__(): Constructor method to initialize the Grouper object and load data from a CSV file.
    - group_data(): Method to group data by 'Model Year' and 'Electric Vehicle Type' and calculate the count.
                    Returns a DataFrame with 'Model Year', 'Electric Vehicle Type', and 'Count' columns.
    - plot_grouped_data(): Method to plot grouped data using Seaborn and Matplotlib.
    """

    def __init__(self):
        """
        Constructor for the Grouper class.

        Loads Electric Vehicle Population Data from a CSV file and initializes the df attribute.
        """
        self.df = pd.read_csv("https://raw.githubusercontent.com/saidadi/codingscripts/main/sai/Electric_Vehicle_Population_Data%20(3).csv")

    def group_data(self):
        """
        Group data by 'Model Year' and 'Electric Vehicle Type' and calculate the count.

        Returns:
        pd.DataFrame: DataFrame with 'Model Year', 'Electric Vehicle Type', and 'Count' columns.
        """
        grouped_data = self.df.groupby(['Model Year', 'Electric Vehicle Type']).size().reset_index(name='Count')
        return grouped_data

    def plot_grouped_data(self):
        """
        Plot grouped data using Seaborn and Matplotlib.
        """
        grouped_data = self.group_data()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model Year', y='Count', data=grouped_data, hue='Electric Vehicle Type')
        
        plt.title('Grouped Data: Count by Model Year and EV Type')

        plt.show()