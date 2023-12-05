"""The purpose of this code is to aggregate data from the Electric Vehicle Population Dataset based on the 'State' and 'Make' attributes. 
The aggregated data includes the mean of 'Electric Range' and the sum of 'Base MSRP'. 
The code also provides a method (plot_aggregated_data) to visualize the aggregated data using Seaborn and Matplotlib in the form of a bar chart, displaying 'Base MSRP' by 'State' with different colors for each 'Make'."""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Aggregator:
    """
    Aggregator class for handling Electric Vehicle Population Data aggregation and plotting.

    Attributes:
    - df (pd.DataFrame): DataFrame containing Electric Vehicle Population Data.

    Methods:
    - __init__(): Constructor method to initialize the Aggregator object and load data from a CSV file.
    - aggregate_data(): Method to aggregate data based on 'State' and 'Make' attributes.
                       Returns a DataFrame with mean 'Electric Range' and sum of 'Base MSRP'.
    - plot_aggregated_data(): Method to plot aggregated data using Seaborn and Matplotlib.
    """

    def __init__(self):
        """
        Constructor for the Aggregator class.

        Loads Electric Vehicle Population Data from a CSV file and initializes the df attribute.
        """
        self.df = pd.read_csv("https://raw.githubusercontent.com/saidadi/codingscripts/main/sai/Electric_Vehicle_Population_Data%20(3).csv")

    def aggregate_data(self):
        """
        Aggregate data based on 'State' and 'Make' attributes.

        Returns:
        pd.DataFrame: DataFrame with mean 'Electric Range' and sum of 'Base MSRP'.
        """
        aggregated_data = self.df.groupby(['State', 'Make']).agg({
            'Electric Range': 'mean',
            'Base MSRP': 'sum'
        }).reset_index()

        return aggregated_data

    def plot_aggregated_data(self):
        """
        Plot aggregated data using Seaborn and Matplotlib.
        """
        aggregated_data = self.aggregate_data()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='State', y='Base MSRP', data=aggregated_data.tail(20), hue='Make')
        
        plt.title('Aggregated Data: Base MSRP by State and Make')

        plt.show()