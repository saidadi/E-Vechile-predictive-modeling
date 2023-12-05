"""The purpose of this code is to reshape the Electric Vehicle Population Dataset using different Pandas functions (pivot_table, melt, and stack) and visualize the reshaped data. 
The code provides methods for reshaping the data and plotting the results using Matplotlib and Seaborn for exploratory data analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Reshaper:
    """
    Reshaper class for handling Electric Vehicle Population Data reshaping and visualization.

    Attributes:
    - df (pd.DataFrame): DataFrame containing Electric Vehicle Population Data.

    Methods:
    - __init__(): Constructor method to initialize the Reshaper object and load data from a CSV file.
    - reshape_data_with_pivot(): Method to reshape data using Pandas pivot_table function.
    - reshape_data_with_melt(): Method to reshape data using Pandas melt function.
    - reshape_data_with_stack(): Method to reshape data using Pandas stack function.
    - reshape_data_with_pivot_plot(): Method to plot reshaped data using Pandas pivot_table and Matplotlib.
    - reshape_data_with_melt_plot(): Method to plot reshaped data using Pandas melt and Seaborn.
    - reshape_data_with_stack_plot(): Method to plot reshaped data using Pandas stack and Seaborn.
    """

    def __init__(self):
        """
        Constructor for the Reshaper class.

        Loads Electric Vehicle Population Data from a CSV file and initializes the df attribute.
        """
        self.df = pd.read_csv("https://raw.githubusercontent.com/saidadi/codingscripts/main/sai/Electric_Vehicle_Population_Data%20(3).csv")

    def reshape_data_with_pivot(self):
        """
        Reshape data using Pandas pivot_table function.

        Returns:
        pd.DataFrame: Reshaped data with mean values of 'Electric Range' and 'Base MSRP'.
        """
        reshaped_data = pd.pivot_table(self.df, values=['Electric Range', 'Base MSRP'], index=['VIN (1-10)', 'Model Year', 'Make', 'Model'], columns='State', aggfunc='mean', fill_value=0)
        reshaped_data_pivot = reshaped_data.reset_index()
        return reshaped_data_pivot

    def reshape_data_with_melt(self):
        """
        Reshape data using Pandas melt function.

        Returns:
        pd.DataFrame: Reshaped data with 'VIN (1-10)', 'Model Year', 'Make', 'Model', 'Metric', and 'Value' columns.
        """
        reshaped_data_melt = pd.melt(self.df, id_vars=['VIN (1-10)', 'Model Year', 'Make', 'Model'], value_vars=['Electric Range', 'Base MSRP'], var_name='Metric', value_name='Value')
        return reshaped_data_melt

    def reshape_data_with_stack(self):
        """
        Reshape data using Pandas stack function.

        Returns:
        pd.DataFrame: Reshaped data with 'VIN (1-10)', 'Model Year', 'Make', 'Model', 'Variable', and 'Value' columns.
        """
        reshaped_data_stack = self.df.set_index(['VIN (1-10)', 'Model Year', 'Make', 'Model']).stack().reset_index()
        reshaped_data_stack.columns = ['VIN (1-10)', 'Model Year', 'Make', 'Model', 'Variable', 'Value']
        return reshaped_data_stack

    def reshape_data_with_pivot_plot(self):
        """
        Plot reshaped data using Pandas pivot_table and Matplotlib.
        """
        reshaped_data_pivot = self.reshape_data_with_pivot().tail(20)
        reshaped_data_pivot.plot(kind='bar', x='Model', y='Electric Range', stacked=True)
        plt.title('Stacked Barplot of Electric Range by Model and State')
        plt.show()

    def reshape_data_with_melt_plot(self):
        """
        Plot reshaped data using Pandas melt and Seaborn.
        """
        reshaped_data_melt = self.reshape_data_with_melt()
        sns.catplot(x='Metric', y='Value', hue='Make', data=reshaped_data_melt.head(10), kind='bar')
        plt.title('Barplot of Electric Range and Base MSRP by Make')
        plt.show()

    def reshape_data_with_stack_plot(self):
        """
        Plot reshaped data using Pandas stack and Seaborn.
        """
        reshaped_data_stack = self.reshape_data_with_stack()
        reshaped_data_stack['Value'] = reshaped_data_stack['Value'].astype(str)
        sns.lineplot(x='Variable', y='Value', hue='Make', data=reshaped_data_stack.head(10), marker='o')
        plt.title('Line Plot of Variables by Make')
        plt.show()