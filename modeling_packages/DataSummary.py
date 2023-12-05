import pandas as pd

class DataSummary:
    """
    The DataSummary class is designed to provide a quick overview of a given pandas DataFrame. Upon initialization, it takes a pandas DataFrame as input. The class offers three main methods for summarizing the data:

    1. basic_info(): Returns the concise summary of the DataFrame using the info() method, displaying data types, non-null counts, and memory usage.
    
    2. missing_values(): Returns a Series showing the count of missing values for each column using isnull() and sum(). This method helps identify the extent of missing data in the DataFrame.
    
    3. summary_statistics(): Returns a statistical summary of the DataFrame using the describe() method. This includes count, mean, std deviation, min, 25th percentile, median, 75th percentile, and max values for each numeric column, providing a quick insight into the distribution of numerical data.

    These methods can be utilized to perform an initial exploration of the dataset, identifying data types, missing values, and obtaining key statistical metrics, facilitating the data analysis and cleaning process.
    """

    def __init__(self, data):
        """
        Initializes the DataSummary object with the provided pandas DataFrame.

        Parameters:
        - data (pd.DataFrame): The input DataFrame for which the summary will be generated.
        """
        self.data = data

    def basic_info(self):
        """
        Returns the concise summary of the DataFrame using the info() method, displaying data types, non-null counts, and memory usage.

        Returns:
        - None
        """
        return self.data.info()

    def missing_values(self):
        """
        Returns a Series showing the count of missing values for each column using isnull() and sum(). This method helps identify the extent of missing data in the DataFrame.

        Returns:
        - pd.Series: A Series displaying the count of missing values for each column.
        """
        return self.data.isnull().sum()

    def summary_statistics(self):
        """
        Returns a statistical summary of the DataFrame using the describe() method. This includes count, mean, std deviation, min, 25th percentile, median, 75th percentile, and max values for each numeric column, providing a quick insight into the distribution of numerical data.

        Returns:
        - pd.DataFrame: A DataFrame containing the summary statistics for each numeric column.
        """
        return self.data.describe()
