#%%
from statsmodels.graphics.gofplots import qqplot
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
from scipy.stats import normaltest
from sqlalchemy import inspect
from sqlalchemy import text
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import csv


#Load the database credentials from a YAML file
class RDSDatabaseConnector:
    def __init__(self, credentials_file):
        self.credentials = self.load_credentials(credentials_file)
        self.db_engine = self.create_db_engine(self.credentials)
        
    def load_credentials(self, file_path):
        try:
            with open(file_path, 'r') as file:
                credentials = yaml.safe_load(file)
            return credentials
        except Exception as e:
            print(f"Error loading credentials from {file_path}: {str(e)}")
            return None
        
    def create_db_engine(self, credentials):
        try:
            engine = create_engine(f"postgresql://{credentials['RDS_USER']}:{credentials['RDS_PASSWORD']}@{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/{credentials['RDS_DATABASE']}")
            engine.execution_options(isolation_level='AUTOCOMMIT').connect()
            return engine
        except Exception as e:
            print(f"Error creating database engine: {str(e)}")
            return None
        
    def extract_data_to_csv(self, table_name, csv_file_path):
        if not self.db_engine:
            print("Database connection or engine not available. Exiting.")
            return

        try:
            conn = self.db_engine.connect()
            
            # Execute SQL query
            query = text(f'SELECT * FROM {table_name}')
            result = conn.execute(query)
            
            # Fetch data and write to CSV file
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(result.keys())  # Write column headers
                csv_writer.writerows(result.fetchall())  # Write data rows
                print(f"Data saved to {csv_file_path}")
        except Exception as e:
            print(f"Error extracting data to CSV: {str(e)}")
        finally:
            conn.close()
            
    def load_csv_to_dataframe(self, csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            return df
        except Exception as e:
            print(f"Error loading CSV file into DataFrame: {str(e)}")
            return None
        
    def get_table_names(self):
        if not self.db_engine:
            print("Database engine not available. Exiting.")
            return []

        inspector = inspect(self.db_engine)
        table_names = inspector.get_table_names()
        return table_names
    
    
    
class DataTransform: 
    def __init__(self, data, date_cols, cat_cols):
        self.data = data
        self.date_columns = date_cols
        self.to_categorical_columns = cat_cols
           
    def transform_date_columns(self, input_format='%b-%Y'):
        for col in self.date_columns:
            # Parse the date in "MMM-yyyy" format
            self.data[col] = pd.to_datetime(self.data[col], format=input_format, errors='coerce')
    
    #Remove the word 'months', handle the Nan values with a '0' and convert the datatype to int    
    def transform_term_column(self, term_column):
        self.data[term_column] = self.data[term_column].str.strip(' months').fillna('0').astype(int)
    
    #Removes the word 'year' and convert the datatype to int
    def transform_employment_length(self, employment_length_column):
        self.data[employment_length_column] = self.data[employment_length_column].str.strip(' years').astype(int)
    
    #Convert specified columns to categorical type.
##add emplyment to this category
    def convert_to_categorical(self):
        for column in self.to_categorical_columns:
            self.data[column] = self.data[column].astype('category')    



class DataFrameInfo:
##### Things to do:  
    def __init__(self, data):
        self.data = data
    
    #Describes all columns in the dataframe to check their data type.    
    def column_type(self):
        return self.data.dtypes
    
    #Extract statistical values from dataframe.
    def extract_statistics(self):
        return self.data.describe()
     
    # Count distinct values in categorical columns.
    def count_distinct_values(self):
        return self.data.select_dtypes(include='category').nunique()
    
    # Print out the shape of the dataframe.
    def print_shape(self):
        return self.data.shape
    
    # Generate a count/percentage count of NULL values in each column.
    def count_null_values(self):
        null_counts = self.data.isnull().sum()
        total_counts = len(self.data)
        null_percentages = (null_counts / total_counts) * 100
        null_info = pd.DataFrame({
            'Null Count': null_counts,
            'Null Percentage': null_percentages})
        return null_info

'''print("Column Types:")
print(info.column_type()) 

print("\nStatistics:")
print(info.extract_statistics())

print("\nDistinct Values in Categorical Columns:")
print(info.count_distinct_values())

print("\nShape of the DataFrame:")
print(info.print_shape())

print("\nNULL Value Counts:")
print(info.count_null_values())'''
    
class Plotter:
    def __init__(self, data):
        self.data = data
        self.DataFrameTransform_instance = DataFrameTransform(data)
    
    def plot_null_percentage(self):
        null_percentage_before = (self.data.isnull().sum() / len(self.data)) * 100
        self.data.dropna()  # Remove rows with missing values
        null_percentage_after = (self.data.isnull().sum() / len(self.data)) * 100
    
    def plot_correlation_heatmap(self):
        # Compute the correlation matrix
        corr_matrix = self.data.select_dtypes(include='number').corr()

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the heatmap style
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap
        plt.figure(figsize=(30, 25))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.show()
    
    def remove_highly_correlated_columns(self, correlation_threshold=0.9):
        # Compute the correlation matrix
        corr_matrix = self.data.select_dtypes(include='number').corr()

        # Identify highly correlated column pairs
        highly_correlated_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    highly_correlated_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

        # Decide which columns to remove (keep one column from each pair)
        columns_to_remove = set()
        for col1, col2, _ in highly_correlated_pairs:
            if col1 not in cat_cols and col1 not in date_cols:
                columns_to_remove.add(col1)

        # Remove the highly correlated columns from the dataset
        df_cleaned = self.data.drop(columns=columns_to_remove)
        print(f"Removed highly correlated columns: {columns_to_remove}")

        return df_cleaned
    
    def display_updated_heatmap(self, df_cleaned, removed_columns):
        # Compute the correlation matrix for the cleaned DataFrame
        corr_cleaned = df_cleaned.corr()
        
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_cleaned, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        
        # Set up the heatmap style
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap
        plt.figure(figsize=(35, 35))  # Adjust the figure size as needed
        sns.heatmap(corr_cleaned, mask=mask, square=True, linewidths=.5, annot=True, cmap=cmap)
        plt.yticks(rotation=0)
        plt.title('Updated Correlation Matrix of all Numerical Variables', fontsize=36)
        plt.show()
        
        print("Removed columns:", removed_columns)
        
      
    def plot_skewed_columns(self, threshold=0.5):
        
        skewed_columns = self.DataFrameTransform_instance.identify_skewed_columns(threshold)

        if len(skewed_columns) == 0:
            print("No significantly skewed columns found.")
            return

        # Create histograms for skewed columns
        for column, skewness in skewed_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data, x=column, kde=True)
            plt.title(f'Distribution of {column} (Skewness: {skewness:.2f})')
            plt.show()
            
    def create_boxplot(self, data, y_cols, title="Box Plots"):
        data_traces = []
        for y_col in y_cols:
            boxplot = go.Box(y=data[y_col], name=y_col, boxpoints="outliers")
            data_traces.append(boxplot)

        layout = go.Layout(
            title=title,
            boxmode="group")

        fig = go.Figure(data=data_traces, layout=layout)
        fig.show()
             
class DataFrameTransform:
    def __init__(self, data):
        self.data = data

    def calculate_null_percentage(self):
        total_rows = len(self.data)
        null_percentages = (self.data.isnull().sum() / total_rows) * 100
        return null_percentages

    def drop_rows_and_impute(self):
        null_percentages = self.calculate_null_percentage()

        # Drop rows with less than 1% null values
        rows_to_drop = null_percentages[null_percentages < 1].index
        self.data.drop(index=rows_to_drop, inplace=True)

        # Impute columns with 1% to 10% null values (you can modify this threshold as needed)
        columns_to_impute = null_percentages[(null_percentages >= 1) & (null_percentages <= 10)].index
        for column in columns_to_impute:
            # You can choose a different imputation method here (e.g., mean, median, etc.)
            imputer = SimpleImputer(strategy='median')
            self.data[column] = imputer.fit_transform(self.data[column].values.reshape(-1, 1))

    def drop_columns_with_missing_values(self, threshold=0.4):
        # Drop columns where the percentage of missing values exceeds the threshold
        null_percentages = self.calculate_null_percentage()
        columns_to_drop = null_percentages[null_percentages > threshold].index
        self.data.drop(columns=columns_to_drop, inplace=True)

    def identify_skewed_columns(self, threshold=0.5):
        # Identify skewed columns
        skewed_columns = []
        
        for column in self.data.select_dtypes(include='number').columns:
            skewness = self.data[column].skew()
            if abs(skewness) > threshold:
                skewed_columns.append((column, skewness))
            
        return skewed_columns
    
    def transform_skewed_columns(self, transformation=['log', 'sqrt']):
        skewed_columns = self.identify_skewed_columns()
        
        for column, skewness in skewed_columns:
            if transformation == 'log':
                self.data[column] = np.log1p(self.data[column])
            elif transformation == 'sqrt':
                self.data[column] = np.sqrt(self.data[column])
    
    def handle_outliers_iqr(data, columns):
        # Create a copy of the DataFrame to avoid modifying the original data
        data_copy = data

        for column in columns:
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace values below the lower bound with the lower bound
            data_copy[column] = data_copy[column].apply(lambda x: lower_bound if x < lower_bound else x)
            # Replace values above the upper bound with the upper bound
            data_copy[column] = data_copy[column].apply(lambda x: upper_bound if x > upper_bound else x)

        return data_copy
    
       
if __name__ == "__main__":
    # Initialize the RDSDatabaseConnector and specify the table and CSV file path
    connector = RDSDatabaseConnector('credentials.yaml')
    table_name = 'loan_payments'
    table_names = connector.get_table_names()
    csv_file_path = 'loan_payments_2.csv'
    # Extract data and save it to a CSV file
    connector.extract_data_to_csv(table_name, csv_file_path)
    # Load the CSV file into a Pandas DataFrame
    df = connector.load_csv_to_dataframe(csv_file_path)
    original_df = df.copy()
    
    #For DataTransform
    info = DataFrameInfo(df)
    date_cols = ("issue_date", "earliest_credit_line", "last_payment_date", "next_payment_date", "last_credit_pull_date")
    cat_cols = ("grade", "sub_grade", "home_ownership", "term_in_months", "verification_status", "loan_status", "payment_plan", "purpose", "application_type")
    transformer = DataTransform(df, date_cols, cat_cols)
    transformer.transform_date_columns()
    transformer.transform_term_column('term')
    df.rename(columns={'term': 'term_in_months'}, inplace=True)
    transformer.convert_to_categorical()         
    transformer.data
    
    #for DataFrameTransform
    frame_transformer = DataFrameTransform(df)
    # Automatically drop rows with less than 1% null values and impute columns with 1% to 10% null values
    ####frame_transformer.drop_rows_and_impute() ##not working still
    frame_transformer.transform_skewed_columns()
    # Drop columns with more than 40% missing values
    frame_transformer.drop_columns_with_missing_values(threshold=0.4)
    
    
    #Plotter
    plotter = Plotter(df)
    '''plotter.plot_skewed_columns()
    plotter.plot_correlation_heatmap()
      # Remove highly correlated columns and display the updated heatmap
    removed_columns = plotter.remove_highly_correlated_columns()
    plotter.display_updated_heatmap(df, removed_columns)
    
    # Additional visualizations
    msno.matrix(df)
    
    columns_of_interest_1 = ['instalment', 'dti', 'delinq_2yrs', 'open_accounts']
    plotter.create_boxplot(df, columns_of_interest_1, title="Box Plots with Outliers 1")
    #df_outliers_handled = frame_transformer.handle_outliers_iqr(columns_of_interest_1)
    #plotter.create_boxplot(df_outliers_handled, columns_of_interest_1, title="Box Plots with Outliers Handled 1(IQR)")
    
    columns_of_interest_2 = ['loan_amount', 'out_prncp', 'total_payment']
    plotter.create_boxplot(df, columns_of_interest_2, title="Box Plots with Outliers 2")
    #df_outliers_handled = frame_transformer.handle_outliers_iqr(columns_of_interest_2)
    #plotter.create_boxplot(df_outliers_handled, columns_of_interest_2, title="Box Plots with Outliers Handled 2 (IQR)")

    columns_of_interest_3 = ['annual_inc']
    plotter.create_boxplot(df, columns_of_interest_3, title="Box Plots with Outliers 3")
    #df_outliers_handled = frame_transformer.handle_outliers_iqr(columns_of_interest_3)
    #plotter.create_boxplot(df_outliers_handled, columns_of_interest_3, title="Box Plots with Outliers Handled 3(IQR)")
'''

#############needs commiting still###############