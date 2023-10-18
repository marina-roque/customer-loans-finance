#%%
from statsmodels.graphics.gofplots import qqplot
from sklearn.impute import SimpleImputer
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from scipy.stats import normaltest
from sqlalchemy import inspect
from sqlalchemy import text
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
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
###Things to change :   datestimes columns --use .to_datetime()-- (issue_date, earliest_credit_line, last_payment_date, next_payment_date, last_credit_pull_date)
#                       term column --use .strip to remove 'months'; fill the Nan with '0' values, rename column name--
#                       employement_length  -- use strip to remove 'years'; how to deal with odd values (10+)??--
#                       transform the column into categorical (grade, sub_grade, home_ownership, verification_status, loan_status, payment_plan, purpose, application_type)
## split date columns into months and years
#("issue_date", "earliest_credit_line", "last_payment_date", "next_payment_date", "last_credit_pull_date")
#("grade", "sub_grade", "home_ownership", "verification_status", "loan_status", "payment_plan", "purpose", "application_type")   
    
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

print("Column Types:")
#print(info.column_type()) 

print("\nStatistics:")
#print(info.extract_statistics())

print("\nDistinct Values in Categorical Columns:")
#print(info.count_distinct_values())

print("\nShape of the DataFrame:")
#print(info.print_shape())

print("\nNULL Value Counts:")
#print(info.count_null_values())
    
class Plotter:
    def __init__(self, data):
        self.data = data
        self.DataFrameTransform_instance = DataFrameTransform(data)
    
    #def plot_null_percentage(self):
        #null_percentage_before = (self.data.isnull().sum() / len(self.data)) * 100
        #self.data.dropna()  # Remove rows with missing values
        #null_percentage_after = (self.data.isnull().sum() / len(self.data)) * 100
    
    def plot_correlation_heatmap(self):
        # Compute the correlation matrix
        corr = self.data.corr()
    
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
    
        # Set up the heatmap style
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
        # Draw the heatmap
        plt.figure(figsize=(35, 35))  # Adjust the figure size as needed
        sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=True, cmap=cmap)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables', fontsize=36)
        plt.show()
        
      
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
    
    ''' def plot_scatter(self):  ###### Not working #########
        num_cols = len(self.data.columns)
        rows = cols = num_cols

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=self.data.columns, shared_xaxes=True, shared_yaxes=True)
        for i, column in enumerate(self.data.columns):
            for j, other_column in enumerate(self.data.columns):
                fig.add_trace(
                    go.Scatter(x=self.data[other_column], y=self.data[column], mode="markers", showlegend=False),
                    row=i + 1, col=j + 1
                )

        fig.update_layout(title_text="Scatter Plots of Columns")
        fig.update_xaxes(title_text="X-Axis")
        fig.update_yaxes(title_text="Y-Axis")
        fig.show()'''
    
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
    
    def transform_skewed_columns(self, transformations=['log', 'sqrt']):
        skewed_columns = self.identify_skewed_columns()
        
        for column, skewness in skewed_columns:
            for transformation in transformations:
                if transformation == 'log':
                    self.data[column] = np.log1p(self.data[column])
                elif transformation == 'sqrt':
                    self.data[column] = np.sqrt(self.data[column])
                    
    
        
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
    # Create a backup of the original DataFrame
    original_df = df.copy()
    # Save the original DataFrame to a backup file (e.g., CSV)
    #backup_file_path = 'original_data_backup.csv'
    #original_df.to_csv(backup_file_path, index=False)
    
    #For DataTransform
    info = DataFrameInfo(df)
    date_cols = ("issue_date", "earliest_credit_line", "last_payment_date", "next_payment_date", "last_credit_pull_date")
    cat_cols = ("grade", "sub_grade", "home_ownership", "term_in_months", "verification_status", "loan_status", "payment_plan", "purpose", "application_type")
    transformer = DataTransform(df, date_cols, cat_cols)
    transformer.transform_date_columns()
    transformer.transform_term_column('term')
    df.rename(columns={'term': 'term_in_months'}, inplace=True)
    #transformer.transform_employment_length('employment_length')
    transformer.convert_to_categorical()
    
    #for DataFrameTransform
    frame_transformer = DataFrameTransform(df)
    # Automatically drop rows with less than 1% null values and impute columns with 1% to 10% null values
   # Automatically drop rows with less than 1% null values and impute columns with 1% to 10% null values
    ##frame_transformer.drop_rows_and_impute() ####still not working
    frame_transformer.transform_skewed_columns(transformations=['log', 'sqrt'])  # Apply skewness transformation

    # Drop columns with more than 40% missing values
    frame_transformer.drop_columns_with_missing_values(threshold=0.4)
    
    
    #Plotter
    plotter = Plotter(df)
    #plotter.plot_skewed_columns()
    #msno.matrix(df)
    #plt.show()
    plotter.plot_scatter()
    
#################
'''
- frame_transformer drop rows not working at the moment
(KeyError: "['id', 'member_id', 'loan_amount', 'funded_amount_inv', 'term_in_months', 'instalment', 'grade', 'sub_grade', 'home_ownership', 'annual_inc', 'verification_status', 'issue_date', 'loan_status', 'payment_plan', 'purpose', 'dti', 'delinq_2yrs', 'earliest_credit_line', 'inq_last_6mths', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_payment_date', 'last_payment_amount', 'last_credit_pull_date', 'collections_12_mths_ex_med', 'policy_code', 'application_type'] not found in axis")

'''

