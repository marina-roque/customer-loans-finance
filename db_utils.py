#%%
from sklearn.impute import SimpleImputer
from sqlalchemy import text, create_engine, inspect
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
        self.db_url = self.create_db_url(self.credentials)
    
    def create_db_url(self, credentials):
        db_url = f"postgresql://{credentials['RDS_USER']}:{credentials['RDS_PASSWORD']}@{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/{credentials['RDS_DATABASE']}"
        return db_url
        
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
    def convert_to_categorical(self):
        for column in self.to_categorical_columns:
            self.data[column] = self.data[column].astype('category')    



class DataFrameInfo:
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
    
    def transform_data(self):
        return DataFrameTransform(self.data)
    
    def plot_null_percentage(self):
        null_percentage_before = (self.data.isnull().sum() / len(self.data)) * 100
        self.data.dropna()  # Remove rows with missing values
        null_percentage_after = (self.data.isnull().sum() / len(self.data)) * 100
    
    def plot_correlation_heatmap(self, data, title="Correlation Matrix Heatmap"):
        corr_matrix = data.select_dtypes(include='number').corr()

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the heatmap style
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap
        plt.figure(figsize=(15, 8))
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
        columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'annual_inc', 'dti', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries']
        data_transformer = self.transform_data()
        skewed_columns = data_transformer.identify_skewed_columns(columns, threshold)

        if len(skewed_columns) == 0:
            print("No significantly skewed columns found.")
            return
        
        for column, skewness in skewed_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data, x=column, kde=True)
            plt.title(f'Distribution of {column} (Skewness: {skewness:.2f})')
            plt.show()
    
    def plot_before_after_skewness(self, columns, transformations=['log', 'sqrt']):
        original_data = original_df

        for column in columns:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(original_data[column], kde=True)
            skewness_before = original_data[column].skew()
            plt.title(f'Before Transformation: {column}\nSkewness: {skewness_before:.2f}')
            
            plt.subplot(1, 2, 2)
            transformed_data = self.transform_skewness(column, transformations)
            skewness_after = transformed_data.skew()
            sns.histplot(transformed_data, kde=True)
            plt.title(f'After Transformation: {column}\nSkewness: {skewness_after:.2f}')
            
            plt.tight_layout()
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
    
    def transform_skewness(self, column, transformations=['log', 'sqrt']):
        transformed_data = self.data[column].copy()
        for transformation in transformations:
            if transformation == 'log':
                transformed_data = np.log1p(transformed_data)
            elif transformation == 'sqrt':
                transformed_data = np.sqrt(transformed_data)
        return transformed_data
    
    def visualize_loan_indicators(self, indicator_columns):
        df = self.data
        # Identify customers currently behind on loan payments
        behind_customers = df[df['loan_status'].isin(['Late (16-30 days)', 'Late (31-120 days)'])]

        # Identify customers already charged off
        charged_off_customers = df[df['loan_status'] == 'Charged Off']

        # Create subsets for visualization
        behind_subset = behind_customers[indicator_columns]
        charged_off_subset = charged_off_customers[indicator_columns]

        # Visualize correlation matrices
        self.plot_correlation_heatmap(behind_subset, title='Correlation Matrix for Currently Behind Customers')
        self.plot_correlation_heatmap(charged_off_subset, title='Correlation Matrix for Charged Off Customers')
    
    def plot_indicators_comparison(self, indicator_column):
        df = self.data
        
        # Identify customers currently behind on loan payments
        behind_customers = df[df['loan_status'].isin(['Late (16-30 days)', 'Late (31-120 days)'])]

        # Identify customers already charged off
        charged_off_customers = df[df['loan_status'] == 'Charged Off']

        # Plot indicators comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.countplot(x=indicator_column, data=behind_customers, hue='loan_status')
        plt.title(f'Comparison of {indicator_column} for Currently Behind Customers')
        plt.xlabel(indicator_column)
        plt.ylabel('Count')
        plt.legend(title='Loan Status')

        plt.subplot(1, 2, 2)
        sns.countplot(x=indicator_column, data=charged_off_customers, hue='loan_status')
        plt.title(f'Comparison of {indicator_column} for Charged Off Customers')
        plt.xlabel(indicator_column)
        plt.ylabel('Count')
        plt.legend(title='Loan Status')

        plt.tight_layout()
        plt.show()
            
class DataFrameTransform:
    def __init__(self, data):
        self.data = data

    def calculate_null_percentage(self):
        total_rows = len(self.data)
        null_percentages = (self.data.isnull().sum() / total_rows) * 100
        return null_percentages

    def drop_rows_and_impute(self):
        null_percentages = self.calculate_null_percentage()
        numeric_columns_to_impute = null_percentages[(null_percentages >= 1) & (null_percentages <= 10)].index
        for column in numeric_columns_to_impute:
            if self.data[column].dtype in ['int64', 'float64']:
                imputer = SimpleImputer(strategy='median')
                self.data[column] = imputer.fit_transform(self.data[column].values.reshape(-1, 1))
        
    def drop_columns_with_missing_values(self, threshold=0.4):
        # Drop columns where the percentage of missing values exceeds the threshold
        null_percentages = self.calculate_null_percentage()
        columns_to_drop = null_percentages[null_percentages > threshold].index
        self.data.drop(columns=columns_to_drop, inplace=True)

    def identify_skewed_columns(self, columns, threshold=0.5):
        # Identify skewed columns
        skewed_columns = []
        
        for column in columns:
            if self.data[column].dtype in ['int64', 'float64']:
                    skewness = self.data[column].skew()
                    if abs(skewness) > threshold:
                        skewed_columns.append((column, skewness)) 
        return skewed_columns
    
    def transform_skewed_columns(self, transformation=['log', 'sqrt']):
        columns_for_skewness = ('loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'annual_inc', 'dti', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries')
        skewed_columns = self.identify_skewed_columns(columns_for_skewness)
        
        for column, skewness in skewed_columns:
            if transformation == 'log':
                self.data[column] = np.log1p(self.data[column])
            elif transformation == 'sqrt':
                self.data[column] = np.sqrt(self.data[column])
    
    def handle_outliers_iqr(self, data, columns):
        data_copy = data.copy()  # Use copy() to avoid modifying the original DataFrame
        for column in columns:
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1

            # Define the bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with the bounds
            data_copy[column] = np.where(data_copy[column] < lower_bound, lower_bound, data_copy[column])
            data_copy[column] = np.where(data_copy[column] > upper_bound, upper_bound, data_copy[column])

        return data_copy
    
class DataInsight:
    def __init__(self, data):
        self.data = data
        
    def calculate_projected_payments(self):
        self.data['end_date'] = self.data.apply(lambda row: row['issue_date'] + pd.DateOffset(months=row['term_in_months']), axis=1)

        # Calculate the difference in months
        self.data['months_left_to_pay'] = (self.data['end_date'].dt.year - self.data['last_payment_date'].dt.year) * 12 + self.data['end_date'].dt.month - self.data['last_payment_date'].dt.month

        def calculate_payment(row):
            if row['months_left_to_pay'] <= 6:
                return row['instalment'] * row['months_left_to_pay']
            else:
                return row['instalment'] * 6

        self.data['project_pay_6_months'] = self.data.apply(calculate_payment, axis=1)
        return self.data
            
    def calculate_totals(self):
        totals = {}

        if 'project_pay_6_months' in self.data.columns:
            # Remaining outstanding principal for the total amount funded
            totals['total_outstanding_amount'] = np.sum(self.data['out_prncp'])
            # Payments received to date for the total amount funded
            totals['total_payment_sum'] = np.sum(self.data['total_payment'])
            # The total amount committed by investors for that loan
            totals['total_inv_amount'] = np.sum(self.data['funded_amount_inv'])
            # The total amount committed to the loan
            totals['total_amount'] = np.sum(self.data['funded_amount'])
            # Total of the projection of the amount to be paid in the next 6 months
            totals['projected_amount'] = np.sum(self.data['project_pay_6_months'])
        else:
            print("'project_pay_6_months' column not found in the DataFrame.")

        return totals
    
    def calculate_charged_off_loss(self):
        df = self.calculate_projected_payments()

        # Select only Charged Off loans
        charged_off_loans = df[df['loan_status'] == 'Charged Off']

        # Calculate the total projected revenue for all Charged Off loans
        total_projected_revenue_charged_off = charged_off_loans['project_pay_6_months'].sum()

        # Calculate the total revenue that was actually received for Charged Off loans
        total_actual_revenue_charged_off = charged_off_loans['total_payment'].sum()

        # Calculate the percentage of expected revenue that was lost
        percentage_revenue_loss = ((total_projected_revenue_charged_off - total_actual_revenue_charged_off) / total_projected_revenue_charged_off) * 100

        return {
            'total_projected_revenue_charged_off': total_projected_revenue_charged_off,
            'total_actual_revenue_charged_off': total_actual_revenue_charged_off,
            'percentage_revenue_loss': percentage_revenue_loss
        }

    def calculate_risk_metrics(self):
        df = self.calculate_projected_payments()

        # Identify customers currently behind on loan payments
        behind_customers = df[df['loan_status'].isin(['Late (16-30 days)', 'Late (31-120 days)'])]

        # Calculate the percentage of users in this bracket as a percentage of all loans
        percentage_behind_users = (len(behind_customers) / len(df)) * 100

        # Calculate the total amount of customers in this bracket
        total_behind_customers = len(behind_customers)

        # Calculate the loss if these users' status changed to Charged Off
        loss_if_charged_off = behind_customers['project_pay_6_months'].sum()

        # Calculate the projected loss if the customer were to finish the loan's term
        projected_loss_if_finished = behind_customers['project_pay_6_months'].sum()

        # Identify customers who have already defaulted on their loan
        charged_off_customers = df[df['loan_status'] == 'Charged Off']

        # Calculate the percentage of total revenue these customers represent
        percentage_charged_off_revenue = (charged_off_customers['total_payment'].sum() / df['total_payment'].sum()) * 100

        # Calculate the percentage of total revenue defaulting customers represent
        percentage_defaulted_revenue = (loss_if_charged_off / df['total_payment'].sum()) * 100

        return {
            'percentage_behind_users': percentage_behind_users,
            'total_behind_customers': total_behind_customers,
            'loss_if_charged_off': loss_if_charged_off,
            'projected_loss_if_finished': projected_loss_if_finished,
            'percentage_charged_off_revenue': percentage_charged_off_revenue,
            'percentage_defaulted_revenue': percentage_defaulted_revenue
        }

    def calculate_charged_off_stats(self):
        charged_off_loans = self.data[self.data['loan_status'] == 'Charged Off']

        # Calculate the percentage of charged-off loans
        charged_off_percentage = (len(charged_off_loans) / len(self.data)) * 100

        # Calculate the amount paid towards charged-off loans before being charged off
        amount_paid_before_charge_off = charged_off_loans['total_payment'].sum()

        return {
            'charged_off_percentage': charged_off_percentage,
            'amount_paid_before_charge_off': amount_paid_before_charge_off}
        
       
if __name__ == "__main__":
    # Initialize the RDSDatabaseConnector and specify the table and CSV file path
    connector = RDSDatabaseConnector('credentials.yaml')
    table_name = 'loan_payments'
    table_names = connector.get_table_names()
    csv_file_path = 'db_utils.csv'
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
    
    #for DataFrameTransform
    frame_transformer = DataFrameTransform(df)
    null_percentages = frame_transformer.calculate_null_percentage()
    frame_transformer.drop_rows_and_impute()
    columns_for_skewness = ('loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'annual_inc', 'dti', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries')
    frame_transformer.transform_skewed_columns(transformation=['log', 'sqrt'])
    frame_transformer.drop_columns_with_missing_values(threshold=0.4)
    
    #Insights
    insights = DataInsight(df)
    df_with_projection = insights.calculate_projected_payments()
    charged_off_stats = insights.calculate_charged_off_stats()
    charged_off_percentage = charged_off_stats['charged_off_percentage']
    amount_paid_before_charge_off = charged_off_stats['amount_paid_before_charge_off']
    charged_off_loss_info = insights.calculate_charged_off_loss()
    risk_metrics_info = insights.calculate_risk_metrics()

    print(f"Percentage of Users Currently Behind on Payments: {risk_metrics_info['percentage_behind_users']:.2f}%")
    print(f"Total Number of Customers Currently Behind: {risk_metrics_info['total_behind_customers']}")
    print(f"Loss if Users in Behind Bracket Converted to Charged Off: ${risk_metrics_info['loss_if_charged_off']:.2f}")
    print(f"Projected Loss if Users in Behind Bracket Finished Loans: ${risk_metrics_info['projected_loss_if_finished']:.2f}")
    print(f"Percentage of Total Revenue Represented by Charged Off Customers: {risk_metrics_info['percentage_charged_off_revenue']:.2f}%")
    print(f"Percentage of Total Revenue Represented by Defaulted Customers: {risk_metrics_info['percentage_defaulted_revenue']:.2f}%")

    print(f"Total Projected Revenue for Charged Off Loans: ${charged_off_loss_info['total_projected_revenue_charged_off']:.2f}")
    print(f"Total Actual Revenue Received for Charged Off Loans: ${charged_off_loss_info['total_actual_revenue_charged_off']:.2f}")
    print(f"Percentage of Expected Revenue Lost: {charged_off_loss_info['percentage_revenue_loss']:.2f}%")

    print(f"Percentage of Charged-Off Loans: {charged_off_percentage:.2f}%")
    print(f"Amount Paid Towards Charged-Off Loans Before Being Charged Off: ${amount_paid_before_charge_off:.2f}")
    totals = insights.calculate_totals()
    total_outstanding_amount = totals['total_outstanding_amount']
    total_payment_sum = totals['total_payment_sum']
    total_inv_amount = totals['total_inv_amount']
    total_amount = totals['total_amount']
    projected_amount = totals['projected_amount']
    
    percentage_rec = total_payment_sum / (total_amount / 100)
    print(f"Current General Recovery Percentage: {percentage_rec:.2f}%")
    
    recovery_inv = total_payment_sum / (total_inv_amount / 100) 
    print(f"Current Investor Recovery Percentage: {recovery_inv:.2f}%")
    
    project_percentage_recovery = (total_payment_sum + projected_amount) / (total_amount / 100)
    print(f"Projection of General Recovery Percentage in 6 months: {project_percentage_recovery:.2f}%")
    
    #Plotter
    plotter = Plotter(df)
    removed_columns = plotter.remove_highly_correlated_columns()
    plotter.display_updated_heatmap(removed_columns, removed_columns)
    plotter.plot_correlation_heatmap(df)
    plotter.plot_before_after_skewness(columns_for_skewness, transformations=['log', 'sqrt'])
   
    columns_of_interest_1 = ['instalment', 'dti', 'delinq_2yrs', 'open_accounts']
    plotter.create_boxplot(df, columns_of_interest_1, title="Box Plots with Outliers 1")
    df_outliers_handled = frame_transformer.handle_outliers_iqr(df, columns_of_interest_1)
    plotter.create_boxplot(df_outliers_handled, columns_of_interest_1, title="Box Plots with Outliers Handled 1(IQR)")

    columns_of_interest_2 = ['loan_amount', 'out_prncp', 'total_payment']
    plotter.create_boxplot(df, columns_of_interest_2, title="Box Plots with Outliers 2")
    df_outliers_handled = frame_transformer.handle_outliers_iqr(df, columns_of_interest_2)
    plotter.create_boxplot(df_outliers_handled, columns_of_interest_2, title="Box Plots with Outliers Handled 2 (IQR)")

    columns_of_interest_3 = ['annual_inc']
    plotter.create_boxplot(df, columns_of_interest_3, title="Box Plots with Outliers 3")
    df_outliers_handled = frame_transformer.handle_outliers_iqr(df, columns_of_interest_3)
    plotter.create_boxplot(df_outliers_handled, columns_of_interest_3, title="Box Plots with Outliers Handled 3(IQR)")

    indicator_columns = ['grade', 'purpose', 'dti']
    plotter.visualize_loan_indicators(indicator_columns)
# Plot comparison of indicators for currently behind and charged off customers
    for column in indicator_columns:
        plotter.plot_indicators_comparison(column)
