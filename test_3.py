import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text
import yaml
import csv
import numpy as np


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
            query = f'SELECT * FROM {table_name}'
            data = conn.execute(query)
            
            # Fetch data and write to CSV file
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(data.keys())  # Write column headers
                csv_writer.writerows(data.fetchall())  # Write data rows
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
    
if __name__ == "__main__":
    # Initialize the RDSDatabaseConnector and specify the table and CSV file path
    connector = RDSDatabaseConnector('credentials.yaml')
    table_name = 'loan_payments'
    table_names = connector.get_table_names()
    csv_file_path = 'loan_payments.csv'
    
    # Extract data and save it to a CSV file
    connector.extract_data_to_csv(table_name, csv_file_path)
    # Load the CSV file into a Pandas DataFrame
    df = connector.load_csv_to_dataframe(csv_file_path)

if table_names:
    print("Table Names:")
    for table_name in table_names:
        print(table_name)
else:
    print("No tables found or an error occurred.")
    
