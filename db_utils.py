#%%
import pandas as pd
from sqlalchemy import create_engine
import yaml
import psycopg2
import csv

# Load the database credentials from a YAML file
class RDSDatabaseConnector:
    def __init__(self, credentials_file):
        self.credentials = self.load_credentials(credentials_file)
        self.connection = self.connect_to_database(self.credentials)
        self.db_engine = self.create_db_engine(self.credentials)
        
    def load_credentials(self, file_path):
        try:
            with open(file_path, 'r') as file:
                credentials = yaml.safe_load(file)
            return credentials
        except Exception as e:
            print(f"Error loading credentials from {file_path}: {str(e)}")
            return None

    # Connect to the remote database
    def connect_to_database(self, credentials):
        try:
            connection = psycopg2.connect(
                host=credentials['RDS_HOST'],
                port=credentials['RDS_PORT'],
                database=credentials['RDS_DATABASE'],
                user=credentials['RDS_USER'],
                password=credentials['RDS_PASSWORD']
            )
            return connection
        except Exception as e:
            print(f"Error connecting to the database: {str(e)}")
            return None

    # Create the SQLAlchemy engine
    def create_db_engine(self, credentials):
        try:
            engine = create_engine(f"postgresql://{credentials['RDS_USER']}:{credentials['RDS_PASSWORD']}@{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/{credentials['RDS_DATABASE']}")
            return engine
        except Exception as e:
            print(f"Error creating database engine: {str(e)}")
            return None
        
    def extract_data_to_csv(self, table_name, csv_file_path):
        if not self.connection or not self.db_engine:
            print("Database connection or engine not available. Exiting.")
            return

        cursor = self.connection.cursor()
        
        try:
            query = f'SELECT * FROM {table_name}'
            cursor.execute(query)
            data = cursor.fetchall()
        except Exception as e:
            print(f"Error executing SQL query: {str(e)}")
            return

        try:
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([desc[0] for desc in cursor.description])  # Write column headers
                csv_writer.writerows(data)  # Write data rows
                print(f"Data saved to {csv_file_path}")
        except Exception as e:
            print(f"Error saving data to CSV file: {str(e)}")
        finally:
            cursor.close()
            self.connection.close()

if __name__ == "__main__":
    # Initialize the RDSDatabaseConnector and specify the table and CSV file path
    connector = RDSDatabaseConnector('credentials.yaml')
    table_name = 'loan_payments'
    csv_file_path = 'loan_payments.csv'
    
    # Extract data and save it to a CSV file
    connector.extract_data_to_csv(table_name, csv_file_path)




