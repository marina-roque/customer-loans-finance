#%%
import pandas as pd
import yaml
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    def __init__(self, credentials_file_path):
        self.credentials_file_path = credentials_file_path
        self.load_credentials()
        self.initialize_engine()

    def load_credentials(self):
        # Load database credentials from the YAML file
        try:
            with open(self.credentials_file_path, 'r') as file:
                self.credentials = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading credentials from {self.credentials_file_path}: {str(e)}")
            return None
        
    def initialize_engine(self):
        # Initialize the SQLAlchemy engine.
        try:
            db_url = f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
            self.engine = create_engine(db_url)
        except Exception as e:
            print(f"Error creating database engine: {str(e)}")
            return None
        
    def extract_data(self, loan_payments):
        # Extract data from the specified table and return it as a Pandas DataFrame.
        try:
            query = f'SELECT * FROM {loan_payments}'
            data = pd.read_sql_query(query, self.engine)
            return data
        except Exception as e:
            print(f"Error extracting data from {loan_payments}: {str(e)}")
            return None
        
    def save_to_csv(self, data, file_path):
        try:
            data.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error saving data to {file_path}: {str(e)}")
            return None
        
if __name__ == "__main__":
    # Step 1: Load the credentials from the YAML file
    credentials_file_path = 'credentials.yaml'

    # Step 2: Create an instance of the RDSDatabaseConnector class
    db_connector = RDSDatabaseConnector(credentials_file_path)

    # Step 3: Replace 'your_table_name' with the name of the table you want to extract data from
    table_name = 'loans_payments'

    # Step 4: Extract data from the specified table
    data = db_connector.extract_data(table_name)

    # Step 5: Specify the file path where you want to save the CSV file
    csv_file_path = 'loans_payments.csv'

    # Step 6: Save the data to a CSV file
    db_connector.save_to_csv(data, csv_file_path)




