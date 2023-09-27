#%%
import psycopg2
import csv
import yaml

# Load the credentials from the YAML file
with open('credentials.yaml', 'r') as yaml_file:
    credentials = yaml.safe_load(yaml_file)

# Extract the credentials
host = credentials['RDS_HOST']
dbname = credentials['RDS_DATABASE']
user = credentials['RDS_USER']
password = credentials['RDS_PASSWORD']
port = credentials['RDS_PORT']

# Connect to the database
try:
    connection = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port
    )

    # Create a cursor
    cursor = connection.cursor()

    # Execute SQL query to extract all da"ta from the 'loan_payments' table
    query = 'SELECT * FROM loan_payments'
    cursor.execute(query)

    # Fetch all the data
    data = cursor.fetchall()

    # Define the path to the CSV file where you want to save the data
    csv_file_path = 'loans_payments.csv'

    # Write the data to the CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([desc[0] for desc in cursor.description])  # Write column headers
        csv_writer.writerows(data)  # Write data rows

    # Close the cursor and connection when done
    cursor.close()
    connection.close()

    print(f'Data from the loans_payments table has been saved to {csv_file_path}')

except psycopg2.Error as error:
    print("Error connecting to the database:", error)

