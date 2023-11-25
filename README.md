# Exploratory Data Analysis - Customer Loans in Finance

## Overview

This project focuses on the analysis of loan payment data to gain insights into loan performance and financial projections. The code includes functionality for data extraction from a database, data transformation, analysis of projected payments, and visualization of key indicators.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Extraction](#data-extraction)
  - [Data Transformation](#data-transformation)
  - [Data Analysis](#data-analysis)
  - [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project is structured into several modules, each serving a specific purpose:

- **Data Extraction**: The `RDSDatabaseConnector` class facilitates the extraction of loan payment data from a database and saves it to a CSV file.

- **Data Transformation**: The `DataTransform` class handles the transformation of the dataset, including the conversion of date columns, categorical encoding, and other necessary preprocessing steps.

- **Data Analysis**: The `DataInsight` class provides functions for calculating various financial metrics, including projected payments, charged-off loss, risk metrics, and totals.

- **Visualization**: The `Plotter` class offers visualizations such as correlation heatmaps, box plots with outliers, and comparisons of loan indicators.

## Preequisites

Libraries needed for this project:
- Pandas
- Numpy
- YAML
- CSV
- SQLAlchemy
- Plotly
- Matplotlib
- Sklearn

Example installation commands
pip install pandas numpy seaborn matplotlib


## Instalation

Clone the repository:

`git clone https://github.com/marina-roque/customer-loans-finance.git`

Navigate to the project directory:

`cd loan-payments-analysis`


Data Transformation
Overview
The data transformation process involves preparing the dataset for analysis. Key steps include handling date columns, converting categorical variables, and preprocessing the data.

Execution
Transform the dataset using the following command:
python data_transformation.py


## Usage
These are some examples on how to use this project and how to obtain information needed for analysis.

### Data Extraction

Extract loan payment data from the database and save it to a CSV file:

```
connector.extract_data_to_csv(table_name, csv_file_path)
df = connector.load_csv_to_dataframe(csv_file_path)
```

### Data Transformation

The data transformation process involves preparing the dataset for analysis. Key steps include handling date columns, converting categorical variables, and preprocessing the data.
Example how to transform the dataset:

```
transformer.transform_date_columns()
transformer.transform_term_column('term')
transformer.convert_to_categorical()
```
### Data Analysis

The data analysis phase focuses on calculating various financial metrics to gain insights into loan performance. This includes projected payments, charged-off loss, risk metrics, and overall totals.
Example how to analyze the loan payment data to calculate various financial metrics:
```
df_with_projection = insights.calculate_projected_payments()
charged_off_percentage = charged_off_stats['charged_off_percentage']
amount_paid_before_charge_off = charged_off_stats['amount_paid_before_charge_off']
risk_metrics_info = insights.calculate_risk_metrics()
```
### Visualization
Visualization plays a crucial role in understanding complex datasets. The Plotter class provides functions for visualizing loan indicators, correlation heatmaps, and box plots with outliers.
Examples how to visualize the data using various plots and visualizations:
```
plotter.plot_before_after_skewness(columns_for_skewness, transformations=['log', 'sqrt'])
plotter.visualize_loan_indicators(indicator_columns)
plotter.display_updated_heatmap(removed_columns, removed_columns)
plotter.plot_correlation_heatmap(df)
```

## Contributing
Feel free to contribute by opening issues or submitting pull requests. All contributions are welcome!

