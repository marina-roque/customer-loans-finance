
#%%
import pandas as pd
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

df = pd.read_csv('loan_payments_2.csv')
df.dtypes

data = df['int_rate']
stat, p = normaltest(data, nan_policy='omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))

df['int_rate'].hist(bins=100)
qq_plot = qqplot(df['int_rate'] , scale=1 ,line='q')
pyplot.show()

print(f'The median of HouseAge is {df["int_rate"].median()}')
print(f'The mean of HouseAge is {df["int_rate"].mean()}')



############ guardar codigo para test-2#################

class Plotter:
    def __init__(self, data):
        self.data = data
    
    def plot_null_percentage(self):
        null_percentage_before = (self.data.isnull().sum() / len(self.data)) * 100
        self.data.dropna()  # Remove rows with missing values ** I deleted the inplace = true from the brackets
        null_percentage_after = (self.data.isnull().sum() / len(self.data)) * 100
               
class DataFrameTransform:
    def __init__ (self, data):
        self.data = data
        
    def count_nulls(self):
        return self.data.isnull().sum()
    
    # Drop columns where the percentage of missing values exceeds the threshold
    def drop_columns_with_missing_values(self, threshold=0.7):
        null_counts = self.count_nulls()
        columns_to_drop = null_counts[null_counts / len(self.data) > threshold].index
        self.data.drop(columns=columns_to_drop)#** I deleted the inplace = true from the brackets
        
    def impute_missing_values(self, method='median'):
        if method == 'median':
            self.data.fillna(self.data.median()) #** I deleted the inplace = true from the brackets
        elif method == 'mean':
            self.data.fillna(self.data.mean()) #** I deleted the inplace = true from the brackets
            
############## Use of simpleimputer to imput new values into the missing ones##########

#print(msno.matrix(df)) ##creates an graphic that shows null values per column
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
print(numeric_columns.isnull().sum())
median_imputer = SimpleImputer(strategy="median")
numeric_columns_imputed = median_imputer.fit_transform(numeric_columns)
median_df = pd.DataFrame(data=numeric_columns_imputed, columns=numeric_columns.columns)
median_df.isnull().sum()

###################
# %%

