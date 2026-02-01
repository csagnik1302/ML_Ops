import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import copy

data = pd.read_csv("/home/sagnik-chandra/MLOps Project/Dataset/Bank-term-deposit.csv")

cat_list=data.select_dtypes(include=["object"]).columns.to_list()    

# data.select_dtypes(include=["object"]) : Extracts only Categorical data Columns from the Dataset, TO FORM A NEW DATAFRAME. This method only needs 1 parameter, Include(or Exclude)  
# .columns: holds the column labels of a DataFrame as a pandas.Index object. 
# to_list():  stores the columns labels generated using .columns into a list


# Initialize the OneHotEncoder (Yes OHE needs an initialization before it could be used)
# Read scikit doc to understand parameters

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

ohe_transform = ohe.fit_transform(data[cat_list])


data1=copy.deepcopy(data)

data1 = data1.drop(columns=cat_list)  # Drop prexisting non-encoded Categorical feature columns (To replace them with encoded ones)

data1 = data1 = pd.concat([data1,ohe_transform],axis=1)

data1 = data1.drop(columns=['y_no'])



data1.to_csv("/home/sagnik-chandra/MLOps Project/Dataset/Final_Encoded_Data",index=False)

# index=False ensures that the index column (The column of index that pandas genereates by itself when data was firest imported and cnoverted into a dataframe) is dropped before exporting

