
import pandas as pd

def load_data():
  df= pd.read_csv('./src/data/dataset.csv')
  return df

# df = load_data()
# print(df)