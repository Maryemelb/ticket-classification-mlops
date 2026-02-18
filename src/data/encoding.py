
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# from src.data.cleaning import cleaning
# from src.data.load_data import load_data

def encoding(df):
        
        columns = ['priority', 'queue']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        for col in columns:
                # Convert column to 2D and handle NaN
                df[col]= df[col].str[0]
                transformed = encoder.fit_transform(df[[col]])
                # Convert to DataFrame with proper column names
                transformed_df = pd.DataFrame(
                    transformed,
                    columns=encoder.get_feature_names_out([col]),
                    index=df.index
                ) 
                # delete original column and add encoded columns
                df = pd.concat([df.drop(columns=[col]), transformed_df], axis=1)
        le= LabelEncoder()
        df['type']= le.fit_transform(df['type'].str[0])
        return df

# df= load_data()
# clean_df= cleaning(df)
# encode_df= encoding(clean_df)
# encode_df.head()