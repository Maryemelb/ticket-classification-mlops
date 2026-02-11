
from load_data import load_data

def fustion_text(df):
    df['fusion_email']= df['subject'].fillna('')+ df['body'].fillna('')
    return df
df= load_data()
fdf= fustion_text(df)
fdf.head()