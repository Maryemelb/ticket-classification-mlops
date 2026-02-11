
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from load_data import load_data
from preprocessing import fustion_text
def cleaning(df):
    #normalisation
    features = df.columns
    for feature in features:
      df[feature]= df[feature].str.lower()
    
    #delete ponctuations
    tokenizer= RegexpTokenizer(r'\w+')
    for feature in features:
      df[feature]= df[feature].apply(lambda x: tokenizer.tokenize(str(x)))
    
    # implement stopwords for each language
    stopwrords_en= stopwords.words('english')
    stopwords_de= stopwords.words('german')

    #filter df whene it is in en

    filtered_df_en= df[df['language'].str[0]== 'en']
    for feature in filtered_df_en.columns:
      filtered_df_en[feature]= filtered_df_en[feature].apply(lambda x:[word for word in x if word not in stopwrords_en])       
    df.update(filtered_df_en)

    # filter df whene it is in germany

    filtered_df_de= df[df['language'].str[0]== 'de']
    for feature in filtered_df_de.columns:
       filtered_df_de[feature]= filtered_df_de[feature].apply(lambda x:[ word for word in x if word not in stopwords_de])
    df.update(filtered_df_de)

df= load_data()
fdf= fustion_text(df)
clean_df= cleaning(fdf)
clean_df.head()