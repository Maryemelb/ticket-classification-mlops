
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from src.data.load_data import load_data
def cleaning(df):
    
    #fustion_text
    df['fusion_email']= df['subject'].fillna('')+ df['body'].fillna('')

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
    filtered_df_en = df[df['language'].str[0] == 'en'].copy()  

    for feature in filtered_df_en.columns:
      filtered_df_en[feature]= filtered_df_en[feature].apply(lambda x:[word for word in x if word not in stopwrords_en])       
    df.update(filtered_df_en)

    # filter df whene it is in germany
    filtered_df_de = df[df['language'].str[0] == 'de'].copy()  

    for feature in filtered_df_de.columns:
       filtered_df_de[feature]= filtered_df_de[feature].apply(lambda x:[ word for word in x if word not in stopwords_de])
       
    df.update(filtered_df_de)

    #fusion tags
    df['tags']= df['tag_1']+df['tag_2']+df['tag_3']+df['tag_4']+df['tag_5']+df['tag_5']+df['tag_6']+df['tag_7']+df['tag_8']
    df.drop(['subject', 'body','tag_1','tag_2','tag_3','tag_4','tag_5', 'tag_6', 'tag_7','tag_8'], axis=1, inplace=True)
    return df

# df= load_data()
# clean_df= cleaning(df)
# clean_df.head()