import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors


def word_cloud_plot (data):
    """
    creates a word cloud from a specified column of a dataframe 
    """
    # create set of stopwords
    stopwords = set(STOPWORDS)

    # Instantiate the word cloud object
    word_cloud = WordCloud(background_color='white',max_words=200,stopwords=stopwords, width=800, height=400)
    
    # generate the word cloud
    word_cloud.generate(' '.join(data))
    
    # To display the word cloud
    plt.figure( figsize=(20,10) )
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
def regex_filter(sentence):
    import re
    return re.sub('[^a-zA-Z]', ' ', sentence)
    
def filter_stop_words(token):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    filtered_token = [word.lower() for word in token if word.lower() not in stop_words]
    return filtered_token

def stem_words(token):
    ps = PorterStemmer()
    stemmed_token = [ps.stem(word) for word in token]
    return stemmed_token

def lemmatize_words(token):
    lem = WordNetLemmatizer()
    lemmatized_token = [lem.lemmatize(word, 'v') for word in token]
    return lemmatized_token


def join_token(token):
    return ' '.join(token)

from datasets import load_dataset
dataset = load_dataset('climate_fever')

df = dataset['test'].to_pandas()
df2 = pd.json_normalize(dataset['test'], 'evidences', ['claim', 'claim_id','claim_label'], errors='ignore',record_prefix='Evidence_')

data1 = df[['claim', 'claim_label']]
data2 = df2[['Evidence_evidence','Evidence_evidence_label', 'claim', 'claim_label']]


sns.histplot(data = data1, x ='claim_label', bins = 20)
sns.histplot(data = data2, x ='claim_label', bins = 20)
sns.histplot(data = data2, x ='Evidence_evidence_label', bins = 20)


# visualizing word frequency in all claims
word_cloud_plot(data1['claim'])


# filter with regex
data1.loc[:, 'claim_token'] = data1.loc[:, 'claim'].apply(regex_filter)

# Tokenizing the claims
nltk.download('punkt')
data1.loc[:, 'claim_token'] = data1.loc[:, 'claim_token'].apply(nltk.word_tokenize)

# Removing stop words from the claclaim_tokenim tokens
data1.loc[:,'claim_token'] = data1.loc[:,'claim_token'].apply(filter_stop_words)

# Stemming the words
data1.loc[:,'stemmed_words'] = data1.loc[:,'claim_token'].apply(stem_words)

# lemmatizing the words
data1.loc[:,'lemmatized_words'] = data1.loc[:,'claim_token'].apply(lemmatize_words)

# Visualizing the word cloud again
word_cloud_plot(data1['claim_token'].apply(join_token))

# getting the length of unique stemmed words
unique_set = [word for token in list(data1['stemmed_words']) for word in token]
unique_set = set(unique_set)
len(unique_set)

# getting the length of unique lemmatized words
unique_set2 = [word for token in list(data1['lemmatized_words']) for word in token]
unique_set2 = set(unique_set2)
len(unique_set2)

# creating the stemmed corpus and lemmatized corpus
corpus_stem = list(data1['stemmed_words'])
corpus_lem = list(data1['lemmatized_words'])

# Embeding with Word2Vec
model_stem_claim = Word2Vec(corpus_stem, min_count=2)
model_lem_claim = Word2Vec(corpus_lem, min_count=2)

# Adding the evidences to increase corpus size

# filer with regex
data2.loc[:, 'evidence_token'] = data2.loc[:, 'Evidence_evidence'].apply(regex_filter)

# Tokenizing the claims
data2.loc[:, 'evidence_token'] = data2.loc[:, 'evidence_token'].apply(nltk.word_tokenize)

# Removing stop words from the evidence_token tokens
data2.loc[:,'evidence_token'] = data2.loc[:,'evidence_token'].apply(filter_stop_words)

# Stemming the words
data2.loc[:,'stemmed_words'] = data2.loc[:,'evidence_token'].apply(stem_words)

# lemmatizing the words
data2.loc[:,'lemmatized_words'] = data2.loc[:,'evidence_token'].apply(lemmatize_words)

# getting the length of unique stemmed words
unique_set3 = [word for token in list(data2['stemmed_words']) for word in token]
unique_set3 = set(unique_set3)
len(unique_set3)

# getting the length of unique lemmatized words
unique_set4 = [word for token in list(data2['lemmatized_words']) for word in token]
unique_set4 = set(unique_set4)
len(unique_set4)

# adding the sentences in the evidence to the corpus
corpus_stem = corpus_stem + list(data2['stemmed_words'])
corpus_lem = corpus_lem + list(data2['lemmatized_words'])

# Embeding with Word2Vec
model_stem_total = Word2Vec(corpus_stem, min_count=2)
model_lem_total = Word2Vec(corpus_lem, min_count=2)

# Loading pretrained Word2Vec model
import gensim.downloader as api
ddd = api.load("text8")
W2V_pretrained = Word2Vec(ddd) 

# Loading pretrained GloVe model
GloVe_pretrained = api.load("glove-wiki-gigaword-50")










