import requests
from bs4 import BeautifulSoup
import gensim
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer



urls=[
   ['https://www.yahoo.com/news/','h3','Mb(5px)'],
   ['https://www.pinkvilla.com/','div','ypromoted'],
   ['https://news.google.com','a','DY5T1d'],
   ['https://www.tentaran.com/category/trending-update/','header','entry-header']
]
no_of_titles=0
titles=[]
for url in urls:
    r1 = requests.get(url[0])
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, 'html.parser')
    coverpage_news = soup1.find_all(url[1],class_=url[2])
    for heading in coverpage_news:
        titles.append(heading.get_text())
        no_of_titles += 1

"""Pre processs the text and convert to tokens"""
stemmer = PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

tokanized_titles=[]

for title in titles:
    tokanized_titles.append( preprocess(title))


dictionary = gensim.corpora.Dictionary(tokanized_titles)
corpus = [dictionary.doc2bow(list_of_tokens) for list_of_tokens in tokanized_titles]

num_topics = 20
lda_model = gensim.models.LdaModel(corpus,
                                    num_topics=num_topics, 
                                    id2word=dictionary_LDA,
                                    passes=4, 
                                    alpha=[0.01]*num_topics,
                                    eta=[0.01]*len(dictionary.keys()))


for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")