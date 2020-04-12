"""

    Program to extract headlines from news websites and perform topic modeling.

"""

import requests
from bs4 import BeautifulSoup
import gensim
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import itertools

# Step 1 : Extract the headlines from news websites
# urls= [[url, attribute , class],......]
urls=[
    ['https://www.scoopwhoop.com/','a','article-title'],
   ['https://www.yahoo.com/news/','h3','Mb(5px)'],
   ['https://www.pinkvilla.com/','div','ypromoted'],
   ['https://www.buzzfeednews.com/','h2','newsblock-story-card__title']
  
]
"""
Some extra urls if required:
['https://www.tentaran.com/category/trending-update/','header','entry-header'],
['https://news.google.com','a','DY5T1d'],
 ['https://www.rediff.com/news','h2','hdtitle']
"""

no_of_titles=0
titles=[]
title_url=[]
print("\n\n\n   Extracting headlines of ",len(urls) ," news websites")
for url in urls:
    r1 = requests.get(url[0])
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, 'html.parser')
    coverpage_news = soup1.find_all(url[1],class_=url[2])
    no_headlines_per_website=0
    for heading in coverpage_news:
        headline=heading.get_text()
        headline=headline.strip()
        titles.append(headline)
        title_url.append([url[0],headline])
        no_of_titles += 1
        no_headlines_per_website+=1
    print("\n\t\tWebsite : " , url[0], '\n\t\tNo. of headlines collected : ', no_headlines_per_website)

print("\n   Total No. of headlines : ",no_of_titles)

# Step 2 : Pre processs the text and convert to tokens
stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))       
    return result

tokanized_titles=[preprocess(title) for title in titles]

# Step 3 : Build the Bow and LDA models

dictionary = gensim.corpora.Dictionary(tokanized_titles)
corpus = [dictionary.doc2bow(list_of_tokens) for list_of_tokens in tokanized_titles]
num_topics = 15
lda_model = gensim.models.LdaModel(corpus,
                                    num_topics=num_topics, 
                                    id2word=dictionary,
                                    passes=4, 
                                    alpha=[0.01]*num_topics,
                                    eta=[0.01]*len(dictionary.keys()))


# Results

print('\n\n   Results :')
topic_words=[]
for topic,words in lda_model.show_topics(num_topics=num_topics, num_words=4, log=False, formatted=False):
    word_list=[]
    for word,prob in words:
        word_list.append(word)
    topic_words.append(word_list)

index_list=[]
for topic in topic_words:
    for i in range(len(tokanized_titles)):
        if(set(topic).issubset(set(tokanized_titles[i]))):
            index_list.append(i)

trending=[]
for i in set(index_list):
    trending.append(title_url[i])
trends=list(k for k,_ in itertools.groupby(trending))

for aritcle in trends:
    print('\n\t Headline : ',aritcle[1],'\n\t Website : ',aritcle[0])