import pandas as pd
import numpy as np
import os

os.chdir('/Users/saketguddeti/Desktop/git/NY-Times-Article-Recommender')

news_desk = pd.read_csv("data/news_desk.csv")
news_desk.head()



import time, threading

def rate_limited(max_per_second):
  '''Decorator that make functions not to be called faster than 1 call/second'''
  lock = threading.Lock()
  minInterval = 1.0 / float(max_per_second)
  def decorate(func):
    lastTimeCalled = [0.0]
    def rateLimitedFunction(args,*kargs):
      lock.acquire()
      elapsed = time.clock() - lastTimeCalled[0]
      leftToWait = minInterval - elapsed
      if leftToWait>0:
        time.sleep(leftToWait)
      lock.release()
      ret = func(args,*kargs)
      lastTimeCalled[0] = time.clock()
      return ret
    return rateLimitedFunction
  return decorate


from threading import Thread
import requests

@rate_limited(0.9)
def process_id(id):
    try:
        r = requests.get(url % id)
        json_data = r.json()
        print('Appended '+str(page_index.index(id))+ ' out of '+ str(len(page_index)))
        return json_data
    except:
        json_data = ''
        print('Skipping...')
        return json_data

def process_range(id_range, store=None):
    if store is None:
        store = {}
    for id in id_range:
        store[id] = process_id(id)
    return store


def threaded_process_range(nthreads, id_range):
    store = {}
    threads = []

    for i in range(nthreads):
        ids = id_range[i::nthreads]
        t = Thread(target=process_range, args=(ids,store))
        threads.append(t)

    [ t.start() for t in threads ]
    [ t.join() for t in threads ]
    return store

news_desk = list(news_desk['Section'])
base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?api-key=a141a689509b459ba12e2b93b83883fd'

article_raw = []
for nd in news_desk:
    param_url = '&fq=news_desk:'+str(nd)+'&sort=newest&page=%s'
    url = base_url + param_url
    print(str(nd)+":")
    page_index = list(range(7))
    try:
        articles_1 = threaded_process_range(2, page_index)
        articles_2 = [articles_1[k]['response']['docs'] for k in page_index if (type(articles_1[k]) is dict) and ('response' in articles_1[k])]
        articles_3 = [item for sublist in articles_2 for item in sublist]
        articles_4 = [{key:item[key] for key in ['web_url','pub_date']} for item in articles_3]
        articles_5 = pd.DataFrame(articles_4)
        articles_5['news_desk'] = str(nd)
        article_raw.append(articles_5)
    except:
        print('Skipping...')

url_data = pd.concat(article_raw).reset_index(drop = True)
#url_data.to_csv('data/url_data.csv', index = False)



# Scraping the articles

from bs4 import BeautifulSoup

@rate_limited(1)
def extract_content(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    name_box = soup.findAll('p', attrs={'class': 'story-body-text story-content'})
    content = [x.text for x in name_box]
    content_final = ' '.join(content)
    
    if(content_final == ''):
        name_box = soup.findAll('p')
        content = [x.text for x in name_box]
        content_final = ' '.join(content)

    return(content_final)

url_data = pd.read_csv('data/url_data.csv')
url_data['video_flag'] = url_data['web_url'].str.contains('/video/')
url_data['slideshow_flag'] = url_data['web_url'].str.contains('/slideshow/')
url_data = url_data.loc[(url_data['video_flag'] == False) & (url_data['slideshow_flag'] == False),]
url_data = url_data.drop(['video_flag','slideshow_flag'], axis = 1)

content = []
for index, i in enumerate(url_data['web_url']):
    try:
        print(index)
        a = extract_content(i)    
        content.append(a)
    except:
        content.append('')
        print('Skipping...')
        
content_data = url_data
content_data['content'] = content
content_data.to_csv('data/content_data.csv', index = False)

content_data['length'] = content_data['content'].str.strip().str.len()

import bokeh.plotting as bp
from bokeh.io import show
from bokeh.models import HoverTool

array = content_data['length'][content_data['length'].values < 100000].values
hist, edges = np.histogram(array, bins=50)

source = bp.ColumnDataSource(data = dict(data_value = hist))

p = bp.figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", source = source)
p.add_tools(HoverTool(tooltips= [("Value:", "@data_value")]))
p.xaxis.axis_label = 'Length of Article'
p.yaxis.axis_label = 'Frequency'
show(p)


content_data = content_data.loc[content_data['length'] > 60,]
content_data = content_data.loc[content_data['length'] < 160000,]




import gensim
from sklearn.feature_extraction.text import CountVectorizer


vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', token_pattern='(?u)\\b\\w\\w\\w+\\b')
X = vect.fit_transform(content_data['content'])
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())


from sklearn.metrics.pairwise import cosine_similarity
def topic_sim(lda_model, num_topic):
    topic_word = lda_model.get_topics()
    avg_sim = []
    for i in range(num_topic):
        arr1 = topic_word[i]
        sim = []
        for j in range(num_topic):
            arr2 = topic_word[j]
            sim_value = KL(list(arr1), list(arr2))
#            sim_value = cosine_similarity(arr1.reshape(1,-1),arr2.reshape(1,-1))
            sim.append(sim_value)
        avg_sim.append(np.mean(sim))
    return(np.mean(avg_sim))        


x = np.linspace(1, 100, 10).astype(int)
topic_overlap = []
for i in x:
    print(i)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = i, 
                                           id2word = id_map, passes = 20, random_state = 34)
    topic_overlap.append(topic_sim(ldamodel, i))


p = bp.figure(plot_width=400, plot_height=400)
p.line(list(x), topic_overlap, line_width=2)
p.circle(list(x), topic_overlap, fill_color="white", size=8)
show(p)

# remove ___ from the corpus
ldamodel.print_topics()
x = ldamodel.get_topics()

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))






