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

