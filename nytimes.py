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








url = 'https://www.nytimes.com/aponline/2018/02/08/sports/basketball/ap-bkn-trade-deadline.html'


from bs4 import BeautifulSoup
import re


r = requests.get(url)
html_soup = BeautifulSoup(r.text, 'html.parser')
pretty_soup = html_soup.prettify()
    
    import sys
    sys.stdout = open('html.txt','w')
    print(pretty_soup)
    sys.stdout.close()
    
    awards = html_soup.find('div', {'id':'content-2-wide'})
    
    try:
        ## Getting total awards and nominations
        won_nom = awards.find('div',{'class':'header'}).text
        tot_awards2 = [int(j) for j in re.findall(r'\d+',won_nom)]
        tot_awards.append({'id':movie_id,
                           'tot_wins':tot_awards2[0],'tot_noms':tot_awards2[1]})
        
        
        ## Getting only Oscars, BAFTA, Golden Globe and Screen Actors Guild Awards
        p_wins = awards.findAll('h3',limit=5)
        p_wins = p_wins[1:]
        award_title = [j.text.strip() for j in p_wins]
        award_title = [j[:-6] for j in award_title]
        
        p_wins = awards.findAll('table',limit=4)
        dat = []
        for index, j in enumerate(p_wins):
            trs = j.findAll('tr')
            dat1 = []    
            for k in trs:
                dic = {}
                try:
                    dic['type'] = award_title[index]
                    dic['status'] = k.find('b').text
                    dic['title'] = k.find('td',{'class':'award_description'}).text
                    dat1.append(dic)
                except:
                    dic['type'] = award_title[index]
                    dic['status'] = dat1[-1]['status']
                    dic['title'] = k.find('td',{'class':'award_description'}).text
                    dat1.append(dic)
            dat.append(dat1)
        
        award_info2 = [pd.DataFrame(j) for j in dat]
        award_info2 = pd.concat(award_info2, axis = 0)
        award_info2['id'] = movie_id
        award_info = pd.concat([award_info, award_info2], axis = 0)
    except:
        print('Movie has no wins or nominations. Skipping.....')
