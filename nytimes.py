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
content_data = content_data.reset_index(drop = True)




import gensim
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import string

def pre_process(text):    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'[0-9a-zA-Z]+')

    def convert_tag(tag):
        """
        Convert the tag given by nltk.pos_tag to the tag used by wordnet
        """
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return 'a'

    cl_text = (" ").join(tokenizer.tokenize(text))
    cl_text = (" ").join([s for s in cl_text.lower().split() if s not in stopwords])
    cl_text = ("").join([s for s in cl_text if s not in punctuation])
    cl_text = nltk.word_tokenize(cl_text)
    pos = nltk.pos_tag(cl_text)
    pos = [convert_tag(t[1]) for t in pos]
    cl_text = [lemmatizer.lemmatize(cl_text[i], pos[i]) for i in range(len(cl_text))]
    return cl_text

content_data['new_content'] = content_data['content'].apply(pre_process)

vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', token_pattern='(?u)\\b\\w\\w\\w+\\b')
X = vect.fit_transform(content_data['new_content'].apply(lambda x: (" ").join(x)))
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
dct = gensim.corpora.Dictionary.from_corpus(corpus, id_map)

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 35, id2word = id_map, passes = 40)
def topic_corpus():
    bow_corpus = [dct.doc2bow(content_data.loc[i,'new_content']) for i in range(content_data.shape[0])]
    topic_corpus = []
    for i in range(len(bow_corpus)):
        topic_dist = ldamodel[bow_corpus[i]]
        topic_dist = {x[0]:x[1] for x in topic_dist}
        topic_corpus.append(topic_dist)
    topic_corpus = pd.DataFrame(topic_corpus).fillna(0)
    return(topic_corpus)

topic_corpus = topic_corpus()



    
    
    
    
test = ["https://www.nytimes.com/2012/05/25/business/energy-environment/summer-gas-prices-expected-to-be-modestly-lower.html",
        "https://www.nytimes.com/2018/01/19/sports/soccer/alexis-sanchez-manchester-united-city.html?rref=collection%2Ftimestopic%2FArsenal%20Football%20Club&action=click&contentCollection=soccer&region=stream&module=stream_unit&version=latest&contentPlacement=2&pgtype=collection",
        "https://www.nytimes.com/2018/02/02/opinion/border-wall.html",
        "https://www.nytimes.com/2005/01/03/business/03invest.html",
        "https://www.nytimes.com/2018/01/18/science/earthquakes-moon-cycles.html"]    

df = query_article_topic(test)

from sklearn.metrics.pairwise import cosine_similarity

def dot_product(arr1, arr2):
    return(cosine_similarity(arr1.reshape(1,-1),arr2.reshape(1,-1))[0][0])

score = []
for i in range(4937):
    score.append(dot_product(topic_corpus.iloc[i,].values, df.iloc[1,].values))
    

score = sorted(range(len(score)), key=lambda k: score[k], reverse = True)

content_data['web_url'][1667]


from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

doc1 = extract_content("https://www.nytimes.com/2018/02/23/sports/olympics/cross-country-skiing-food.html?rref=collection%2Fsectioncollection%2Fsports&action=click&contentCollection=sports&region=rank&module=package&version=highlights&contentPlacement=1&pgtype=sectionfront")
doc2 = extract_content("https://www.nytimes.com/2017/08/28/sports/soccer/arsenal-liverpool.html?rref=collection%2Ftimestopic%2FArsenal%20Football%20Club&action=click&contentCollection=soccer&region=stream&module=stream_unit&version=latest&contentPlacement=5&pgtype=collection")

doc2 = "The timing of the statement, signed by 37 MPs and a dozen peers, including Helena Kennedy and Doreen Lawrence, will infuriate Corbyn as he prepares to announce a significant shift in policy tomorrow that is expected to commit Labour to backing permanent membership of a customs union with the EU. His speech will open a far clearer divide between Labour and the Tories over Brexit and raises the possibility that the party’s MPs could join forces with pro-EU Tories, the SNP and the Liberal Democrats to inflict damaging defeats on Theresa May in a series of Commons votes in the spring. But signatories to the statement, co-ordinated by the Labour Campaign for the Single Market, up the ante by saying that while the customs union shift is a step forward, it falls way short of where the party should be on Brexit. They argue that if Corbyn were to back single market membership too, he could deliver a parliamentary vote in favour of staying in, thereby safeguarding the country’s economic future and allowing his plans for boosting spending on public services to go ahead."

def pre_process_doc2vec(text):    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation).union(set(['“','”','—','’','‘']))
    punctuation.remove('-')
    cl_text = (" ").join([s for s in text.lower().split() if s not in stopwords])
    cl_text = ("").join([s for s in cl_text if s not in punctuation])
    cl_text = nltk.word_tokenize(cl_text)
    return cl_text

class LabeledLineSentence(object):
    def __init__(self, doc_list):
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words = pre_process_doc2vec(doc), tags = [idx])
            

it = LabeledLineSentence(list(content_data['content']))


model = Doc2Vec(size=50, min_count=5, alpha=0.025, min_alpha=0.025, workers=8)
model.build_vocab(it)

for epoch in range(10):
    model.train(it, total_examples = model.corpus_count, epochs = model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    
model.save("data/doc2vec.model")
model = Doc2Vec.load("data/doc2vec.model")

docvec = model.docvecs[106]
similar_doc = model.docvecs.most_similar(4412) 
docvec = model.infer_vector(pre_process_doc2vec(doc2))
model.docvecs.most_similar([docvec])

for i in model.docvecs.most_similar([docvec]):
    print(content_data['web_url'][i[0]])



pca_data = pd.DataFrame(columns = list(range(50)))
for i in range(len(model.docvecs)):
    x = pd.DataFrame(model.docvecs[i].reshape(1,-1), columns = list(range(50)))
    pca_data = pd.concat([pca_data, x], axis = 0)
pca_data = pca_data.reset_index(drop = True)    


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_fln = pca.fit_transform(pca_data)
pca_fln = pd.DataFrame(pca_fln)
pca_fln['news_desk'] = content_data['news_desk']
pca_fln.columns = ['col1','col2','col3']

pca_fln = pca_fln.loc[pca_fln['col3'].isin(['Politics', 'Business', 'Sports','Culture','World','Health','Technology','Retail','Wealth','Travel']), ]

source = bp.ColumnDataSource.from_df(pca_fln)

from bokeh.palettes import d3
import bokeh
palette = d3['Category10'][len(pca_fln['col3'].unique())]
color_map = bokeh.models.CategoricalColorMapper(factors=list(pca_fln['col3'].unique()), palette=palette)

p = bp.figure(title = "Visualization of Article Semantics using PCA")
p.scatter('col1','col2',source = source, color = {'field': 'col3', 'transform': color_map},
          legend = {'field':'col3'}, alpha = 0.4, size = 10)
p.border_fill_color = "whitesmoke"
p.add_tools(HoverTool(tooltips= [("Article:","@col3")]))
p.xaxis.axis_label = 'Dimension 1'
p.yaxis.axis_label = 'Dimension 2'
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
show(p)
