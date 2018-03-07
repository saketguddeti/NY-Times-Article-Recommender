import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import os
os.chdir('/Users/saketguddeti/Desktop/git/NY-Times-Article-Recommender')


from bs4 import BeautifulSoup

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


content_data = pd.read_csv('data/content_data.csv')

# In[21]:
import ast

content_data['content_clean'] = content_data['content_clean'].apply(lambda x: ast.literal_eval(x))

vect = CountVectorizer(min_df=10, max_df=0.6, stop_words='english', token_pattern='(?u)\\b\\w\\w\\w+\\b')

X = vect.fit_transform(content_data['content_clean'].apply(lambda x: (" ").join(x)))

corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

id_map = dict((v, k) for k, v in vect.vocabulary_.items())

dct = gensim.corpora.Dictionary.from_corpus(corpus, id_map)

# In[22]:
num_topics = 30
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word = id_map, passes = 40)

ldamodel = gensim.models.ldamodel.LdaModel.load("data/lda.model")

topic_corpus = pd.read_csv('data/topic_corpus.csv')

def query_article_topic(list):
    query_content = []
    for at in list:
        try:
            query_content.append(pre_process(extract_content(at)))
        except:
            query_content.append([''])
        
    bow = [dct.doc2bow(i) for i in query_content]
    query_corpus = []
    for i in range(len(bow)):
        topic_dist = ldamodel[bow[i]]
        topic_dist = {x[0]:x[1] for x in topic_dist}
        query_corpus.append(topic_dist)
    query_corpus.append({i:0 for i in range(num_topics)})    
    query_corpus = pd.DataFrame(query_corpus).fillna(0)
    return(query_corpus.iloc[:-1,])

# In[24]:
from gensim.models import Doc2Vec

def pre_process_doc2vec(text):    
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation).union(set(['“','”','—','’','‘']))
    punctuation.remove('-')
    cl_text = (" ").join([s for s in text.lower().split() if s not in stopwords])
    cl_text = ("").join([s for s in cl_text if s not in punctuation])
    cl_text = nltk.word_tokenize(cl_text)
    return cl_text

model = Doc2Vec.load("data/doc2vec.model")

# In[29]:
from sklearn.metrics.pairwise import cosine_similarity
from bokeh.io import curdoc, show, output_notebook
from bokeh.layouts import column, row
from bokeh.models.widgets import TextInput, Button, Paragraph, Div, RadioButtonGroup
from bokeh.models import CustomJS
from functools import partial
from datetime import datetime
import webbrowser
import requests
import os

def default_recs():
    base_url = 'https://api.nytimes.com/svc/mostpopular/v2/mostviewed/all-sections/1.json?api-key=c77ddf1d1b594f76b2773928f324615f'
    url = base_url

    r = requests.get(url)
    json_data = r.json()
    article_meta_data = json_data['results']

    headlines = []
    urls = []
    snippets = []
    for artc in article_meta_data[:10]:
        url = artc['url']
        headline = artc['title']
        snippet = artc['abstract']
        headlines.append(headline)
        urls.append(url)
        snippets.append(snippet)

    article_topics = query_article_topic(urls)
    article_topics.to_csv('data/article_topics.csv', index = False)

    return(headlines, urls, snippets)

headlines_ls, urls_ls, snippets_ls = default_recs()

    
def reset_pref():
    usr = text_user.value
    df = pd.read_csv('data/df.csv')
    df = df.loc[df['name'] != usr,]
    df.to_csv('data/df.csv', index = False)
    try:
        os.remove('data/user_pref/%s.csv' %usr)
    except:
        None

def rec_generate():
    if(text_user.value == ''):
        return

    bar.text = "<p style='color:red;font-size:120%;text-align:center'><b>Loading...</b></p>"
    
    pref_data = pd.read_csv('data/df.csv')
    usr = text_user.value
    user_list = list(pref_data['name'].values)

    def get_headline(url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        name_box = soup.findAll('h1')
        return(name_box[0].text)

    def get_snippet(url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        name_box = soup.findAll('p')
        for i in name_box:
            try:
                if(len(i.text)>150):
                    snipp = i.text
                    return(snipp)
            except:
                return("No Snippet Available")

    if(usr in user_list):
        usr_data = pref_data.loc[pref_data['name'] == usr,]
        usr_arr = usr_data.iloc[0].values[1:]
#             usr_arr = usr_data.iloc[0].values[1:]/sum(usr_arr)

        from sklearn.metrics.pairwise import cosine_similarity
        def dot_product(arr1, arr2):
            return(cosine_similarity(arr1.reshape(1,-1),arr2.reshape(1,-1))[0][0])
        score = []
        for i in range(topic_corpus.shape[0]):
            score.append(dot_product(topic_corpus.iloc[i,].values, usr_arr))

        id_score = pd.DataFrame(list(zip(list(content_data['_id']), score)), columns = ['doc_id','lda_sim'])
        doc_data = pd.read_csv('data/user_pref/%s.csv' % usr)
        similarity_df = pd.merge(left = doc_data[['doc_id','cos_sim']], right = id_score, how = 'left', on = 'doc_id')
        similarity_df['age'] = content_data['pub_date'].apply(lambda x: 
                                    datetime.today()-datetime.strptime(x[:19], '%Y-%m-%dT%H:%M:%S')).apply(lambda y:
                                    int(y.total_seconds()/3600))
        similarity_df['age_string'] = similarity_df['age'].apply(lambda x: 
                                                          str(int(x/24))+' day(s)' if x>24 else str(x)+' hour(s)')
        similarity_df['rank'] = 0.3*similarity_df['cos_sim']+0.7*similarity_df['lda_sim']
        churn_rank = list((1000*similarity_df['rank'])/(np.sqrt(1.00001**similarity_df['age'])))
        sorted_index = sorted(range(len(churn_rank)), key=lambda k: churn_rank[k], reverse = True)[:10]

        url = [content_data['web_url'][i] for i in sorted_index]
        time_stamp = [similarity_df['age_string'][i] for i in sorted_index]
        hdls = [get_headline(i) for i in url]
        snipps = [get_snippet(i) for i in url]
        snipps = ['['+time_stamp[i]+' old] '+snipps[i] for i in range(len(snipps))]
        ui_fill(hdls, snipps, url)
    else:
        base_url = 'https://api.nytimes.com/svc/mostpopular/v2/mostviewed/all-sections/1.json?api-key=c77ddf1d1b594f76b2773928f324615f'
        url = base_url

        r = requests.get(url)
        json_data = r.json()
        article_meta_data = json_data['results']

        headlines = []
        urls = []
        snippets = []
        for artc in article_meta_data[:10]:
            url = artc['url']
            headline = artc['title']
            snippet = artc['abstract']
            headlines.append(headline)
            urls.append(url)
            snippets.append(snippet)

        ui_fill(headlines, snippets, urls)


def on_value_change(attr, old, new, foo):
    if(text_user.value == '' or new == -1):
        return
    
    # Building user preference based on liked article content
    scale = {0:-1,-1:0,1:1}
    pref_data = pd.read_csv('data/df.csv')
    user_list = list(pref_data['name'].values)
    usr = text_user.value

    if(usr not in user_list):
        # Adding user profile to the topic Similarity Data
        df = pd.DataFrame(columns = ['name'] + [str(i) for i in range(num_topics)])
        df.loc[0] = [usr] + [0 for i in range(num_topics)]
        pref_data = pd.concat([pref_data, df], axis = 0)
        pref_data = pref_data.fillna(0)
        pref_data.to_csv('data/df.csv', index = False)

        # Adding user profile to the semantic similarity Data
        df_sem = pd.DataFrame(columns = ['doc_id','cos_sim','count'])
        df_sem.to_csv('data/user_pref/%s.csv' %usr, index = False)

    article_topics = pd.read_csv('data/article_topics.csv')
    pref_data = pd.read_csv('data/df.csv')
    pref_change = scale[new] - scale[old]
    for i in range(num_topics):
        pref_data.loc[pref_data['name'] == usr, str(i)] += pref_change*article_topics.iloc[foo, i]
    pref_data.to_csv('data/df.csv', index = False)

    # Building document similarity data utilizing semantics
    docvec = model.infer_vector(pre_process_doc2vec(extract_content(urls_ls[foo])))
    doc_sim = []
    for i in range(len(model.docvecs)):
        x = cosine_similarity(model.docvecs[i].reshape(1,-1), docvec.reshape(1,-1))
        doc_sim.append(x[0][0])
    doc_data = pd.DataFrame(np.column_stack([list(content_data['_id']), doc_sim, 
                                             [1 for i in range(len(model.docvecs))]]), 
                            columns=['doc_id', 'cos_sim', 'count'])

    doc_data_tmp = pd.read_csv('data/user_pref/%s.csv' %usr)
    doc_data = pd.merge(left = doc_data_tmp, right = doc_data, on = 'doc_id', how = 'outer')
    doc_data = doc_data.drop_duplicates('doc_id')
    doc_data = doc_data.loc[pd.notnull(doc_data['doc_id']),]
    doc_data = doc_data.loc[pd.notnull(doc_data['cos_sim_y']),]

    doc_data['cos_sim'] = np.where(pd.isnull(doc_data['cos_sim_x']) & pd.isnull(doc_data['count_x']), 
                                   doc_data['cos_sim_y'], doc_data['cos_sim_x'])
    doc_data['count'] = np.where(pd.isnull(doc_data['cos_sim_x']) & pd.isnull(doc_data['count_x']), 
                                   doc_data['count_y'], doc_data['count_x'])

    doc_data['cos_sim'] = np.where(pd.notnull(doc_data['cos_sim_x']) & pd.notnull(doc_data['count_x']), 
                                   (doc_data['cos_sim_x']*doc_data['count_x']+
                                    doc_data.fillna(0)['cos_sim_y'].astype(float)*doc_data.fillna(0)['count_y'].astype(float))/
                                   (doc_data.fillna(0)['count_x'].astype(float) + doc_data.fillna(0)['count_y'].astype(float)),
                                   doc_data['cos_sim'])
    doc_data['count'] = np.where(pd.notnull(doc_data['cos_sim_x']) & pd.notnull(doc_data['count_x']), 
                                   doc_data['count_x'].astype(float)+doc_data.fillna(0)['count_y'].astype(float), doc_data['count'])

    doc_data = doc_data.drop(['cos_sim_x','cos_sim_y','count_x','count_y'], axis = 1)
    doc_data.to_csv('data/user_pref/%s.csv' %usr, index = False)


def ui_fill(headlines, snippets, urls):

    article_topics = query_article_topic(urls)
    article_topics.to_csv('data/article_topics.csv', index = False)
    
    bar.text = "<p></p>"

    for i in range(len(headlines)):
        items_hl[i].label = headlines[i]
        items_sn[i].text = snippets[i]
        items_rate[i].active = -1
        
    global urls_ls
    urls_ls = urls


def onclick():

    if(text_query.value == ''):
        return
    
    bar.text = "<p style='color:red;font-size:120%;text-align:center'><b>Loading...</b></p>"
    
    base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?api-key=c77ddf1d1b594f76b2773928f324615f'
    param_url = '&q='+str(text_query.value)+'&page=0'
    url = base_url + param_url

    r = requests.get(url)
    json_data = r.json()
    article_meta_data = json_data['response']['docs']

    headlines = []
    urls = []
    snippets = []
    for artc in article_meta_data:
        url = artc['web_url']
        headline = artc['headline']['main']
        snippet = artc['snippet']
        headlines.append(headline)
        urls.append(url)
        snippets.append(snippet)

    ui_fill(headlines, snippets, urls)



output = Paragraph()
text_query = TextInput(value = "", width=220, height=10, title = "Enter article query")

get_alert2 = CustomJS(args=dict(text_query=text_query), code = """

var str = text_query.value;
if (str == ""){
    alert("Input cannot be blank")
}
"""
                    )

button_query = Button(label = 'Search articles', width=220, 
                      height=10, button_type='success', callback = get_alert2)

empty_div = Div(text="", width=70, height=1)
query_grid = row(text_query, column(empty_div, button_query), name = 'query_grid')

text_user = TextInput(placeholder='Enter your name', value = 'saket', width = 340)
username = Div(text="Username:", width=70, height=10)

get_alert = CustomJS(args=dict(text_user=text_user), code = """

var usr = text_user.value;
var val = cb_obj.active
if (usr == ""){
    if (val == 1) 
        {cb_obj.active = -1;} 
    else 
        {cb_obj.active = -1;}
    alert("Enter your name before giving preferences")
}"""
                    )

rec_button = Button(label = 'Get recommendations', width=110, 
                    height=5, button_type='warning', callback = get_alert)

reset_button = Button(label = 'Reset Preferences', width = 150, 
                      height = 5, button_type = 'warning', callback = get_alert)

user_grid1 = row(column(empty_div, username), text_user, name = 'user_grid1')
user_grid2 = row(row(Div(text="", width = 70), rec_button), 
                 row(Div(text="", width = 5), reset_button),
                 name = 'user_grid2')
user_grid = column(user_grid1, user_grid2, name = 'user_grid')
input_grid = row(query_grid, Div(text="", width = 60), user_grid, name = 'input_grid')
listing_grid = empty_div
grid = column(input_grid,
              Div(text=
                  "<p style='color:#808080'>Rate the article 'Less' or 'More' to personalize recommendations</p>",
                    height = 20), 
              listing_grid,
              listing_grid,
              name = 'grid')

items_hl = [Button(label=w, width = 800, height = 40, button_type = 'primary') 
                   for w in headlines_ls]

items_sn = [Paragraph(text=w, width = 800, height = 61, style = {'background-color':'#F2F3F4'}) 
                   for w in snippets_ls]

items_rate = [RadioButtonGroup(labels=['Less', 'More'], active = -1, callback = get_alert) 
                     for i in range(len(headlines_ls))]

items = column([column(row(items_hl[i], items_rate[i]),
                        items_sn[i]) for i in range(len(headlines_ls))])
    
def redirect_link(foo):
    webbrowser.open_new_tab(urls_ls[foo])

for i, hl in enumerate(items_hl):
    hl.on_click(partial(redirect_link, foo = i))

for i, hl in enumerate(items_rate):
    hl.on_change('active', partial(on_value_change, foo = i))
    
bar = Div(text = "<p></p>", height = 20, width = 700)
load_grid = row(Div(text="", width=100, height=1), bar, Div(text="", width=100, height=1))

grid.children[2] = load_grid
grid.children[3] = items

curdoc().add_root(grid)

button_query.on_click(onclick)
rec_button.on_click(rec_generate)
reset_button.on_click(reset_pref)    
