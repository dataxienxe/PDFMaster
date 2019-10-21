from tika import parser
import glob, os
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
import re
import os
import spacy
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from nltk.tag import pos_tag
from sklearn import feature_extraction
import mpld3

path = r'Users\Rohan.Gupta.USNIIT-TECH\Downloads\LeaseModel\static\upload'


def cluster_run(path):

    df = pd.DataFrame(columns=['File_Name', 'Content'])
    companies = []
    indX = []
    os.chdir(path)
    for file in glob.glob("*.pdf"):

        raw = parser.from_file(file)
        text = raw['content']
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\xa0', ' ')
        text = text.lower()
        companies.append(file)
        df1 = {'File_Name': file,'Content': text[0:30000]}
        df = df.append(df1, ignore_index=True)

    for i in range(0,len(companies)):
        indX.append(i)


    df.to_csv('pdf_Files_Details1.csv', encoding='utf-8', index=False)

    df = pd.read_csv("pdf_Files_Details1.csv")

    text = []
    for i in df['Content']:
        text.append(BeautifulSoup(i, 'html.parser').getText())


    stopwords = nltk.corpus.stopwords.words('english')

    stemmer = SnowballStemmer("english")

    def tokenize_and_stem(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems

    def tokenize_only(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for q in text:
        allwords_stemmed = tokenize_and_stem(q)
        totalvocab_stemmed.extend(allwords_stemmed)

        allwords_tokenized = tokenize_only(q)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

    words = vocab_frame['words']
    words = words.tolist()

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(text)

    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)

    #K-Means clustering

    num_clusters = 4
    km = KMeans(n_clusters=num_clusters)

    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()

    joblib.dump(km,  'doc_cluster.pkl')
    km = joblib.load('doc_cluster.pkl')
    clusters = km.labels_.tolist()

    findata = {'companies': companies, 'index': indX, 'text': text, 'cluster': clusters}

    frame = pd.DataFrame(findata, index = [clusters] , columns = ['index', 'text', 'cluster', 'companies'])

    frame = frame.sort_values(by='index')


    MDS()

    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    #strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text


    def strip_proppers_POS(text):
        tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
        non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
        return non_propernouns

    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {0: 'Cluster 0',
                     1: 'Cluster 1',
                     2: 'Cluster 2',
                     3: 'Cluster 3'}

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    data = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=companies))

    #group by cluster
    groups = data.groupby('label')


    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(data)):
        ax.text(data.iloc[i]['x'], data.iloc[i]['y'], data.iloc[i]['title'], size=8)

    p = os.path.abspath(r'C:\Users\Rohan.Gupta.USNIIT-TECH\Downloads\LeaseModel\templates\plot.html')

    clust = mpld3.save_html(fig, p)
    #show the plot


    #uncomment the below to save the plot if need be
