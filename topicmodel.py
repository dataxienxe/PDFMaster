#Importing PDFs and converting into a CSV output.
from tika import parser
import glob, os
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import string
from nltk.stem.snowball import SnowballStemmer



def topicmodel(file):
    df = pd.DataFrame(columns = ['File_Name','Content'])

    raw = parser.from_file(file)
    text = raw['content']
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\xa0', ' ')
    text = text.lower()
    df1 = {'File_Name': file, 'Content': text[0:30000]}
    df.append(df1, ignore_index=True)

    for i in df['Content']:
        text = [BeautifulSoup(text, 'html.parser').getText()]

    companies = []
    for x in df['File_Name']:
        x,y = x.split('_')
        companies.append(x)
    indX = []
    for i in range(0,len(df)):
        indX.append(i)


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

    allwords_stemmed = tokenize_and_stem(text)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(text)
    totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

    words = vocab_frame['words']
    words = words.tolist()


    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # Remove Stop Words
    data_words_nostops = remove_stopwords(words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]


    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
    vis = pyLDAvis.save_html(vis, 'templates/lda.html')
