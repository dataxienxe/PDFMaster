from flask import Flask,render_template,url_for,request, redirect, flash, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from io import StringIO
import random
from werkzeug.utils import secure_filename
from gensim.summarization import summarize
from spacy_summarization import text_summarizer
from nltk_summarization import nltk_summarizer
from labelextract import label_extraction, input_taker
from pdftotext import pdftocsv
from topicmodel import topicmodel
from cluster import cluster_run
import time
import spacy
nlp = spacy.load('en_core_web_sm')
import os
import re

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen


app = Flask(__name__)

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext) ])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page, "lxml")
	fetched_text = ' '.join(map(lambda p:p.text, soup.find_all('p')))
	return fetched_text



@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		filetext = request.form['filename']
		rawtext = pdftocsv(filetext)
		final_reading_time = readingTime(rawtext)
		final_summary = nltk_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end - start
	return render_template('index.html', filetext=filetext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = nltk_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end - start
	return render_template('index.html',final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/topic_model')
def topic_model():
    return render_template('topic_model.html')


@app.route('/LDA', methods=['GET','POST'])
def LDA():
    if request.method == 'POST':
        filename2 = request.form['filename2']
        LDAchart = topicmodel(filename2)
        return render_template('lda.html')


    return render_template('topic_model.html', )

@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_nltk = NLTk_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_nltk)
		# Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time_spacy = readingTime(final_summary_spacy)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy)

		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)

@app.route('/cluster')
def cluster():
    return render_template('cluster.html')

@app.route('/plot', methods=['GET', 'POST'])
def plot_png():
    if request.method == 'POST':
        path = os.path.abspath(r'C:\Users\Rohan.Gupta.USNIIT-TECH\Downloads\LeaseModel\docs')
        fig = cluster_run(path)
        return render_template('plot.html')


@app.route('/labels')
def labels():
    return render_template('labels.html')

@app.route('/label_extract', methods=['GET', 'POST'])
def label_extract():
    if request.method == 'POST':
        filename3 = request.form['filename3']
        label_extraction(filename3)
        keyval = request.form['keyval']
        val = input_taker(filename3, keyval)
    return render_template('labels.html', keyval=keyval, val=val)


if __name__ == '__main__':
	app.run(debug=True)
