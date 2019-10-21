#Importing PDFs and converting into a CSV output.
from tika import parser
import glob, os
import pandas as pd
from bs4 import BeautifulSoup

def pdftocsv(filename):


    raw = parser.from_file(filename)
    text = raw['content']
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\xa0', ' ')

    text = BeautifulSoup(text, 'html.parser').getText()

    return text
