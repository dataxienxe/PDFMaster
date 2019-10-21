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

def label_extraction(filename):
    df = pd.DataFrame(columns = ['File_Name','Content'])

    raw = parser.from_file(filename)
    text = raw['content']
    text = text.replace('\r', ' ')
    text = text.replace('\xa0', ' ')
    df1 = {'File_Name': filename, 'Content': text[0:30000]}
    df.append(df1, ignore_index=True)

    for i in df['Content']:
        text = [BeautifulSoup(text, 'html.parser').getText()]

    text = re.sub('[^A-Za-z0-9.$ ]+', '', text)
    text = re.sub(r"\s+", " ", text)
    regex_num = re.compile('Unit .{19}')
    units = regex_num.findall(text)
    size =[]
    for d in units:
        size.append(d[7:])
    alph = ['Unit A', 'Unit B','Unit F', 'Unit C', 'Unit D',]

    dictionary = dict(zip(alph, size))

    secdep = re.compile('Security Deposit of.{11}' )
    sec_d = secdep.findall(text)
    for a in sec_d:
        sec_a = a[20:]
    rec = re.compile('RECITALS .{776}')
    Recitals = rec.findall(text)
    for r in Recitals:
        Recitals = r[12:]

    rent = re.compile('RENT .{5655}')
    Rent = rent.findall(text)
    Rent = re.split(r'( a\.| b\.| c\.| d\.| e\.| f\.+ )', Rent[0])

    clause_a = Rent[2]
    clause_b = Rent[4]
    clause_c = Rent[6]
    clause_d = Rent[8]
    clause_e = Rent[10]
    clause_f = Rent[12]


    dictionary['Security Deposit'] = sec_a
    dictionary['Recitals'] = Recitals
    dictionary['Rent Clause A'] = clause_a
    dictionary['Rent Clause B'] = clause_b
    dictionary['Rent Clause C'] = clause_c
    dictionary['Rent Clause D'] = clause_d
    dictionary['Rent Clause E'] = clause_e
    dictionary['Rent Clause F'] = clause_f

    return dictionary

def input_taker(filename, keyval):
    dictionary = label_extraction(filename)
    for keys,values in dictionary.items():
        if keyval == keys:
            return values
