from flask import Flask
import math
import nltk
import os
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import num2words
import contractions
from textblob import Word
import string
from dateutil.parser import parse
import csv
from anyascii import anyascii
from breame.spelling import get_american_spelling, get_british_spelling
import numpy as np
from argparse import ArgumentParser
from dateutil import parser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def remove_dates(text):
    words = text.split()
    filtered_words = [word for word in words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def replace_all(text, mydict):
    for gb, us in mydict.items():
        text = text.replace(us, gb)
    return text

def remove_punctuation(text):
    punctuation_list = string.punctuation
    translator = str.maketrans('', '', punctuation_list)
    return text.translate(translator)

def remove_extra_spaces(text):
    return ' '.join(text.split())

def normalize_text(text):
    normalized_text = text.lower()
    normalized_text = ''.join(char for char in normalized_text if char.isalnum() or char.isspace())
    normalized_text = ' '.join(normalized_text.split())
    return normalized_text

def create_acronym_dic(acronym_file_path):
    acronym = {}
    with open(acronym_file_path + ".csv", encoding="utf8") as file:
        tsv_file = csv.reader(file)
        for line in tsv_file:
            acronym[line[0].lower()] = line[1]
    return acronym

def replace_acronyms(text, acronym_dict):
    words = text.split()
    replaced_words = [acronym_dict.get(word.lower(), word) for word in words]
    return ' '.join(replaced_words)

def penn2morphy(penntag, returnNone=True):
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ, 'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    lemmatized_words = []
    for word, tag in pos_tagged:
        pos = penn2morphy(tag)
        if pos is None:
            lemmatized_word = lemmatizer.lemmatize(word)
        else:
            lemmatized_word = lemmatizer.lemmatize(word, pos)
        lemmatized_words.append(lemmatized_word)
    return ' '.join(lemmatized_words)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def process_text(text, acronym_dict):
    text = contractions.fix(text)
    text = replace_all(text, {"us": "gb", "gb": "us"})
    text = replace_acronyms(text, acronym_dict)
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)
    text = normalize_text(text)
    text = remove_dates(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def read_and_process_file(input_file_path, output_file_path, acronym_file_path):
    acronym_dict = create_acronym_dic(acronym_file_path)
    df = pd.read_csv(input_file_path, sep='\t', header=None)
    df.columns = ['Document Number', 'Content']


    df['Content'] = df['Content'].fillna('')

    # df = df.head(1000)
    with open(output_file_path, 'w', encoding='utf8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for _, row in df.iterrows():
            processed_data = process_text(row['Content'], acronym_dict)
            writer.writerow([row['Document Number'], processed_data])
            print(f"{row['Document Number']}: {processed_data}")

input_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/collection.tsv"
output_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/collection1.tsv"
acronym_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/acronyms"


read_and_process_file(input_file_path, output_file_path, acronym_file_path)

print("////////////////////////data set antique/////////////////////")
input_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/collection.tsv"
output_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/collection1.tsv"
acronym_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/acronyms"

read_and_process_file(input_file_path_antique, output_file_path_antique, acronym_file_path_antique)