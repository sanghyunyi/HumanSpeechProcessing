import csv
import pandas as pd
from datetime import datetime
import numpy as np
from scipy import interpolate
import math

# DA tagger
import os, sys
DA_path = os.path.join(sys.path[0], '../../DialogueAct-Tagger')
sys.path.insert(1, DA_path)
from config import Config, Model
import argparse
from predictors.svm_predictor import SVMPredictor
import logging

# Other NLP featuers
# import spacy
import gensim.downloader as api
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Handle stimuli.

def srt2df(path):
    srt_in = open(path, 'r', encoding='utf-8-sig').read().split('\n\n')
    starts = []
    ends = []
    transcripts = []

    for line in srt_in:
        line = line.split('\n')

        time_stamp = line[1].split('-->')

        start_time = datetime.strptime(time_stamp[0].strip(), '%H:%M:%S,%f')
        start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond * 1e-6

        end_time = datetime.strptime(time_stamp[1].strip(), '%H:%M:%S,%f')
        end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond * 1e-6

        transcript = ' '.join(line[2:])

        starts.append(start_time)
        ends.append(end_time)
        transcripts.append(transcript)

    d = {'Start': starts, 'End': ends, 'Transcript': transcripts}
    df = pd.DataFrame(data=d)
    return df

def googleSTT2df(path):
    googleSTT = open(path, 'r').read().split('Transcript:')
    starts = []
    ends = []
    sentences = []
    words = []
    for line in googleSTT[1:]:
        try:
            sentence = line.strip().split('\n')[0]
            word_list = line.strip().split('Word:')
            for word_and_times in word_list[1:]:
                word_and_times = word_and_times.strip().split()
                word = word_and_times[0][:-1]
                start_time = float(word_and_times[2][:-1])
                end_time = float(word_and_times[-1])
                starts.append(start_time)
                ends.append(end_time)
                sentences.append(sentence)
                words.append(word)
        except:
            pass
    d = {'Start': starts, 'End': ends, 'Transcript': sentences, 'Word': words}
    df = pd.DataFrame(data=d)
    return df


def add_DA_tags(df):
    cfg = Config.from_json(os.path.join(DA_path, 'models/Model.SVM/meta.json'))
    cfg.out_folder = os.path.join(DA_path, 'models/Model.SVM')
    tagger = SVMPredictor(cfg)
    DA_dimension = []
    DA_communicative_function = []
    for sentence in df['Transcript']:
        DA_tag = tagger.dialogue_act_tag(sentence)
        if len(DA_tag) == 0:
            DA_dimension.append("Other")
            DA_communicative_function.append("Other")
        else:
            DA_dimension.append(DA_tag[0]['dimension'])
            DA_communicative_function.append(DA_tag[0]['communicative_function'])
    df['dimension'] = DA_dimension
    df['communicative_function'] = DA_communicative_function
    return df

def add_sentiment_features(df):
    senti_class = []
    senti_p_pos = []
    for sentence in df['Transcript']:
        blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
        senti_class.append(blob.sentiment.classification)
        senti_p_pos.append(blob.sentiment.p_pos)

    df['senti_class'] = senti_class
    df['senti_p_pos'] = senti_p_pos
    return df

def add_sentvec_features(df, model):
    # It's just averaging the word vectors
    # Check available models at https://github.com/RaRe-Technologies/gensim-data
    # Eg.
    # fasttext-wiki-news-subwords-300
    # glove-wiki-gigaword-300
    # word2vec-google-news-300
    sent_vecs = []
    wordvec_model = api.load(model)
    for sentence in df['Transcript']:
        word_list = sentence.lower().strip().split()
        word_vecs = np.array([wordvec_model[word] for word in word_list])
        sent_vec = np.average(word_vecs, axis=0)
        setn_vecs.append(sent_vec)

    df['sent_vecs'] = sent_vecs
    return df

def add_wordvec_features(df, model):
    # Check available models at https://github.com/RaRe-Technologies/gensim-data
    # Eg.
    # fasttext-wiki-news-subwords-300
    # glove-wiki-gigaword-300
    # word2vec-google-news-300
    word_vecs = []
    wordvec_model = api.load(model)
    for word in df['Word']:
        word_vec = wordvec_model[word.lower()]
        word_vecs.append(word_vec)

    df['word_vecs'] = word_vecs
    return df

def vectorize(df, feature):
    values = df[feature].unique()
    dic = {}
    for i, value in enumerate(values):
        dic[value] = i
    vectorized = []
    for value in df[feature]:
        vec = np.zeros(len(dic))
        i = dic[value]
        vec[i] = 1.
        vec = vec.reshape(1, vec.shape[0])
        vectorized.append(vec)
    df[feature] = vectorized
    return df

def resample(df, rate, last_end_time):
    # rate is Hz
    tr = 1./rate
    count = 0
    time_stamps = []
    lines = []
    time_stamp = count * tr
    while time_stamp <= last_end_time:
        line = df[(df['Start'] <= time_stamp) & (df['End'] > time_stamp)]
        if len(line) == 0:
            line = pd.Series([np.nan])
        time_stamps.append(time_stamp)
        lines.append(line)
        count += 1
        time_stamp = count * tr
    df = pd.concat(lines)
    df['time_stamp'] = time_stamps
    return df

def replace_na(df, features):
    for feature in features:
        default_value = df[feature].mean()
        default_value -= default_value
        new_values = []
        for value in df[feature]:
            if np.isnan(value).any():
                new_values.append(default_value)
            else:
                new_values.append(value)
        df[feature] = new_values
    return df

def interpolation(df, kind, feature):
    #values should be float dor int
    values = df[feature].values #values should be float or int
    values = np.concatenate(values, axis=0)
    time_stamp = df['time_stamp']
    f_dic = {}
    for i in range(values.shape[-1]):
        dim = values[:, i]
        f = interpolate.interp1d(time_stamp, dim, kind=kind)
        f_dic[feature + str(i)] = f
    return f_dic

def resample_from_interpolation(functions_dic, tr, last_end_time):
    count = 0
    time_stamps = []
    time_stamp = count * tr
    out = {}
    for key in functions_dic.keys():
        out[key] = []
    while time_stamp <= last_end_time:
        for key in functions_dic.keys():
            out[key].append(functions_dic[key](time_stamp))
        time_stamps.append(time_stamp)
        count += 1
        time_stamp = count * tr

    out['time_stamp'] = time_stamps
    df = pd.DataFrame(out)
    return df # return type should be np array

if __name__ == "__main__":
    #sbt = srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')
    #print(sbt)
    sbt = googleSTT2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/seg0_vid.txt')
    sbt = sbt.iloc[:2]
    sbt = add_DA_tags(sbt)
    sbt = vectorize(sbt, 'dimension')
    print(sbt)
    sbt = resample(sbt, 4, 192) #886
    print(sbt)
    sbt = replace_na(sbt, ['dimension'])
    print(sbt)
    f_dic = interpolation(sbt, 'nearest', 'dimension')
    sbt = resample_from_interpolation(f_dic, 2, 192)
    print(sbt)

