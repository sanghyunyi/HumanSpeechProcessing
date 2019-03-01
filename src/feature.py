import csv
import pandas as pd
from datetime import datetime
import numpy as np
from scipy import interpolate
import math
import glob
import data

# DA tagger
import os, sys
DA_path = os.path.join(sys.path[0], '../../DialogueAct-Tagger')
sys.path.insert(1, DA_path)
from config import Config, Model
import argparse
from predictors.svm_predictor import SVMPredictor
import logging

# Other NLP featuers
import spacy
import gensim.downloader as api
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import g2p_en as g2p

# Extract features from stimuli.

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

def concat_sessions(df_list, offset_list):
    # The format of offset_list is in config_.py
    assert len(df_list) == len(offset_list) - 1
    out_df_list = []
    for i, df in enumerate(df_list):
        offset = offset_list[i][0]
        df['Start'] += offset
        df['End'] += offset
        out_df = df[df['End'] <= offset_list[i + 1][0]]
        out_df_list.append(out_df)
    return pd.concat(out_df_list)

def add_DA_features(df):
    # Need vectorize
    cfg = Config.from_json(os.path.join(DA_path, 'models/Model.SVM/meta.json'))
    cfg.out_folder = os.path.join(DA_path, 'models/Model.SVM')
    tagger = SVMPredictor(cfg)
    DA_dimension_list = []
    DA_communicative_function_list = []
    past_sentence = ""
    for sentence in df['Transcript']:
        if sentence != past_sentence:
            DA_tag = tagger.dialogue_act_tag(sentence)
            if len(DA_tag) == 0:
                DA_dimension = "Other"
                DA_communicative_function = "Other"
            else:
                DA_dimension = DA_tag[0]['dimension']
                DA_communicative_function = DA_tag[0]['communicative_function']
            past_sentence = sentence
        DA_dimension_list.append(DA_dimension)
        DA_communicative_function_list.append(DA_communicative_function)
    df['DA_dimension'] = DA_dimension_list
    df['DA_communicative_function'] = DA_communicative_function_list
    return df

def add_sentiment_features(df):
    # Need vectorize
    senti_class_list = []
    senti_p_pos_list = []
    past_sentence = ""
    for sentence in df['Transcript']:
        if sentence != past_sentence:
            blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
            senti_class = blob.sentiment.classification
            senti_p_pos = blob.sentiment.p_pos
            past_sentence = sentence
        senti_class_list.append(senti_class)
        senti_p_pos_list.append(senti_p_pos)

    df['senti_class'] = senti_class_list
    df['senti_p_pos'] = senti_p_pos_list
    return df

def add_POS_features(df):
    # Need vectorize
    current_sentence = ""
    current_sentence_list = []
    POS_tags = []
    for i, new_sentence in enumerate(df['Transcript']):
        if new_sentence != current_sentence:
            blob = TextBlob(new_sentence)
            POSs = blob.tags
            current_sentence = new_sentence
            current_sentence_list = current_sentence.split()
            count = 0

        current_word = df['Word'][i]

        if current_word == POSs[count][0]:
            POS_tags.append([POSs[count][-1]])
            count += 1
        else:
            subword = POSs[count][0]
            POS_tags_of_a_word = [POSs[count][-1]]
            while current_word != subword:
                count += 1
                subword += POSs[count][0]
                POS_tags_of_a_word.append(POSs[count][-1])
            count += 1
            POS_tags.append(POS_tags_of_a_word)

    df['POS'] = POS_tags
    return df

def add_word_rate_features(df):
    word_rates = []
    for i, word in enumerate(df['Word']):
        word_rate = 1./(df['End'][i] - df['Start'][i])
        word_rates.append(word_rate)
    df['word_rate'] = word_rates
    return df

def add_phoneme_features(df):
    # Need vectorize
    past_sentence = ""
    phoneme_list = []
    with g2p.Session():
        for sentence in df['Transcript']:
            if sentence != past_sentence:
                phonemes_of_sentence = g2p.g2p(sentence)
                phonemes_of_sentence = ','.join(phonemes_of_sentence)
                phonemes_of_sentence = phonemes_of_sentence.split(', ,')
                phonemes_of_sentence = [phonemes_chunk.split(',') for phonemes_chunk in phonemes_of_sentence]
                past_sentence = sentence
            phoneme_list.append(phonemes_of_sentence.pop(0))
    df['phoneme'] = phoneme_list
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
    nlp = spacy.load('en')
    past_sentence = ""
    for sentence in df['Transcript']:
        if sentence != past_sentence:
            tokens = nlp(sentence)
            word_list = [token.text.lower() for token in tokens]
            word_vecs = []
            for word in word_list:
                try:
                    word_vecs.append(wordvec_model[word])
                except:
                    continue
            if len(word_vecs) > 0:
                word_vecs = np.array(word_vecs)
                sent_vec = np.average(word_vecs, axis=0)
            else:
                sent_vec = np.nan
            past_sentence = sentence
        sent_vecs.append(sent_vec)
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
    nlp = spacy.load('en')
    for word in df['Word']:
        tokens = nlp(word)
        sub_word_list = [token.text.lower() for token in tokens]
        sub_word_vecs = []
        for sub_word in sub_word_list:
            try:
                sub_word_vecs.append(wordvec_model[sub_word])
            except:
                continue
        if len(sub_word_vecs) > 0:
            sub_word_vecs = np.array(sub_word_vecs)
            word_vec = np.average(sub_word_vecs, axis=0)
        else:
            word_vec = np.nan
        word_vecs.append(word_vec)
    df['word_vecs'] = word_vecs
    return df

def vectorize(df, features):
    for feature in features:
        assert feature not in ['word_vecs', 'sent_vecs', 'word_rate', 'senti_p_pos']
        if feature in ['POS', 'phoneme']:
            values = set(sum(df[feature], []))
            dic = {}
            for i, value in enumerate(values):
                dic[value] = i
            vectorized = []
            for value_list in df[feature]:
                vec = np.zeros(len(dic))
                for value in value_list:
                    i = dic[value]
                    vec[i] += 1.
                vectorized.append(vec)
            df[feature] = vectorized
        else:
            values = df[feature].unique()
            dic = {}
            for i, value in enumerate(values):
                dic[value] = i
            vectorized = []
            for value in df[feature]:
                vec = np.zeros(len(dic))
                i = dic[value]
                vec[i] += 1.
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

def interpolation(df, kind, features):
    f_dic = {}
    for feature in features:
        #values should be float dor int
        values = df[feature].values #values should be float or int
        values = np.stack(values)
        time_stamp = df['time_stamp']
        if len(values.shape) == 1:
            values = values.reshape(values.shape[0], 1)
        for i in range(values.shape[-1]):
            component = values[:, i]
            f = interpolate.interp1d(time_stamp, component, kind=kind)
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

def delay_and_concat(df):
    # Assume df has TR(2s) of fMRI
    # Refer to Huth et al., Nature, 2016.
    df_list = [df.copy()]
    df = df.drop('time_stamp', axis='columns')
    for i in range(4):
        df.loc[-1] = [0.] * len(df.loc[0])
        df.index += 1
        df.sort_index(inplace=True)
        df_list.append(df.copy()[:-1])

    df = df_list[0]
    df = df.rename(lambda x: '0s_delayed_'+str(x), axis='columns')
    for i, dfs in enumerate(df_list[1:]):
        dfs = dfs.rename(lambda x: str(i + 1) + 's_delayed_' + str(x), axis='columns')
        df = df.join(dfs)

    return df

def full_preproc(path, wordvec_model, interpolation_kind, tr):
    stimuli = []
    for text in glob.glob(os.path.join(path, '*.txt')):
        stimuli.append(googleSTT2df(text))
    stimuli = concat_sessions(stimuli, data.SEGMENTS_OFFSETS)

    stimuli = add_sentiment_features(stimuli)
    stimuli = add_POS_features(stimuli)
    stimuli = add_word_rate_features(stimuli)
    stimuli = add_phoneme_features(stimuli)
    stimuli = add_sentvec_features(stimuli, wordvec_model)
    stimuli = add_wordvec_features(stimuli, wordvec_model)

    stimuli = vectorize(stimuli, ['DA_dimension', 'DA_communicative_function', 'senti_class', 'senti_p_pos', 'POS', 'phoneme'])
    stimuli = resample(stimuli, 4, data.SEGMENTS_OFFSETS[-1][0])
    stimuli = replace_na(stimuli, ['DA_dimension', 'DA_communicative_function', 'senti_class', 'senti_p_pos', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
    ftn_dic = interpolate(stimuli, interpolation_kind, ['DA_dimension', 'DA_communicative_function', 'senti_class', 'senti_p_pos', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
    stimuli = resample_from_interpolation(f_dic, tr, data.SEGMENTS_OFFSETS[-1][0])
    stimuli = delay_and_concat(stimuli)

    return stimuli

if __name__ == "__main__":
    #sbt = srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')
    #print(sbt)
    sbt = []
    for i in range(8):
        sbt.append(googleSTT2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/seg{}_vid.txt'.format(i)))
    sbt = concat_sessions(sbt, data.SEGMENTS_OFFSETS)
    print(sbt)
    #sbt = sbt.iloc[:6] #for testing
    sbt = add_DA_features(sbt)
    print("DA done")
    sbt = add_sentiment_features(sbt)
    print("Senti done")
    sbt = add_POS_features(sbt)
    print("POS done")
    sbt = add_word_rate_features(sbt)
    print("Word rate done")
    sbt = add_phoneme_features(sbt)
    print("Phoneme done")
    sbt = add_sentvec_features(sbt, 'glove-wiki-gigaword-50')
    print("sentvec done")
    sbt = add_wordvec_features(sbt, 'glove-wiki-gigaword-50')
    print("wordvec done")
    print(sbt)
    sbt = vectorize(sbt, ['DA_dimension', 'DA_communicative_function', 'senti_class', 'POS', 'phoneme'])
    print(sbt)
    sbt = resample(sbt, 4, data.SEGMENTS_OFFSETS[-1][0])# 194)
    print(sbt)
    sbt = replace_na(sbt, ['DA_dimension', 'DA_communicative_function', 'senti_class', 'senti_p_pos', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
    print(sbt)
    f_dic = interpolation(sbt, 'nearest', ['DA_dimension', 'DA_communicative_function', 'senti_class', 'senti_p_pos', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
    sbt = resample_from_interpolation(f_dic, 2, data.SEGMENTS_OFFSETS[-1][0])# 194)
    print(sbt)
    sbt = delay_and_concat(sbt)
    print(sbt)
    sbt.to_pickle('../data/feature.pkl')
