import csv
import pandas as pd
from datetime import datetime
import numpy as np
from scipy import interpolate
import math, glob, re

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
from textblob.sentiments import PatternAnalyzer
import g2p_en as g2p

# Extract features from stimuli.

def srt2df(path):
    '''
    Read a srt(a subtitle file of a movie) file and convert it to a pandas dataframe.
    Input
    - path: the path to the srt file
    Output
    - df: the pandas dataframe of the srt file. It has 3 columns: Start, End and Transcript
    '''
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
    '''
    Read a output text file from transcribe.py and convert it to a pandas dataframe.
    Input
    - path: the path to the text file from transcribe.py
    Output
    - df: the pandas dataframe of the text file. It has 4 colums: Start, End, Transcript and Word.
    '''
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

def add_DA_features(df):
    '''
    Add Dialog Acts as columns.
    Use DialogueAct-Tagger(https://github.com/ColingPaper2018/DialogueAct-Tagger) to calculate DA tags.
    Input
    - df: the pandas dataframe that has Transciprt column
    Output
    - df: the pandas dataframe with the added columns: DA_dimension and DA_communicative_function
    '''
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
    '''
    Add sentiments as columns.
    Use TextBlob to do the sentiment analysis.
    Input
    - df: the pandas dataframe that has Transciprt column
    Output
    - df: the pandas dataframe with the added columns: senti_p_postivie, senti_polarity, senti_subjectivity
    '''
    p_positive_list = []
    polarity_list = []
    subjectivity_list = []
    past_sentence = ""
    for sentence in df['Transcript']:
        if sentence != past_sentence:
            blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
            p_positive = blob.sentiment.p_pos
            blob = TextBlob(sentence, analyzer=PatternAnalyzer())
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            past_sentence = sentence
        p_positive_list.append(p_positive)
        polarity_list.append(polarity)
        subjectivity_list.append(subjectivity)

    df['senti_p_positive'] = p_positive_list
    df['senti_polarity'] = polarity_list
    df['senti_subjectivity'] = subjectivity_list
    return df

def add_POS_features(df):
    '''
    Add Part Of Speech tag as columns.
    Use spcay to get the tags.
    Input
    - df: the pandas dataframe that has Transciprt and Word columns
    Output
    - df: the pandas dataframe with the added column: POS
    '''
    current_sentence = ""
    POS_tags = []
    nlp = spacy.load('en')
    for i, new_sentence in enumerate(df['Transcript']):
        if new_sentence != current_sentence:
            tokens = nlp(new_sentence)
            POSs = list(zip([token.text for token in tokens], [token.tag_ for token in tokens]))
            current_sentence = new_sentence
            POS_len = len(POSs)
            count = 0

        current_word = df['Word'].iloc[i]

        if current_word == POSs[count][0] or bool(re.search(r'\d', current_word)):
            POS_tags.append([POSs[count][-1]])
            count = (count + 1) % POS_len
        else:
            subword = POSs[count][0]
            POS_tags_of_a_word = [POSs[count][-1]]
            while current_word != subword:
                count = (count + 1) % POS_len
                subword += POSs[count][0]
                POS_tags_of_a_word.append(POSs[count][-1])
            count = (count + 1) % POS_len
            POS_tags.append(POS_tags_of_a_word)

    df['POS'] = POS_tags
    return df

def add_syntactic_dependencies_features(df):
    '''
    Add syntactic dependencies as a column.
    Use spacy to get the tags.
    Input
    - df: the pandas dataframe that has Transciprt and Word columns
    Output
    - df: the pandas dataframe with the added column: syntactic_dependencies
    '''
    current_sentence = ""
    DEP_tags = []
    nlp = spacy.load('en')
    for i, new_sentence in enumerate(df['Transcript']):
        if new_sentence != current_sentence:
            tokens = nlp(new_sentence)
            DEPs = list(zip([token.text for token in tokens], [token.dep_ for token in tokens]))
            current_sentence = new_sentence
            DEP_len = len(DEPs)
            count = 0

        current_word = df['Word'].iloc[i]

        if current_word == DEPs[count][0] or bool(re.search(r'\d', current_word)):
            DEP_tags.append([DEPs[count][-1]])
            count = (count + 1) % DEP_len
        else:
            subword = DEPs[count][0]
            DEP_tags_of_a_word = [DEPs[count][-1]]
            while current_word != subword:
                count = (count + 1) % DEP_len
                subword += DEPs[count][0]
                DEP_tags_of_a_word.append(DEPs[count][-1])
            count = (count + 1) % DEP_len
            DEP_tags.append(DEP_tags_of_a_word)

    df['syntactic_dependencies'] = DEP_tags
    return df

def add_word_rate_features(df):
    '''
    Add word rate as a column.
    Input
    - df: the pandas dataframe that has Word, Start and End columns
    Output
    - df: the pandas dataframe with the added column: word_rate
    '''
    word_rates = []
    for i, word in enumerate(df['Word']):
        word_rate = 1./(df['End'].iloc[i] - df['Start'].iloc[i])
        word_rates.append(word_rate)
    df['word_rate'] = word_rates
    return df

def add_phoneme_features(df):
    '''
    Add phonmes as a column.
    Use g2p_en (https://github.com/Kyubyong/g2p) to get the phonemes.
    Input
    - df: the pandas dataframe that has Transcript column
    Output
    - df: the pandas dataframe with the added column: phoneme
    '''
    def get_idx_of_numbers_in_sentence(sentence):
        word_list = sentence.split()
        idx_list = []
        for i, word in enumerate(word_list):
            if bool(re.search(r'\d', word)):
                idx_list.append(i)
        return idx_list

    def split_sentence_by_number(sentence):
        idx_of_numbers = get_idx_of_numbers_in_sentence(sentence)
        word_list = sentence.split()
        sentence_list = []
        sub_sentence = ""
        for i in range(len(word_list)):
            if i in idx_of_numbers:
                if len(sub_sentence) > 0:
                    sentence_list.append(sub_sentence.strip())
                    sub_sentence = ""
                sentence_list.append(word_list[i])
            else:
                sub_sentence += ' ' + word_list[i]
        sentence_list.append(sub_sentence.strip())
        return sentence_list

    past_sentence = ""
    phoneme_list = []
    with g2p.Session():
        for i, sentence in enumerate(df['Transcript']):
            if sentence != past_sentence:
                splitted_sentence = split_sentence_by_number(sentence)
                phonemes_of_sentence = []
                for sub_sentence in splitted_sentence:
                    if bool(re.search(r'\d', sub_sentence)):
                        phonemes_of_sub_sentence = g2p.g2p(sub_sentence)
                        phonemes_of_sentence.append(phonemes_of_sub_sentence)
                    else:
                        phonemes_of_sub_sentence = g2p.g2p(sub_sentence)
                        phonemes_of_sub_sentence = ','.join(phonemes_of_sub_sentence)
                        phonemes_of_sub_sentence = phonemes_of_sub_sentence.split(', ,')
                        phonemes_of_sub_sentence = [phonemes_chunk.split(',') for phonemes_chunk in phonemes_of_sub_sentence]
                        phonemes_of_sub_sentence = [phonemes for phonemes in phonemes_of_sub_sentence if phonemes != ['.']]
                        phonemes_of_sentence += phonemes_of_sub_sentence

                past_sentence = sentence
                phonemes_len = len(phonemes_of_sentence)
                count = 0

            assert len(sentence.split()) == phonemes_len
            phoneme_list.append(phonemes_of_sentence[count])
            count = (count + 1) % phonemes_len
    df['phoneme'] = phoneme_list
    return df


def add_sentvec_features(df, model):
    '''
    Add sentence vectors as a column.
    Use spacy and gensim to tokenize and vectorize the sentences.
    Now, it just makes an average of the word vectors.
    Check available models at https://github.com/RaRe-Technologies/gensim-data

    Eg.
    fasttext-wiki-news-subwords-300
    glove-wiki-gigaword-300
    word2vec-google-news-300

    Input
    - df: the pandas dataframe that has Transcript column
    - model: word vector model
    Output
    - df: the pandas dataframe with the added column: sent_vecs
    '''
    sent_vecs = []
    wordvec_model = api.load(model)
    nlp = spacy.load('en')
    past_sentence = ""
    vec_dim = int(model.split('-')[-1])
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
                sent_vec = np.zeros(vec_dim)
            past_sentence = sentence
        sent_vecs.append(sent_vec)
    df['sent_vecs'] = sent_vecs
    return df

def add_wordvec_features(df, model):
    '''
    Add word vectors as a column.
    Use spacy and gensim to tokenize and vectorize the sentences.
    Check available models at https://github.com/RaRe-Technologies/gensim-data

    Eg.
    fasttext-wiki-news-subwords-300
    glove-wiki-gigaword-300
    word2vec-google-news-300

    Input
    - df: the pandas dataframe that has Word column
    - model: word vector model
    Output
    - df: the pandas dataframe with the added column: word_vecs
    '''
    word_vecs = []
    wordvec_model = api.load(model)
    nlp = spacy.load('en')
    vec_dim = int(model.split('-')[-1])
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
            word_vec = np.zeros(vec_dim)
        word_vecs.append(word_vec)
    df['word_vecs'] = word_vecs
    return df

def vectorize(df, features):
    '''
    Convert the categorical data to one hot vectors
    Input
    - df: the dataframe that contains columns which are categorical
    - features: the list of the features that to be converted
    Output
    - df: the converted dataframe
    - onehot2feature: the dictionary that convert one hot vectors to corresponding categories.
    '''
    onehot2feature = {}
    for feature in features:
        assert feature in ['POS', 'syntactic_dependencies', 'phoneme', 'DA_dimension', 'DA_communicative_function']
        if feature in ['POS', 'syntactic_dependencies', 'phoneme']:
            values = set(sum(df[feature], []))
            dic = {}
            for i, value in enumerate(values):
                dic[value] = i
            onehot2feature[feature] = {v: k for k, v in dic.items()}
            vectorized = []
            for value_list in df[feature]:
                vec = np.zeros(len(dic))
                for value in value_list:
                    i = dic[value]
                    vec[i] += 1.
                vectorized.append(vec)
            df[feature] = vectorized
        else: #DA
            values = df[feature].unique()
            dic = {}
            for i, value in enumerate(values):
                dic[value] = i
            onehot2feature[feature] = {v: k for k, v in dic.items()}
            vectorized = []
            for value in df[feature]:
                vec = np.zeros(len(dic))
                i = dic[value]
                vec[i] += 1.
                vectorized.append(vec)
            df[feature] = vectorized
    return df, onehot2feature

def resample(df, rate, last_end_time):
    '''
    Get more data points which will be used to calculate interopolation functions.
    Input
    - df: the dataframe which will be resampled
    - rate: the frequency of the resampling
    - last_end_time: the end time point of the considered dataframe
    Output
    - df: the resampled dataframe
    '''
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
    '''
    Remove NA in the dataframe
    Input
    - df: the dataframe which NAs will be removed
    - features: the list of the features that contain NAs.
    Output
    - df: the dataframe without NA
    '''
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

def interpolation(df, kind, features, onehot2feature):
    '''
    Calculate interpolation functions
    Input
    - df: the dataframe
    - kind: the way of the interpolation
    - features: the list of features on which we want to get the interpolation functions
    - onehot2feature: the dictionary that converts one hot vectors to the corresponding feature names. This is from vectorize function.
    Ouput
    - f_dic: the dictionary of feature names to interpolation functions.
    '''
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
            if feature in ['POS', 'syntactic_dependencies', 'phoneme', 'DA_dimension', 'DA_communicative_function']:
                feature_name = onehot2feature[feature][i]
                f_dic[feature + '_' + feature_name] = f
            else:
                f_dic[feature + str(i)] = f
    return f_dic

def resample_from_interpolation(functions_dic, tr, last_end_time):
    '''
    Resample the features from the interpolation functions and return a dataframe.
    The sampling rate should be aligned to the sampling rate of the brain data.
    Input
    - functions_dic: the dictionary of feature names to interpolation functions.
    - tr: the temporal resolution. i.e. inverse of the sampling rate.
    - last_end_time: the end time point of the considered data.
    Output
    - df: the dataframe containing resampled features
    '''
    count = 0
    time_stamps = []
    time_stamp = count * tr + 1
    out = {}
    for key in functions_dic.keys():
        out[key] = []
    while time_stamp <= last_end_time:
        for key in functions_dic.keys():
            out[key].append(functions_dic[key](time_stamp))
        time_stamps.append(time_stamp)
        count += 1
        time_stamp = count * tr + 1

    out['time_stamp'] = time_stamps
    df = pd.DataFrame(out)
    return df

def concat_sessions(df_list, len_list):
    '''
    Concatenate the list of dataframes from different seessions to one dataframe.
    Input
    - df_list: the list of dataframes from different sessions
    - len_list: the length of each session in second. The format of it is in data.py
    Output
    - out: the concatenated dataframe
    '''
    assert len(df_list) == len(len_list)
    out_df_list = []
    for i, df in enumerate(df_list):
        out_df = df[(df['time_stamp'] >= 6) & (df['time_stamp'] <= len_list[i] - 10)]
        out_df_list.append(out_df)
    out = pd.concat(out_df_list, axis=0, ignore_index=True, sort=False)
    out = out.fillna(0)
    return out

def delay_and_concat(df, tr):
    '''
    Make delayed feature dataframes and concatenate those dataframes.
    Refer to Huth et al., Nature, 2016
    Input
    - df: the original dataframe
    - tr: the temporal resolution of the brain data. In Huth et al., it is 2s.
    Output
    - df: the delayed and concatenated dataframe
    '''
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
        dfs = dfs.rename(lambda x: str(tr*(i + 1)) + 's_delayed_' + str(x), axis='columns')
        df = df.join(dfs)

    return df

def full_preproc(path, wordvec_model, interpolation_kind, tr):
    '''
    Do all the feature extraction in feature.py module.
    Input
    - path: Path to the directory where the transcription files are.
    - wordvec_model: The model to make word and sentence vectors
    - interpolation_kind: The way of doing the interpolation
    - tr: the temoporal resolution of the brain data
    Output
    - stimuli: the dataframe that contains all the features
    '''
    stimuli_list = []
    text_files = list(glob.glob(os.path.join(path, '*.txt')))
    text_files.sort()
    for text in text_files:
        print(text)
        stimuli_list.append(googleSTT2df(text))

    proc_stimuli_list = []
    for i, stimuli in enumerate(stimuli_list):
        print('{}th session'.format(i))
        stimuli = add_DA_features(stimuli)
        print('DA done')
        stimuli = add_sentiment_features(stimuli)
        print('Senti done')
        stimuli = add_POS_features(stimuli)
        print('POS done')
        stimuli = add_syntactic_dependencies_features(stimuli)
        print('syntactic dependencies done')
        stimuli = add_word_rate_features(stimuli)
        print('Word rate done')
        stimuli = add_phoneme_features(stimuli)
        print('Phoneme done')
        stimuli = add_sentvec_features(stimuli, wordvec_model)
        print('Sent vec done')
        stimuli = add_wordvec_features(stimuli, wordvec_model)
        print('Word vec done')
        stimuli.to_pickle('../data/correct_feature_text_{}.pkl'.format(i))
        print('Pickled')
        stimuli, onehot2feature = vectorize(stimuli, ['DA_dimension', 'DA_communicative_function', 'POS', 'syntactic_dependencies', 'phoneme'])
        stimuli = resample(stimuli, 4, data.SEGMENTS_LEN[i])
        stimuli = replace_na(stimuli, ['DA_dimension', 'DA_communicative_function', 'senti_p_positive','senti_polarity', 'senti_subjectivity', 'POS', 'syntactic_dependencies', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
        ftn_dic = interpolation(stimuli, interpolation_kind, ['DA_dimension', 'DA_communicative_function', 'senti_p_positive','senti_polarity', 'senti_subjectivity', 'POS', 'syntactic_dependencies', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'], onehot2feature)
        stimuli = resample_from_interpolation(ftn_dic, tr, data.SEGMENTS_LEN[i])

        proc_stimuli_list.append(stimuli)
    stimuli = concat_sessions(proc_stimuli_list, data.SEGMENTS_LEN)
    stimuli = delay_and_concat(stimuli, tr)

    return stimuli

if __name__ == "__main__":
    sbt = full_preproc('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/', 'glove-wiki-gigaword-50', 'nearest', 2)
    '''
    print(len(sbt))
    sbt.to_pickle('../data/correct_feature.pkl')
    print(pd.read_pickle('../data/correct_feature.pkl'))
    sbt = srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')
    print(sbt)
    sbt = googleSTT2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/seg0_vid.txt')
    print(sbt)
    sbt = sbt.iloc[:6] #for testing
    sbt = add_DA_features(sbt)
    print("DA done")
    sbt = add_sentiment_features(sbt)
    print("Senti done")
    sbt = add_POS_features(sbt)
    print("POS done")
    print(sbt)
    sbt = add_syntactic_dependencies_features(sbt)
    print("syntactic dependencies done")
    print(sbt)
    sbt = add_word_rate_features(sbt)
    print("Word rate done")
    print(sbt)
    sbt = add_phoneme_features(sbt)
    print("Phoneme done")
    print(sbt)
    sbt = add_sentvec_features(sbt, 'glove-wiki-gigaword-50')
    print("sentvec done")
    print(sbt)
    sbt = add_wordvec_features(sbt, 'glove-wiki-gigaword-50')
    print("wordvec done")
    print(sbt)
    sbt = pd.read_pickle('../data/feature_text.pkl')
    sbt, onehot2feature = vectorize(sbt, ['DA_dimension', 'DA_communicative_function', 'POS', 'phoneme'])
    print(sbt)
    sbt = resample(sbt, 4, 194)
    print(sbt)
    sbt = replace_na(sbt, ['DA_dimension', 'DA_communicative_function', 'senti_p_positive','senti_polarity', 'senti_subjectivity', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'])
    print(sbt)
    f_dic = interpolation(sbt, 'nearest',  ['DA_dimension', 'DA_communicative_function', 'senti_p_positive','senti_polarity', 'senti_subjectivity', 'POS', 'phoneme', 'word_vecs', 'sent_vecs', 'word_rate'], onehot2feature)
    sbt = resample_from_interpolation(f_dic, 2, 194)
    print(sbt)
    sbt = delay_and_concat(sbt)
    print(sbt)
    '''
