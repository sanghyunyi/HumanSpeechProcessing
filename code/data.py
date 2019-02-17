import os, sys
DA_path = os.path.join(sys.path[0], '../../DialogueAct-Tagger')
sys.path.insert(1, DA_path)
import csv
import pandas as pd
from datetime import datetime

import nibabel as nib

# DA tagger
from config import Config, Model
import argparse
from predictors.svm_predictor import SVMPredictor
import logging

# Other NLP featuers
# import spacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

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

def add_nlp_features(df):
    senti_class = []
    senti_p_pos = []
    for sentence in df['Transcript']:
        blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
        senti_class.append(blob.sentiment['classification'])
        senti_p_pos.append(blob.sentiment['p_pos'])

    df['senti_class'] = senti_class
    df['senti_p_pos'] = senti_p_pos
    return df

def align_fMRI2data(df, img):
    img = img.get_fdata() #ndarry
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        start = row['Start']/2.
        end = row['End']/2.

        res = start % 1
        if res < 0.5:
            start = int(start)
        else:
            start = int(start) + 1

        res = end % 1
        if res < 0.5:
            end = int(end) - 1
        else:
            end = int(end)

        out.append(np.average(img[:,:,:,start:end+1], axis=0))

    return np.array(out)

sbt = srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')

print(align_fMRI_with_data(sbt, None))

data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-forrestgump/func'
img = nib.load(os.path.join(data_path, 'sub-03_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz'))
#print(img.get_fdata()[:,:,:,0].shape)

