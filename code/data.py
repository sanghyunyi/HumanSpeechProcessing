import os
import csv
import pandas
from datetime import datetime
# The module to handle data
# Onset time
# Duration
# Name of the feature
# feature Content
# Corresponding voxel matrices
# handel tsv and csv

# idx

def srt2df(path):
    srt_in = open(path, 'r', encoding='utf-8-sig').read().split('\n\n')
    starts = []
    ends = []
    transcripts = []

    for line in srt_in:
        line = line.split('\n')

        time_stamp = line[1].split('-->')

        start_time = datetime.strptime(time_stamp[0].strip(), '%H:%M:%S,%f')
        start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time * 1e-6

        end_time = datetime.strptime(time_stamp[1].strip(), '%H:%M:%S,%f')
        end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time * 1e-6

        transcript = ' '.join(line[2:])

        starts.append(start_time)
        ends.append(end_time)
        transcripts.append(subtitle)

    d = {'Start': starts, 'End': ends, 'Transcript': transcripts}
    df = pd.DataFrame(data=d)
    return df

class fMRIData:
    def __init__(self, path):
        # need to know DICOM format, dir structure
        return None

