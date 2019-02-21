import image
import feature
import nibabel as nib
import os
import numpy as np

# Align brain imaging data and stimuli

def average_align(df, img_data, tr): # tr = 2
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        print(row)
        start = row['Start']/tr
        end = row['End']/tr

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

        out.append(np.average(img_data[:,:,:,start:end+1], axis=0))

    return np.array(out)


if __name__ == "__main__":
    sbt = stimuli.srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')
    print(sbt)

    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-forrestgump/func'
    img = nib.load(os.path.join(data_path, 'sub-03_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz'))
    #img_data = img.get_data()

    print(img.header)

    print(average_align(sbt, img.get_data(), 2))
