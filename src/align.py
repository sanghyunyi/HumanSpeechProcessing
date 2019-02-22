import image
import feature
import nibabel as nib
import os
import numpy as np
# Align brain imaging data and stimuli

def nn_align(df, brain_img, tr): # tr = 2 #naive nearest neighbor interpolation
    out_i = []
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

        matched_image = brain_img.img_data[:,:,:,start:end+1]
        if matched_image.size:
            out_i.append(i)
            out.append(np.average(matched_image, axis=-1))

    return df.iloc[out_i], np.array(out)

if __name__ == "__main__":
    sbt = feature.srt2df('/Users/YiSangHyun/Dropbox/Study/Graduate/2018-Winter/Ralphlab/FG/FG_delayed10s_seg0.srt')
    print(sbt)

    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-forrestgump/func'
    img = image.fMRIimage(os.path.join(data_path, 'sub-03_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz'))
    #img_data = img.get_data()

    print(img)

    print(nn_align(sbt, img, 2))
