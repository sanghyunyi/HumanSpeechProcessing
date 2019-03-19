from sklearn.linear_model import Ridge
import nibabel as nib
import feature, brain, fit
PATH_TO_TRANSCRIPTIONS = '../data/transcription/'
PATH_TO_FMRIS = '../data/fmri/'
PATH_TO_SAVE = '../data/result/encoding.nii.gz'
NUM_OF_SESSIONS = 8 # StudyForrest has 8 sessions
WORD_VEC_MODEL = 'glove-wiki-gigaword-50'
INTERPOLATION = 'nearest'
TR = 2
feature = feature.full_preproc(PATH_TO_TRANSCRIPTIONS, WORD_VEC_MODEL, INTERPOLATION, TR)
feature = feature.filters(regex='DA_c')
brain_img = brain.full_preproc(PATH_TO_FMRIS, NUM_OF_SESSIONS)
reg = Ridge()
corr = fit.encoding(reg, feature, brain_img)
corr[corr < 0.2] = 0.
corr_img = nib.Nifti1Image(corr, affine=np.eye(4))
corr_img.to_filename(PATH_TO_SAVE)

