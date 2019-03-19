from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LogisticRegression # Encoding
from sklearn.svm import LinearSVC # Decoding
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle as pkl
import pandas as pd
import numpy as np
import nibabel as nib


ALPHAS = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] # Considered hyper parameters

def flatten_brain(images):
    '''
    Make the 3d x volumes brain image array be the 1d x volumes array.
    Input
    - images: 3d x volumes numpy array
    Output
    - out: 1d x volumes numpy array
    - original_shape: the shape of the input
    '''
    original_shape = images.shape
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, :, :, volume_idx]
        one_volume = one_volume.reshape(-1)
        out.append(one_volume)
    out = np.array(out)
    return out, original_shape

def flat_to_3ds(images, original_shape):
    '''
    Make the 1d x volumes array be the 3d x volumes brain image array.
    This is the inverse function of flattern_brain
    Input
    - images: 1d x volumes numpy array
    - original_shape: the original shape of the images before being applied to flatten_brain
    Output
    - out: 3d x volumes numpy array
    '''
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, volume_idx]
        one_volume = one_volume.reshape(original_shape[:-1])
        out.append(one_volume)
    out = np.array(out)
    return out

def flat_to_3d(images, original_shape):
    '''
    Make the 1d array be the 3d brain image array.
    Input
    - images: 1d numpy array
    - original_shape: the original shape of the images before being applied to flatten_brain
    Output
    - out: 3d numpy array
    '''
    images = images.reshape(original_shape[:-1])
    return images

def cross_validation(reg, X, y, n_splits):
    '''
    Do n-fold cross validation with a regression algorithm.
    Input
    - reg: A regression algorithm from sklean
    - X: the input to the model
    - y: the output of the model
    - n_splits: the number of folds
    Output
    - corr_from_CV: correlation score from the cross validation
    '''
    kf = KFold(n_splits=n_splits, shuffle=False)
    corr_from_CV = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)
        corr_list = []
        for i in range(prediction.shape[-1]):
            corr = np.corrcoef(prediction[:, i], y_test[:, i])[0, 1]
            corr_list.append(corr)
        corr_list = np.nan_to_num(np.array(corr_list))
        corr_from_CV.append(corr_list)
    corr_from_CV = np.average(corr_from_CV, axis=0)
    return corr_from_CV


def encoding(reg, features, images):
    '''
    Run encoding models voxel by voxel with features and brain images.
    10-folds cross validation is used and the hypermarameter is tuned
    based on the cross validation result.
    Input
    - reg: A regression algorithm from sklearn
    - features: the feature array from feature.py module
    - images: the brain image array from brain.py module
    Output
    - corr_3d: 3d array that contains correlation scores of each voxel
    '''
    X = features
    y, image_shape = flatten_brain(images)
    corr_from_alpha = []
    for alpha in ALPHAS:
        reg.alpha = alpha
        corr_from_CV = cross_validation(reg, X, y, 10)
        corr_from_alpha.append(corr_from_CV)
    best_corr = np.amax(corr_from_alpha, axis=0)
    corr_3d = flat_to_3d(best_corr, image_shape)
    return corr_3d

def decoding(clf, features, images):
    '''
    Run a decoding model with features and brain images.
    10-folds cross validation is used and the hyperparameter is tuned
    based on the cross validation result.
    The implementation now, is running on a whole brain.
    Input
    - clf: A classificaion algorithm from sklearn
    - features: the feature array from feature.py module
    - images: the barin image array from barin.py module
    Output
    - best_score: the best accuracy score from the hyperparameter tuning using the cross validation.
    '''
    X, image_shape = flatten_brain(images)
    y = features
    score_from_alpha = []
    for alpha in ALPHAS:
        reg.C = 1./alpha
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        score = np.mean(scores)
        score_from_alpha.append(score)
    best_score = np.amax(score_from_alpha)
    return best_score


if __name__ == "__main__":
    brain_img = pkl.load(open('../data/brain.pkl', 'rb'))
    feature = pd.read_pickle('../data/correct_feature.pkl').filter(regex='DA_com')
    feature = np.argmax(np.array(feature), axis=1)
    print(len(feature))
    en = False
    if en:
        reg = Ridge()
        corr = encoding(reg, feature, brain_img)
        corr[corr < 0.2] = 0.
        corr_img = nib.Nifti1Image(corr, affine=np.eye(4))
        corr_img.to_filename('../data/corr.nii.gz')
        single_brain_img = brain_img[:, :, :, 0]
        single_brain_img = nib.Nifti1Image(single_brain_img, affine=np.eye(4))
        single_brain_img.to_filename('../data/brain.nii.gz')
    else:
        clf = LinearSVC()
        print(decoding(clf, feature, brain_img))

