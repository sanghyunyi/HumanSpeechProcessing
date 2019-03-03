from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV # Encoding
from sklearn.svm import LinearSVC # Decoding
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle as pkl
import pandas as pd
import numpy as np
import nibabel as nib

ALPHAS = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

def flatten_brain(images):
    original_shape = images.shape
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, :, :, volume_idx]
        one_volume = one_volume.reshape(-1)
        out.append(one_volume)
    out = np.array(out)
    return out, original_shape

def flat_to_3ds(images, original_shape):
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, volume_idx]
        one_volume = one_volume.reshape(original_shape[:-1])
        out.append(one_volume)
    out = np.array(out)
    return out

def flat_to_3d(images, original_shape):
    images = images.reshape(original_shape[:-1])
    return images

def cross_validation(reg, X, y, n_splits): #reg
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

def decoding(reg, features, images):
    X, image_shape = flatten_brain(images)
    y = features
    score_from_alpha = []
    for alpha in ALPHAS:
        reg.C = 1./alpha
        scores = cross_val_score(reg, X, y, cv=10, scoring='accuracy')
        score = np.mean(scores)
        score_from_alpha.append(score)
    best_score = np.amax(score_from_alpha)
    return best_score


if __name__ == "__main__":
    brain_img = pkl.load(open('../data/brain.pkl', 'rb'))
    feature = pd.read_pickle('../data/correct_feature.pkl').filter(regex='POS')
    reg = Ridge()
    corr = encoding(reg, feature, brain_img)
    corr[corr < 0.2] = 0.
    corr_img = nib.Nifti1Image(corr, affine=np.eye(4))
    corr_img.to_filename('../data/corr.nii.gz')
    single_brain_img = brain_img[:, :, :, 0]
    single_brain_img = nib.Nifti1Image(single_brain_img, affine=np.eye(4))
    single_brain_img.to_filename('../data/brain.nii.gz')

