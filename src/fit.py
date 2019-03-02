from sklearn.linear_model import LinearRegression, Ridge, Lasso # Encoding
from sklearn.svm import LinearSVC # Decoding
import pickle as pkl
import pandas as pd
import numpy as np

def flatten_brain(images):
    original_shape = images.shape
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, :, :, volume_idx]
        one_volume = one_volume.reshape(-1)
        out.append(one_volume)
    out = np.array(out)
    return out, original_shape

def flat_to_3d(images, original_shape):
    out = []
    for volume_idx in range(images.shape[-1]):
        one_volume = images[:, volume_idx]
        one_volume = one_volume.reshape(original_shape[:-1])
        out.append(one_volume)
    out = np.array(out)
    return out

def encoding(reg, features, images):
    X = features
    y, image_shape = flatten_brain(images)
    reg.fit(X, y)
    return reg.score(X, y)

def decoding(reg, features, images):
    X, image_shape = flatten_brain(images)
    y = features
    return None


if __name__ == "__main__":
    brain_img = pkl.load(open('../data/brain.pkl', 'rb'))
    print(brain_img.shape)
    feature = pd.read_pickle('../data/feature_number.pkl')
    print(len(feature))
    reg = Ridge()
    print(encoding(reg, feature, brain_img))

