import scipy.io
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def extract_features(image, n):
    patches = []

    for i in range(n, 255-n):
        for j in range(n, 255-n):
            patch = image[i-n:i+n+1, j-n:j+n+1]
            patches.append(patch.flatten())

    return patches
    

def flatten_mask(mask, n):
    return mask[n:255-n, n:255-n].astype(int).flatten()
    # patches = []

    # for i in range(n, 255-n):
    #     for j in range(n, 255-n):
    #         patch = mask[i-n:i+n+1, j-n:j+n+1]
    #         patches.append(patch.flatten())

    # return patches

data = scipy.io.loadmat('../data/data.mat')['data'][0]

def create_training_set():
    masks = []
    t2_images = []

    for i in range(0, 61):
        masks.append(data[i][0])
        t2_images.append(data[i][1])

    feature_vectors = []
    mask_vectors = []

    for i in range(len(masks)):
        # np.concatenate((feature_vectors, extract_features(t2_images[i], 3)))
        patient = extract_features(t2_images[i], 3)
        for j in range(len(patient)):
            feature_vectors.append(patient[j])

        mask_vectors +=  flatten_mask(masks[i], 3).tolist()

    print(len(feature_vectors))
    print(len(mask_vectors))

    # model = RandomForestClassifier(n_estimators=10)
    # model.fit(feature_vectors, mask_vectors)
    model = pickle.load(open('finalized_model.sav', 'rb'))


    test_features = feature_vectors[22000:26000]
    test_mask = mask_vectors[22000:26000]
    print(test_mask)
    pred = model.predict(test_features)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    #print(roc_auc_score(y_true=test_mask, y_score=pred))
    print(accuracy_score(y_true=test_mask, y_pred=pred))


create_training_set()


