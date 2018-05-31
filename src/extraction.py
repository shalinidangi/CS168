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

def reduce_mask(flattened_mask):
	mask = []

	for i in range(len(flattened_mask)):
		if flattened_mask[i] == 2:
			mask.append(1)
		else:
			mask.append(0)

	return mask


data = scipy.io.loadmat('../data/data.mat')['data'][0]

def create_training_set():
    masks = []
    t2_images = []
    test_patient = 8

    for i in range(0, 62):
    	if i == test_patient:
    		continue
    	masks.append(data[i][0])
    	t2_images.append(data[i][1])

    feature_vectors = []
    mask_vectors = []
    auc_vectors = []

    for i in range(len(masks)):
        # np.concatenate((feature_vectors, extract_features(t2_images[i], 3)))
        patient = extract_features(t2_images[i], 3)
        for j in range(len(patient)):
            feature_vectors.append(patient[j])

        accuracy_mask = flatten_mask(masks[i], 3).tolist()
        # print(accuracy_mask)
        auc_mask = reduce_mask(accuracy_mask)
        # print(auc_mask)
        mask_vectors += accuracy_mask
        auc_vectors += auc_mask
        # mask_vectors += flatten_mask(masks[i], 3).tolist()

    print(len(feature_vectors))
    print(len(mask_vectors))
    print(len(auc_vectors))

    model = RandomForestClassifier(n_estimators=10)
    model.fit(feature_vectors, mask_vectors)
    # model = pickle.load(open('finalized_model.sav', 'rb'))

    n = len(feature_vectors) / 62

    test_features = extract_features(data[test_patient][1], 3)
    test_mask = flatten_mask(data[test_patient][0], 3)
    auc_test_mask = reduce_mask(test_mask)

    print(test_mask)
    pred = model.predict(test_features)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    print(roc_auc_score(y_true=auc_test_mask, y_score=pred))
    print(accuracy_score(y_true=test_mask, y_pred=pred))


create_training_set()


