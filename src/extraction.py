import scipy.io
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def extract_features(image, mask, n):
    patches = []
    trimmed_mask = []

    for i in range(n, 255-n):
        for j in range(n, 255-n):
            if int(mask[i, j]) != 0:
            # if 0 not in mask[i-n:i+n+1, j-n:j+n+1].astype(int):
                patch = image[i-n:i+n+1, j-n:j+n+1]
                patches.append(patch.flatten())   
                trimmed_mask.append(int(mask[i,j]))             

    # print(patches)
    return patches, trimmed_mask
    

def flatten_mask(mask, n):
    return mask[n:255-n, n:255-n].astype(int).flatten()


def reduce_mask(flattened_mask):
    # Mark all cancerous pixels as 1 and all others as 0
    mask = []

    for i in range(len(flattened_mask)):
        # Reduce 1 -> 0 and 2 -> 1
        if flattened_mask[i] == 2:
            mask.append(1)
        elif flattened_mask[i] == 1:
            mask.append(0)

    return mask


data = scipy.io.loadmat('../data/data.mat')['data'][0]

def create_training_set(test_patient):
    masks = []
    t2_images = []

    for i in range(0, 62):
        # Gather masks and images for all patients (excluding test patient)
        if i == test_patient or i == 27:
            continue
        masks.append(data[i][0])
        t2_images.append(data[i][1])

    feature_vectors = []
    # mask_vectors = []
    auc_vectors = []

    for i in range(len(masks)):
        # Extract features for all patients
        patient, trimmed_mask = extract_features(t2_images[i], masks[i], 6)
        for j in range(len(patient)):
            feature_vectors.append(patient[j])

        # Flatten and reduce masks for use in accuracy and AUC testing
        accuracy_mask = flatten_mask(masks[i], 3).tolist()
        # mask_vectors += accuracy_mask
        auc_vectors += reduce_mask(trimmed_mask)
        # print(auc_mask)

    print(len(feature_vectors))
    print(len(auc_vectors))
    print(auc_vectors.count(1))
    print(auc_vectors.count(0))

    return feature_vectors, auc_vectors


def build_model(feature_vectors, auc_vectors, classifier, filename, train=False):
    if train:
        model = classifier
        model.fit(feature_vectors, auc_vectors)
        pickle.dump(model, open(filename, 'wb'))
    else:
        model = pickle.load(open(filename, 'rb'))

    return model


def test_model(test_patient, model):
    # Determine features and masks for test patient
    test_mask = data[test_patient][0]
    test_features, trimmed_test_mask = extract_features(data[test_patient][1], test_mask, 6)
    pred = model.predict_proba(test_features)
    print(pred)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(reduce_mask(trimmed_test_mask), pred[:, 1])
    roc_auc= auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # print(trimmed_test_mask)
    # print(roc_auc_score(y_true=reduce_mask(trimmed_test_mask), y_score=pred))
    # print(accuracy_score(y_true=test_mask, y_pred=pred))


def run_model(test_patient):
    classifier1 = RandomForestClassifier(n_estimators=10)
    classifier2 = AdaBoostClassifier()
    classifier3 = GradientBoostingClassifier()


    feature_vectors, auc_vectors = create_training_set(test_patient)
    # print(feature_vectors)
    # print(auc_vectors)
    model = build_model(feature_vectors, auc_vectors, classifier3, "gbc.sav", True)
    test_model(test_patient, model)


run_model(18)


