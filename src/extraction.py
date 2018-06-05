import scipy.io
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt

# np.set_printoptions(threshold='nan')

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

def create_training_set(train_set, n):
    masks = []
    t2_images = []

    for patient_index in train_set:
        # Gather masks and images for all patients (excluding test patient)
        masks.append(data[patient_index][0])
        t2_images.append(data[patient_index][1])

    feature_vectors = []
    auc_vectors = []

    for i in range(len(masks)):
        # Extract features for all patients
        patient, trimmed_mask = extract_features(t2_images[i], masks[i], n)
        for j in range(len(patient)):
            feature_vectors.append(patient[j])

        # Flatten and reduce masks for use in accuracy and AUC testing
        accuracy_mask = flatten_mask(masks[i], 3).tolist()
        auc_vectors += reduce_mask(trimmed_mask)

    # print(len(feature_vectors))
    # print(len(auc_vectors))
    print(auc_vectors.count(1))
    print(auc_vectors.count(0))
    print(auc_vectors.count(1)/float(auc_vectors.count(0)+ auc_vectors.count(1)))

    return feature_vectors, auc_vectors, auc_vectors.count(1)/float(auc_vectors.count(0)+ auc_vectors.count(1))


def build_model(feature_vectors, auc_vectors, classifier, filename, train=False):
    if train:
        model = classifier
        model.fit(feature_vectors, auc_vectors)
        pickle.dump(model, open(filename, 'wb'))
    else:
        model = pickle.load(open(filename, 'rb'))

    return model


def test_model(test_set, model, n):
    # Determine features and masks for test patient
    test_features = []
    test_masks = []

    for patient_index in test_set:
        test_feature, test_mask = extract_features(data[patient_index][1], data[patient_index][0], n)
        test_features += test_feature
        test_masks += test_mask

    pred = model.predict_proba(test_features)
    print(pred)

    my_theta_times_X = 0.3  # Our custom threshold
    predict_theta = pred[:, 1] > my_theta_times_X
    
    reduced_test_masks = reduce_mask(test_masks)
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(reduced_test_masks, pred[:, 1])
    false_positive_rate, true_positive_rate, thresholds = roc_curve(reduced_test_masks, predict_theta)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)

    precision, recall, thresholds = precision_recall_curve(reduced_test_masks, pred[:,1], pos_label=1)
    pr_auc = auc(recall, precision)
    print(pr_auc)

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')

    return roc_auc, pr_auc


def run_model(classifier, train_set, test_set, n):
    feature_vectors, auc_vectors, baseline = create_training_set(train_set, n)
    # print(feature_vectors)
    # print(auc_vectors)
    model = build_model(feature_vectors, auc_vectors, classifier, "forest.sav", True)
    roc_auc, pr_auc = test_model(test_set, model, n)
    return roc_auc, pr_auc, baseline



if __name__ == "__main__":
    forest = RandomForestClassifier(n_estimators=10)
    ada = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    k_neighbors = KNeighborsClassifier(3)
    tree = DecisionTreeClassifier()
    neural_network = MLPClassifier(alpha=1)
    svc = SVC()
    classifiers = [forest, ada, gbc, k_neighbors, tree, neural_network, svc]
    names = ['forest', 'ada', 'gbc', 'k_neighbors', 'tree', 'neural_network', 'svc']

    patients = list(range(0,62))
    patients.remove(27)

    kf = KFold(n_splits=10)
    output = open('output2.txt','w') 

    for n in range (2,7):
        output.write("\\\\\\\\\\\\\\\\\\\\\\\\\\       " + str(n) + "      \\\\\\\\\\\\\\\\\\\\\\\\\\ \n")

        for i in range(len(classifiers)):
            output.write(names[i] + "\n")
            roc_aucs = []
            pr_aucs = []
            adjusted_scores = []

            for train_set, test_set in kf.split(patients):
                roc_auc, pr_auc, baseline = run_model(classifiers[i], train_set, test_set, n)

                output.write("roc_auc: " + str(roc_auc) + "\n")
                output.write("pr_auc: " + str(pr_auc) + "\n")

                roc_aucs.append(roc_auc)
                pr_aucs.append(pr_auc)

                if roc_auc < 0.5:
                    adjusted_scores.append(1 - roc_auc)
                else:
                    adjusted_scores.append(roc_auc)

            output.write(names[i] + "\n")
            output.write("AUROC: " + str(sum(roc_aucs) / float(len(roc_aucs))) + "\n")
            output.write("Adj. AUROC: " + str(sum(adjusted_scores) / float(len(adjusted_scores))) + "\n")
            output.write("AUPRC: " + str(sum(pr_aucs) / float(len(pr_aucs))) + "\n")
            output.write("Ratio of Positive/Total: " + str(baseline) + "\n")
            output.write("Difference in Average Precision: " + str(sum(pr_aucs) / float(len(pr_aucs)) - baseline) + "\n")

