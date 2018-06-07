import scipy.io
import numpy as np
import sys
import pickle
import operator
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)

# Load patient data from data.mat
data = scipy.io.loadmat('../data/data.mat')['data'][0]

def extract_features(image, mask, n):
    """ 
    Extracts features and corresponding target values for a single patient
    (to be used later for model fitting)

    Args:
        image: T2-weighted image
        mask: mask image corresponding to T2 image
        n: feature vector radius

    Returns:
        patches: array of feature vectors obtained from this patient
        trimmed_mask: array of target values associated with feature vectors 
                      (center pixel of each feature vector)
    """

    patches = []
    trimmed_mask = []

    for i in range(n, 255-n):
        for j in range(n, 255-n):
            # Option 1: If a feature vector contains all 0s, discard it. 
            #   if np.count_nonzero(mask[i-n:i+n+1, j-n:j+n+1].astype(int)) > 0:
            # Option 2: If a feature vectore contains at least one 0, discard it.
            #   if 0 not in mask[i-n:i+n+1, j-n:j+n+1].astype(int):
            # Option 3: If a feature vector's center pixel is a 0, discard it.
            if int(round(mask[i,j])) != 0:
                patch = image[i-n:i+n+1, j-n:j+n+1]
                patches.append(patch.flatten())   
                trimmed_mask.append(int(round(mask[i,j])))             

    return patches, trimmed_mask


def create_training_set(train_set, n):
    """
    Compiles feature vectors and target values for all patients in training set

    Args:
        train_set: array of patient indices to be used in training set
        n: feature vector radius

    Returns:
        feature_vectors: array of feature vectors for all patients in training set
        auc_vectors: array of target values for all patients in training set
        baseline: percentage of 2s (cancerous pixels) in target values
    """

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
        auc_vectors += trimmed_mask

    baseline = auc_vectors.count(2)/float(len(auc_vectors))

    return feature_vectors, auc_vectors, baseline


def build_model(feature_vectors, auc_vectors, classifier, filename, train=False):
    """
    Fits model based off of specified classifier and provided feature vectors and target values

    Args:
        feature_vectors: array of feature vectors extracted from patients in training set
        auc_vectors: array of target values extracted from patients in training set
        classifier: machine learning classifier chosen for model
        filename: name of file to save/load model to/from
        train: (default=False) boolean indicating whether we want to load existing model or fit a new one

    Returns:
        model: fitted model/loaded model
    """

    if train:
        model = classifier
        model.fit(feature_vectors, auc_vectors)
    else:
        model = pickle.load(open(filename, 'rb'))

    return model


def test_model(test_set, model, n):
    """
    Evaluates given model on 3 metrics: ROC-AUC, PR-AUC, and accuracy

    Args:
        test_set: array of patient indices to be used in testing set
        model: model to be tested
        n: feature vector radius

    Returns: 
        roc_auc: ROC-AUC score
        pr_auc: PR-AUC score
        accuracy: accuracy
    """

    # Determine features and masks for test patient
    test_features = []
    test_masks = []

    for patient_index in test_set:
        test_feature, test_mask = extract_features(data[patient_index][1], data[patient_index][0], n)
        test_features += test_feature
        test_masks += test_mask

    pred = model.predict_proba(test_features)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_masks, pred[:, 1], pos_label=2)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    optimal_idx = np.argmax(true_positive_rate - false_positive_rate)
    optimal_threshold = thresholds[optimal_idx]

    precision, recall, thresholds = precision_recall_curve(test_masks, pred[:,1], pos_label=2)
    pr_auc = auc(recall, precision)

    pred = model.predict(test_features)
    accuracy = accuracy_score(y_true=test_masks, y_pred=pred)

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    return roc_auc, pr_auc, accuracy


def run_model(classifier, train_set, test_set, n):
    """
    Creates training set for model fitting, builds model, and tests model

    Args: 
        classifier: machine learning classifier chosen for model
        train_set: array of patient indices to be used in training set
        test_set: array of patient indices to be used in testing set
        n: feature vector radius

    Returns:
        roc_auc: ROC-AUC score
        pr_auc: PR-AUC score
        accuracy: accuracy
        baseline: percentage of 2s (cancerous pixels) in target values
    """

    feature_vectors, auc_vectors, baseline = create_training_set(train_set, n)
    model = build_model(feature_vectors, auc_vectors, classifier, "forest.sav", True)
    roc_auc, pr_auc, accuracy = test_model(test_set, model, n)
   
    return roc_auc, pr_auc, accuracy, baseline


if __name__ == "__main__":
    forest = RandomForestClassifier(n_estimators=10, class_weight="balanced")
    ada = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    k_neighbors = KNeighborsClassifier(3)
    tree = DecisionTreeClassifier(class_weight="balanced")
    neural_network = MLPClassifier(alpha=1)
    classifiers = [forest, ada, gbc, k_neighbors, tree, neural_network]
    names = ["Random Forest", "AdaBoost", "Gradient Boosting", "K-Nearest Neighbors", "Decision Tree", "Multilayer Perceptron (Neural Network)"]

    patients = list(range(0,62))

    kf = KFold(n_splits=10)
    output = open('results.txt','w') 

    for n in range(3,8):
        output.write("\\\\\\\\\\\\\\\\\\\\\\\\\\       " + str(n) + "      \\\\\\\\\\\\\\\\\\\\\\\\\\ \n")
        print("N = " + str(n))
        for i in range(len(classifiers)):
            output.write("\n" + names[i] + "\n")
            print(names[i])
            roc_aucs = []
            baselines = []
            pr_aucs = []
            accuracies = []

            for train_set, test_set in kf.split(patients):
                roc_auc, pr_auc, accuracy, baseline = run_model(classifiers[i], train_set, test_set, n)

                output.write("roc_auc: " + str(roc_auc) + "\n")
                output.write("pr_auc: " + str(pr_auc) + "\n")
                output.write("accuracy: " + str(pr_auc) + "\n")

                print("roc_auc: " + str(roc_auc))
                print("pr_auc: " + str(pr_auc))
                print("accuracy: " + str(accuracy))

                roc_aucs.append(roc_auc)
                baselines.append(baseline)
                pr_aucs.append(pr_auc)
                accuracies.append(accuracy)


            print("AUROC: " + str(sum(roc_aucs) / float(len(roc_aucs))))
            print("Baseline: " + str(sum(baselines) / float(len(baselines))))
            print("AUPRC: " + str(sum(pr_aucs) / float(len(pr_aucs))))
            print("Accuracy: " + str(sum(accuracies) / float(len(accuracies))))

            output.write(names[i] + "\n")
            output.write("AUROC: " + str(sum(roc_aucs) / float(len(roc_aucs))) + "\n")
            output.write("Baseline: " + str(sum(baselines) / float(len(baselines))) + "\n")
            output.write("AUPRC: " + str(sum(pr_aucs) / float(len(pr_aucs))) + "\n")
            output.write("Accuracy: " + str(sum(accuracies) / float(len(accuracies))) + "\n")


    # test_set = [18]
    # train_set = patients
    # train_set.remove(18)

    # # Optional testing for single patient
    # output = open('patient18.txt','w') 
    # for n in range(2,8):
    #     print("N = " + str(n))
    #     output.write("N = " + str(n) + "\n")
    #     for i in range(len(classifiers)):
    #         print(names[i])
    #         output.write(names[i] + "\n")
    #         roc_auc, pr_auc, accuracy, baseline = run_model(classifiers[i], train_set, test_set, n)

    #         print("AUROC: " + str(roc_auc))
    #         print("AUPRC: " + str(pr_auc))
    #         print("Accuracy: " + str(accuracy))

    #         output.write("AUROC: " + str(roc_auc) + "\n")
    #         output.write("Baseline: " + str(baseline) + "\n")
    #         output.write("AUPRC: " + str(pr_auc) + "\n")
    #         output.write("Accuracy: " + str(accuracy) + "\n")


