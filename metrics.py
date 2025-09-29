import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score

def PlotROCCurve(probs,y_test_roc, ci= 95, random_seed=0):
    
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    # print("AUC:",roc_auc)
    # print("AUPRC:", average_precision)
    # print("Sensitivity:",sensitivity)
    # print("Specificity:",specificity)
    # print("Score thresold:",threshold)
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)
    # print("AUC CI:", lower_auroc, upper_auroc)
    # print("AUC Standard Deviation:", std_auroc)
    # print("AUPRC CI:", lower_ap, upper_ap)
    # print("AUPRC Standard Deviation:", std_ap)
    # print("Sensitivity CI:", lower_sensitivity, upper_sensitivity)
    # print("Sensitivity Standard Deviation:", std_sensitivity)
    # print("Specificity CI:", lower_specificity, upper_specificity)
    # print("Specificity Standard Deviation:", std_specificity)
    return roc_auc, average_precision, sensitivity, specificity, threshold


def auc_with_ci(probs,y_test_roc, lower = 2.5, upper = 97.5, n_bootstraps=200, rng_seed=10):
    print(lower, upper)
    y_test_roc = np.asarray(y_test_roc)
    bootstrapped_auroc = []
    bootstrapped_ap = []
    bootstrapped_sensitivity = []
    bootstrapped_specificity = []

    rng = np.random.default_rng(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.integers(0, len(y_test_roc)-1, len(y_test_roc))
        if len(np.unique(y_test_roc[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        fpr, tpr, threshold = metrics.roc_curve(y_test_roc[indices],probs[indices])
        auroc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(y_test_roc[indices], probs[indices])
        a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
        sensitivity = tpr[a]
        specificity = 1-fpr[a]
        bootstrapped_auroc.append(auroc)
        bootstrapped_ap.append(ap)
        bootstrapped_sensitivity.append(sensitivity)
        bootstrapped_specificity.append(specificity)

    lower_auroc,upper_auroc = np.percentile(bootstrapped_auroc, [lower, upper])
    lower_ap,upper_ap = np.percentile(bootstrapped_ap, [lower, upper])
    lower_sensitivity,upper_sensitivity = np.percentile(bootstrapped_sensitivity, [lower, upper])
    lower_specificity,upper_specificity = np.percentile(bootstrapped_specificity, [lower, upper])

    std_auroc = np.std(bootstrapped_auroc)
    std_ap = np.std(bootstrapped_ap)
    std_sensitivity = np.std(bootstrapped_sensitivity)
    std_specificity = np.std(bootstrapped_specificity)

    return lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity



def Calculatemetric_CI(probs,y_test_roc, ci= 95, random_seed=0):
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    
    lower_auroc, upper_auroc, std_auroc, lower_ap, upper_ap, std_ap, lower_sensitivity, upper_sensitivity, std_sensitivity, lower_specificity, upper_specificity, std_specificity = auc_with_ci(probs,y_test_roc, lower = (100-ci)/2, upper = 100-(100-ci)/2, n_bootstraps=20, rng_seed=random_seed)
    print("AUC CI:", lower_auroc, upper_auroc)
    print("AUC Standard Deviation:", std_auroc)
    print("AUPRC CI:", lower_ap, upper_ap)
    print("AUPRC Standard Deviation:", std_ap)
    print("Sensitivity CI:", lower_sensitivity, upper_sensitivity)
    print("Sensitivity Standard Deviation:", std_sensitivity)
    print("Specificity CI:", lower_specificity, upper_specificity)
    print("Specificity Standard Deviation:", std_specificity)
    return lower_auroc, upper_auroc, lower_ap, upper_ap, lower_sensitivity, upper_sensitivity, lower_specificity, upper_specificity