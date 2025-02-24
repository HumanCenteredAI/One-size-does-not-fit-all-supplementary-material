
# %%
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    'pandas',
    'scikit-learn',
    'numpy',
    'matplotlib',
    'imblearn',
    'xgboost',
]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# %%
# IMPORT PACKAGES AND LIBRARIES
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, cohen_kappa_score

from sklearn.model_selection import GroupKFold
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import precision_score, make_scorer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from collections import defaultdict

import shap


# %%
# READ CSV FILES
df_all_clean = pd.read_csv("all_clean.csv")
df_typical_clean = pd.read_csv("neurotypical_clean.csv")
df_divergent_clean = pd.read_csv("neurodivergent_clean.csv")
df_d1_clean = pd.read_csv("d1_clean.csv")
df_d2_clean = pd.read_csv("d2_clean.csv")
df_d3_clean = pd.read_csv("d3_clean.csv")
df_d4_clean = pd.read_csv("d4_clean.csv")
df_d5_clean = pd.read_csv("d5_clean.csv")
df_d6_clean = pd.read_csv("d6_clean.csv")
df_d7_d8_clean = pd.read_csv("d7_d8_clean.csv")
df_all_data_no_null = pd.read_csv("all_data_no_null.csv")

# %%
# NUMBER OF PARTICIPANTS, INSTANCES AND BASE RATE
# List of DataFrames
dataframes = [
    ('df_all_clean', df_all_clean),
    ('df_typical_clean', df_typical_clean),
    ('df_divergent_clean', df_divergent_clean),
    ('df_d1_clean', df_d1_clean),
    ('df_d2_clean', df_d2_clean),
    ('df_d3_clean', df_d3_clean),
    ('df_d4_clean', df_d4_clean),
    ('df_d5_clean', df_d5_clean),
    ('df_d6_clean', df_d6_clean),
    ('df_d7_d8_clean', df_d7_d8_clean)
]

def calculate_statistics(df):
    num_participants = df['Participant'].nunique()
    num_instances = len(df) 
    base_rate = df['TUT'].mean()
    return num_participants, num_instances, base_rate

for i, (name, df) in enumerate(dataframes, start=1):
    num_participants, num_instances, base_rate = calculate_statistics(df)
    print(f"{name}:")
    print(f"  Number of participants = {num_participants}")
    print(f"  Number of instances = {num_instances}")
    print(f"  Base Rate: {base_rate:.2f}")

# %%
#FUNCTION FOR MODELING
def train_and_evaluate(df, features):
    X = df[features]
    y = df['TUT']
    
    # Define classifiers and parameter grids
    classifiers = {
        'Chance': DummyClassifier(strategy="stratified"),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(random_state=42, probability=True), 
        "XGBoost": xgb.XGBClassifier(random_state=42)
    }

    param_grid_RF = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [50, 100, 200],
    }

    param_grid_SVM = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']
    }

    param_grid_XGB = {
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [10, 20],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
    }

    param_grids = {
        'Random Forest': param_grid_RF,
        'SVM': param_grid_SVM,
        'XGBoost': param_grid_XGB
    }

    precision_dict = {}
    recall_dict = {}
    kappa_dict = {}
    auroc_score_barplot = {name: [] for name in classifiers}
    
    # Fit PCA to determine the number of components that explain 95% of the variance
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1

    # Define the number of folds for cross-validation
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    # Loop through the classifiers
    for name, clf in classifiers.items():
        accuracy_scores = []
        weighted_f1_scores = []
        precision_scores = []
        recall_scores = []
        kappa_scores = []
        auroc_scores = []
        confusion_matrices = []
        prediction_rates = []

        # Perform cross-validation
        for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Apply PCA on the training data
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(X_train_pca.shape[1])])

            # Perform oversampling on minority class
            smote = SMOTE()
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca_df, y_train)

            # Apply the transformation to the test data
            X_test_pca = pca.transform(X_test)
            X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i}' for i in range(X_test_pca.shape[1])])

            if name != "Chance":
                grid = GridSearchCV(clf, param_grids[name], refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train_resampled, y_train_resampled)
                best_clf = grid.best_estimator_
                best_clf = best_clf.fit(X_train_resampled, y_train_resampled)
                y_pred = best_clf.predict(X_test_pca_df)
                y_pred_proba = best_clf.predict_proba(X_test_pca_df)[:, 1]
            else:
                best_clf = clf.fit(X_train, y_train)
                y_pred = best_clf.predict(X_test)
                y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_pred_proba)
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
            prediction_rate = (confusion_matrix[1, 1] + confusion_matrix[0, 1]) / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 0])

            accuracy_scores.append(accuracy)
            weighted_f1_scores.append(weighted_f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            kappa_scores.append(kappa)
            auroc_scores.append(auroc)
            confusion_matrices.append(confusion_matrix)
            prediction_rates.append(prediction_rate)

            auroc_score_barplot[name].append(auroc)

        avg_accuracy = np.mean(accuracy_scores)
        avg_weighted_f1 = np.mean(weighted_f1_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_kappa = np.mean(kappa_scores)
        avg_auroc = np.mean(auroc_scores)
        avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
        avg_prediction_rate = np.mean(prediction_rates)

        print(f"{name} Results:")
        print(f"Avg Accuracy: {avg_accuracy:.2f}")
        print(f"Avg Weighted F1 Score: {avg_weighted_f1:.2f}")
        print(f"Avg Precision_1: {avg_precision:.2f}")
        print(f"Avg Recall_1: {avg_recall:.2f}")
        print(f"Avg Kappa: {avg_kappa:.2f}")
        print(f"Avg AUROC: {avg_auroc:.2f}")
        print("Avg Confusion Matrix:")
        print(avg_confusion_matrix)
        print(f"Avg Prediction Rate: {avg_prediction_rate:.2f}\n")

        precision_dict[name] = avg_precision
        recall_dict[name] = avg_recall
        kappa_dict[name] = avg_kappa

    return precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers

def plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers, n_splits=5):
    colors = ['#003F5C', '#FFA600', '#7A5195', '#D62728']

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].bar(precision_dict.keys(), precision_dict.values(), color=colors)
    ax[0].set_title('Precision Scores by Classifier')
    ax[0].set_xlabel('Classifier')
    ax[0].set_ylabel('Precision Score')
    ax[0].set_ylim([0, 1])

    ax[1].bar(recall_dict.keys(), recall_dict.values(), color=colors)
    ax[1].set_title('Recall Scores by Classifier')
    ax[1].set_xlabel('Classifier')
    ax[1].set_ylabel('Recall Score')
    ax[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 7))
    bar_width = 0.15  
    gap_width = 0.05 

    bar_positions = np.arange(len(classifiers)) * (n_splits * (bar_width + gap_width))

    for idx, name in enumerate(classifiers):
        scores = auroc_score_barplot[name]
        avg_score = np.mean(scores)
        for fold_idx, score in enumerate(scores):
            bar_pos = bar_positions[idx] + (fold_idx * bar_width)
            ax.bar(bar_pos, score, bar_width, label=name if fold_idx == 0 else "",
                   color=colors[idx], edgecolor='white', linewidth=0.1, alpha=1)
        ax.axhline(y=avg_score, color=colors[idx], linestyle='dashed', linewidth=1.5)
    ax.set_xlabel('Classifiers', fontsize=15)
    ax.set_ylabel('AUROC Score', fontsize=15)
    ax.set_title('AUROC Scores for each Fold, Grouped by Classifiers', fontsize=17)

    ax.set_xticks(bar_positions + ((n_splits * bar_width) / 2) - (bar_width / 2))
    ax.set_xticklabels(classifiers.keys())

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc='lower right', fontsize=13)

    plt.tight_layout()
    plt.show()

# %%
# MODELING ALL PARTICIPANTS
# NLP
df = df_all_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_all_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_all_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING NEUROTYPICAL PARTICIPANTS
# NLP
df = df_typical_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_typical_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_typical_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING NEURODIVERGENT PARTICIPANTS
# NLP
df = df_divergent_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_divergent_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_divergent_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING ADD/ADHD PARTICIPANTS
# NLP
df = df_d1_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d1_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d1_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING AUTISM/ASPERGER'S/ASD PARTICIPANTS
# NLP
df = df_d2_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d2_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d2_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING DYSLEXIA/DYSPRAXIA/DYSCALCULIA/DYSGRAPHIA PARTICIPANTS
# NLP
df = df_d3_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d3_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d3_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING ANY OTHER LANGUGE, READING, MATH AND NON-VERBAL LEARNING DISORDER PARTICIPANTS
# NLP
df = df_d4_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d4_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d4_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING GENERALIZED ANXIETY DISORDER PARTICIPANTS
# NLP
df = df_d5_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d5_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d5_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING OTHER PARTICIPANTS
# NLP
df = df_d6_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d6_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d6_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# MODELING "Prefer not to respond" and "I have never been diagnosed with any listed diagnosis"
# NLP
df = df_d7_d8_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df = df_d7_d8_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df = df_d7_d8_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

#%%
# SHAP FUNCTION - RF


def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    
    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}

    # Define the number of folds for cross-validation
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    accuracy_scores = []
    weighted_f1_scores = []
    precision_scores = []
    recall_scores = []
    auroc_scores = []
    conf_matrices = []
    prediction_rates = []

    # Perform cross-validation
    for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Perform oversampling on minority class
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_resampled, y_train_resampled)
        best_clf = grid.best_estimator_
        best_clf = best_clf.fit(X_train_resampled, y_train_resampled)
        y_pred = best_clf.predict(X_test)
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
        # SHAP Explainer
        explainer = shap.Explainer(best_clf)
        shap_values = explainer(X_test)
        
        # Visualize the SHAP values for the first prediction (USE FOR RANDOM FOREST)
        shap.plots.waterfall(shap_values[0][:,1])
        
        # Beeswarm plot for all observations
        shap.plots.beeswarm(shap_values[:, :, 1])
        
        # Summary plot for all observations
        shap.summary_plot(shap_values[:, :, 1], X_test)

        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        auroc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        prediction_rate = (conf_matrix[1, 1] + conf_matrix[0, 1]) / (conf_matrix[1, 1] + conf_matrix[1, 0] + conf_matrix[0, 1] + conf_matrix[0, 0])

        accuracy_scores.append(accuracy)
        weighted_f1_scores.append(weighted_f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        auroc_scores.append(auroc)
        conf_matrices.append(conf_matrix)
        prediction_rates.append(prediction_rate)

        auroc_score_barplot[classifier_name].append(auroc)

    avg_accuracy = np.mean(accuracy_scores)
    avg_weighted_f1 = np.mean(weighted_f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_auroc = np.mean(auroc_scores)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)
    avg_prediction_rate = np.mean(prediction_rates)

    print(f"{classifier_name} Results:")
    print(f"Avg Accuracy: {avg_accuracy:.2f}")
    print(f"Avg Weighted F1 Score: {avg_weighted_f1:.2f}")
    print(f"Avg Precision_1: {avg_precision:.2f}")
    print(f"Avg Recall_1: {avg_recall:.2f}")
    print(f"Avg AUROC: {avg_auroc:.2f}")
    print("Avg Confusion Matrix:")
    print(avg_conf_matrix)
    print(f"Avg Prediction Rate: {avg_prediction_rate:.2f}\n")

    precision_dict[classifier_name] = avg_precision
    recall_dict[classifier_name] = avg_recall

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: clf}

#%%
# NLP, OTHER LANGUAGE/ READING.., RF
df = df_d4_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}

precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)



# %%
# SHAP FUNCTION - XGB
import shap
from collections import defaultdict

def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    

    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}
    
#     # Fit PCA to determine the number of components that explain 95% of the variance
#     pca = PCA().fit(X)
#     cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#     n_components = np.argmax(cumulative_variance >= 0.95) + 1

    # Define the number of folds for cross-validation
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    accuracy_scores = []
    weighted_f1_scores = []
    precision_scores = []
    recall_scores = []
    auroc_scores = []
    conf_matrices = []
    prediction_rates = []

    # Perform cross-validation
    for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         # Apply PCA on the training data
#         pca = PCA(n_components=n_components)
#         X_train_pca = pca.fit_transform(X_train)
#         X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(X_train_pca.shape[1])])

        # Perform oversampling on minority class
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#         # Apply the transformation to the test data
#         X_test_pca = pca.transform(X_test)
#         X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i}' for i in range(X_test_pca.shape[1])])

        grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_resampled, y_train_resampled)
        best_clf = grid.best_estimator_
        best_clf = best_clf.fit(X_train_resampled, y_train_resampled)
        y_pred = best_clf.predict(X_test)
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
        # SHAP Explainer 
        explainer = shap.Explainer(best_clf)
        shap_values = explainer(X_test)


        # Visualize the SHAP values for the first prediction (USE FOR XGB)
        shap.plots.waterfall(shap_values[0])
        
        # Beeswarm plot for all observations
        shap.plots.beeswarm(shap_values)
        
        # Summary plot for all observations
        shap.summary_plot(shap_values, X_test)

        
        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        auroc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        prediction_rate = (conf_matrix[1, 1] + conf_matrix[0, 1]) / (conf_matrix[1, 1] + conf_matrix[1, 0] + conf_matrix[0, 1] + conf_matrix[0, 0])

        accuracy_scores.append(accuracy)
        weighted_f1_scores.append(weighted_f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        auroc_scores.append(auroc)
        conf_matrices.append(conf_matrix)
        prediction_rates.append(prediction_rate)

        auroc_score_barplot[classifier_name].append(auroc) 
        
    avg_accuracy = np.mean(accuracy_scores)
    avg_weighted_f1 = np.mean(weighted_f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_auroc = np.mean(auroc_scores)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)
    avg_prediction_rate = np.mean(prediction_rates)

    print(f"{classifier_name} Results:")
    print(f"Avg Accuracy: {avg_accuracy:.2f}")
    print(f"Avg Weighted F1 Score: {avg_weighted_f1:.2f}")
    print(f"Avg Precision_1: {avg_precision:.2f}")
    print(f"Avg Recall_1: {avg_recall:.2f}")
    print(f"Avg AUROC: {avg_auroc:.2f}")
    print("Avg Confusion Matrix:")
    print(avg_conf_matrix)
    print(f"Avg Prediction Rate: {avg_prediction_rate:.2f}\n")

    precision_dict[classifier_name] = avg_precision
    recall_dict[classifier_name] = avg_recall

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: clf}

#%%
# NLP, GENERALIZED ANXIETY DISORDER, XGB
df = df_d5_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

clf = XGBClassifier()
param_grid_XGB = {
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [10, 20],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
    }
precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_XGB, clf)

# %%
# SHAP FUNCTION - SVM
def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    
    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}

    # Define the number of folds for cross-validation
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    accuracy_scores = []
    weighted_f1_scores = []
    precision_scores = []
    recall_scores = []
    auroc_scores = []
    conf_matrices = []

    # Perform cross-validation
    for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Perform oversampling on minority class
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Train the model
        grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_resampled, y_train_resampled)
        best_clf = grid.best_estimator_
        best_clf = best_clf.fit(X_train_resampled, y_train_resampled)
        y_pred = best_clf.predict(X_test)
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
        # SHAP Explainer
        explainer = shap.KernelExplainer(best_clf.predict, X_train_resampled)
        shap_values = explainer.shap_values(X_test)

        # # Beeswarm plot for all observations
        # shap.plots.beeswarm(shap_values)
        
        # Summary plot for all observations
        shap.summary_plot(shap_values, X_test)
 

        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        auroc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)

        accuracy_scores.append(accuracy)
        weighted_f1_scores.append(weighted_f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        auroc_scores.append(auroc)
        conf_matrices.append(conf_matrix)

        auroc_score_barplot[classifier_name].append(auroc)

    avg_accuracy = np.mean(accuracy_scores)
    avg_weighted_f1 = np.mean(weighted_f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_auroc = np.mean(auroc_scores)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print(f"{classifier_name} Results:")
    print(f"Avg Accuracy: {avg_accuracy:.2f}")
    print(f"Avg Weighted F1 Score: {avg_weighted_f1:.2f}")
    print(f"Avg Precision_1: {avg_precision:.2f}")
    print(f"Avg Recall_1: {avg_recall:.2f}")
    print(f"Avg AUROC: {avg_auroc:.2f}")
    print("Avg Confusion Matrix:")
    print(avg_conf_matrix)

    precision_dict[classifier_name] = avg_precision
    recall_dict[classifier_name] = avg_recall

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: clf}

#%%
# GAZE+FIX+NLP, DYSLEXIA, SVM
df = df_d3_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]
clf = SVC(probability=True)
param_grid_SVM = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']
    }

precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_SVM, clf)

#--------------------------------------------------------------------
# %%
# SHAP plots for a single fold

# SHAP FUNCTION - RF
import shap
from collections import defaultdict
from sklearn.model_selection import GroupKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    groups = df['Participant']

    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Perform oversampling on the minority class using SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Perform hyperparameter tuning using GridSearchCV
    grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)
    best_clf = grid.best_estimator_

    # Train the best model on the resampled training data
    best_clf.fit(X_train_resampled, y_train_resampled)

    # Predict on the test set
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
    # SHAP Explainer for model interpretation
    explainer = shap.Explainer(best_clf)
    shap_values = explainer(X_test)
        
    # Visualize the SHAP values
    shap.plots.waterfall(shap_values[0][:, 1])  # Waterfall plot for the first prediction
    shap.plots.beeswarm(shap_values[:, :, 1])  # Beeswarm plot for all observations
    shap.summary_plot(shap_values[:, :, 1], X_test)  # Summary plot for all observations

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    prediction_rate = (conf_matrix[1, 1] + conf_matrix[0, 1]) / np.sum(conf_matrix)

    # Store metrics
    precision_dict[classifier_name] = precision
    recall_dict[classifier_name] = recall
    auroc_score_barplot[classifier_name].append(auroc)

    # Print average results
    print(f"{classifier_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted F1 Score: {weighted_f1:.2f}")
    print(f"Precision_1: {precision:.2f}")
    print(f"Recall_1: {recall:.2f}")
    print(f"AUROC: {auroc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Prediction Rate: {prediction_rate:.2f}\n")

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: best_clf}

#%%
# NLP, ALL, RF
df = df_all_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}

precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)


#%%
# NLP, OTHER LANGUAGE/ READING.., RF
df = df_d4_clean
# Mapping of abbreviated feature names to full names
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}

precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)

# %%
# SHAP FUNCTION - XGB
def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    groups = df['Participant']

    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Perform oversampling on the minority class using SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Perform hyperparameter tuning using GridSearchCV
    grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)
    best_clf = grid.best_estimator_

    # Train the best model on the resampled training data
    best_clf.fit(X_train_resampled, y_train_resampled)

    # Predict on the test set
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
    # SHAP Explainer for model interpretation
    explainer = shap.Explainer(best_clf)
    shap_values = explainer(X_test)
        
    # Visualize the SHAP values for the first prediction (USE FOR XGB)
    shap.plots.waterfall(shap_values[0])
        
    # Beeswarm plot for all observations
    shap.plots.beeswarm(shap_values)
        
    # Summary plot for all observations
    shap.summary_plot(shap_values, X_test)


    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    prediction_rate = (conf_matrix[1, 1] + conf_matrix[0, 1]) / np.sum(conf_matrix)

    # Store metrics
    precision_dict[classifier_name] = precision
    recall_dict[classifier_name] = recall
    auroc_score_barplot[classifier_name].append(auroc)

    # Print average results
    print(f"{classifier_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted F1 Score: {weighted_f1:.2f}")
    print(f"Precision_1: {precision:.2f}")
    print(f"Recall_1: {recall:.2f}")
    print(f"AUROC: {auroc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Prediction Rate: {prediction_rate:.2f}\n")

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: best_clf}


#%%
# NLP, NEUROTYPICAL, XGB
df = df_typical_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

clf = XGBClassifier()
param_grid_XGB = {
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [10, 20],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
    }
precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_XGB, clf)

# %%
import matplotlib.pyplot as plt

# Data from the table
diagnoses = [
    "All Neurodivergent \nParticipants (N=176)",
    "ADD/ADHD (N=45)", 
    "Autism/Asperger's/ASD (N=30)", 
    "Dyslexia/Dyspraxia/Dyscalculia/\nDysgraphia (N=25)", 
    "Other language/reading/math/\nnon-verbal learning disorders (N=20)", 
    "Generalized Anxiety Disorder (N=35)", 
    "Other (N=21)"
]
auroc_scores = [0.57, 0.61, 0.59, 0.65, 0.70, 0.59, 0.62]

# Define the colors
colors = ['blue'] + ['green'] * (len(diagnoses) - 1)

# Creating the bar chart with different colors
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(diagnoses, auroc_scores, color=colors)

# Set x-axis limits to start at 0.5
ax.set_xlim(0.5, 0.75)

# Adding title and labels
ax.set_title('AUROC Scores for Predicting Mind Wandering in Neurodivergent Students by Diagnosis', fontsize = 20, loc='right')
ax.set_xlabel('AUROC', fontsize = 20)
ax.set_ylabel('Diagnoses', fontsize = 20)

ax.tick_params(axis='y', labelsize=16)

# Invert y-axis for better readability
ax.invert_yaxis()

# Display the chart
plt.show()


# %%
# SHAP FUNCTION FOR ALL RF MODELS
def train_and_evaluate(df, features, param_grid, classifier_name):
    X = df[features]
    y = df['TUT']
    groups = df['Participant']

    precision_dict = {}
    recall_dict = {}
    auroc_score_barplot = {classifier_name: []}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Perform oversampling on the minority class using SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Perform hyperparameter tuning using GridSearchCV
    grid = GridSearchCV(clf, param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)
    best_clf = grid.best_estimator_

    # Train the best model on the resampled training data
    best_clf.fit(X_train_resampled, y_train_resampled)

    # Predict on the test set
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        
    # SHAP Explainer for model interpretation
    explainer = shap.Explainer(best_clf)
    shap_values = explainer(X_test)
        
    # Visualize the SHAP values
    shap.plots.waterfall(shap_values[0][:, 1])  # Waterfall plot for the first prediction
    shap.plots.beeswarm(shap_values[:, :, 1])  # Beeswarm plot for all observations
    shap.summary_plot(shap_values[:, :, 1], X_test)  # Summary plot for all observations

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    prediction_rate = (conf_matrix[1, 1] + conf_matrix[0, 1]) / np.sum(conf_matrix)

    # Store metrics
    precision_dict[classifier_name] = precision
    recall_dict[classifier_name] = recall
    auroc_score_barplot[classifier_name].append(auroc)

    # Print average results
    print(f"{classifier_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted F1 Score: {weighted_f1:.2f}")
    print(f"Precision_1: {precision:.2f}")
    print(f"Recall_1: {recall:.2f}")
    print(f"AUROC: {auroc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Prediction Rate: {prediction_rate:.2f}\n")

    return precision_dict, recall_dict, auroc_score_barplot, {classifier_name: best_clf}


#%%
# GAZE + FIXATION + NLP, ALL, RF
df = df_all_clean
features = ["PP_Gazes","PP_AOI_Gazes","PP_OffscreenPix","PP_OffscreenProp","PP_2_Gazes","PP_2_AOI_Gazes",
            "PP_2_OffscreenPix","PP_2_OffscreenProp","PP_3_Gazes","PP_3_AOI_Gazes","PP_3_OffscreenPix",
            "PP_3_OffscreenProp","PP_cluster_num_clusters","PP_cluster_avg_duration","PP_cluster_sd_duration",
            "PP_cluster_skew_duration","PP_dispersion","PP_2_cluster_num_clusters","PP_2_cluster_avg_duration",
            "PP_2_cluster_sd_duration","PP_2_cluster_skew_duration","PP_2_dispersion","PP_3_cluster_num_clusters",
            "PP_3_cluster_avg_duration","PP_3_cluster_sd_duration","PP_3_cluster_skew_duration","PP_3_dispersion",
            "wordCount_with_stopwords","syllableCount_with_stopwords","ease_of_reading","wordCount_without_stopwords",
            "syllableCount_without_stopwords","sentiment","PP_CD","PP_NN","PP_JJ","PP_VBN","PP_VBP","PP_NNS","PP_VBD",
            "PP_VBG","PP_DT","PP_IN","PP_JJS","PP_MD","PP_VB","PP_RB","PP_VBZ","PP_RBS","PP_JJR","PP_WRB","PP_RBR",
            "PP_2_CD","PP_2_NN","PP_2_JJ","PP_2_VBN","PP_2_VBP","PP_2_NNS","PP_2_VBD","PP_2_VBG","PP_2_DT","PP_2_IN",
            "PP_2_JJS","PP_2_MD","PP_2_VB","PP_2_RB","PP_2_VBZ","PP_2_RBS","PP_2_JJR","PP_2_WRB","PP_2_RBR","PP_3_CD",
            "PP_3_NN","PP_3_JJ","PP_3_VBN","PP_3_VBP","PP_3_NNS","PP_3_VBD","PP_3_VBG","PP_3_DT","PP_3_IN","PP_3_JJS",
            "PP_3_MD","PP_3_VB","PP_3_RB","PP_3_VBZ","PP_3_RBS","PP_3_JJR","PP_3_WRB","PP_3_RBR"
]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}
precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)


#%%
# GAZE + FIXATION + NLP, DIVERGENT, RF
df = df_divergent_clean
features = ["PP_Gazes","PP_AOI_Gazes","PP_OffscreenPix","PP_OffscreenProp","PP_2_Gazes","PP_2_AOI_Gazes",
            "PP_2_OffscreenPix","PP_2_OffscreenProp","PP_3_Gazes","PP_3_AOI_Gazes","PP_3_OffscreenPix",
            "PP_3_OffscreenProp","PP_cluster_num_clusters","PP_cluster_avg_duration","PP_cluster_sd_duration",
            "PP_cluster_skew_duration","PP_dispersion","PP_2_cluster_num_clusters","PP_2_cluster_avg_duration",
            "PP_2_cluster_sd_duration","PP_2_cluster_skew_duration","PP_2_dispersion","PP_3_cluster_num_clusters",
            "PP_3_cluster_avg_duration","PP_3_cluster_sd_duration","PP_3_cluster_skew_duration","PP_3_dispersion",
            "wordCount_with_stopwords","syllableCount_with_stopwords","ease_of_reading","wordCount_without_stopwords",
            "syllableCount_without_stopwords","sentiment","PP_CD","PP_NN","PP_JJ","PP_VBN","PP_VBP","PP_NNS","PP_VBD",
            "PP_VBG","PP_DT","PP_IN","PP_JJS","PP_MD","PP_VB","PP_RB","PP_VBZ","PP_RBS","PP_JJR","PP_WRB","PP_RBR",
            "PP_2_CD","PP_2_NN","PP_2_JJ","PP_2_VBN","PP_2_VBP","PP_2_NNS","PP_2_VBD","PP_2_VBG","PP_2_DT","PP_2_IN",
            "PP_2_JJS","PP_2_MD","PP_2_VB","PP_2_RB","PP_2_VBZ","PP_2_RBS","PP_2_JJR","PP_2_WRB","PP_2_RBR","PP_3_CD",
            "PP_3_NN","PP_3_JJ","PP_3_VBN","PP_3_VBP","PP_3_NNS","PP_3_VBD","PP_3_VBG","PP_3_DT","PP_3_IN","PP_3_JJS",
            "PP_3_MD","PP_3_VB","PP_3_RB","PP_3_VBZ","PP_3_RBS","PP_3_JJR","PP_3_WRB","PP_3_RBR"
]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}
precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)


#%%
# GAZE + FIXATION + NLP, DYSLEXIA, RF
df = df_d3_clean
features = ["PP_Gazes","PP_AOI_Gazes","PP_OffscreenPix","PP_OffscreenProp","PP_2_Gazes","PP_2_AOI_Gazes",
            "PP_2_OffscreenPix","PP_2_OffscreenProp","PP_3_Gazes","PP_3_AOI_Gazes","PP_3_OffscreenPix",
            "PP_3_OffscreenProp","PP_cluster_num_clusters","PP_cluster_avg_duration","PP_cluster_sd_duration",
            "PP_cluster_skew_duration","PP_dispersion","PP_2_cluster_num_clusters","PP_2_cluster_avg_duration",
            "PP_2_cluster_sd_duration","PP_2_cluster_skew_duration","PP_2_dispersion","PP_3_cluster_num_clusters",
            "PP_3_cluster_avg_duration","PP_3_cluster_sd_duration","PP_3_cluster_skew_duration","PP_3_dispersion",
            "wordCount_with_stopwords","syllableCount_with_stopwords","ease_of_reading","wordCount_without_stopwords",
            "syllableCount_without_stopwords","sentiment","PP_CD","PP_NN","PP_JJ","PP_VBN","PP_VBP","PP_NNS","PP_VBD",
            "PP_VBG","PP_DT","PP_IN","PP_JJS","PP_MD","PP_VB","PP_RB","PP_VBZ","PP_RBS","PP_JJR","PP_WRB","PP_RBR",
            "PP_2_CD","PP_2_NN","PP_2_JJ","PP_2_VBN","PP_2_VBP","PP_2_NNS","PP_2_VBD","PP_2_VBG","PP_2_DT","PP_2_IN",
            "PP_2_JJS","PP_2_MD","PP_2_VB","PP_2_RB","PP_2_VBZ","PP_2_RBS","PP_2_JJR","PP_2_WRB","PP_2_RBR","PP_3_CD",
            "PP_3_NN","PP_3_JJ","PP_3_VBN","PP_3_VBP","PP_3_NNS","PP_3_VBD","PP_3_VBG","PP_3_DT","PP_3_IN","PP_3_JJS",
            "PP_3_MD","PP_3_VB","PP_3_RB","PP_3_VBZ","PP_3_RBS","PP_3_JJR","PP_3_WRB","PP_3_RBR"
]

clf = RandomForestClassifier()
param_grid_RF = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200],
}
precision_dict, recall_dict, auroc_score_barplot, classifiers = train_and_evaluate(df, features, param_grid_RF, clf)


#%%
# TRAIN ON ALL, TEST ON NEUROTYPICAL/NEURODIVERGENT
def train_and_evaluate(df_train, df_test, features):
    X_train = df_train[features]
    y_train = df_train['TUT']
    X_test = df_test[features]
    y_test = df_test['TUT']
    
    # Define classifiers and parameter grids
    classifiers = {
        'Chance': DummyClassifier(strategy="stratified"),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(random_state=42, probability=True), 
        "XGBoost": xgb.XGBClassifier(random_state=42)
    }

    param_grid_RF = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [50, 100, 200],
    }

    param_grid_SVM = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']
    }

    param_grid_XGB = {
        'colsample_bytree': [0.8, 1.0],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [10, 20],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
    }

    param_grids = {
        'Random Forest': param_grid_RF,
        'SVM': param_grid_SVM,
        'XGBoost': param_grid_XGB
    }

    precision_dict = {}
    recall_dict = {}
    kappa_dict = {}
    auroc_score_barplot = {name: [] for name in classifiers}

    # Fit PCA to determine the number of components that explain 95% of the variance on the training data
    pca = PCA().fit(X_train)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1

    # Apply PCA to the training data
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(X_train_pca.shape[1])])

    # Apply the transformation to the test data
    X_test_pca = pca.transform(X_test)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i}' for i in range(X_test_pca.shape[1])])

    # Perform oversampling on minority class in the training data
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca_df, y_train)

    # Loop through the classifiers
    for name, clf in classifiers.items():
        if name != "Chance":
            grid = GridSearchCV(clf, param_grids[name], refit=True, verbose=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train_resampled, y_train_resampled)
            best_clf = grid.best_estimator_
        else:
            best_clf = clf.fit(X_train, y_train)

        # Fit the best classifier on the resampled training data
        best_clf = best_clf.fit(X_train_resampled, y_train_resampled)

        # Predict on the test data
        y_pred = best_clf.predict(X_test_pca_df)
        y_pred_proba = best_clf.predict_proba(X_test_pca_df)[:, 1]

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        prediction_rate = (confusion_matrix[1, 1] + confusion_matrix[0, 1]) / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 0])

        # Store results in dictionaries
        precision_dict[name] = precision
        recall_dict[name] = recall
        kappa_dict[name] = kappa
        auroc_score_barplot[name].append(auroc)

        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Weighted F1 Score: {weighted_f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Kappa: {kappa:.2f}")
        print(f"AUROC: {auroc:.2f}")
        print(f"Prediction Rate: {prediction_rate:.2f}")
        print(f"Confusion Matrix:\n{confusion_matrix}\n")

    return precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers

def plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers):
    colors = ['#003F5C', '#FFA600', '#7A5195', '#D62728']

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].bar(precision_dict.keys(), precision_dict.values(), color=colors)
    ax[0].set_title('Precision Scores by Classifier')
    ax[0].set_xlabel('Classifier')
    ax[0].set_ylabel('Precision Score')
    ax[0].set_ylim([0, 1])

    ax[1].bar(recall_dict.keys(), recall_dict.values(), color=colors)
    ax[1].set_title('Recall Scores by Classifier')
    ax[1].set_xlabel('Classifier')
    ax[1].set_ylabel('Recall Score')
    ax[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 7))
    bar_width = 0.15  
    gap_width = 0.05 

    bar_positions = np.arange(len(classifiers)) * bar_width * 2

    for idx, name in enumerate(classifiers):
        scores = auroc_score_barplot[name]
        avg_score = np.mean(scores)
        for fold_idx, score in enumerate(scores):
            bar_pos = bar_positions[idx] + (fold_idx * bar_width)
            ax.bar(bar_pos, score, bar_width, label=name if fold_idx == 0 else "",
                   color=colors[idx], edgecolor='white', linewidth=0.1, alpha=1)
        ax.axhline(y=avg_score, color=colors[idx], linestyle='dashed', linewidth=1.5)
    
    ax.set_xlabel('Classifiers', fontsize=15)
    ax.set_ylabel('AUROC Score', fontsize=15)
    ax.set_title('AUROC Scores by Classifier', fontsize=17)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(classifiers.keys())

    plt.tight_layout()
    plt.show()


#%%
# MODELING NEUROTYPICAL PARTICIPANTS
# NLP
df_train = df_all_clean  
df_test = df_typical_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df_train = df_all_clean  
df_test = df_typical_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df_train = df_all_clean  
df_test = df_typical_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)



#%%
# MODELING NEURODIVERGENT PARTICIPANTS
# NLP
df_train = df_all_clean  
df_test = df_divergent_clean
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation
df_train = df_all_clean  
df_test = df_divergent_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)

# %%
# Gaze + Fixation + NLP
df_train = df_all_clean  
df_test = df_divergent_clean
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"
]

precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers = train_and_evaluate(df_train, df_test, features)
plot_metrics(precision_dict, recall_dict, kappa_dict, auroc_score_barplot, classifiers)


# %%
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score

# Create directory for storing fold predictions
if not os.path.exists('fold_predictions'):
    os.makedirs('fold_predictions')

# FUNCTION FOR MODELING
def train_and_evaluate(df, features, n_splits=5):
    X = df[features]  # Use only feature columns
    y = df['TUT']
    
    # Define classifiers 
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    gkf = GroupKFold(n_splits=n_splits)
    
    for name, clf in classifiers.items():
        fold_counter = 1
        for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            test_participants = df.iloc[test_index]['Participant']
            test_paragraphs = df.iloc[test_index]['Paragraph']
            
            # Train the classifier
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            # Save the fold predictions to a file (without Condition and Diagnoses)
            fold_predictions = pd.DataFrame({
                'Participant': test_participants,
                'Paragraph': test_paragraphs,
                'Prediction': y_pred,
                'Prediction_Proba': y_pred_proba,
                'True_Label': y_test
            })
            fold_predictions.to_csv(f'fold_predictions/predictions_fold_{fold_counter}.csv', index=False)
            fold_counter += 1

    print(f"All fold predictions saved to 'fold_predictions/' directory.")

# Merging predictions with the original file and extracting Condition and Diagnoses
def merge_predictions_with_original(df, prediction_files):
    all_predictions = pd.concat([pd.read_csv(file) for file in prediction_files], ignore_index=True)

    # Merge with the original file (excluding condition and diagnosis columns)
    merged_df = df[['Participant', 'Paragraph', 'Condition']].merge(all_predictions, on=['Participant', 'Paragraph'], how='left')

    # For neurodivergent condition, extract diagnosis columns (D1 to D8)
    neurodivergent_df = df[df['Condition'] == 'Neurodivergent'][['Participant', 'Paragraph', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']]

    # Merge diagnosis columns for neurodivergent participants
    merged_df = merged_df.merge(neurodivergent_df, on=['Participant', 'Paragraph'], how='left')

    return merged_df

# Splitting by condition and diagnosis, calculating AUROC and Kappa
def evaluate_by_conditions(merged_df):
    conditions = ['Neurotypical', 'Neurodivergent']
    
    for condition in conditions:
        condition_df = merged_df[merged_df['Condition'] == condition]

        # Calculate AUROC and Kappa for the condition
        if len(condition_df) > 0:
            auroc = roc_auc_score(condition_df['True_Label'], condition_df['Prediction_Proba'])
            kappa = cohen_kappa_score(condition_df['True_Label'], condition_df['Prediction'])
            
            print(f"Condition: {condition}")
            print(f"  AUROC: {auroc:.2f}")
            print(f"  Kappa: {kappa:.2f}")
        
        # If Neurodivergent, further split by diagnosis using D1-D8
        if condition == 'Neurodivergent':
            for diagnosis in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']:
                diagnosis_df = condition_df[condition_df[diagnosis] == 1]
                if len(diagnosis_df) > 0:
                    auroc = roc_auc_score(diagnosis_df['True_Label'], diagnosis_df['Prediction_Proba'])
                    kappa = cohen_kappa_score(diagnosis_df['True_Label'], diagnosis_df['Prediction'])
                    
                    print(f"  Diagnosis: {diagnosis}")
                    print(f"    AUROC: {auroc:.2f}")
                    print(f"    Kappa: {kappa:.2f}")

# Main execution
df = df_all_data_no_null
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

# Train model and save predictions for each fold
train_and_evaluate(df, features)

# Collect prediction file paths
prediction_files = [f'fold_predictions/predictions_fold_{i}.csv' for i in range(1, 6)]

# Merge predictions with original data and extract neurodivergent diagnoses
merged_df = merge_predictions_with_original(df, prediction_files)

# Evaluate by conditions and diagnoses
evaluate_by_conditions(merged_df)


# %%
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Create directory for storing fold predictions
if not os.path.exists('fold_predictions'):
    os.makedirs('fold_predictions')

# FUNCTION FOR MODELING
def train_and_evaluate(df, features, n_splits=5):
    X = df[features]  # Use only feature columns
    y = df['TUT']

    # Perform PCA to select the number of components explaining 95% of the variance
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1  # Number of components explaining 95% of variance
    print(f"Selected number of PCA components: {n_components}")

    # Redefine PCA with selected number of components
    pca = PCA(n_components=n_components)

    # Define classifiers and their parameter grids for hyperparameter tuning
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }

    param_grids = {
        'RandomForest': {
            'criterion': ['entropy', 'gini'],
            'max_depth': [10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [50, 100, 200],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear']
        },
        'XGBoost': {
            'colsample_bytree': [0.8, 1.0],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [10, 20],
            'min_child_weight': [1, 3, 5],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
        }
    }

    # SMOTE for oversampling the minority class
    smote = SMOTE()

    gkf = GroupKFold(n_splits=n_splits)

    fold_counter = 1
    for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_participants = df.iloc[test_index]['Participant']
        test_paragraphs = df.iloc[test_index]['Paragraph']

        # Apply PCA for dimensionality reduction
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Apply SMOTE to balance the training data after PCA
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

        for name, clf in classifiers.items():
            # Perform hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(clf, param_grids[name], scoring='roc_auc', cv=3, n_jobs=-1)
            grid_search.fit(X_train_resampled, y_train_resampled)

            # Get the best model from GridSearchCV
            best_clf = grid_search.best_estimator_

            # Train the best model and make predictions
            best_clf.fit(X_train_resampled, y_train_resampled)
            y_pred = best_clf.predict(X_test_pca)
            y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]

            # Save the fold predictions to a file (without Condition and Diagnoses)
            fold_predictions = pd.DataFrame({
                'Participant': test_participants,
                'Paragraph': test_paragraphs,
                'Classifier': name,
                'Prediction': y_pred,
                'Prediction_Proba': y_pred_proba,
                'True_Label': y_test
            })
            fold_predictions.to_csv(f'fold_predictions/predictions_fold_{fold_counter}_{name}.csv', index=False)

        fold_counter += 1

    print(f"All fold predictions saved to 'fold_predictions/' directory.")

# Merging predictions with the original file and extracting Condition and Diagnoses
def merge_predictions_with_original(df, prediction_files):
    all_predictions = pd.concat([pd.read_csv(file) for file in prediction_files], ignore_index=True)

    # Merge with the original file (excluding condition and diagnosis columns)
    merged_df = df[['Participant', 'Paragraph', 'Condition']].merge(all_predictions, on=['Participant', 'Paragraph'], how='left')

    # For neurodivergent condition, extract diagnosis columns (D1 to D6) and merge D7 and D8
    neurodivergent_df = df[df['Condition'] == 'Neurodivergent'][['Participant', 'Paragraph', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']]
    neurodivergent_df['D7_D8'] = df[['D7', 'D8']].max(axis=1)  # Merge D7 and D8

    # Merge diagnosis columns for neurodivergent participants
    merged_df = merged_df.merge(neurodivergent_df, on=['Participant', 'Paragraph'], how='left')

    return merged_df

# Splitting by condition and diagnosis, calculating AUROC and Kappa
def evaluate_by_conditions(merged_df):
    conditions = ['Neurotypical', 'Neurodivergent']

    for condition in conditions:
        condition_df = merged_df[merged_df['Condition'] == condition]

        # Calculate AUROC and Kappa for the condition
        if len(condition_df) > 0:
            auroc = roc_auc_score(condition_df['True_Label'], condition_df['Prediction_Proba'])
            kappa = cohen_kappa_score(condition_df['True_Label'], condition_df['Prediction'])

            print(f"Condition: {condition}")
            print(f"  AUROC: {auroc:.2f}")
            print(f"  Kappa: {kappa:.2f}")

        # If Neurodivergent, further split by diagnosis using D1-D6 and merged D7_D8
        if condition == 'Neurodivergent':
            for diagnosis in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7_D8']:
                diagnosis_df = condition_df[condition_df[diagnosis] == 1]
                if len(diagnosis_df) > 0:
                    auroc = roc_auc_score(diagnosis_df['True_Label'], diagnosis_df['Prediction_Proba'])
                    kappa = cohen_kappa_score(diagnosis_df['True_Label'], diagnosis_df['Prediction'])

                    print(f"  Diagnosis: {diagnosis}")
                    print(f"    AUROC: {auroc:.2f}")
                    print(f"    Kappa: {kappa:.2f}")
# Main execution
df = df_all_data_no_null
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

# Train model and save predictions for each fold
train_and_evaluate(df, features)

# Collect prediction file paths
prediction_files = [f'fold_predictions/predictions_fold_{i}_{name}.csv' for i in range(1, 6) for name in ['RandomForest', 'SVM', 'XGBoost']]

# Merge predictions with original data and extract neurodivergent diagnoses
merged_df = merge_predictions_with_original(df, prediction_files)

# Evaluate by conditions and diagnoses
evaluate_by_conditions(merged_df)



# %%
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Create directory for storing fold predictions
if not os.path.exists('fold_predictions'):
    os.makedirs('fold_predictions')

# FUNCTION FOR MODELING
def train_and_evaluate(df, features, n_splits=5):
    X = df[features]  # Use only feature columns
    y = df['TUT']

    # Perform PCA to select the number of components explaining 95% of the variance
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1  # Number of components explaining 95% of variance
    print(f"Selected number of PCA components: {n_components}")

    # Redefine PCA with selected number of components
    pca = PCA(n_components=n_components)

    # Define classifiers and their parameter grids for hyperparameter tuning
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }

    param_grids = {
        'RandomForest': {
            'criterion': ['entropy', 'gini'],
            'max_depth': [10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [50, 100, 200],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear']
        },
        'XGBoost': {
            'colsample_bytree': [0.8, 1.0],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [10, 20],
            'min_child_weight': [1, 3, 5],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
        }
    }

    # SMOTE for oversampling the minority class
    smote = SMOTE()

    gkf = GroupKFold(n_splits=n_splits)

    fold_counter = 1
    for train_index, test_index in gkf.split(X, y, groups=df['Participant']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_participants = df.iloc[test_index]['Participant']
        test_paragraphs = df.iloc[test_index]['Paragraph']

        # Apply PCA for dimensionality reduction
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Apply SMOTE to balance the training data after PCA
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

        for name, clf in classifiers.items():
            # Perform hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(clf, param_grids[name], scoring='roc_auc', cv=3, n_jobs=-1)
            grid_search.fit(X_train_resampled, y_train_resampled)

            # Get the best model from GridSearchCV
            best_clf = grid_search.best_estimator_

            # Train the best model and make predictions
            best_clf.fit(X_train_resampled, y_train_resampled)
            y_pred = best_clf.predict(X_test_pca)
            y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]

            # Save the fold predictions to a file (without Condition and Diagnoses)
            fold_predictions = pd.DataFrame({
                'Participant': test_participants,
                'Paragraph': test_paragraphs,
                'Classifier': name,
                'Prediction': y_pred,
                'Prediction_Proba': y_pred_proba,
                'True_Label': y_test
            })
            fold_predictions.to_csv(f'fold_predictions/predictions_fold_{fold_counter}_{name}.csv', index=False)

        fold_counter += 1

    print(f"All fold predictions saved to 'fold_predictions/' directory.")

# Merging predictions with the original file and extracting Condition and Diagnoses
def merge_predictions_with_original(df, model_name):
    # Get all prediction files for the given model
    prediction_files = [f'fold_predictions/predictions_fold_{i}_{model_name}.csv' for i in range(1, 6)]
    all_predictions = pd.concat([pd.read_csv(file) for file in prediction_files], ignore_index=True)

    # Merge with the original file (excluding condition and diagnosis columns)
    merged_df = df[['Participant', 'Paragraph', 'Condition']].merge(all_predictions, on=['Participant', 'Paragraph'], how='left')

    # For neurodivergent condition, extract diagnosis columns (D1 to D6) and merge D7 and D8
    neurodivergent_df = df[df['Condition'] == 'Neurodivergent'][['Participant', 'Paragraph', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']]
    neurodivergent_df['D7_D8'] = df[['D7', 'D8']].max(axis=1)  # Merge D7 and D8

    # Merge diagnosis columns for neurodivergent participants
    merged_df = merged_df.merge(neurodivergent_df, on=['Participant', 'Paragraph'], how='left')

    return merged_df

# Splitting by condition and diagnosis, calculating AUROC and Kappa
def evaluate_by_conditions(merged_df, model_name):
    print(f"\nEvaluating for model: {model_name}")
    conditions = ['Neurotypical', 'Neurodivergent']

    for condition in conditions:
        condition_df = merged_df[merged_df['Condition'] == condition]

        # Calculate AUROC and Kappa for the condition
        if len(condition_df) > 0:
            auroc = roc_auc_score(condition_df['True_Label'], condition_df['Prediction_Proba'])
            kappa = cohen_kappa_score(condition_df['True_Label'], condition_df['Prediction'])

            print(f"Condition: {condition}")
            print(f"  AUROC: {auroc:.2f}")
            print(f"  Kappa: {kappa:.2f}")

        # If Neurodivergent, further split by diagnosis using D1-D6 and merged D7_D8
        if condition == 'Neurodivergent':
            for diagnosis in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7_D8']:
                diagnosis_df = condition_df[condition_df[diagnosis] == 1]
                if len(diagnosis_df) > 0:
                    auroc = roc_auc_score(diagnosis_df['True_Label'], diagnosis_df['Prediction_Proba'])
                    kappa = cohen_kappa_score(diagnosis_df['True_Label'], diagnosis_df['Prediction'])

                    print(f"  Diagnosis: {diagnosis}")
                    print(f"    AUROC: {auroc:.2f}")
                    print(f"    Kappa: {kappa:.2f}")

#%%
#GAZE + FIX + NLP
# Main execution
df = df_all_data_no_null
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            "wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

# Train model and save predictions for each fold
train_and_evaluate(df, features)

# List of models
models = ['RandomForest', 'SVM', 'XGBoost']

# Evaluate each model
for model in models:
    # Merge predictions with original data for each model
    merged_df = merge_predictions_with_original(df, model)

    # Evaluate by conditions and diagnoses for each model
    evaluate_by_conditions(merged_df, model)

# %%
#GAZE + FIX
# Main execution
df = df_all_data_no_null
features = ["PP_Gazes", "PP_AOI_Gazes", "PP_OffscreenPix", "PP_OffscreenProp", "PP_2_Gazes", "PP_2_AOI_Gazes", 
            "PP_2_OffscreenPix", "PP_2_OffscreenProp", "PP_3_Gazes", "PP_3_AOI_Gazes", "PP_3_OffscreenPix", 
            "PP_3_OffscreenProp", "PP_cluster_num_clusters", "PP_cluster_avg_duration", "PP_cluster_sd_duration", 
            "PP_cluster_skew_duration", "PP_dispersion", "PP_2_cluster_num_clusters", "PP_2_cluster_avg_duration", 
            "PP_2_cluster_sd_duration", "PP_2_cluster_skew_duration", "PP_2_dispersion", "PP_3_cluster_num_clusters", 
            "PP_3_cluster_avg_duration", "PP_3_cluster_sd_duration", "PP_3_cluster_skew_duration", "PP_3_dispersion", 
            ]

# Train model and save predictions for each fold
train_and_evaluate(df, features)

# List of models
models = ['RandomForest', 'SVM', 'XGBoost']

# Evaluate each model
for model in models:
    # Merge predictions with original data for each model
    merged_df = merge_predictions_with_original(df, model)

    # Evaluate by conditions and diagnoses for each model
    evaluate_by_conditions(merged_df, model)

#%%
#NLP
# Main execution
df = df_all_data_no_null
features = ["wordCount_with_stopwords", "syllableCount_with_stopwords", "ease_of_reading", "wordCount_without_stopwords", 
            "syllableCount_without_stopwords", "sentiment", "PP_CD", "PP_NN", "PP_JJ", "PP_VBN", "PP_VBP", "PP_NNS", "PP_VBD", 
            "PP_VBG", "PP_DT", "PP_IN", "PP_JJS", "PP_MD", "PP_VB", "PP_RB", "PP_VBZ", "PP_RBS", "PP_JJR", "PP_WRB", "PP_RBR", 
            "PP_2_CD", "PP_2_NN", "PP_2_JJ", "PP_2_VBN", "PP_2_VBP", "PP_2_NNS", "PP_2_VBD", "PP_2_VBG", "PP_2_DT", "PP_2_IN", 
            "PP_2_JJS", "PP_2_MD", "PP_2_VB", "PP_2_RB", "PP_2_VBZ", "PP_2_RBS", "PP_2_JJR", "PP_2_WRB", "PP_2_RBR", "PP_3_CD", 
            "PP_3_NN", "PP_3_JJ", "PP_3_VBN", "PP_3_VBP", "PP_3_NNS", "PP_3_VBD", "PP_3_VBG", "PP_3_DT", "PP_3_IN", "PP_3_JJS", 
            "PP_3_MD", "PP_3_VB", "PP_3_RB", "PP_3_VBZ", "PP_3_RBS", "PP_3_JJR", "PP_3_WRB", "PP_3_RBR"]

# Train model and save predictions for each fold
train_and_evaluate(df, features)

# List of models
models = ['RandomForest', 'SVM', 'XGBoost']

# Evaluate each model
for model in models:
    # Merge predictions with original data for each model
    merged_df = merge_predictions_with_original(df, model)

    # Evaluate by conditions and diagnoses for each model
    evaluate_by_conditions(merged_df, model)

# %%
