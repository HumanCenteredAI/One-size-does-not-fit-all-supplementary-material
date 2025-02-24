#%%
#IMPORT LIBRARIES
import pandas as pd
import numpy as np

#%%
# FUNCTION FOR FINDING BEST MODEL AND BEST FEATURE COMBINATION
def evaluate_models(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    # Assign weights to each metric
    weights = {
#         'Accuracy': 0.2,
        'Weighted F1': 0.23,
        'Precision_1': 0.23,
        'Recall_1': 0.23,
        'AUROC': 0.31
    }

    df_copy = df.copy()
    metrics = ['Weighted F1', 'Precision_1', 'Recall_1', 'AUROC']

    # Calculate the composite score for each model
    df_copy['Composite Score'] = 0
    for metric in metrics:
        df_copy['Composite Score'] += df_copy[metric] * weights[metric]

    # Get the composite score for the chance model
    chance_score = df_copy[df_copy['Model'] == 'Chance']['Composite Score'].values[0]
    
    # Filter out the chance model and sort the remaining models by Composite Score in descending order
    sorted_df = df_copy[df_copy['Model'] != 'Chance'].sort_values(by='Composite Score', ascending=False)

    # Display the results
    print(sorted_df[['Model', 'Composite Score']])

    # Determine the best model that performs better than chance
    better_than_chance = sorted_df[sorted_df['Composite Score'] > chance_score]
    
    if not better_than_chance.empty:
        best_model = better_than_chance.iloc[0]['Model']
        print(f"The best model that performs better than chance is: {best_model}")
    else:
        best_model = sorted_df.iloc[0]['Model']
        print(f"No model performs better than chance. The model with the highest composite score is: {best_model}")

#%%
# ALL PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.55, 0.54, 0.56, 0.55],
    'Weighted F1': [0.56, 0.55, 0.57, 0.56],
    'Precision_1': [0.32, 0.37, 0.36, 0.37],
    'Recall_1': [0.35, 0.55, 0.49, 0.52],
    'Kappa': [0.00, 0.08, 0.07, 0.08],
    'AUROC': [0.50, 0.57, 0.53, 0.56],
    'Prediction Rate': [0.35, 0.49, 0.44, 0.46]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.56, 0.59, 0.56, 0.57],
    'Weighted F1': [0.56, 0.57, 0.57, 0.56],
    'Precision_1': [0.33, 0.33, 0.35, 0.32],
    'Recall_1': [0.34, 0.27, 0.44, 0.28],
    'Kappa': [0.01, 0.01, 0.05, -0.01],
    'AUROC': [0.48, 0.50, 0.52, 0.50],
    'Prediction Rate': [0.34, 0.26, 0.40, 0.29]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.56, 0.57, 0.53, 0.56],
    'Weighted F1': [0.56, 0.57, 0.54, 0.56],
    'Precision_1': [0.33, 0.33, 0.35, 0.32],
    'Recall_1': [0.33, 0.31, 0.54, 0.30],
    'Kappa': [0.01, 0.00, 0.06, -0.01],
    'AUROC': [0.49, 0.52, 0.54, 0.50],
    'Prediction Rate': [0.32, 0.30, 0.50, 0.31]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# NEUROTYPICAL PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.61, 0.48, 0.52, 0.47],
    'Weighted F1': [0.61, 0.51, 0.54, 0.50],
    'Precision_1': [0.26, 0.28, 0.29, 0.28],
    'Recall_1': [0.26, 0.60, 0.58, 0.65],
    'Kappa': [-0.01, 0.02, 0.05, 0.04],
    'AUROC': [0.53, 0.53, 0.49, 0.52],
    'Prediction Rate': [0.27, 0.57, 0.52, 0.60]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.60, 0.63, 0.49, 0.61],
    'Weighted F1': [0.61, 0.61, 0.52, 0.60],
    'Precision_1': [0.28, 0.24, 0.27, 0.23],
    'Recall_1': [0.33, 0.19, 0.52, 0.19],
    'Kappa': [0.02, -0.03, 0.01, -0.05],
    'AUROC': [0.47, 0.47, 0.48, 0.47],
    'Prediction Rate': [0.30, 0.21, 0.52, 0.22]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.59, 0.61, 0.52, 0.62],
    'Weighted F1': [0.59, 0.61, 0.54, 0.62],
    'Precision_1': [0.23, 0.26, 0.29, 0.28],
    'Recall_1': [0.24, 0.25, 0.57, 0.28],
    'Kappa': [-0.04, -0.01, 0.05, 0.02],
    'AUROC': [0.49, 0.48, 0.49, 0.49],
    'Prediction Rate': [0.27, 0.26, 0.52, 0.27]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# NEURODIVERGENT PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.51, 0.52, 0.56],
    'Weighted F1': [0.52, 0.50, 0.52, 0.56],
    'Precision_1': [0.39, 0.41, 0.41, 0.45],
    'Recall_1': [0.40, 0.59, 0.53, 0.53],
    'Kappa': [-0.01, 0.05, 0.04, 0.10],
    'AUROC': [0.49, 0.55, 0.52, 0.57],
    'Prediction Rate': [0.40, 0.56, 0.50, 0.47]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.53, 0.55, 0.59, 0.54],
    'Weighted F1': [0.53, 0.54, 0.58, 0.54],
    'Precision_1': [0.40, 0.41, 0.47, 0.41],
    'Recall_1': [0.40, 0.37, 0.42, 0.39],
    'Kappa': [0.02, 0.04, 0.12, 0.04],
    'AUROC': [0.53, 0.55, 0.57, 0.53],
    'Prediction Rate': [0.38, 0.35, 0.34, 0.37]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.54, 0.57, 0.52, 0.53],
    'Weighted F1': [0.53, 0.57, 0.52, 0.54],
    'Precision_1': [0.39, 0.44, 0.42, 0.40],
    'Recall_1': [0.36, 0.44, 0.57, 0.40],
    'Kappa': [0.01, 0.09, 0.06, 0.02],
    'AUROC': [0.49, 0.56, 0.56, 0.54],
    'Prediction Rate': [0.36, 0.38, 0.53, 0.39]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# ADD/ADHD PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.58, 0.58, 0.58],
    'Weighted F1': [0.53, 0.58, 0.58, 0.59],
    'Precision_1': [0.53, 0.58, 0.58, 0.60],
    'Recall_1': [0.53, 0.54, 0.54, 0.52],
    'Kappa': [0.05, 0.15, 0.15, 0.17],
    'AUROC': [0.48, 0.59, 0.59, 0.59],
    'Prediction Rate': [0.50, 0.46, 0.46, 0.43]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.48, 0.49, 0.50, 0.51],
    'Weighted F1': [0.49, 0.48, 0.46, 0.51],
    'Precision_1': [0.49, 0.50, 0.62, 0.51],
    'Recall_1': [0.49, 0.50, 0.36, 0.51],
    'Kappa': [-0.03, 0.00, 0.04, 0.02],
    'AUROC': [0.49, 0.51, 0.58, 0.54],
    'Prediction Rate': [0.51, 0.50, 0.33, 0.49]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.54, 0.58, 0.55],
    'Weighted F1': [0.53, 0.53, 0.58, 0.55],
    'Precision_1': [0.53, 0.54, 0.59, 0.56],
    'Recall_1': [0.53, 0.57, 0.53, 0.56],
    'Kappa': [0.06, 0.08, 0.15, 0.10],
    'AUROC': [0.54, 0.57, 0.61, 0.56],
    'Prediction Rate': [0.50, 0.53, 0.45, 0.50]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# AUTISM/ASPERGER'S/ASD PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.55, 0.54, 0.52, 0.52],
    'Weighted F1': [0.56, 0.53, 0.52, 0.52],
    'Precision_1': [0.51, 0.51, 0.49, 0.50],
    'Recall_1': [0.50, 0.65, 0.64, 0.62],
    'Kappa': [0.10, 0.12, 0.09, 0.10],
    'AUROC': [0.52, 0.59, 0.55, 0.59],
    'Prediction Rate': [0.45, 0.58, 0.58, 0.55]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.49, 0.51, 0.49, 0.49],
    'Weighted F1': [0.50, 0.52, 0.49, 0.50],
    'Precision_1': [0.45, 0.46, 0.41, 0.45],
    'Recall_1': [0.40, 0.49, 0.39, 0.48],
    'Kappa': [-0.01, 0.03, -0.08, 0.01],
    'AUROC': [0.51, 0.53, 0.48, 0.52],
    'Prediction Rate': [0.42, 0.48, 0.44, 0.49]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.50, 0.53, 0.51, 0.51],
    'Weighted F1': [0.49, 0.54, 0.51, 0.52],
    'Precision_1': [0.44, 0.48, 0.48, 0.47],
    'Recall_1': [0.45, 0.48, 0.55, 0.44],
    'Kappa': [0.01, 0.06, 0.06, 0.03],
    'AUROC': [0.52, 0.54, 0.54, 0.55],
    'Prediction Rate': [0.42, 0.46, 0.51, 0.43]
}


# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# DYSLEXIA/DYSPRAXIA/DYSCALCULIA/DYSGRAPHIA PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.45, 0.48, 0.56, 0.50],
    'Weighted F1': [0.44, 0.47, 0.56, 0.49],
    'Precision_1': [0.54, 0.60, 0.69, 0.58],
    'Recall_1': [0.59, 0.48, 0.48, 0.52],
    'Kappa': [-0.05, 0.05, 0.16, 0.03],
    'AUROC': [0.47, 0.50, 0.47, 0.53],
    'Prediction Rate': [0.61, 0.45, 0.39, 0.49]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.46, 0.57, 0.63, 0.60],
    'Weighted F1': [0.46, 0.54, 0.62, 0.59],
    'Precision_1': [0.53, 0.65, 0.78, 0.66],
    'Recall_1': [0.55, 0.69, 0.61, 0.71],
    'Kappa': [-0.07, 0.20, 0.30, 0.24],
    'AUROC': [0.49, 0.65, 0.63, 0.67],
    'Prediction Rate': [0.60, 0.61, 0.47, 0.61]
}

data3 = {
    'Model': ['Chance', 'Random Forest', 'SVM', 'XGBoost'],
    'Accuracy': [0.53, 0.53, 0.64, 0.57],
    'Weighted F1': [0.53, 0.51, 0.65, 0.56],
    'Precision_1': [0.58, 0.60, 0.70, 0.66],
    'Recall_1': [0.64, 0.65, 0.69, 0.66],
    'Kappa': [0.06, 0.12, 0.30, 0.19],
    'AUROC': [0.53, 0.60, 0.65, 0.57],
    'Prediction Rate': [0.61, 0.62, 0.55, 0.58]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# ANY OTHER LANGUGE, READING, MATH AND NON-VERBAL LEARNING DISORDER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.63, 0.68, 0.59, 0.59],
    'Weighted F1': [0.65, 0.69, 0.61, 0.61],
    'Precision_1': [0.61, 0.69, 0.55, 0.59],
    'Recall_1': [0.54, 0.57, 0.59, 0.62],
    'Kappa': [0.25, 0.35, 0.20, 0.24],
    'AUROC': [0.42, 0.70, 0.59, 0.67],
    'Prediction Rate': [0.38, 0.30, 0.42, 0.41]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.47, 0.45, 0.44, 0.49],
    'Weighted F1': [0.50, 0.41, 0.40, 0.49],
    'Precision_1': [0.41, 0.37, 0.32, 0.56],
    'Recall_1': [0.50, 0.55, 0.48, 0.56],
    'Kappa': [-0.06, 0.14, 0.09, 0.14],
    'AUROC': [0.60, 0.64, 0.65, 0.65],
    'Prediction Rate': [0.49, 0.42, 0.35, 0.43]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.50, 0.60, 0.53],
    'Weighted F1': [0.55, 0.49, 0.61, 0.53],
    'Precision_1': [0.45, 0.40, 0.62, 0.38],
    'Recall_1': [0.43, 0.63, 0.67, 0.52],
    'Kappa': [0.03, 0.18, 0.27, 0.16],
    'AUROC': [0.40, 0.67, 0.61, 0.63],
    'Prediction Rate': [0.49, 0.47, 0.43, 0.39]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# GENERALIZED ANXIETY DISORDER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.58, 0.58, 0.57],
    'Weighted F1': [0.52, 0.57, 0.57, 0.56],
    'Precision_1': [0.46, 0.49, 0.49, 0.50],
    'Recall_1': [0.47, 0.49, 0.49, 0.53],
    'Kappa': [0.05, 0.11, 0.11, 0.11],
    'AUROC': [0.49, 0.59, 0.56, 0.59],
    'Prediction Rate': [0.42, 0.44, 0.44, 0.47]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.49, 0.53, 0.56, 0.49],
    'Weighted F1': [0.50, 0.53, 0.54, 0.49],
    'Precision_1': [0.42, 0.46, 0.50, 0.42],
    'Recall_1': [0.44, 0.40, 0.36, 0.37],
    'Kappa': [-0.03, 0.05, 0.08, -0.04],
    'AUROC': [0.52, 0.55, 0.54, 0.50],
    'Prediction Rate': [0.45, 0.38, 0.31, 0.40]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.50, 0.53, 0.58, 0.52],
    'Weighted F1': [0.50, 0.53, 0.57, 0.52],
    'Precision_1': [0.43, 0.48, 0.48, 0.48],
    'Recall_1': [0.44, 0.43, 0.53, 0.45],
    'Kappa': [-0.01, 0.05, 0.11, 0.05],
    'AUROC': [0.51, 0.55, 0.58, 0.55],
    'Prediction Rate': [0.44, 0.41, 0.48, 0.43]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# OTHER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.51, 0.57, 0.55, 0.55],
    'Weighted F1': [0.51, 0.57, 0.55, 0.55],
    'Precision_1': [0.43, 0.50, 0.49, 0.49],
    'Recall_1': [0.39, 0.64, 0.70, 0.68],
    'Kappa': [-0.01, 0.14, 0.14, 0.12],
    'AUROC': [0.51, 0.59, 0.53, 0.62],
    'Prediction Rate': [0.41, 0.54, 0.60, 0.60]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.50, 0.51, 0.53, 0.56],
    'Weighted F1': [0.50, 0.50, 0.52, 0.55],
    'Precision_1': [0.43, 0.42, 0.49, 0.49],
    'Recall_1': [0.50, 0.38, 0.60, 0.45],
    'Kappa': [0.01, 0.00, 0.08, 0.09],
    'AUROC': [0.59, 0.54, 0.63, 0.58],
    'Prediction Rate': [0.48, 0.37, 0.53, 0.39]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.50, 0.50, 0.53, 0.48],
    'Weighted F1': [0.50, 0.51, 0.52, 0.48],
    'Precision_1': [0.44, 0.45, 0.45, 0.39],
    'Recall_1': [0.47, 0.48, 0.57, 0.38],
    'Kappa': [0.01, 0.02, 0.08, -0.06],
    'AUROC': [0.48, 0.55, 0.44, 0.51],
    'Prediction Rate': [0.45, 0.46, 0.50, 0.41]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)

#%%
# "Prefer not to respond" and "I have never been diagnosed with any listed diagnosis"
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.57, 0.49, 0.45, 0.54],
    'Weighted F1': [0.57, 0.53, 0.50, 0.57],
    'Precision_1': [0.18, 0.25, 0.24, 0.29],
    'Recall_1': [0.28, 0.43, 0.46, 0.43],
    'Kappa': [-0.08, -0.04, -0.06, 0.04],
    'AUROC': [0.50, 0.47, 0.45, 0.48],
    'Prediction Rate': [0.27, 0.50, 0.56, 0.45]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.63, 0.65, 0.49, 0.58],
    'Weighted F1': [0.64, 0.63, 0.52, 0.58],
    'Precision_1': [0.33, 0.32, 0.24, 0.19],
    'Recall_1': [0.34, 0.22, 0.40, 0.20],
    'Kappa': [0.08, 0.05, -0.05, -0.08],
    'AUROC': [0.44, 0.47, 0.45, 0.42],
    'Prediction Rate': [0.27, 0.18, 0.46, 0.24]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.58, 0.61, 0.49, 0.58],
    'Weighted F1': [0.59, 0.61, 0.51, 0.59],
    'Precision_1': [0.30, 0.28, 0.21, 0.25],
    'Recall_1': [0.29, 0.25, 0.34, 0.19],
    'Kappa': [0.01, 0.01, -0.07, -0.04],
    'AUROC': [0.50, 0.44, 0.41, 0.43],
    'Prediction Rate': [0.28, 0.25, 0.43, 0.25]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)


# %%
# BEST FEATURE COMBINATION
# ALL PARTICIPANTS 
data1 = {
    'Model': ['Chance', 'RF', 'SVM1', 'SVM2'],
    'Accuracy': [0.55, 0.54, 0.56, 0.53],
    'Weighted F1': [0.56, 0.55, 0.57, 0.54],
    'Precision_1': [0.32, 0.37, 0.35, 0.35],
    'Recall_1': [0.35, 0.55, 0.44, 0.54],
    'Kappa': [0.00, 0.08, 0.05, 0.06],
    'AUROC': [0.50, 0.57, 0.52, 0.54]
}

# NEUROTYPICAL PARTICIPANTS
data2 = {
    'Model': ['Chance', 'XGB', 'SVM1', 'SVM2'],
    'Accuracy': [0.61, 0.47, 0.49, 0.52],
    'Weighted F1': [0.61, 0.50, 0.52, 0.54],
    'Precision_1': [0.26, 0.28, 0.27, 0.29],
    'Recall_1': [0.26, 0.65, 0.52, 0.57],
    'Kappa': [-0.01, 0.04, 0.01, 0.05],
    'AUROC': [0.53, 0.52, 0.48, 0.49]
}

# NEURODIVERGENT PARTICIPANTS
data3 = {
    'Model': ['Chance', 'XGB', 'SVM1', 'SVM2'],
    'Accuracy': [0.52, 0.56, 0.59, 0.52],
    'Weighted F1': [0.52, 0.56, 0.58, 0.52],
    'Precision_1': [0.39, 0.45, 0.47, 0.42],
    'Recall_1': [0.40, 0.53, 0.42, 0.57],
    'Kappa': [-0.01, 0.10, 0.12, 0.06],
    'AUROC': [0.49, 0.57, 0.57, 0.56]
}

# ADD/ADHD PARTICIPANTS
data4 = {
    'Model': ['Chance', 'XGB1', 'XGB2', 'SVM'],
    'Accuracy': [0.52, 0.58, 0.51, 0.58],
    'Weighted F1': [0.53, 0.59, 0.51, 0.58],
    'Precision_1': [0.53, 0.60, 0.51, 0.59],
    'Recall_1': [0.53, 0.52, 0.51, 0.53],
    'Kappa': [0.05, 0.17, 0.02, 0.15],
    'AUROC': [0.48, 0.59, 0.54, 0.61]
}

# AUTISM/ASPERGER'S/ASD PARTICIPANTS
data5 = {
    'Model': ['Chance', 'RF1', 'RF2', 'SVM'],
    'Accuracy': [0.55, 0.54, 0.51, 0.51],
    'Weighted F1': [0.56, 0.53, 0.52, 0.51],
    'Precision_1': [0.51, 0.51, 0.46, 0.48],
    'Recall_1': [0.50, 0.65, 0.49, 0.55],
    'Kappa': [0.10, 0.12, 0.03, 0.06],
    'AUROC': [0.52, 0.59, 0.53, 0.54]
}

# DYSLEXIA/DYSPRAXIA/DYSCALCULIA/DYSGRAPHIA PARTICIPANTS
data6 = {
    'Model': ['Chance', 'SVM1', 'XGB', 'SVM2'],
    'Accuracy': [0.45, 0.56, 0.60, 0.64],
    'Weighted F1': [0.44, 0.56, 0.59, 0.65],
    'Precision_1': [0.54, 0.69, 0.66, 0.70],
    'Recall_1': [0.59, 0.48, 0.71, 0.69],
    'Kappa': [-0.05, 0.16, 0.24, 0.30],
    'AUROC': [0.47, 0.47, 0.67, 0.65]
}

# ANY OTHER LANGUGE, READING, MATH AND NON-VERBAL LEARNING DISORDER PARTICIPANTS
data7 = {
    'Model': ['Chance', 'RF', 'XGB', 'SVM'],
    'Accuracy': [0.63, 0.68, 0.49, 0.60],
    'Weighted F1': [0.65, 0.69, 0.49, 0.61],
    'Precision_1': [0.61, 0.69, 0.56, 0.62],
    'Recall_1': [0.54, 0.57, 0.56, 0.67],
    'Kappa': [0.25, 0.35, 0.14, 0.27],
    'AUROC': [0.42, 0.70, 0.65, 0.61]
}

# GENERALIZED ANXIETY DISORDER PARTICIPANTS
data8 = {
    'Model': ['Chance', 'XGB', 'RF', 'SVM'],
    'Accuracy': [0.52, 0.57, 0.53, 0.58],
    'Weighted F1': [0.52, 0.56, 0.53, 0.57],
    'Precision_1': [0.46, 0.50, 0.46, 0.48],
    'Recall_1': [0.47, 0.53, 0.40, 0.53],
    'Kappa': [0.05, 0.11, 0.05, 0.11],
    'AUROC': [0.49, 0.59, 0.55, 0.58]
}

# OTHER PARTICIPANTS
data9 = {
    'Model': ['Chance', 'XGB', 'SVM', 'RF'],
    'Accuracy': [0.51, 0.55, 0.53, 0.50],
    'Weighted F1': [0.51, 0.55, 0.52, 0.51],
    'Precision_1': [0.43, 0.49, 0.49, 0.45],
    'Recall_1': [0.39, 0.68, 0.60, 0.48],
    'Kappa': [-0.01, 0.12, 0.08, 0.02],
    'AUROC': [0.51, 0.62, 0.63, 0.55]
}

# "Prefer not to respond" and "I have never been diagnosed with any listed diagnosis"
data10 = {
    'Model': ['Chance', 'XGB', 'RF1', 'RF2'],
    'Accuracy': [0.57, 0.54, 0.65, 0.61],
    'Weighted F1': [0.57, 0.57, 0.63, 0.61],
    'Precision_1': [0.18, 0.29, 0.32, 0.28],
    'Recall_1': [0.28, 0.43, 0.22, 0.25],
    'Kappa': [-0.08, 0.04, 0.05, 0.01],
    'AUROC': [0.50, 0.48, 0.47, 0.44]
}

evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)
evaluate_models(data6)
evaluate_models(data7)
evaluate_models(data8)
evaluate_models(data9)
evaluate_models(data10)
# %%

# %%

def evaluate_models(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    # Assign weights to each metric
    weights = {
        'Weighted F1': 0.23,
        'Precision_1': 0.23,
        'Recall_1': 0.23,
        'AUROC': 0.31
    }

    # Normalize the metrics using min-max normalization
    normalized_df = df.copy()
    metrics = ['Weighted F1', 'Precision_1', 'Recall_1', 'AUROC']

    for metric in metrics:
        min_value = df[metric].min()
        max_value = df[metric].max()
        if min_value != max_value:
            normalized_df[metric] = (df[metric] - min_value) / (max_value - min_value)
        else:
            normalized_df[metric] = 0  
            
    # Calculate the composite score for each model
    normalized_df['Composite Score'] = 0
    for metric in metrics:
        normalized_df['Composite Score'] += normalized_df[metric] * weights[metric]

    # Sort the models by Composite Score in descending order
    sorted_df = normalized_df.sort_values(by='Composite Score', ascending=False)

    # Display the results
    print(sorted_df[['Model', 'Composite Score']])

    # Determine the best model based on the highest composite score
    best_model = sorted_df.iloc[0]['Model']
    print(f"The model with the highest composite score is: {best_model}")

data = {
    'Model': ['RF1', 'XGB1', 'XGB2', 'SVM1', 'RF2', 'SVM2', 'RF3', 'XGB3', 'XGB4', 'XGB5'],
    'Base rate': [0.33, 0.26, 0.39, 0.50, 0.45, 0.56, 0.43, 0.43, 0.36, 0.27],
    'Weighted F1': [0.55, 0.50, 0.56, 0.58, 0.53, 0.65, 0.69, 0.66, 0.59, 0.57],
    'Precision_1': [0.37, 0.28, 0.45, 0.59, 0.51, 0.70, 0.69, 0.50, 0.50, 0.29],
    'Recall_1': [0.55, 0.65, 0.53, 0.53, 0.55, 0.69, 0.57, 0.53, 0.51, 0.43],
    'Kappa': [0.08, 0.04, 0.10, 0.15, 0.12, 0.30, 0.35, 0.11, 0.02, 0.04],
    'AUROC': [0.57, 0.52, 0.57, 0.61, 0.59, 0.65, 0.70, 0.59, 0.54, 0.48]
}
evaluate_models(data)

# %%
