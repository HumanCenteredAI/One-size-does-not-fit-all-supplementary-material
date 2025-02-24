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
    'Accuracy': [0.66, 0.69, 0.61, 0.70],
    'Weighted F1': [0.66, 0.71, 0.64, 0.72],
    'Precision_1': [0.78, 0.87, 0.86, 0.87],
    'Recall_1': [0.79, 0.72, 0.59, 0.74],
    'Kappa': [-0.02, 0.26, 0.17, 0.27],
    'AUROC': [0.50, 0.71, 0.67, 0.71]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.63, 0.52, 0.64],
    'Weighted F1': [0.66, 0.65, 0.56, 0.65],
    'Precision_1': [0.79, 0.79, 0.79, 0.79],
    'Recall_1': [0.78, 0.73, 0.53, 0.74],
    'Kappa': [-0.01, 0.00, 0.01, 0.00],
    'AUROC': [0.50, 0.50, 0.52, 0.51]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.67, 0.66, 0.66],
    'Weighted F1': [0.66, 0.69, 0.69, 0.68],
    'Precision_1': [0.79, 0.84, 0.88, 0.83],
    'Recall_1': [0.79, 0.72, 0.66, 0.72],
    'Kappa': [-0.01, 0.17, 0.24, 0.15],
    'AUROC': [0.50, 0.64, 0.68, 0.63]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.67, 0.63, 0.67],
    'Weighted F1': [0.66, 0.67, 0.66, 0.67],
    'Precision_1': [0.79, 0.79, 0.81, 0.79],
    'Recall_1': [0.79, 0.78, 0.70, 0.79],
    'Kappa': [0.00, 0.03, 0.06, 0.01],
    'AUROC': [0.50, 0.53, 0.55, 0.52]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.71, 0.66, 0.70],
    'Weighted F1': [0.66, 0.72, 0.69, 0.71],
    'Precision_1': [0.78, 0.84, 0.88, 0.83],
    'Recall_1': [0.78, 0.78, 0.66, 0.78],
    'Kappa': [-0.02, 0.20, 0.24, 0.17],
    'AUROC': [0.50, 0.68, 0.59, 0.67]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# NEUROTYPICAL PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.68, 0.70, 0.66, 0.72],
    'Weighted F1': [0.67, 0.72, 0.68, 0.74],
    'Precision_1': [0.79, 0.87, 0.86, 0.87],
    'Recall_1': [0.80, 0.72, 0.68, 0.76],
    'Kappa': [0.03, 0.27, 0.21, 0.29],
    'AUROC': [0.50, 0.71, 0.69, 0.71]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.63, 0.54, 0.62],
    'Weighted F1': [0.66, 0.64, 0.58, 0.64],
    'Precision_1': [0.79, 0.78, 0.79, 0.79],
    'Recall_1': [0.78, 0.72, 0.56, 0.71],
    'Kappa': [0.00, 0.00, 0.03, 0.01],
    'AUROC': [0.49, 0.50, 0.53, 0.50]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.67, 0.68, 0.66, 0.68],
    'Weighted F1': [0.66, 0.70, 0.69, 0.70],
    'Precision_1': [0.78, 0.84, 0.88, 0.84],
    'Recall_1': [0.79, 0.74, 0.66, 0.73],
    'Kappa': [-0.00, 0.20, 0.24, 0.19],
    'AUROC': [0.49, 0.67, 0.68, 0.67]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.67, 0.60, 0.66],
    'Weighted F1': [0.66, 0.67, 0.63, 0.66],
    'Precision_1': [0.79, 0.79, 0.81, 0.79],
    'Recall_1': [0.79, 0.79, 0.64, 0.78],
    'Kappa': [0.00, 0.02, 0.07, 0.02],
    'AUROC': [0.51, 0.53, 0.57, 0.52]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.72, 0.66, 0.71],
    'Weighted F1': [0.66, 0.73, 0.69, 0.72],
    'Precision_1': [0.78, 0.84, 0.88, 0.84],
    'Recall_1': [0.79, 0.79, 0.65, 0.79],
    'Kappa': [-0.01, 0.22, 0.25, 0.21],
    'AUROC': [0.49, 0.70, 0.73, 0.68]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# NEURODIVERGENT PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.67, 0.67, 0.56, 0.67],
    'Weighted F1': [0.67, 0.70, 0.60, 0.70],
    'Precision_1': [0.79, 0.87, 0.87, 0.87],
    'Recall_1': [0.79, 0.69, 0.52, 0.67],
    'Kappa': [-0.00, 0.24, 0.15, 0.24],
    'AUROC': [0.51, 0.71, 0.67, 0.71]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.67, 0.63, 0.32, 0.63],
    'Weighted F1': [0.67, 0.65, 0.30, 0.65],
    'Precision_1': [0.79, 0.80, 0.80, 0.79],
    'Recall_1': [0.80, 0.71, 0.20, 0.73],
    'Kappa': [0.00, 0.02, -0.01, 0.01],
    'AUROC': [0.50, 0.51, 0.48, 0.51]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.67, 0.65, 0.67],
    'Weighted F1': [0.66, 0.69, 0.68, 0.69],
    'Precision_1': [0.79, 0.84, 0.88, 0.84],
    'Recall_1': [0.79, 0.72, 0.65, 0.71],
    'Kappa': [-0.02, 0.17, 0.23, 0.17],
    'AUROC': [0.50, 0.65, 0.68, 0.66]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.65, 0.70, 0.43, 0.67],
    'Weighted F1': [0.65, 0.70, 0.44, 0.67],
    'Precision_1': [0.78, 0.81, 0.78, 0.80],
    'Recall_1': [0.77, 0.82, 0.36, 0.77],
    'Kappa': [-0.05, 0.08, -0.01, 0.04],
    'AUROC': [0.48, 0.55, 0.50, 0.54]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.68, 0.70, 0.66, 0.70],
    'Weighted F1': [0.68, 0.71, 0.69, 0.71],
    'Precision_1': [0.79, 0.84, 0.88, 0.83],
    'Recall_1': [0.81, 0.76, 0.66, 0.78],
    'Kappa': [0.01, 0.19, 0.24, 0.17],
    'AUROC': [0.50, 0.68, 0.71, 0.66]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# ADD/ADHD PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.66, 0.63, 0.67],
    'Weighted F1': [0.66, 0.69, 0.66, 0.70],
    'Precision_1': [0.79, 0.88, 0.87, 0.88],
    'Recall_1': [0.79, 0.66, 0.63, 0.68],
    'Kappa': [-0.01, 0.24, 0.17, 0.24],
    'AUROC': [0.46, 0.71, 0.66, 0.70]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.67, 0.64, 0.48, 0.65],
    'Weighted F1': [0.67, 0.65, 0.51, 0.66],
    'Precision_1': [0.80, 0.80, 0.81, 0.80],
    'Recall_1': [0.79, 0.73, 0.45, 0.74],
    'Kappa': [0.03, 0.03, 0.03, 0.04],
    'AUROC': [0.48, 0.53, 0.54, 0.53]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.65, 0.67, 0.66, 0.66],
    'Weighted F1': [0.66, 0.69, 0.69, 0.68],
    'Precision_1': [0.79, 0.84, 0.88, 0.84],
    'Recall_1': [0.77, 0.72, 0.66, 0.73],
    'Kappa': [-0.01, 0.17, 0.24, 0.19],
    'AUROC': [0.51, 0.65, 0.68, 0.67]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.68, 0.57, 0.67],
    'Weighted F1': [0.66, 0.68, 0.60, 0.67],
    'Precision_1': [0.79, 0.80, 0.82, 0.80],
    'Recall_1': [0.78, 0.80, 0.59, 0.78],
    'Kappa': [0.00, 0.06, 0.07, 0.03],
    'AUROC': [0.50, 0.56, 0.56, 0.54]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.69, 0.63, 0.70],
    'Weighted F1': [0.66, 0.71, 0.66, 0.71],
    'Precision_1': [0.79, 0.84, 0.88, 0.84],
    'Recall_1': [0.78, 0.76, 0.62, 0.77],
    'Kappa': [-0.01, 0.19, 0.21, 0.18],
    'AUROC': [0.49, 0.69, 0.72, 0.68]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# AUTISM/ASPERGER'S/ASD PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.65, 0.59, 0.63, 0.60],
    'Weighted F1': [0.65, 0.63, 0.66, 0.64],
    'Precision_1': [0.79, 0.88, 0.89, 0.89],
    'Recall_1': [0.77, 0.58, 0.61, 0.57],
    'Kappa': [-0.06, 0.16, 0.20, 0.19],
    'AUROC': [0.49, 0.70, 0.69, 0.71]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.63, 0.42, 0.61],
    'Weighted F1': [0.69, 0.65, 0.46, 0.63],
    'Precision_1': [0.81, 0.79, 0.80, 0.79],
    'Recall_1': [0.81, 0.73, 0.38, 0.70],
    'Kappa': [0.02, -0.03, -0.02, -0.06],
    'AUROC': [0.51, 0.45, 0.51, 0.44]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.68, 0.68, 0.62, 0.67],
    'Weighted F1': [0.69, 0.70, 0.66, 0.69],
    'Precision_1': [0.81, 0.85, 0.88, 0.86],
    'Recall_1': [0.79, 0.72, 0.61, 0.70],
    'Kappa': [0.04, 0.17, 0.19, 0.18],
    'AUROC': [0.51, 0.67, 0.69, 0.67]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.67, 0.44, 0.69],
    'Weighted F1': [0.69, 0.67, 0.48, 0.69],
    'Precision_1': [0.81, 0.80, 0.79, 0.80],
    'Recall_1': [0.81, 0.79, 0.42, 0.81],
    'Kappa': [0.04, -0.03, -0.03, 0.01],
    'AUROC': [0.51, 0.46, 0.49, 0.49]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.70, 0.58, 0.70],
    'Weighted F1': [0.68, 0.72, 0.62, 0.72],
    'Precision_1': [0.80, 0.85, 0.90, 0.85],
    'Recall_1': [0.81, 0.77, 0.54, 0.77],
    'Kappa': [-0.00, 0.19, 0.17, 0.18],
    'AUROC': [0.51, 0.68, 0.68, 0.67]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# DYSLEXIA/DYSPRAXIA/DYSCALCULIA/DYSGRAPHIA PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.73, 0.74, 0.64, 0.76],
    'Weighted F1': [0.72, 0.77, 0.68, 0.78],
    'Precision_1': [0.83, 0.91, 0.90, 0.91],
    'Recall_1': [0.85, 0.77, 0.64, 0.79],
    'Kappa': [-0.05, 0.28, 0.16, 0.30],
    'AUROC': [0.54, 0.77, 0.68, 0.78]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.72, 0.72, 0.57, 0.70],
    'Weighted F1': [0.72, 0.73, 0.61, 0.71],
    'Precision_1': [0.84, 0.86, 0.83, 0.85],
    'Recall_1': [0.82, 0.80, 0.61, 0.78],
    'Kappa': [-0.00, 0.08, -0.02, 0.08],
    'AUROC': [0.51, 0.59, 0.47, 0.57]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.71, 0.80, 0.66, 0.78],
    'Weighted F1': [0.73, 0.81, 0.70, 0.80],
    'Precision_1': [0.86, 0.91, 0.90, 0.91],
    'Recall_1': [0.78, 0.85, 0.67, 0.83],
    'Kappa': [0.09, 0.35, 0.18, 0.31],
    'AUROC': [0.53, 0.79, 0.70, 0.76]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.74, 0.78, 0.55, 0.71],
    'Weighted F1': [0.75, 0.76, 0.58, 0.70],
    'Precision_1': [0.86, 0.85, 0.84, 0.83],
    'Recall_1': [0.83, 0.90, 0.57, 0.82],
    'Kappa': [0.12, 0.05, 0.03, 0.03],
    'AUROC': [0.54, 0.62, 0.51, 0.52]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.78, 0.67, 0.78],
    'Weighted F1': [0.71, 0.79, 0.71, 0.79],
    'Precision_1': [0.84, 0.89, 0.90, 0.90],
    'Recall_1': [0.78, 0.86, 0.68, 0.83],
    'Kappa': [0.03, 0.26, 0.20, 0.28],
    'AUROC': [0.50, 0.75, 0.74, 0.76]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# ANY OTHER LANGUGE, READING, MATH AND NON-VERBAL LEARNING DISORDER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.60, 0.64, 0.58, 0.63],
    'Weighted F1': [0.61, 0.66, 0.61, 0.65],
    'Precision_1': [0.73, 0.79, 0.80, 0.79],
    'Recall_1': [0.72, 0.71, 0.57, 0.70],
    'Kappa': [0.01, 0.17, 0.14, 0.15],
    'AUROC': [0.55, 0.68, 0.66, 0.68]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.52, 0.63, 0.49, 0.54],
    'Weighted F1': [0.53, 0.63, 0.48, 0.55],
    'Precision_1': [0.68, 0.73, 0.80, 0.73],
    'Recall_1': [0.66, 0.74, 0.53, 0.57],
    'Kappa': [-0.18, 0.03, 0.02, 0.04],
    'AUROC': [0.52, 0.51, 0.57, 0.54]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.62, 0.63, 0.62, 0.64],
    'Weighted F1': [0.63, 0.65, 0.65, 0.65],
    'Precision_1': [0.74, 0.78, 0.81, 0.79],
    'Recall_1': [0.72, 0.71, 0.63, 0.71],
    'Kappa': [0.03, 0.14, 0.20, 0.17],
    'AUROC': [0.49, 0.61, 0.66, 0.65]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.59, 0.61, 0.50, 0.60],
    'Weighted F1': [0.61, 0.61, 0.50, 0.61],
    'Precision_1': [0.73, 0.72, 0.78, 0.72],
    'Recall_1': [0.68, 0.69, 0.50, 0.70],
    'Kappa': [-0.02, -0.02, -0.01, -0.00],
    'AUROC': [0.40, 0.47, 0.50, 0.49]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.63, 0.64, 0.61, 0.65],
    'Weighted F1': [0.63, 0.66, 0.64, 0.65],
    'Precision_1': [0.75, 0.78, 0.81, 0.76],
    'Recall_1': [0.76, 0.72, 0.58, 0.78],
    'Kappa': [0.06, 0.16, 0.18, 0.10],
    'AUROC': [0.47, 0.63, 0.65, 0.60]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# GENERALIZED ANXIETY DISORDER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.70, 0.67, 0.61, 0.63],
    'Weighted F1': [0.70, 0.70, 0.66, 0.67],
    'Precision_1': [0.82, 0.90, 0.89, 0.91],
    'Recall_1': [0.81, 0.67, 0.60, 0.61],
    'Kappa': [0.04, 0.24, 0.18, 0.21],
    'AUROC': [0.50, 0.72, 0.69, 0.72]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.68, 0.65, 0.50, 0.62],
    'Weighted F1': [0.68, 0.67, 0.53, 0.65],
    'Precision_1': [0.81, 0.81, 0.80, 0.80],
    'Recall_1': [0.80, 0.75, 0.51, 0.71],
    'Kappa': [-0.03, -0.02, -0.01, -0.05],
    'AUROC': [0.51, 0.48, 0.49, 0.45]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.70, 0.68, 0.67, 0.67],
    'Weighted F1': [0.70, 0.71, 0.70, 0.70],
    'Precision_1': [0.82, 0.86, 0.90, 0.86],
    'Recall_1': [0.81, 0.73, 0.66, 0.71],
    'Kappa': [0.01, 0.16, 0.24, 0.16],
    'AUROC': [0.51, 0.65, 0.70, 0.66]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.68, 0.38, 0.68],
    'Weighted F1': [0.69, 0.68, 0.41, 0.69],
    'Precision_1': [0.81, 0.81, 0.74, 0.82],
    'Recall_1': [0.81, 0.79, 0.35, 0.79],
    'Kappa': [-0.01, -0.02, -0.05, 0.01],
    'AUROC': [0.50, 0.49, 0.47, 0.50]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.72, 0.64, 0.71],
    'Weighted F1': [0.69, 0.73, 0.68, 0.72],
    'Precision_1': [0.81, 0.85, 0.90, 0.85],
    'Recall_1': [0.81, 0.79, 0.62, 0.78],
    'Kappa': [-0.01, 0.18, 0.22, 0.16],
    'AUROC': [0.49, 0.69, 0.72, 0.66]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# OTHER PARTICIPANTS
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.66, 0.70, 0.66, 0.70],
    'Weighted F1': [0.66, 0.73, 0.69, 0.73],
    'Precision_1': [0.79, 0.89, 0.89, 0.90],
    'Recall_1': [0.78, 0.71, 0.66, 0.70],
    'Kappa': [-0.03, 0.28, 0.24, 0.30],
    'AUROC': [0.49, 0.75, 0.71, 0.74]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.64, 0.63, 0.51, 0.64],
    'Weighted F1': [0.65, 0.65, 0.54, 0.65],
    'Precision_1': [0.79, 0.79, 0.78, 0.79],
    'Recall_1': [0.75, 0.72, 0.52, 0.75],
    'Kappa': [-0.05, -0.02, -0.02, -0.03],
    'AUROC': [0.48, 0.46, 0.47, 0.46]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.68, 0.72, 0.68, 0.72],
    'Weighted F1': [0.68, 0.73, 0.71, 0.74],
    'Precision_1': [0.81, 0.86, 0.91, 0.87],
    'Recall_1': [0.79, 0.77, 0.66, 0.77],
    'Kappa': [0.04, 0.24, 0.30, 0.26],
    'AUROC': [0.53, 0.72, 0.71, 0.72]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.67, 0.64, 0.58, 0.62],
    'Weighted F1': [0.67, 0.65, 0.60, 0.63],
    'Precision_1': [0.80, 0.79, 0.81, 0.78],
    'Recall_1': [0.79, 0.76, 0.64, 0.72],
    'Kappa': [0.00, -0.04, 0.00, -0.05],
    'AUROC': [0.51, 0.49, 0.53, 0.48]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.69, 0.73, 0.69, 0.73],
    'Weighted F1': [0.70, 0.74, 0.72, 0.74],
    'Precision_1': [0.81, 0.85, 0.91, 0.85],
    'Recall_1': [0.80, 0.79, 0.67, 0.80],
    'Kappa': [0.08, 0.25, 0.30, 0.23],
    'AUROC': [0.47, 0.74, 0.75, 0.72]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

#%%
# "Prefer not to respond" and "I have never been diagnosed with any listed diagnosis"
data1 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.61, 0.66, 0.60, 0.66],
    'Weighted F1': [0.62, 0.69, 0.63, 0.69],
    'Precision_1': [0.76, 0.86, 0.84, 0.86],
    'Recall_1': [0.73, 0.67, 0.58, 0.67],
    'Kappa': [-0.02, 0.25, 0.18, 0.25],
    'AUROC': [0.54, 0.68, 0.65, 0.66]
}

data2 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.65, 0.66, 0.50, 0.61],
    'Weighted F1': [0.64, 0.65, 0.52, 0.62],
    'Precision_1': [0.76, 0.77, 0.79, 0.75],
    'Recall_1': [0.78, 0.80, 0.48, 0.73],
    'Kappa': [-0.00, 0.02, 0.03, -0.03],
    'AUROC': [0.54, 0.48, 0.55, 0.45]
}

data3 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.64, 0.61, 0.65, 0.63],
    'Weighted F1': [0.64, 0.63, 0.67, 0.65],
    'Precision_1': [0.76, 0.79, 0.85, 0.82],
    'Recall_1': [0.76, 0.68, 0.65, 0.69],
    'Kappa': [0.01, 0.09, 0.24, 0.16],
    'AUROC': [0.52, 0.61, 0.67, 0.63]
}

data4 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.64, 0.67, 0.46, 0.68],
    'Weighted F1': [0.64, 0.67, 0.48, 0.67],
    'Precision_1': [0.76, 0.78, 0.78, 0.77],
    'Recall_1': [0.77, 0.80, 0.43, 0.82],
    'Kappa': [-0.01, 0.08, 0.02, 0.05],
    'AUROC': [0.52, 0.52, 0.51, 0.52]
}

data5 = {
    'Model': ['Chance', 'RF', 'SVM', 'XGBoost'],
    'Accuracy': [0.61, 0.66, 0.65, 0.64],
    'Weighted F1': [0.61, 0.67, 0.68, 0.66],
    'Precision_1': [0.75, 0.80, 0.86, 0.79],
    'Recall_1': [0.74, 0.75, 0.65, 0.73],
    'Kappa': [-0.05, 0.13, 0.24, 0.11],
    'AUROC': [0.53, 0.62, 0.69, 0.60]
}

# Call the function with the data
evaluate_models(data1)
evaluate_models(data2)
evaluate_models(data3)
evaluate_models(data4)
evaluate_models(data5)

# %%
# BEST FEATURE COMBINATION
# ALL PARTICIPANTS 
data1 = {
    'Model': ['Chance', 'XGB1', 'XGB2', 'SVM1', 'RF1', 'RF2'],
    'Weighted F1': [0.66, 0.72, 0.65, 0.69, 0.67, 0.72],
    'Precision_1': [0.78, 0.87, 0.79, 0.88, 0.79, 0.84],
    'Recall_1': [0.79, 0.74, 0.74, 0.66, 0.78, 0.78],
    'Kappa': [-0.02, 0.27, 0.00, 0.24, 0.03, 0.20],
    'AUROC': [0.50, 0.71, 0.51, 0.68, 0.53, 0.68],
}

# NEUROTYPICAL PARTICIPANTS
data2 = {
    'Model': ['Chance', 'XGB1', 'RF1', 'RF2', 'RF3', 'RF4'],
    'Weighted F1': [0.67, 0.74, 0.64, 0.70, 0.67, 0.73],
    'Precision_1': [0.79, 0.87, 0.78, 0.84, 0.79, 0.84],
    'Recall_1': [0.80, 0.76, 0.72, 0.74, 0.79, 0.79],
    'Kappa': [0.03, 0.29, -0.00, 0.20, 0.02, 0.22],
    'AUROC': [0.50, 0.71, 0.50, 0.67, 0.53, 0.70],
}

# NEURODIVERGENT PARTICIPANTS
data3 = {
    'Model': ['Chance', 'RF1', 'XGB1', 'XGB2', 'RF2', 'RF3'],
    'Weighted F1': [0.67, 0.70, 0.65, 0.69, 0.70, 0.71],
    'Precision_1': [0.79, 0.87, 0.79, 0.84, 0.81, 0.84],
    'Recall_1': [0.79, 0.69, 0.73, 0.71, 0.82, 0.76],
    'Kappa': [-0.00, 0.24, 0.01, 0.17, 0.08, 0.19],
    'AUROC': [0.51, 0.71, 0.51, 0.66, 0.55, 0.68],
}

# ADD/ADHD PARTICIPANTS
data4 = {
    'Model': ['Chance', 'XGB1', 'XGB2', 'XGB3', 'RF1', 'RF2'],
    'Weighted F1': [0.66, 0.70, 0.66, 0.68, 0.68, 0.71],
    'Precision_1': [0.79, 0.88, 0.80, 0.86, 0.80, 0.84],
    'Recall_1': [0.79, 0.68, 0.74, 0.68, 0.80, 0.76],
    'Kappa': [-0.01, 0.24, 0.04, 0.19, 0.06, 0.19],
    'AUROC': [0.46, 0.70, 0.53, 0.65, 0.56, 0.69],
}

# AUTISM/ASPERGER'S/ASD PARTICIPANTS
data5 = {
    'Model': ['Chance', 'SVM1', 'RF1', 'RF2', 'XGB1', 'RF3'],
    'Weighted F1': [0.70, 0.66, 0.65, 0.70, 0.69, 0.72],
    'Precision_1': [0.81, 0.89, 0.79, 0.85, 0.80, 0.85],
    'Recall_1': [0.81, 0.61, 0.73, 0.72, 0.81, 0.77],
    'Kappa': [-0.06, 0.20, -0.03, 0.17, 0.01, 0.19],
    'AUROC': [0.52, 0.69, 0.45, 0.67, 0.49, 0.68],
}

# DYSLEXIA/DYSPRAXIA/DYSCALCULIA/DYSGRAPHIA PARTICIPANTS
data6 = {
    'Model': ['Chance', 'XGB1', 'RF1', 'RF2', 'RF3', 'RF4'],
    'Weighted F1': [0.72, 0.78, 0.73, 0.81, 0.76, 0.79],
    'Precision_1': [0.83, 0.91, 0.86, 0.91, 0.85, 0.89],
    'Recall_1': [0.85, 0.79, 0.80, 0.85, 0.90, 0.86],
    'Kappa': [-0.05, 0.30, 0.08, 0.35, 0.05, 0.26],
    'AUROC': [0.54, 0.78, 0.59, 0.79, 0.62, 0.75],
}

# ANY OTHER LANGUGE, READING, MATH AND NON-VERBAL LEARNING DISORDER PARTICIPANTS
data7 = {
    'Model': ['Chance', 'RF1', 'RF2', 'XGB1', 'XGB2', 'RF3'],
    'Weighted F1': [0.61, 0.66, 0.63, 0.65, 0.61, 0.66],
    'Precision_1': [0.73, 0.79, 0.73, 0.79, 0.72, 0.78],
    'Recall_1': [0.72, 0.71, 0.74, 0.71, 0.70, 0.72],
    'Kappa': [0.01, 0.17, 0.03, 0.17, -0.00, 0.16],
    'AUROC': [0.55, 0.68, 0.51, 0.65, 0.49, 0.63],
}

# GENERALIZED ANXIETY DISORDER PARTICIPANTS
data8 = {
    'Model': ['Chance', 'RF1', 'RF2', 'SVM', 'XGB', 'RF3'],
    'Weighted F1': [0.70, 0.70, 0.67, 0.70, 0.69, 0.73],
    'Precision_1': [0.82, 0.90, 0.81, 0.90, 0.82, 0.85],
    'Recall_1': [0.81, 0.67, 0.75, 0.66, 0.79, 0.79],
    'Kappa': [0.04, 0.24, -0.02, 0.24, 0.01, 0.18],
    'AUROC': [0.50, 0.72, 0.48, 0.70, 0.50, 0.69]
}

# OTHER PARTICIPANTS
data9 = {
    'Model': ['Chance', 'RF1', 'XGB1', 'XGB2', 'RF2', 'RF3'],
    'Weighted F1': [0.66, 0.73, 0.65, 0.74, 0.65, 0.74],
    'Precision_1': [0.79, 0.89, 0.79, 0.87, 0.79, 0.85],
    'Recall_1': [0.78, 0.71, 0.75, 0.77, 0.76, 0.79],
    'Kappa': [-0.03, 0.28, 0.00, 0.26, -0.04, 0.25],
    'AUROC': [0.49, 0.75, 0.46, 0.72, 0.49, 0.74]}

# "Prefer not to respond" and "I have never been diagnosed with any listed diagnosis"
data10 = {
    'Model': ['Chance', 'RF1', 'RF2', 'SVM1', 'XGB', 'SVM2'],
    'Weighted F1': [0.62, 0.69, 0.65, 0.67, 0.67, 0.68],
    'Precision_1': [0.76, 0.86, 0.77, 0.85, 0.77, 0.86],
    'Recall_1': [0.73, 0.67, 0.80, 0.65, 0.82, 0.65],
    'Kappa': [-0.02, 0.25, 0.02, 0.24, 0.05, 0.24],
    'AUROC': [0.54, 0.68, 0.48, 0.67, 0.52, 0.69]
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

#%%
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

# BEST GROUPS
data = {
    'Model': ['XGB1', 'XGB2', 'RF1', 'RF2', 'RF3', 'RF4', 'RF5', 'RF6', 'RF7', 'RF8'],
    'Base rate': [0.79, 0.79, 0.79, 0.79, 0.80, 0.84, 0.73, 0.81, 0.80, 0.76],
    'Weighted F1': [0.72, 0.74, 0.71, 0.71, 0.72, 0.81, 0.66, 0.73, 0.74, 0.69],
    'Precision_1': [0.87, 0.87, 0.84, 0.84, 0.85, 0.91, 0.79, 0.85, 0.85, 0.86],
    'Recall_1': [0.74, 0.76, 0.76, 0.76, 0.77, 0.85, 0.71, 0.79, 0.79, 0.67],
    'Kappa': [0.27, 0.29, 0.19, 0.19, 0.19, 0.35, 0.17, 0.18, 0.25, 0.25],
    'AUROC': [0.71, 0.71, 0.68, 0.69, 0.68, 0.79, 0.68, 0.69, 0.74, 0.68]
    }
evaluate_models(data)
# %%
