import numpy as np
import pandas as pd

def multiclass_acc(y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

df_custormer = pd.read_csv('MMML/LLM/data/output_CustomerTrain.csv')
df_fusion = pd.read_csv('MMML/LLM/data/output_fusion_trian.csv')
df_origin = pd.read_csv('MMML/LLM/data/output_origin_train.csv')


df_origin = df_origin.dropna(subset=['label','Confidence'])

test_truth = df_origin['label'].tolist()
test_preds = df_origin['Confidence'].tolist()
ms_3 = [-0.01, 0.4, 0.7, 1.01]
test_preds_a3 = test_preds.copy()
test_truth_a3 = test_truth.copy()
test_preds_a3 = np.zeros_like(test_preds)
test_truth_a3 = np.zeros_like(test_truth)
for i in range(len(test_preds)):
    for j in range(3):
        if ms_3[j] <= test_preds[i] <= ms_3[j+1]:
            test_preds_a3[i] = j
            break
    for j in range(3):
        if ms_3[j] <= test_truth[i] <= ms_3[j+1]:
            test_truth_a3[i] = j
            break
        
        
mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)

with open("MMML/LLM/data/customer_llama_result.txt",'a') as f:
    f.writelines(f"acc3 result of ollama in origin is {mult_a3}")