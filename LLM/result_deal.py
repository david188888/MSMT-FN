import numpy as np
import pandas as pd
import json
def multiclass_acc(y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

df_custormer = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/LLM/output_CustomerTrain.csv')
df_origin = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/LLM/output_origin_train.csv')
df_gpt3_5 = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/LLM/GPT3.5-test-customer-output.csv')
df_gpt40 = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/LLM/GPT4o-test-customer-output.csv')

df_custormer = df_custormer.dropna(subset=['label','Confidence'])

df_gpt40 = df_gpt3_5.dropna(subset=['label','Original'])

df_gpt40 = df_gpt40.dropna(subset=['label', 'Original'])
original = df_gpt40['Original']
possibilities = original.apply(lambda x: json.loads(x).get('Possibility'))

test_truth = df_gpt40['label'].tolist()
test_preds = possibilities.tolist()

print(test_preds)
print('---------')
print(test_truth)

ms_3 = [-0.01, 0.3, 0.7, 1.0]
ms_5 = [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
ms_4 = [-0.01, 0.3, 0.5, 0.7, 1.0]
ms_2 = [-0.01, 0.5, 1.0]

test_preds_a3 = test_preds.copy()
test_truth_a3 = test_truth.copy()
test_preds_a3 = np.zeros_like(test_preds)
test_truth_a3 = np.zeros_like(test_truth)


test_preds_a2 = test_preds.copy()
test_truth_a2 = test_truth.copy()
test_preds_a2 = np.zeros_like(test_preds)
test_truth_a2 = np.zeros_like(test_truth)

test_preds_a4 = test_preds.copy()
test_truth_a4 = test_truth.copy()
test_preds_a4 = np.zeros_like(test_preds)
test_truth_a4 = np.zeros_like(test_truth)

test_preds_a5 = test_preds.copy()
test_truth_a5 = test_truth.copy()
test_preds_a5 = np.zeros_like(test_preds)
test_truth_a5 = np.zeros_like(test_truth)


for i in range(len(test_preds)):
    for j in range(3):  # 这里使用3是因为有三个区间
        if ms_3[j] <= test_preds[i] < ms_3[j+1]:  # 确保值在开区间内避免边界重叠
            test_preds_a3[i] = j
            break

# 映射真实标签到三分类
for i in range(len(test_truth)):
    for j in range(3):
        if ms_3[j] <= test_truth[i] < ms_3[j+1]:  # 同样的逻辑应用于真实标签
            test_truth_a3[i] = j
            break
            
            
for i in range(len(test_preds)):
    if ms_2[0] <= test_preds[i] < ms_2[1]:  # 注意：边界处理应避免等于的情况，确保值能正确分类
        test_preds_a2[i] = 0
    else:  # 大于或等于最高边界值
        test_preds_a2[i] = 1

# 映射真实标签
for i in range(len(test_truth)):
    if ms_2[0] <= test_truth[i] < ms_2[1]:  # 同样的边界处理逻辑
        test_truth_a2[i] = 0
    else:
        test_truth_a2[i] = 1
        
        
for i in range(len(test_preds)):
    for j in range(4):  # 这里使用3是因为有三个区间
        if ms_4[j] <= test_preds[i] < ms_4[j+1]:  # 确保值在开区间内避免边界重叠
            test_preds_a4[i] = j
            break


for i in range(len(test_truth)):
    for j in range(4):
        if ms_4[j] <= test_truth[i] < ms_4[j+1]:  # 同样的逻辑应用于真实标签
            test_truth_a4[i] = j
            break
    
        
for i in range(len(test_preds)):
    for j in range(5):  # 这里使用3是因为有三个区间
        if ms_5[j] <= test_preds[i] < ms_5[j+1]:  # 确保值在开区间内避免边界重叠
            test_preds_a5[i] = j
            break

for i in range(len(test_truth)):
    for j in range(5):
        if ms_5[j] <= test_truth[i] < ms_5[j+1]:  # 同样的逻辑应用于真实标签
            test_truth_a5[i] = j
            break
        
    
    
mult_a2 = multiclass_acc(test_preds_a2, test_truth_a2)
mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)
mult_a4 = multiclass_acc(test_preds_a4, test_truth_a4)
mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)


print(mult_a2)

with open("/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/LLM/data/LLM_result.txt",'a') as f:
    f.write("\n")
    f.writelines("The result of gpt3.5 in customer is as follows:")
    f.write("\n")
    f.writelines(f"acc2 result of gpt3.5 in customer is {mult_a2}")
    f.write("\n")
    f.writelines(f"acc3 result of gpt3.5 in customer is {mult_a3}")
    f.write("\n")
    f.writelines(f"acc4 result of gpt3.5 in customer is {mult_a4}")
    f.write("\n")
    f.writelines(f"acc5 result of gpt3.5 in customer is {mult_a5}")