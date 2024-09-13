import requests
import pandas as pd
from tqdm import tqdm

url = "http://llama.rekeymed.com:8080/v1/chat/completions"  # 替换为实际的URL

headers = {
    "Content-Type": "application/json"
}

# 读取CSV文件
df = pd.read_csv("data/Origin/trainlabel.csv")

# 初始化新的列
df["Confidence"] = None
df["Description"] = None

for index, row in tqdm(df.iterrows()):
    print(index,row)
    # if index > 10:
    #     break

    data = {
        "model": "/home/ubuntu/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "你是一个购买意向判断机器人，你需要接收自动营销系统的录音ASR结果,录音中包含自动营销系统的声音和客户声音，营销系统的语料不应该对识别结果有影响，请识别出来这个用户对产品的兴趣有多少，只能用json返回如下格式的数据，不能有额外描述例如“Here is the recongition result”：{\"Confidence\":0.5 , \"Description\":\"因为...而判断为有购买意向\"}"
            },
            {
                "role": "user",
                "content": row["text"]
            }
        ],
        "stop_token_ids": [128001, 128009]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # 提取Confidence和Description
    content = result['choices'][0]['message']['content']
    print(content)
    # extracted_data = eval(content)  # 使用eval将字符串转换为字典
    try:
        extracted_data = eval(content)  # 使用 eval 将字符串转换为字典
        df.at[index, "Confidence"] = extracted_data["Confidence"]
        df.at[index, "Description"] = extracted_data["Description"]
    except:
        df.at[index, "Original"] = content


    # df.at[index, "Confidence"] = extracted_data["Confidence"]
    # df.at[index, "Description"] = extracted_data["Description"]

# 保存到新的CSV文件
df.to_csv("output_OriginTrain.csv", index=False)
