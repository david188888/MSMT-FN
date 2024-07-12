import requests
import pandas as pd
from tqdm import tqdm
import re

url = "http://cnllama.rekeymed.com:8080/api/chat"  # 替换为实际的URL

headers = {
    "Content-Type": "application/json"
}

# 读取CSV文件
df = pd.read_csv("data/Customer/trainlabel.csv")

# 初始化新的列
df["Confidence"] = None
df["Description"] = None

def remove_markdown(text):
    markdown_pattern = re.compile(r'(```.*?```|`[^`]*`|\*\*[^*]*\*\*|\*[^*]*\*|_[^_]*_|__[^_]*__|~~[^~]*~~)', re.DOTALL)
    return re.sub(markdown_pattern, '', text)

for index, row in tqdm(df.iterrows()):
    # if index < 331:
    #     index = 331
    #     continue
    print(index)
    # if index > 10:
    #     break

    data = {
        # "model": "/home/ubuntu/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct",
        "model":"wangshenzhi/llama3-8b-chinese-chat-ollama-q4",
        "messages": [
            {
                "role": "system",
                "content": "你是一个购买意向判断机器人，你需要接收自动营销系统的录音ASR结果,录音中包含自动营销系统的声音和客户声音，营销系统的语料不应该对识别结果有影响，请识别出来这个用户对产品的兴趣有多少，只用json返回数据，不能有额外描述,也一定不能有markdown标记，例如“Here is the recongition result”，“```json```”都是错误的，以下是正确格式：{Confidence:0.5 , Description:因为...而判断为有购买意向}"
            },
            {
                "role": "user",
                "content": row["text"]
            }
        ],
        "stream":False,
        # "stop_token_ids": [151329, 151336, 151338]
    }

    response = requests.post(url, headers=headers, json=data)
    # print(response)
    result = response.json()
    # print(result)

    # 提取Confidence和Description
    content = result['message']['content']
    print(content)
    content = remove_markdown(content)  # 去除markdown标记
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
df.to_csv("ollama/output_FusionTrain.csv", index=False)
