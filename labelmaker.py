import re
import csv
import os


def extract_info(file_content):
    result = {}
    pattern = r"The audio\\(\d+)\.wav file has been transcribed as follows:\n(.*?)\n\n"
    matches = re.findall(pattern, file_content, re.DOTALL)
    
    for match in matches:
        id = match[0]
        asr_result = match[1].strip()
        result[id] = asr_result
    
    return result
asr_dict = {}
# 读取文件内容
with open('data/Output-cn.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 提取信息并存入字典
extracted_info = extract_info(content)

# 打印结果
for id, asr_result in extracted_info.items():
    # print(f"ID: {id}")
    # print(f"ASR结果: {asr_result}")
    asr_dict[id] = asr_result


# 创建label.csv文件
with open('label.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 写入表头
    writer.writerow(['audio_id', 'text', 'annotation', 'mode'])
    
    # 遍历data文件夹中的音频文件
    for filename in os.listdir('data/audio'):
        if filename.endswith('.wav') or filename.endswith('.mp3'):  # 假设音频文件是wav或mp3格式
            audio_id = os.path.splitext(filename)[0]  # 获取不带扩展名的文件名作为audio_id
            
            # 从ASR_text字典中获取对应的文本
            text = asr_dict.get(audio_id, '')  # 如果找不到对应的文本，则使用空字符串
            
            # 这里的annotation和mode需要根据实际情况填写
            # 在这个例子中，我们使用占位符
            annotation = ''  # 这里应该填入实际的annotation
            mode = 'train'  # 假设所有数据都是训练集
            
            # 写入一行数据
            writer.writerow([audio_id, text, annotation, mode])

print("label.csv文件已创建完成。")
