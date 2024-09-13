import re
import csv

def extract_data_from_text(text):
    """
    从文本中提取文件名、手机号和转录结果
    """
    pattern = re.compile(r"The Out/processed_(\d+)_left\.wav file has been transcribed as follows:\n(.+?)\n\n", re.DOTALL)
    matches = pattern.findall(text)
    data = []
    
    for match in matches:
        file_path = match[0].strip()
        transcription = match[1].strip()
        # 提取手机号（假设手机号是文件名的一部分，如 'processed_19880698005_left.wav'）
        phone_number_match = re.search(r'processed_(\d+)_left\.wav', file_path)
        if phone_number_match:
            phone_number = phone_number_match.group(1)
            data.append([file_path, phone_number, transcription])
    
    return data

def save_to_csv(data, output_file):
    """
    将提取的数据保存到CSV文件中
    """
    with open(output_file, mode='a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path", "Phone Number", "Transcription"])
        writer.writerows(data)

if __name__ == '__main__':
    input_file = 'Output-cn_7.10PM.txt'  # 替换为你的文本文件路径
    output_file = 'output_7.10.csv'
    
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    data = extract_data_from_text(text)
    save_to_csv(data, output_file)
    print(f"提取的数据已保存到 {output_file}")
