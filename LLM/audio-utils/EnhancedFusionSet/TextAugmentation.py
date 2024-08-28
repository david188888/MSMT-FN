import pandas as pd
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 读取多音字表格数据
multi_tone_df = pd.read_csv('8000hans.csv')

# 创建一个按拼音分组的字典
pinyin_dict = {}
for index, row in multi_tone_df.iterrows():
    word = row['Hans']
    pinyin = row['py2']
    if pinyin not in pinyin_dict:
        pinyin_dict[pinyin] = []
    pinyin_dict[pinyin].append(word)

# 定义一个替换函数
def replace_with_similar_pinyin(text, pinyin_dict, replace_prob=0.5):
    words = list(text)
    for i, word in enumerate(words):
        for pinyin, chars in pinyin_dict.items():
            if word in chars and random.random() < replace_prob:
                words[i] = random.choice(chars)
                break
    return ''.join(words)

# 定义多进程处理函数
def process_chunk(args):
    chunk, pinyin_dict, replace_prob = args
    return chunk.apply(lambda x: replace_with_similar_pinyin(x, pinyin_dict, replace_prob))

# 读取目标文件
target_df = pd.read_csv('newTrain_TextEnhan.csv')

# 多次读取源文件，进行替换并追加到目标文件中
num_iterations = 50  # 设置重复次数
replace_prob = 0.1  # 设置替换概率
num_processes = cpu_count()  # 使用所有CPU核心

for _ in tqdm(range(num_iterations), desc="Iterations"):
    source_df = pd.read_csv('newTrain.csv')
    
    # 将数据分割成多片进行处理
    chunks = [source_df['text'][i::num_processes] for i in range(num_processes)]
    args = [(chunk, pinyin_dict, replace_prob) for chunk in chunks]
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_chunk, args), total=num_processes, desc="Processing Chunks"))
    
    # 合并处理结果
    source_df['text'] = pd.concat(results).sort_index()
    target_df = pd.concat([target_df, source_df], ignore_index=True)

# 保存最终结果到目标文件
target_df.to_csv('newTrain_TextEnhan.csv', index=False)

print("处理完成并保存到目标文件。")
