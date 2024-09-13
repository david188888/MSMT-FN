import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 读取多音字表格数据
multi_tone_df = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/8000hans.csv')
target_df = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/dialog_train.csv')

# 创建一个按拼音分组的字典
pinyin_dict = {}
for index, row in multi_tone_df.iterrows():
    word = row['Hans']
    pinyin = row['py']
    if pinyin not in pinyin_dict:
        pinyin_dict[pinyin] = []
    pinyin_dict[pinyin].append(word)

# 定义一个替换函数
def replace_with_similar_pinyin(text, pinyin_dict, replace_prob=0.5):
    words = list(text)
    skip_words = ['客','服','顾']
    for i, word in enumerate(words):
        if word in skip_words:
            continue
        for pinyin, chars in pinyin_dict.items():
            if word in chars and random.random() < replace_prob:
                words[i] = random.choice(chars)
                break
    return ''.join(words)



# 定义多线程处理函数
def process_chunk(chunk, pinyin_dict, replace_prob):
    return chunk.apply(lambda x: replace_with_similar_pinyin(x, pinyin_dict, replace_prob))

# 读取目标文件


# 使用tqdm添加进度条
tqdm.pandas(desc="Processing")

# 多次读取源文件，进行替换并追加到目标文件中
num_iterations = 5  # 设置重复次数
replace_prob = 0.3  # 设置替换概率
num_threads = 4  # 设置线程数

for _ in tqdm(range(num_iterations), desc="Iterations"):
    source_df = pd.read_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/dialog_train.csv')
    
    # 将数据分割成多片进行处理
    chunks = [source_df['text'][i::num_threads] for i in range(num_threads)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(lambda chunk: process_chunk(chunk, pinyin_dict, replace_prob), chunks), total=num_threads, desc="Processing Chunks"))
    
    # 合并处理结果
    source_df['text'] = pd.concat(results).sort_index().values
    pd.concat([target_df, source_df], ignore_index=True).to_csv('/home/lhy/MM-LLMs/MM-purchase-judgment/MMML/data/qa_new_data/train_text_augment.csv', index=False)

print("处理完成并保存到目标文件。")
