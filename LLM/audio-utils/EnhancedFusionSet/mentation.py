import os
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import soundfile as sf

# 定义增强方法
augmentations = Compose([
    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# 读取csv文件
csv_path = "trainlabel.csv"  # 替换为你的csv文件路径
data = pd.read_csv(csv_path)

# 定义wav文件所在的文件夹
wav_folder = "train"  # 替换为你的wav文件夹路径
NewWav = "Enhanced"

# 创建一个新的DataFrame来保存增强后的数据
new_data = []

for idx, row in data.iterrows():
    phone_number = row['audio_id']
    wav_path = os.path.join(wav_folder, f"{phone_number}.wav")
    
    if os.path.exists(wav_path):
        # 读取音频文件
        samples, sample_rate = sf.read(wav_path)
        
        # 进行音频增强
        augmented_samples = augmentations(samples=samples, sample_rate=sample_rate)
        
        # 保存增强后的音频文件
        new_wav_filename = f"{phone_number}_Shift_augmented.wav"
        new_wav_path = os.path.join(NewWav, new_wav_filename)
        sf.write(new_wav_path, augmented_samples, sample_rate)
        
        # 创建新的记录
        new_row = row.copy()
        new_row['audio_id'] = new_wav_filename
        new_data.append(new_row)

# 将新数据保存到新的csv文件
new_csv_path = "your_new_csv_file3.csv"  # 替换为新的csv文件路径
new_data_df = pd.DataFrame(new_data)
new_data_df.to_csv(new_csv_path, index=False)

print("增强后的音频文件已保存，新CSV文件已创建。")
