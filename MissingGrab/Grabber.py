import csv
import os
import shutil

def copy_files_from_csv(csv_file, source_folder, destination_folder, filename_column):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 读取CSV文件中的文件名
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row[filename_column]
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # 复制文件
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"Copied: {filename}")
            else:
                print(f"File not found: {filename}")

# 示例用法
csv_file = 'MissingGrab/MissingData.csv'  # CSV文件路径
source_folder = 'data/audio'  # 源文件夹路径
destination_folder = 'Out'  # 目标文件夹路径
filename_column = 'Filename'  # CSV文件中表示文件名的列名

copy_files_from_csv(csv_file, source_folder, destination_folder, filename_column)
