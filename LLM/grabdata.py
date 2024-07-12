import re

# 定义文件路径
file_path = "LLM/result.txt"

# 初始化空列表来保存提取的数据
data = []

# 打开并读取文件内容
with open(file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        # 使用正则表达式提取手机号和tensor值
        match = re.search(r"The result of (\d+) is tensor\(\[\[(\d+\.\d+)\]\],", line)
        if match:
            phone_number = match.group(1)
            tensor_value = float(match.group(2))
            data.append((phone_number, tensor_value))

# 打印提取的数据
for phone, tensor in data:
    print(f"Phone: {phone}, Tensor: {tensor}")

# 可选：将提取的数据保存到CSV文件
import csv

output_file = "output2.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Phone Number", "Tensor Value"])
    writer.writerows(data)

print(f"Data saved to {output_file}")
