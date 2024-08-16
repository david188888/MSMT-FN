import base64
import hashlib
import hmac
import json
import os
import time
import requests
import sqlite3
import urllib
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'
conn = sqlite3.connect('asr_status.db')
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS asr (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        appId TEXT,
        signa TEXT,
        ts INTEGER,
        fileSize INTEGER,
        fileName TEXT,
        duration TEXT,
        orderId TEXT,
        text TEXT
    )
''')

def insert_into_table(data_dict, table_name='asr', db_file='asr_status.db'):
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()

        # 构造插入语句
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join('?' * len(data_dict))
        sql = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'

        # 执行插入操作
        cur.execute(sql, tuple(data_dict.values()))

        # 提交事务
        conn.commit()
        print(f"Inserted data into {table_name} successfully.")

    except sqlite3.Error as e:
        print(f"Error inserting data into {table_name}: {e}")

    finally:
        # 关闭数据库连接
        if conn:
            conn.close()


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        print("上传部分：")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            "fileSize": file_len,
            "fileName": file_name,
            "duration": "200"
        }
        print("upload参数:", param_dict)
        data = open(upload_file_path, 'rb').read(file_len)

        response = requests.post(url=lfasr_host + api_upload + "?" + urllib.parse.urlencode(param_dict),
                                 headers={"Content-type": "application/json"}, data=data)
        print("upload_url:", response.request.url)
        result = json.loads(response.text)
        print("upload resp:", result)
        return result, param_dict

    def get_result(self):
        uploadresp, previous_dict = self.upload()
        if uploadresp['code'] == '26625':
            raise TimeoutError
        
        orderId = uploadresp['content']['orderId']
        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            'orderId': orderId,
            'resultType': "transfer,predict"
        }

        db_dict = {}
        db_dict.update(previous_dict)
        db_dict['OrderId'] = orderId
        db_dict['result'] = ''
        db_dict['status'] = ''
        # print("")
        # print("查询部分：")
        # print("get result参数:", param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            # print(result)
            status = result['content']['orderInfo']['status']
            print("status=", status)
            if status == 4:
                break
            time.sleep(5)
            try:
                data = result['content']['orderResult']
                all_text = []
                data = json.loads(data)
                lattice = data.get('lattice') if data.get('lattice') else []
                for i in lattice:
                    json_1best = json.loads(i.get('json_1best'))
                    for i in json_1best['st']['rt']:
                        for j in i['ws']:
                            all_text.append(j['cw'][0]['w'])
                all_text = ''.join(all_text)
            except:
                file_name = os.path.basename(self.upload_file_path)
                print(f"exception! in {file_name}")
        return all_text


def convert_to_wav(source_file, target_file):
    # 使用libroso 
    data, sr = librosa.load(source_file, sr=None)
    sf.write(target_file, data, sr)
    return True

def collect_and_convert_audios(folder_path):
    """
    遍历指定文件夹,将所有音频文件转换为wav格式,并返回一个包含所有wav文件路径的列表。
    """
    wav_files = [] # 用于存储所有处理后的.wav文件路径

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            continue # 忽略非文件项
        
        file_name, file_extension = os.path.splitext(file)
        
        # 定义输出文件的路径为.wav格式
        target_file = os.path.join(folder_path, f"{file_name}.wav")
        
        # 如果文件已是.wav格式，直接添加到列表
        if file_extension.lower() == '.wav':
            wav_files.append(file_path)
        else: # 否则，尝试转换并添加
            conversion_success = convert_to_wav(file_path, target_file)
            if conversion_success:
                wav_files.append(target_file)
            else:
                print(f"警告：文件{file}转换失败，未添加至列表。")
    
    return wav_files


def remove_mp3_files(folder_path):
    """
    遍历指定文件夹,删除所有非wav格式的音频文件。
    """
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            continue # 忽略非文件项
        
        file_name, file_extension = os.path.splitext(file)
        
        # 如果文件不是.wav格式，删除
        if file_extension.lower() != '.wav':
            os.remove(file_path)


def write_file(result, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"转写结果文件1:\n{result['orderResult']}\n")

# 输入讯飞开放平台的appid，secret_key和待转写的文件路径
if __name__ == '__main__':
    folder_path = 'Out' # 音频文件夹路径
    wav_files = collect_and_convert_audios(folder_path) # 转换音频文件为.wav格式

    with open('Output-cn_7.10PM.txt', 'a+', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(RequestApi(appid="c1f61b19",
                                              secret_key="77d4ec24b213f1d88ab0327e9e76a325",
                                              upload_file_path=i).get_result): i for i in wav_files}
        for future in as_completed(futures):
            i = futures[future]
            try:
                time.sleep(0.3)
                result = future.result()
                f.write(f"The {i} file has been transcribed as follows:\n{result}\n\n")
                print(result)
            except TimeoutError as e:
                print(f'failed when upload {i}')
                break
            except Exception as e:
                print(f"Error processing file {i}: {e}")

