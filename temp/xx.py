import time
import random

def generate_unique_id():
    timestamp = str(int(time.time()))  # 获取当前时间戳并转换为字符串
    random_part = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))  # 生成6位随机字母和数字组成的字符串
    
    unique_id = timestamp + '-' + random_part  # 将时间戳和随机数部分连接起来形成唯一ID
    
    return unique_id

s=generate_unique_id()
print(s)