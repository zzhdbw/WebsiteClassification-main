
import re
import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

def read(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    } # 设置请求头信息
    response = requests.get(url, headers=headers, timeout=(3.00, 5)) # 发送HTTP请求

    # 如果响应状态码为200，则说明请求成功
    if response.status_code == 200:
        html_doc = response.text # 获取HTML文档内容
        soup = BeautifulSoup(html_doc, 'html.parser') # 创建BeautifulSoup对象
        text = soup.get_text().replace(" ", "").replace("\n", "") # 获取网页的全部文本内容

        pattern = re.compile('[\u4e00-\u9fa5]+')
        result = pattern.findall(text)

        # 将匹配到的中文和数字连接起来，形成新字符串
        new_str = ''.join(result)


        return new_str
    else:
        return ""

with open("data/processed/train1补充.csv", "r", encoding="utf8") as f,\
        open("data/processed/train1补充_spyder.csv", "w", encoding="utf8") as w:
    i = 0
    for line in tqdm(f.readlines()):
        i += 1

        try:
            line = line.split(",")
        except Exception:
            continue
        link, label = line[0], line[1].strip()

        text = "None"
        try:
            text = read("http://" + link)
        except:
            pass
            # print(link+"访问不到！")
        else:
            # print(link)
            pass

        # print(link, text[:3], label)

        w.write(link+","+text+","+label+"\n")

# print(read(url))