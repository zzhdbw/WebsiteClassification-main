import multiprocessing
import time
import re
import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

def read(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    } # 设置请求头信息
    response = requests.get(url, headers=headers, timeout=(2.00, 2)) # 发送HTTP请求

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

def frocess_test(start, end):
    with open("data/processed/test(unlabeled).csv", "r", encoding="utf8") as f,\
            open("data/processed/test(unlabeled)_spyder_{}.csv".format(end), "w", encoding="utf8") as w:
        bar =  tqdm(f.readlines()[start:end])
        for line in bar:
            link = line.strip()

            text = "None"

            try:
                text = read("http://" + link)
            except:
                pass

            w.write(link+","+text+"\n")

            bar.set_postfix_str(str(end))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 8)

    for i in range(8):
        start = i * 100000
        end = (i+1) * 100000
        pool.apply_async(frocess_test, (start, end))

    print("Mark")
    pool.close()
    pool.join()
    print("程序结束")