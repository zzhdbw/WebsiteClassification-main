# -*- coding: utf-8 -*-
import os

basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'model')
modelfile_path = os.path.join(model_path, 'model.pkl')
vectorfile_path = os.path.join(model_path, 'vector.pkl')

# 分词
def getTokens(input):
    web_url = input.lower()
    urltoken = []
    dot_slash = []
    slash = str(web_url).split('/')

    for i in slash:
        r1 = str(i).split('-')
        token_slash = []
        for j in range(0, len(r1)):
            r2 = str(r1[j]).split('.')
            token_slash = token_slash + r2
        dot_slash = dot_slash + r1 + token_slash

    urltoken = list(set(dot_slash))

    if 'com' in urltoken:
        urltoken.remove('com')
    if 'cn' in urltoken:
        urltoken.remove('cn')
    return urltoken

if __name__ == '__main__':
    text = "svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt"
    result = getTokens(text)
    # print(result)
