import torch
from modules import Model_BERT, Model_BERT_BILSTM
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id2label = {0: "正常网址", 1: "购物消费", 2: "婚恋交友", 3: "假冒身份", 4: "钓鱼网站", 5: "冒充公检法",
            6: "平台诈骗", 7: "招聘兼职", 8: "杀猪盘", 9: "博彩赌博", 10: "信贷理财", 11: "刷单诈骗", 12: "中奖诈骗"}
label2id = dict([(v, k) for k, v in id2label.items()])

max_len = 256
batch_size = 4

model = Model_BERT_BILSTM()
model.load_state_dict(torch.load("./ckpt/bert-bilstm/best_model_0.8684237786898426.pth"))
model.to(device)

LINK = []
TEXT = []
with open("data/test(unlabeled)_spyder.csv", "r", encoding="utf8") as f:
    for line in f:
        link, text = line.strip().split(",")
        if (not text):
            continue
        LINK.append(link)
        TEXT.append(text.strip())

print(len(LINK))
print(len(TEXT))

class PredictDataset(Dataset):
    def __init__(self, LINK, TEXT):
        super(PredictDataset, self).__init__()
        self.LINK = LINK  # X
        self.TEXT = TEXT  # Y

    def __getitem__(self, item):
        return self.LINK[item], torch.LongTensor(tokenizer.encode(self.LINK[item]+"#"+self.TEXT[item],
                                                 padding="max_length",
                                                 max_length=max_len,
                                                 pad_to_max_length=True,
                                                 truncation=True)).to(device)

    def __len__(self):
        return len(self.Y)

tokenizer = BertTokenizer.from_pretrained("./pretrained_model/bert-base-chinese")

predictDataset = PredictDataset(LINK, TEXT)

#bert预测带文本的
link_label_dict = {}
for link, batch_text in predictDataset:
    output = model(batch_text.unsqueeze(0))
    re = torch.argmax(output, dim=-1)
    link_label_dict[link] = id2label[int(re)]

with open("./data/test(labeled).csv","r", encoding="utf8") as r,\
        open("./data/test(labeled)_text.csv", "w", encoding="utf8") as w:
    for line in r:
        link, label = line.strip().split(",")

        if(link in link_label_dict.keys()):
            label = link_label_dict[link]

        w.write(link+","+label+"\n")