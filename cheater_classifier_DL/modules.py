from torch.nn import Module, Embedding, Linear
from transformers import BertModel
import torch

id2label = {0: "正常网址", 1: "购物消费", 2: "婚恋交友", 3: "假冒身份", 4: "钓鱼网站", 5: "冒充公检法", 6: "平台诈骗",
            7: "招聘兼职", 8: "杀猪盘", 9: "博彩赌博", 10: "信贷理财", 11: "刷单诈骗", 12: "中奖诈骗"}
label2id = dict([(v, k) for k, v in id2label.items()])

class Model_BERT(Module):
    def __init__(self):
        super(Model_BERT, self).__init__()
        self.encoder = BertModel.from_pretrained("./pretrained_model/bert-base-chinese")

        self.linear = Linear(768, len(id2label))

    def forward(self, input):  # [batch_size, max_len]
        last_hidden_state, pooler_output = self.encoder(input)  # [batch_size, max_len, word_dim]

        # last_hidden_state = output.last_hidden_state
        # pooler_output = output.pooler_output

        final_out = self.linear(pooler_output)  # [batch_size, 12]

        return final_out


class Model_BERT_BILSTM(Module):
    def __init__(self):
        super(Model_BERT_BILSTM, self).__init__()
        self.encoder = BertModel.from_pretrained("./pretrained_model/bert-base-chinese")
        self.lstm = torch.nn.LSTM(768, 256, bidirectional=True, batch_first=True)
        self.linear = Linear(32768, len(id2label))

    def forward(self, input):  # [batch_size, max_len]
        last_hidden_state, pooler_output = self.encoder(input)  # [batch_size, max_len, word_dim]
        # last_hidden_state = output.last_hidden_state
        # pooler_output = output.pooler_output

        sequence_out, hidden_state = self.lstm(last_hidden_state)#torch.Size([4, 256, 512])

        output = torch.max_pool1d(sequence_out, kernel_size=4)
        output = output.view(output.shape[0], -1)
        final_out = self.linear(output)  # [batch_size, 12]

        return final_out
