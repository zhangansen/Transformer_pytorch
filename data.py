import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E'],
             ['我 是 女 生 P', 'S I am a girl', 'I am a girl E'],
             ['我 爱 你 P P', 'S I love you P', 'I love you P E']]  # P: 占位符号，如果当前句子不足固定长度用P占位

src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8, '女': 9, '爱': 10,
             '你': 11}  # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)  # 字典字的个数
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9,
             'girl': 10, 'love': 11, 'you': 12}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}  # 把目标字典转换成 索引：字的形式
# print("idx2word:",idx2word)
tgt_vocab_size = len(tgt_vocab)  # 目标字典尺寸
src_len = len(sentences[0][0].split(" "))  # Encoder输入的最大长度
print("src_len:", src_len)
tgt_len = len(sentences[0][1].split(" "))  # Decoder输入输出最大长度
print("tgt_len:", tgt_len)


# 把sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
print("enc_inputs:", enc_inputs)  # 对应字变为其索引值
print("dec_outputs:", dec_outputs)
print("dec_outputs:", dec_outputs)


# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 5, False)