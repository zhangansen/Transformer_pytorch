import logging
import datetime
from model import *
import math
from collections import Counter
import numpy as np
from bleu import *

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
# 设置日志文件名和格式
log_file = 'train.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      loss = criterion(outputs, dec_outputs.view(-1))
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # 将训练结果记录到日志文件中
      logging.info('Epoch: %04d, loss: %.6f' % (epoch + 1, loss.item()))



candidates = []
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.to(device)
for i in range(len(enc_inputs)):
  predict_dec_input = test(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
  predict, _, _, _ = model(enc_inputs[i].view(1, -1).to(device), predict_dec_input)
  # print("predict:",predict)
  predict = predict.data.max(1, keepdim=True)[1]  # 贪婪解码的思想，找到概率最大的词的索引
  # 将预测结果转换为对应的单词，并添加到candidates列表中
  translation = [idx2word[n.item()] for n in predict.squeeze()]
  candidates.append(translation)
  print([src_idx2word[int(i)] for i in enc_inputs[i]], '->',
        [idx2word[n.item()] for n in predict.squeeze()])

candidates = [[word for word in candidate if word not in ['E', 'P']] for candidate in candidates]
# print("candidates:", candidates)


references = []

for s in sentences:
    references.append([word for word in s[2].split() if word not in ['S', 'P', 'E']])

print("references:",references)
print("candidates:",candidates)

# print(len(references))
# print(len(candidates))
print("BLEU分数:",get_bleu(candidates, references))