from utils import *
from layers import *
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def test(model, enc_input, start_symbol):
  enc_outputs, enc_self_attns = model.encoder(enc_input)
  dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
  next_symbol = start_symbol  # 一开始为开始符S
  for i in range(0, tgt_len):
    dec_input[0][i] = next_symbol
    dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
    projected = model.projection(dec_outputs)
    # print("projected:",projected)
    prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  # 选择概率最大的
    next_word = prob.data[i]
    next_symbol = next_word.item()  # 将预测值作为下一次的输入
  return dec_input