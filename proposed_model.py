import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # Проверка, что d_model (размерность входного тензора) делится на num_heads (количество голов)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Задаем параметры
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Задаем четыре линейные преобразования для Q, K, V и объединенного вывода
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Матричное умножение для получения матрицы скалярных произведений
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Применение маски, если она имеется
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Применение функции softmax для получения вероятностей
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Матричное умножение вероятностей на значение V
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Разбиение входного тензора на num_heads частей
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Объединение num_heads частей в один тензор
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Применяем линейные преобразования для получения Q, K, V
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Вызываем метод многоголового внимания для получения attn_output
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Объединяем num_heads частей в один тензор и применяем линейное преобразование для получения output
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # Создаем два линейных слоя
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Проходим через первый линейный слой и применяем функцию активации ReLU
        output = self.relu(self.fc1(x))
        # Проходим через второй линейный слой
        output = self.fc2(output)
        # Возвращаем выходной тензор
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # Инициализируем механизм внимания, прямой проход с обратной связью и нормализацию
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Проходим входной тензор через механизм внимания
        attn_output = self.self_attn(x, x, x, mask)
        # Добавляем результат механизма внимания к входному тензору, нормализуем и применяем dropout
        x = self.norm1(x + self.dropout(attn_output))
        # Проходим полученный тензор через прямой проход с обратной связью
        ff_output = self.feed_forward(x)
        # Добавляем результат прямого прохода к входному тензору, нормализуем и применяем dropout
        x = self.norm2(x + self.dropout(ff_output))
        # Возвращаем выходной тензор
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Create a tensor of shape (max_seq_length, d_model) filled with zeros
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a tensor with values from 0 to max_seq_length - 1
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate the denominator of the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        # Apply the positional encoding formula for each position and dimension
        # The sine function is applied to even-indexed dimensions, while cosine is applied to odd-indexed dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding tensor as a buffer of the module
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def position(self, t):
        res = torch.zeros_like(t)
        for i, sep in enumerate((t == tk2num['<SEP>']).nonzero().flatten().tolist()):
            res[sep:] = i + 1
        return res.long().tolist()
        
    def forward(self, x, tokens):
        # Add the positional encoding tensor to the input tensor
        # Note that only the first x.size(1) elements of the positional encoding tensor are added,
        # because the input tensor might have been padded to a certain length
        pos = list(map(self.position, tokens))
        return x + self.pe[0][pos, :]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, output_sz, dropout=0.1, max_seq_length=200, seg=0):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model + seg, output_sz)
    
    def padding(self, x, pad):
        return x.masked_fill(pad[:, :, None].repeat(1, 1, x.shape[-1]), 0)
        
    def forward(self, tokens, pad, seg=None, mask=None):
        x = self.embedding(tokens)
        x = self.pos_enc(x, tokens)
        x = self.padding(x, pad)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        if seg is not None:
            x = torch.cat((x, seg[:, None, :].repeat(1, x.shape[1], 1)), 2)
            
        return self.fc(x)