import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SASRec(nn.Module):
    def __init__(self, num_items, embed_size, max_seq_length, num_heads, num_blocks, d_ff, output_sz, dropout_prob):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Linear(num_items, embed_size)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_size)
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, d_ff, dropout_prob)
            for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(embed_size, output_sz)

    def forward(self, input_seq):
        item_embedded = self.item_embedding(input_seq)  # [batch_size, seq_length, embed_size]
        seq_length = item_embedded.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=item_embedded.device)
        positional_embedded = self.positional_embedding(position_ids).unsqueeze(0)  # [1, seq_length, embed_size]
        hidden_state = item_embedded + positional_embedded

        for attention_block in self.attention_blocks:
            hidden_state = attention_block(hidden_state)

        output = self.fc(hidden_state).squeeze(-1)  # [batch_size, seq_length]
        return output


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
        attn_scores = torch.tril(torch.matmul(Q, K.transpose(-2, -1))) / np.sqrt(self.d_k)
        
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


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, d_ff, dropout_prob):
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embed_size)
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x
        attended = self.self_attention(x, x, x)[0]
        x = self.layer_norm1(residual + self.dropout(attended))

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm2(residual + self.dropout(x))
        return x
