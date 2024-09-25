import torch
import torch.nn as nn
import datetime
import numpy as np
class TimestampDiffEmbedding(nn.Module):
    def __init__(self, embedding_length):
        super().__init__()
        self.embedding_layer = nn.Linear(1, embedding_length)

    def forward(self, timestamp_diffs):
        # timestamp_diffs 是已经计算好的时间差异张量

        # 嵌入表示
        timestamp_embed = self.embedding_layer(timestamp_diffs.unsqueeze(-1))
        return timestamp_embed

# 示例用法
embedding_length = 16  # 假设嵌入长度为 128
model_Time = TimestampDiffEmbedding(embedding_length)

# 假设 t 是您的时间戳数据，形状为 (1672, 33)，这里用示例数据
# t = torch.randint(1_560_000_000, 1_600_000_000, (1672, 33), dtype=torch.float32)

# 计算相邻时间戳之间的差异（以天数为例）
timestamp_diffs = np.load('generated_data_test.npz')['temporal_data']

timestamp_diffs_tensor = torch.tensor(timestamp_diffs, dtype=torch.float32)
timestamp_embed = model_Time(timestamp_diffs_tensor)



class AppEmbedding(nn.Module):
    def __init__(self, num_apps, embedding_dim):
        super().__init__()
        self.app_embedding_layer = nn.Linear(num_apps, embedding_dim)

    def forward(self, one_hot_matrix):
        # 确保 one_hot_matrix 是浮点型
        one_hot_matrix = one_hot_matrix.to(torch.float32)

        # one_hot_matrix 的形状: (batch_size, num_apps, sequence_length)
        batch_size, num_apps, sequence_length = one_hot_matrix.shape

        # 调整矩阵形状
        one_hot_matrix = one_hot_matrix.transpose(1, 2)
        one_hot_matrix_flat = one_hot_matrix.reshape(-1, num_apps)

        # 应用嵌入层
        app_embeddings = self.app_embedding_layer(one_hot_matrix_flat)

        # 重新调整嵌入矩阵的形状
        app_embeddings = app_embeddings.reshape(batch_size, sequence_length, -1)
        return app_embeddings

# 示例用法
num_apps = 48  # 48 种 APP
embedding_dim = 16  # 嵌入维度为 128
model_APP = AppEmbedding(num_apps, embedding_dim)
o = np.load('generated_data_test.npz')['spatial_data']
oo = o.squeeze()
# 假设 one_hot_matrix 是您的独热编码矩阵，形状为 (1672, 48, 33)
one_hot_matrix = torch.tensor(oo)
app_embeddings = model_APP(one_hot_matrix)
time_embeddings = timestamp_embed.squeeze()
np.savez('embedding.npz', time_embedding =timestamp_embed.detach().numpy(),app_embedding = app_embeddings.detach().numpy() )

