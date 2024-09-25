import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# 假设 data_np 是你的主数据数组（1672 x 48 x 33），timestamp_np 是时间戳数据（1672 x 33）
#data_np = np.load('netdata.npz')['OneHotProtocols']
#timestamp_np = np.load('netdata.npz')['Timestamps']
data_np =np.load('d1.npz')['packet_feature'].reshape(2500,50,80)[:1672,:48,:33]
timestamp_np = np.load('d1.npz')['packet_feature'].reshape(2500,80*50)[:1672,:33]
tensor_data = torch.Tensor(data_np).unsqueeze(1)  # 添加一个通道维度

tensor_timestamps = torch.Tensor(timestamp_np)    # 时间戳数据
dataset = TensorDataset(tensor_data, tensor_timestamps, tensor_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # 空间数据的卷积网络
        super(Encoder, self).__init__()
        # 假设输入维度是 [batch_size, 48, 33]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1) # 输出维度: [batch_size, 16, 24, 17]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 输出维度: [batch_size, 32, 12, 9]
       # self.fc = nn.Linear(32 * 12 * 9, latent_dim)
        
        # 时间数据的 LSTM 网络
        self.lstm = nn.LSTM(input_size=33, hidden_size=50, num_layers=32, batch_first=True)
        
        # 合并 LSTM 和卷积网络的输出
        self.fc = nn.Linear(50 + 32 * 12 * 9, latent_dim)

    def forward(self, spatial_data, temporal_data):
        # 处理空间数据
        x = F.relu(self.conv1(spatial_data))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平卷积输出
        #print(x.size())
        # 处理时间数据
        _, (lstm_output, _) = self.lstm(temporal_data)  # 修改这里以获取LSTM的隐藏状态
        #lstm_hidden = lstm_hidden[-1]  # 取 LSTM 输出的最后一个时间步的隐藏状态
        #print(lstm_output.size())
        #print(x.size(0))
        lstm_hidden = lstm_output
        
        lstm_hidden = lstm_hidden.contiguous().view(x.size(0), -1)

        combined = torch.cat((lstm_hidden, x), dim=1)
        return self.fc(combined)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # 为空间数据重建设置卷积转置层
        self.conv_trans1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fc1 = nn.Linear(latent_dim, 32 * 12 * 9)  # 用于转换潜在向量以匹配卷积层的输入尺寸

        # 为时间数据重建设置全连接层或LSTM层
        self.fc2 = nn.Linear(latent_dim, 50)  # 可以调整这个大小
        self.fc3 = nn.Linear(50, 33)  # 输出时间数据的尺寸
        self.adaptive_pool_app= nn.AdaptiveAvgPool2d((48, 33))  # 确保输出尺寸为 48x33
        self.adaptive_pool_time = nn.AdaptiveAvgPool2d((1, 33))
        self.fc4 = nn.Linear(33,33)

    def forward(self, x):
        # 空间数据重建
        x1 = F.relu(self.fc1(x))
        x1 = x1.view(x1.size(0), 32, 12, 9)  # 调整形状以匹配卷积层的输入
        x1 = F.relu(self.conv_trans1(x1))
        x1 = torch.sigmoid(self.conv_trans2(x1))  # 使用 sigmoid 确保输出在 [0, 1] 范围内
        spatial_data = self.adaptive_pool_app(x1)
        # 时间数据重建
        x2 = F.relu(self.fc2(x))
        x2 = self.fc3(x2)
       # x2 = F.softmax(self.fc4(x2))
        x2 = self.fc4(x2)
       # x2 = x2.unsqueeze(2)  # 在第二维上添加一个维度
       # temporal_data = self.adaptive_pool_time(x2)
        temporal_data = torch.sigmoid(x2).reshape(-1,33)

        return spatial_data, temporal_data



# 假设 data_np 是你的主数据数组（1672 x 48 x 33），timestamp_np 是时间戳数据（1672 x 33）
data_np =np.load('d1.npz')['packet_feature'].reshape(2500,50,80)[:1672,:48,:33]
timestamp_np = np.load('d1.npz')['packet_feature'].reshape(2500,80*50)[:1672,:33]
tensor_data = torch.Tensor(data_np).unsqueeze(1)  # 添加一个通道维度

tensor_timestamps = torch.Tensor(timestamp_np)    # 时间戳数据
dataset = TensorDataset(tensor_data, tensor_timestamps, tensor_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True,drop_last=True)

# 定义 num_samples
num_samples = len(data_loader.dataset)



encoder = Encoder(latent_dim=20)
decoder = Decoder(latent_dim=20)
loss_function = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

num_epochs = 500  # 根据需要调整
for epoch in range(num_epochs):
    # 使用 tqdm 显示进度条
    for spatial_data, temporal_data, target_data in tqdm(data_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]'):
        latent = encoder(spatial_data, temporal_data)
        reconstructed_spatial, reconstructed_temporal = decoder(latent)
        # 计算损失，包括空间数据和时间数据
        loss_spatial = loss_function(reconstructed_spatial, spatial_data)
        loss_temporal = loss_function(reconstructed_temporal, temporal_data)
        loss = loss_spatial + loss_temporal  # 组合空间和时间损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 定义生成数据的数量
num_samples = 1672
latent_samples = torch.randn(num_samples, 20)
generated_spatial, generated_temporal = decoder(latent_samples)

# 将生成的数据保存为 npz 文件
np.savez('generated_data_test.npz', spatial_data=generated_spatial.detach().numpy(), temporal_data=generated_temporal.detach().numpy())
