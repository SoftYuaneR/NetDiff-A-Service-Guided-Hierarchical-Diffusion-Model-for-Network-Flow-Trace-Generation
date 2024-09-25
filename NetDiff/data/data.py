import pandas as pd
import numpy as np

# Define a function to process the data for each IP and Protocol
def process_data(group, sequence_length=10, max_zeros=5):
    # Initialize the matrix for traffic features and vector for timestamps
    traffic_features = np.zeros((sequence_length, 4))  # 4 features
    timestamps = np.zeros(sequence_length)

    # Sort the group based on Timestamp to maintain the sequence
    group = group.sort_values('Timestamp')

    # Truncate or pad the data to match the sequence length
    for i, (idx, row) in enumerate(group.iterrows()):
        if i >= sequence_length:
            break  # Stop if we have enough data
        traffic_features[i] = row[['Flow.Duration', 'Average.Packet.Size', 'Total.Fwd.Packets', 'Total.Backward.Packets']]
        timestamps[i] = row['Timestamp']

    # Check if the sequence has too many zeros and should be discarded
    if (traffic_features == 0).sum(axis=0).max() > max_zeros:
        return None, None

    return traffic_features, timestamps

# Load the dataset
df = pd.read_csv('train_data.csv')
import pandas as pd
from datetime import datetime

# 定义将日期时间字符串转换为时间戳的函数
def convert_to_timestamp(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y%H:%M:%S').timestamp()

# 加载CSV文件
#df = pd.read_csv('/mnt/data/train_data.csv')

# 将'Timestamp'列中的所有日期时间字符串转换为时间戳
df['Timestamp'] = df['Timestamp'].apply(convert_to_timestamp)

# ...后续处理代码...

# Filter out the data for each IP and Protocol
processed_data = {
    'ProtocolName': [],
    'Source.IP': [],
    'Traffic_Features': [],
    'Timestamps': []
}

# Process the data
for (protocol, ip), group in df.groupby(['ProtocolName', 'Source.IP']):
    traffic_features, timestamps = process_data(group)
    if traffic_features is not None:
        processed_data['ProtocolName'].append(protocol)
        processed_data['Source.IP'].append(ip)
        processed_data['Traffic_Features'].append(traffic_features)
        processed_data['Timestamps'].append(timestamps)

# Convert lists to arrays and save as .npz
np.savez('test_data.npz', 
         ProtocolName=np.array(processed_data['ProtocolName']),
         Source_IP=np.array(processed_data['Source.IP']),
         Traffic_Features=np.array(processed_data['Traffic_Features'], dtype=object),
         Timestamps=np.array(processed_data['Timestamps'], dtype=object))



