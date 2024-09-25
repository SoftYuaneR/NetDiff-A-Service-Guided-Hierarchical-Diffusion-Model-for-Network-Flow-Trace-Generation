import numpy as np

# Load the dataset
data = np.load('modified_test_data.npz')
traffic_features = data['Traffic_Features']

# Normalize each feature individually across all samples and time steps
normalized_traffic_features = traffic_features.copy()
for feature_index in range(traffic_features.shape[1]):
    # Extract one feature for all samples and time steps
    feature_values = traffic_features[:, feature_index, :].reshape(-1)
    min_val = feature_values.min()
    max_val = feature_values.max()
    # Avoid division by zero if all values are the same
    if min_val != max_val:
        # Normalize the feature values
        normalized_feature_values = (feature_values - min_val) / (max_val - min_val)
    else:
        normalized_feature_values = np.zeros_like(feature_values)
    # Reshape and assign the normalized values back to the corresponding feature
    normalized_traffic_features[:, feature_index, :] = normalized_feature_values.reshape(
        traffic_features.shape[0], traffic_features.shape[2])

# Save the normalized data to a new npz file
normalized_file_path = 'at.npz'
np.savez(normalized_file_path, traffic_features=normalized_traffic_features)

