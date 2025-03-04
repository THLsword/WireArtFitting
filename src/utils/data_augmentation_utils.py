import numpy as np
import torch

def random_rotate_point_cloud(point_cloud, batch_size, angle_range_deg=10):
    """
    對輸入點雲做隨機旋轉數據增強。
    
    參數:
      point_cloud: np.array, cpu, shape [4096, 3]，輸入的點雲數據。
      batch_size: int，需要生成幾組不同旋轉的點雲。
      angle_range_deg: float，每個軸旋轉角度範圍（單位：度），旋轉角度將均勻地從 [-angle_range_deg, angle_range_deg] 中選取。
    
    返回:
      rotated_points: np.array, shape [batch_size, 4096, 3]，旋轉後的點雲。
      rotation_matrices: np.array, shape [batch_size, 3, 3]，每次旋轉所使用的旋轉矩陣。
    """
    rotated_points = []
    rotation_matrices = []
    
    # 將角度範圍轉換為弧度
    angle_range = np.deg2rad(angle_range_deg)
    
    for _ in range(batch_size):
        # 為 x, y, z 軸分別生成隨機旋轉角度（均勻分布在 [-angle_range, angle_range]）
        angle_x = np.random.uniform(-angle_range, angle_range)
        angle_y = np.random.uniform(-angle_range, angle_range)
        angle_z = np.random.uniform(-angle_range, angle_range)
        
        # 計算繞 x 軸旋轉矩陣
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x),  np.cos(angle_x)]])
        # 計算繞 y 軸旋轉矩陣
        Ry = np.array([[ np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])
        # 計算繞 z 軸旋轉矩陣
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z),  np.cos(angle_z), 0],
                       [0, 0, 1]])
        
        # 組合旋轉矩陣，這裡採用 Rz * Ry * Rx 的順序
        R = Rz @ Ry @ Rx
        
        # 對於點雲（每一行代表一個點），由於我們的 R 是針對列向量的，
        # 因此對點雲進行旋轉時需用 R 的轉置：p' = p dot R.T
        rotated = np.dot(point_cloud, R.T)
        
        rotated_points.append(rotated)
        rotation_matrices.append(R)
    
    # 將列表轉換為 numpy array
    rotated_points = np.stack(rotated_points, axis=0)      # shape: [batch_size, 4096, 3]
    rotation_matrices = np.stack(rotation_matrices, axis=0)  # shape: [batch_size, 3, 3]
    
    return rotated_points, rotation_matrices

def add_gaussian_noise(point_cloud, noise_std):
    """
    Add Gaussian noise to a point cloud.

    Parameters:
        point_cloud (numpy.ndarray): Input point cloud of shape (B, N, 3).
        noise_std (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        numpy.ndarray: Point cloud with added Gaussian noise.
    """

    # Generate Gaussian noise with mean=0 and std=noise_std
    noise = np.random.normal(loc=0.0, scale=noise_std, size=point_cloud.shape)

    # Add noise to the original point cloud
    noisy_point_cloud = point_cloud + noise

    return noisy_point_cloud