import numpy as np
from scipy.spatial import ConvexHull

def sample_points_on_triangle(A, B, C, n):
    """
    在三角形 ABC 上均勻採樣 n 個點，利用 barycentric 坐標方法：
    隨機生成 u, v，若 u+v>1 則進行對稱修正，從而保證點均勻分佈在三角形內部。
    """
    u = np.random.rand(n)
    v = np.random.rand(n)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    # 生成的點 = w*A + u*B + v*C
    return np.outer(w, A) + np.outer(u, B) + np.outer(v, C)

def sample_convex_hull_with_normals(points, num_samples=4096):
    """
    輸入 points 為形狀 [n,3] 的點雲，
    先計算凸包，再根據各面面積權重在凸包面上均勻採樣 num_samples 個點，
    並同時計算每個點的 normal（使用對應面外向法向量）。
    
    返回:
      - samples: [num_samples, 3] 採樣點坐標
      - normals: [num_samples, 3] 每個採樣點對應的 normal
    """
    # 計算凸包
    hull = ConvexHull(points)
    simplices = hull.simplices  # 每行為一個三角面頂點索引
    num_faces = simplices.shape[0]
    
    # 使用 hull.equations 的前 3 個係數作為面法向量（根據文件，其方向為外向）
    face_normals = np.zeros((num_faces, 3))
    for i in range(num_faces):
        n = hull.equations[i, :3]
        # 正規化法向量
        face_normals[i] = n / np.linalg.norm(n)
    
    # 計算每個三角形面積
    areas = []
    for simplex in simplices:
        A, B, C = points[simplex]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
        areas.append(area)
    areas = np.array(areas)
    total_area = areas.sum()
    
    # 根據面積建立權重分佈
    probs = areas / total_area
    cum_probs = np.cumsum(probs)
    
    # 根據累積概率隨機選擇每個採樣點所屬的面
    rand_vals = np.random.rand(num_samples)
    face_indices = np.searchsorted(cum_probs, rand_vals)
    
    samples_list = []
    normals_list = []
    
    # 為每個面採樣並記錄其 normal
    unique_faces = np.unique(face_indices)
    for face_idx in unique_faces:
        count = np.sum(face_indices == face_idx)
        simplex = simplices[face_idx]
        A, B, C = points[simplex]
        pts = sample_points_on_triangle(A, B, C, count)
        samples_list.append(pts)
        # 該面所有採樣點均分配相同的法向量
        normals_list.append(np.tile(face_normals[face_idx], (count, 1)))
    
    samples = np.vstack(samples_list)
    normals = np.vstack(normals_list)
    return samples, normals

# 測試代碼
if __name__ == '__main__':
    # 生成隨機三維點雲數據
    pts = np.random.rand(100, 3)
    sampled_points, sampled_normals = sample_convex_hull_with_normals(pts, num_samples=4096)
    print("采樣後點集形狀：", sampled_points.shape)   # 預期為 (4096, 3)
    print("采樣後 normal 形狀：", sampled_normals.shape)  # 預期為 (4096, 3)

    # 將點雲保存到當前文件夾，obj格式
    with open('sampled_points.obj', 'w') as f:
        for i in range(sampled_points.shape[0]):
            f.write(f'v {sampled_points[i, 0]} {sampled_points[i, 1]} {sampled_points[i, 2]}\n')
            f.write(f'vn {sampled_normals[i, 0]} {sampled_normals[i, 1]} {sampled_normals[i, 2]}\n')
