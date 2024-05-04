from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
from torch.nn import functional as F
from multipledispatch import dispatch

def cal_tangent_vector(t, params):
    A = np.array([[-1, 1, 0, 0],
                [2, -4, 2, 0],
                [-1, 3, -3, 1],
                [0, 0, 0, 0]])
    dim = np.array([0, 1, 2, 3]) 
    dt = np.power(t, dim) @ A @ params
    tangent_v = dt / np.sqrt(np.sum(dt**2, axis=-1, keepdims=True))
    
    return tangent_v

@dispatch(np.ndarray, np.ndarray)
def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    # row: coefficient for all control points with different power of t
    A = np.array([[1, 0, 0, 0],
                [-3, 3, 0, 0],
                [3, -6, 3, 0],
                [-1, 3, -3, 1]])
    t = np.power(t, np.array([0, 1, 2, 3])) # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

@dispatch(torch.Tensor, torch.Tensor)
def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    # row: coefficient for all control points with different power of t
    A = params.new_tensor([[1, 0, 0, 0],
                           [-3, 3, 0, 0],
                           [3, -6, 3, 0],
                           [-1, 3, -3, 1]])
    

    t = t.pow(t.new_tensor([0, 1, 2, 3]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def bezier_sample_8(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [-7, 7, 0, 0, 0, 0, 0, 0],
                    [21, -42, 21, 0, 0, 0, 0, 0],
                    [-35, 105, -105, 35, 0, 0, 0, 0],
                    [35, -140, 210, -140, 35, 0, 0, 0],
                    [-21, 105, -210, 210, -105, 21, 0, 0],
                    [7, -42, 105, -140, 105, -42, 7, 0],
                    [-1, 7, -21, 35, -35, 21, -7, 1]
                    ])

    t = t.pow(t.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def process_curves(params, vertex_idxs, curve_idxs):
    """ Process params to curve control points.
    Args: 
        params: (batch_size, output_dim)
        vertex_idxs: (point#, 3)
        curve_idx: (curve#, 4)
    Return:
        vertices: (batch_size, point#, 3)
        curves: (batch_size, curve#, 4, 3)
    """
    vertices = params.clone()[:, vertex_idxs]
    curves = vertices[:, curve_idxs] #(b, curve#, 4, 3)

    return vertices, curves

def process_FoldNet_curves(params, vertex_idxs, curve_idxs):
    """ Process params to curve control points.
    Args: 
        params: Decoder's output (batch_size, num_points(122) ,3)
        vertex_idxs: (point#, 3)
        curve_idx: (curve#, 4)
    Return:
        vertices: (batch_size, point#, 3)
        curves: (batch_size, curve#, 4, 3)
    """
    vertices = params.clone()
    curves = vertices[:, curve_idxs] #(b, curve#, 4, 3)

    return vertices, curves

def write_curve_points(file, curves, control_point_num = 4, res=300):
    """Write Bezier curve points to an obj file."""
    r_linspace = torch.linspace(0, 1, res).to(curves).flatten()[..., None]
    if control_point_num == 4:
        points = bezier_sample(r_linspace, curves)
    elif control_point_num == 8:
        points = bezier_sample_8(r_linspace, curves)

    c = [0.6, 0.6, 0.6] # (r, g, b)

    with open(file, 'w') as f:
        for p, vertice in enumerate(points):
            for x, y, z in vertice:
                f.write(f'v {x} {y} {z} {c[0]} {c[1]} {c[2]}\n')