import trimesh
import numpy as np
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import os
import alphashape

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

def sample_points_from_mesh(mesh, num_points=4096):
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    return points

def render_pcd(pcd, output_path, filename):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    verts = torch.Tensor(pcd).to(device)
    rgb = torch.ones_like(verts).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    list_R=[]
    list_T=[]
    for x in [0,45,90]:
        temp_R, temp_T = look_at_view_transform(1.5, 15, x)
        list_R.append(temp_R)
        list_T.append(temp_T)
    temp_R, temp_T = look_at_view_transform(1.5, 90, 0)
    list_R.append(temp_R)
    list_T.append(temp_T)

    raster_settings = PointsRasterizationSettings(
        image_size=128, 
        radius = 0.010,
        points_per_pixel = 5
    )

    image_list = []
    for i in tqdm(range(len(list_R))):
        # camera_list.append(PerspectiveCameras(R=list_R[i], T=list_T[i], device=device))
        # cameras = PerspectiveCameras(R=list_R[i], T=list_T[i], device=device)
        cameras = FoVOrthographicCameras(device=device, R=list_R[i], T=list_T[i], znear=0.01)
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0))
        )
        images = renderer(point_cloud)
        image_data = images[0, ..., :3].cpu().numpy()
        image_data = (image_data * 255).astype(np.uint8)

        image_list.append(image_data)

        image = Image.fromarray(image_data)
        SAVE_filename=f'{filename}_{i}.png'
        image.save(os.path.join(output_path,SAVE_filename))

    return image_list

def compute_area(img, alpha_value):
    y, x = np.where(img > 0)
    points_2d = list(zip(x, y))
    alpha_shape_pcd = alphashape.alphashape(points_2d, alpha_value)
    area = alpha_shape_pcd.area

    return area


def main(**kwargs):
    np.set_printoptions(threshold=np.inf)

    model_path = kwargs['model_path']
    mesh = trimesh.load(model_path)
    output_path = kwargs['output_path']

    wire_path = kwargs['wire_path']
    wire_mesh = trimesh.load(wire_path)


    # mesh -> pcd
    if isinstance(mesh, trimesh.Trimesh):
        print("Loaded object is a mesh.")
        points = sample_points_from_mesh(mesh, 4096)
    elif isinstance(mesh, trimesh.points.PointCloud):
        print("Loaded object is a point cloud.")
        points = mesh.vertices
    else:
        print("Loaded object is of unknown type.")

    wire_points = sample_points_from_mesh(wire_mesh, 4096)

    # render
    pcd_img_list = render_pcd(points, output_path, 'pcd')
    wire_img_list = render_pcd(wire_points, output_path, 'wire')

    # print(pcd_img_list[0][:,:,0].shape)
    # print(np.where(pcd_img_list[0][:,:,0]>0))

    # alphashape compute area
    print("compute alphashape and IoU")
    alpha_value = kwargs['alpha_value']

    pcd_area = []
    for i in pcd_img_list:
        pcd_area.append(compute_area(i[:,:,0], alpha_value))

    wire_area = []
    for i in wire_img_list:
        wire_area.append(compute_area(i[:,:,0], alpha_value))

    # compute IoU
    IoU_losses = []
    for i in range(len(pcd_area)):
        IoU_losses.append(wire_area[i]/pcd_area[i])

    print(IoU_losses)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default=f"eval/yeh_data/model.obj")
    # parser.add_argument('--wire_path', type=str, default=f"eval/yeh_data/wire.obj")

    parser.add_argument('--model_path', type=str, default=f"eval/my_data/model.obj")
    parser.add_argument('--wire_path', type=str, default=f"eval/my_data/wire.obj")
    parser.add_argument('--output_path', type=str, default=f"eval/outputs")

    parser.add_argument('--alpha_value', type=float, default=0.08) 

    args = parser.parse_args()

    main(**vars(args))