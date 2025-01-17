import os
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
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
from PIL import Image
from tqdm import tqdm


def pcd_renderer(input_pcd, view_angels, device):
    verts = torch.Tensor(input_pcd).to(device)
    rgb = torch.ones_like(verts).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    list_R=[]
    list_T=[]
    for x in view_angels:
        temp_R, temp_T = look_at_view_transform(1.5, 15, x)
        list_R.append(temp_R)
        list_T.append(temp_T)

    raster_settings = PointsRasterizationSettings(
        image_size=128, 
        radius = 0.010,
        points_per_pixel = 5
    )

    rendered_images = []
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
        output = renderer(point_cloud)
        image_data = output[0, ..., :3].cpu().numpy()
        image_data = (image_data * 255).astype(np.uint8)
        rendered_images.append(image_data)

    return rendered_images
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./data/models/cat")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--filename', type=str, default="model_normalized_4096.npz")

    args = parser.parse_args()

    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # create paths
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR, exist_ok=True)

    # load pointcloud
    file_path = os.path.join(args.DATA_DIR, args.filename)
    npzfile = np.load(file_path)
    pointcloud = npzfile['points']

    # view_angels
    view_angels = [45,90,135,225,270,315]

    file_path = os.path.join(args.DATA_DIR, args.filename)
    rendered_images = pcd_renderer(pointcloud, view_angels, device)

    # save image
    for i, data in enumerate(rendered_images):
        image = Image.fromarray(data)
        SAVE_filename=f'{os.path.splitext(args.filename)[0]}_{i}.png'
        print(SAVE_filename)
        image.save(os.path.join(args.SAVE_DIR,SAVE_filename))