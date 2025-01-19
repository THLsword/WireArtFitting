import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image
from multiprocessing import Pool

from visual_prior_utils.pcd_renderer import pcd_renderer
from visual_prior_utils.img_alphashape import img_alphashape, multi_process_image
from visual_prior_utils.visual_training import visual_training

from utils.save_data import save_img
from utils.save_data import save_obj

def main(args):
    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    DATA_DIR = args.DATA_DIR
    FILENAME = args.FILENAME
    VIEW_ANGELS = args.VIEW_ANGELS
    PCD_PATH = os.path.join(DATA_DIR, FILENAME)

    # create folder
    if not os.path.exists(args.RENDER_SAVE_DIR):
        os.makedirs(args.RENDER_SAVE_DIR, exist_ok=True)
    if not os.path.exists(args.ALPHA_SAVE_DIR):
        os.makedirs(args.ALPHA_SAVE_DIR, exist_ok=True)
    if not os.path.exists(args.TRAIN_SAVE_DIR):
        os.makedirs(args.TRAIN_SAVE_DIR, exist_ok=True)

    # load normalized pointcloud
    npzfile = np.load(PCD_PATH)
    pointcloud = npzfile['points']

    # render the pointcloud and save images
    print('Start rendering pointcloud')
    RENDER_SAVE_DIR = args.RENDER_SAVE_DIR
    rendered_images = pcd_renderer(pointcloud, VIEW_ANGELS, device)
    for i, img in enumerate(rendered_images):
        image = Image.fromarray(img)
        SAVE_filename=f'{os.path.splitext(FILENAME)[0]}_{i}.png'
        image.save(os.path.join(RENDER_SAVE_DIR, SAVE_filename))

    # alpha shape(on cpu)
    print('Start computing alphashape')
    ALPHA_SIZE = args.ALPHA_SIZE
    EXPAND_SIZE = args.EXPAND_SIZE
    ALPHA_SAVE_DIR = args.ALPHA_SAVE_DIR
    contour_imgs = []
        # multi process 
    process_num = len(rendered_images) if len(rendered_images) <= 4 else 4
    with Pool(processes=process_num) as pool:
        params_list = [(img, ALPHA_SIZE, EXPAND_SIZE) for img in rendered_images]
        contour_imgs = list(
            tqdm(
                pool.imap(multi_process_image, params_list),
                total=len(params_list),
                desc="Processing"
            )
        )
        # save images
    for i, _ in enumerate(contour_imgs):
        image = Image.fromarray(contour_imgs[i])
        SAVE_filename=f'{os.path.splitext(FILENAME)[0]}_{i}.png'
        image.save(os.path.join(ALPHA_SAVE_DIR, SAVE_filename))


    ''' without multi processing
    for i, img in enumerate(rendered_images):
        contour_img = img_alphashape(img, ALPHA_SIZE, EXPAND_SIZE)
        contour_imgs.append(contour_img)
        # save images
        image = Image.fromarray(contour_img)
        SAVE_filename=f'{os.path.splitext(FILENAME)[0]}_{i}.png'
        image.save(os.path.join(ALPHA_SAVE_DIR, SAVE_filename))
    '''

    contour_imgs = np.stack(contour_imgs)

    # visual training
    print('Start visual training')
    EPOCH = args.EPOCH
    TRAIN_SAVE_DIR = args.TRAIN_SAVE_DIR
    training_outputs = visual_training(pointcloud, contour_imgs, EPOCH, VIEW_ANGELS, device)

    # save training results
    images = training_outputs[0].detach().cpu()
    colors = training_outputs[1].detach().cpu()
    for i, img in enumerate(images):
        save_img(img.numpy(), f'{TRAIN_SAVE_DIR}/output_{i}.png')
    torch.save(colors, f'{TRAIN_SAVE_DIR}/weights.pt')
    mask = (colors > 0.5)
    masked_pcd = pointcloud[mask]
    save_obj(f'{TRAIN_SAVE_DIR}/multi_view.obj', masked_pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./data/models/cat")
    parser.add_argument('--RENDER_SAVE_DIR', type=str, default="./prep_outputs/render_outputs")
    parser.add_argument('--ALPHA_SAVE_DIR', type=str, default="./prep_outputs/alpha_outputs")
    parser.add_argument('--TRAIN_SAVE_DIR', type=str, default="./prep_outputs/train_outputs")
    parser.add_argument('--FILENAME', type=str, default="model_normalized_4096.npz")

    parser.add_argument('--EPOCH', type=int, default=50)
    parser.add_argument('--VIEW_ANGELS', type=float, default=[45,90,135,225,270,315])
    parser.add_argument('--ALPHA_SIZE', type=float, default=50.0)
    parser.add_argument('--EXPAND_SIZE', type=int, default=1)

    args = parser.parse_args()

    main(args)