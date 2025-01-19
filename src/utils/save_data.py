import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def save_img(img: np.ndarray, file_name):
    """
    input np.ndarray [256,256,3]  
    value 0~1
    """
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)

def save_obj(filename, points):
    """
    points.shape: [n, 3]
    """
    # points (p_num, 3)
    with open(filename, 'w') as file:
        # 遍历每个点，将其写入文件
        for point in points:
            # 格式化为OBJ文件中的顶点数据行
            file.write("v {} {} {}\n".format(*point))

def save_loss_fig(loss_list, save_dir):
    """
    imput: list of losses
    """
    plt.figure(figsize=(6,4))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()

def save_lr_fig(lr_list, save_dir):
    """
    imput: list of learning rate
    """
    plt.figure(figsize=(6,4))
    plt.plot(lr_list, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    plt.close()