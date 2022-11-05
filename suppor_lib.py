import matplotlib.cm as cm
import matplotlib
import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt
import config
import cv2
import torchvision.transforms as transforms

pi = math.pi


def rotate_images(input_path, output_path):
    """Rotate 360-degree images"""
    for _, _, files in os.walk(input_path):
        for name in files:
            for i in range(6):
                angle = str(-180 + i * 60)
                # execute rotation cmd: ffmpeg -i input.png  -vf v360=e:e:yaw=angle output.png
                cmd = 'ffmpeg -i ' + input_path + name + ' -vf v360=e:e:yaw=' + angle + ' ' + \
                      output_path + name.split('.')[0] + '_' + str(i) + '.png'
                os.system(cmd)


def image_process(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.image_size[1], config.image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    image = transform(image)
    return image


def sphere2plane(sphere_cord, height_width=None):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y) shape = (n, 2) """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]
    if height_width is None:
        y = (lat + 90) / 180
        x = (lon + 180) / 360
    else:
        y = (lat + 90) / 180 * height_width[0]
        x = (lon + 180) / 360 * height_width[1]
    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)


def plane2sphere(plane_cord, height_width=None):
    """ input:  (x, y) shape = (n, 2)
        output: (lat, lon) shape = (n, 2) """
    y, x = plane_cord[:, 0], plane_cord[:, 1]
    if (height_width is None) & (torch.any(plane_cord <= 1).item()):
        lat = (y - 0.5) * 180
        lon = (x - 0.5) * 360
    else:
        lat = (y / height_width[0] - 0.5) * 180
        lon = (x / height_width[1] - 0.5) * 360
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def sphere2xyz(shpere_cord):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y, z) shape = (n, 3) """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]
    lat = lat / 180 * pi
    lon = lon / 180 * pi
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)


def xyz2sphere(threeD_cord):
    """ input: (x, y, z) shape = (n, 3)
        output: (lat, lon) shape = (n, 2) """
    x, y, z = threeD_cord[:, 0], threeD_cord[:, 1], threeD_cord[:, 2]
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    lat = lat / pi * 180
    lon = lon / pi * 180
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def xyz2plane(threeD_cord, height_width=None):
    """ input: (x, y, z) shape = (n, 3)
        output: (x, y) shape = (n, 2) """
    sphere_cords = xyz2sphere(threeD_cord)
    plane_cors = sphere2plane(sphere_cords, height_width)
    return plane_cors


def plot_scanpaths(scanpaths, img_path, lengths, save_path, img_height=256, img_witdth=512):
    # Plot predicted scanpaths
    # this code is modified on the basis of ScanGAN https://github.com/DaniMS-ZGZ/ScanGAN360/

    image = cv2.resize(matplotlib.image.imread(img_path), (img_witdth, img_height))

    image_name = img_path.split('/')[-1].split('.')[0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axs = plt.subplots(4, 5, constrained_layout=True)
    fig.set_size_inches(48, 24)

    # plot 20 scanpaths for an image.
    for user_i in range(20):
        idx1 = int(user_i / 5)
        idx2 = user_i % 5

        points_x = []
        points_y = []
        for sample_i in range(lengths[user_i]):
            points_x.append(scanpaths[user_i][sample_i][1])
            points_y.append(scanpaths[user_i][sample_i][0])

        colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

        previous_point = None
        for num, x, y, c in zip(range(0, len(points_x)), points_x, points_y, colors):
            x = x * img_witdth
            y = y * img_height
            markersize = 28.
            if previous_point is not None:
                if abs(previous_point[0] - x) < (img_witdth / 2):
                    axs[idx1, idx2].plot([x, previous_point[0]], [y, previous_point[1]], color='blue', linewidth=8.,
                                         alpha=0.35)
                else:
                    h_diff = (y - previous_point[1]) / 2
                    if (x > previous_point[0]):  # X is on the right, Previous is on the Left
                        axs[idx1, idx2].plot([previous_point[0], 0],
                                             [previous_point[1], previous_point[1] + h_diff], color='blue',
                                             linewidth=8., alpha=0.35)
                        axs[idx1, idx2].plot([img_witdth, x], [previous_point[1] + h_diff, y],
                                             color='blue', linewidth=8., alpha=0.35)
                    else:
                        axs[idx1, idx2].plot([previous_point[0], img_witdth],
                                             [previous_point[1], previous_point[1] + h_diff], color='blue',
                                             linewidth=8., alpha=0.35)
                        axs[idx1, idx2].plot([0, x], [previous_point[1] + h_diff, y], color='blue', linewidth=8.,
                                             alpha=0.35)
            previous_point = [x, y]
            axs[idx1, idx2].plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
        axs[idx1, idx2].imshow(image)
        axs[idx1, idx2].axis('off')

    plt.savefig(save_path + '/sp_' + image_name + ".png")
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.clf()
    plt.close('all')
