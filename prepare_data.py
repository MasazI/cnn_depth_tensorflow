#encoding: utf-8
import os
import numpy as np
import h5py
from PIL import Image
import random
import wget


def convert_nyu(path):
    imgdir = os.path.join("data", "nyu_datasets");
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    nyuurl = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
    file = os.path.join("data", "nyu_depth_v2_labeled.mat")
    if not os.path.exists(file):    
        filename = wget.download(nyuurl, out="data")
        print('\n downloaded: ', filename)

    print("load dataset: %s" % (path))
    f = h5py.File(path)

    trains = []
    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (i))
        depth_pil.save(depth_name)

        trains.append((image_name, depth_name))

    random.shuffle(trains)

    if not os.path.exists('train.csv'):  
        os.remove('train.csv')

    with open('train.csv', 'w') as output:
        for (image_name, depth_name) in trains:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'data/nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
