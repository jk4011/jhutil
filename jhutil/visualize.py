from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt
from ipywidgets import interact
import os
from PIL import Image
import numpy as np


def show_video(video_path):
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=600 controls>
      <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
  

def show_images(img_paths):
    def plot_img(img_paths, index=0):
        img_path = img_paths[index]
        img = Image.open(img_path)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.show()

    # Create an interactive slider to browse through the images
    if len(img_paths) > 0:
        interact(plot_img, img_paths=[img_paths], index=(0, len(img_paths)-1))
    else:
        print("No images found in the list.")
