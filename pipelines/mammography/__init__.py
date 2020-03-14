import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from .frcnn_test_vgg import predictt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#pass in the original and mask image, and get the image to be displayed on the website
def get_display_image(orig):
    inp = np.array(Image.open(orig))
    pred=predictt(inp)
    return pred
