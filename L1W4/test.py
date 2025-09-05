from PIL import Image
import numpy as np

img = Image.open("./datasets/images.webp")
np_img = np.array(img)
print(np_img.shape)