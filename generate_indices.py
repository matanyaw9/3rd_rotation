import random
import numpy as np

n_images = 10
images_indices = random.sample(range(879), n_images)
# images_indices = [22] + images_indices
np.save('/home/jonathak/VisualEncoder/DIP_decoder/images_indices.npy', images_indices)
