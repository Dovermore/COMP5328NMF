#The following class deals with all the pre-processing steps to be done on the image data
# Author: Lupita Sahu

import numpy as np

class preprocess():
    def __init__(self, image):
        self.image = image

    # Normalisation method.
    def normalise(self): 
        rowmin = np.ndarray.min(self.image, axis=0)
        rowmax = np.ndarray.max(self.image, axis=0)
        range = rowmax - rowmin
        image = (self.image - rowmin) / range
        return image

    # Centering
    def center(self): 
        n_samples = len(self.image)
        mean = self.image.mean(axis=0)
        #global centering
        image = self.image - mean
        # local centering
        image -= image.mean(axis=1).reshape(n_samples, -1)
        return image