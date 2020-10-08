import numpy as np

class noise():
    def __init__(self, image, p, r):
        self.image = image
        self.p = p
        self.r = r

    # Salt and pepper algs here.
    #p for noise level(0-1), r for salt/pepper ratio (0-1)
    
    def add_snp(self): 
        #row, col = image.shape
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(self.p * self.image.size * self.r)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in self.image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(self.p * self.image.size * (1. - self.r))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in self.image.shape]
        out[coords] = 0
        return out
