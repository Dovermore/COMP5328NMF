import numpy as np

#The following class deals with creating noise
class noise():
    def __init__(self, image):
        self.image = image
        #self.p = p
        #self.r = r

    # salt and pepper noise which is controlled by 2 parameters.
    #Parameters: p for noise level(0-1), r for salt/pepper ratio (0-1)
    def add_snp(self, p, r): 
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(p * self.image.size * r)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in self.image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(p * self.image.size * (1. - r))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in self.image.shape]
        out[coords] = 0
        return out
    #Gaussian noise here, which is controlled by mean and sigma
    def add_gaussian_noise(self, mean=0, sigma=20):
        """Add Gaussian noise to an image"""
        gaussian_noise = np.random.normal(mean, sigma, self.image.shape)
        gaussian_noise = gaussian_noise.reshape(self.image.shape)
        noisy_image = self.image + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255)
        #noisy_image = noisy_image.astype(np.uint8)
        return noisy_image
