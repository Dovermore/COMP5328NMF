# Salt and pepper algs here
def noise(image, p, r): #p for noise level(0-1), r for salt/pepper ratio (0-1)
        row,col = image.shape
        out = np.copy(image)
    # Salt mode
        num_salt = np.ceil(p * image.size * r)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255

    # Pepper mode
        num_pepper = np.ceil(p * image.size * (1. - r))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

#20% noise with 50% salt
V1 = noise(V_hat, 0.2, 0.5)

#40% noise with 20% salt
V2 = noise(V_hat, 0.4, 0.2)

# Plot result (optional, hence commenting out)
#import matplotlib.pyplot as plt
#img_size = [i//3 for i in (92, 112)] # ORL
#ind = 2 # index of demo image.
#plt.figure(figsize=(10,3))
#plt.subplot(131)
#plt.imshow(V_hat[:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
#plt.title('Image(Original)')
#plt.subplot(132)
#plt.imshow(V1[:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
#plt.title('Image(20% noise 50% salt)')
#plt.subplot(133)
#plt.imshow(V2[:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
#plt.title('Image(40% noise 20% salt)')
#plt.show()
