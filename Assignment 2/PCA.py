#%%
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image

#%%
# Read image and split channels
img = imageio.imread('Assignment 2/wah.png')
img_np = np.array(img)
img_r = img_np[:,:,0]
img_g = img_np[:,:,1]
img_b = img_np[:,:,2]

#%%

def PCA_2d(image_2d):
    cov = image_2d - np.mean(image_2d)
    eigVal, eigVec = np.linalg.eigh(np.cov(cov))
    p = np.size(eigVec)
    idx = np.argsort(eigVal)
    idx = idx[::-1]
    eigVec = eigVec[:,idx]
    eigVal = eigVal[idx]

    # Number of principal components
    numPC = 30

    if numPC < p or numPC > 0:
        eigVec = eigVec[:,range(numPC)]
    score = np.dot(eigVec.T, cov)
    recon = np.dot(eigVec,score) + np.mean(image_2d)
    reconImg = np.uint8(np.absolute(recon))
    return reconImg
#%%
# Run PCA on each channel and then reconstruct it
img_r_recon, img_g_recon, img_b_recon = PCA_2d(img_r), PCA_2d(img_g), PCA_2d(img_b)
recon_color = np.dstack((img_r_recon, img_g_recon, img_b_recon))
recon_color = Image.fromarray(recon_color)

# Plot PCA image
plt.figure()
plt.imshow(recon_color)
plt.axis('off')
#%%
