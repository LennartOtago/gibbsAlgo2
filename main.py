import numpy as np
import numpy.random as rd
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.linalg as lin


def sample_laplacian(siz_of_img):
    # ***
    # sample from Laplacian matrix in this case with variance 4 and 0 mean
    # original L L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])
    # ***

    v = np.zeros((siz_of_img, siz_of_img))
    # top to bottom first row
    for i in range(0, siz_of_img):
        rand_num = np.array([-1, 1]) * rd.normal(0, 1)#np.sqrt(2))#(1 / np.sqrt(2)) *
        v[0, i], v[-1, i] = [v[0, i], v[-1, i]] + np.array(rand_num)

    for j in range(0, siz_of_img):
        for i in range(0, siz_of_img - 1):
            rand_num =  np.array([-1, 1]) * rd.normal(0, 1)#(1 / np.sqrt(2)) *
            v[j, i], v[j, i + 1] = [v[j, i], v[j, i + 1]] + np.array(rand_num)

    # all normal up and down neighbours

    for j in range(0, siz_of_img - 1):
        for i in range(0, siz_of_img):
            rand_num =  np.array([-1, 1]) * rd.normal(0,1)#np.sqrt(2))#(1 / np.sqrt(2)) *
            v[j, i], v[j + 1, i] = [v[j, i], v[j + 1, i]] + np.array(rand_num)

    # all left right boundaries neighbours

    for i in range(0, siz_of_img):
        rand_num =  np.array([-1, 1]) * rd.normal(0,1)# np.sqrt(2))#(1 / np.sqrt(2))
        v[i, 0], v[i, -1] = [v[i, 0], v[i, -1]] + np.array(rand_num)

    return v


# import all functions
gray_img = mpimg.imread('jupiter1.tif')
# get psf from satellite
org_img = np.array(gray_img)  # /sum(sum(np.array(gray_img)))
IMG_FOUR = fft2(org_img)
xPos = 234
yPos = 85  # Pixel at centre of satellite
sat_img_org = org_img[yPos - 16: yPos + 16, xPos - 16:xPos + 16]
sat_img = sat_img_org / (sum(sum(sat_img_org)))
sat_img[sat_img < 0.05 * np.max(sat_img)] = 0

siz = len(org_img)

PSF = fft2(sat_img, (siz, siz))
L_org = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
L = fft2(L_org, (siz, siz))

a_gamma = 1
a_rho = 1
b_gamma = 1e-4
b_rho = 1e-4
n_sample = 70
res_img = np.zeros((n_sample, 256, 256))
rho = np.zeros(n_sample)
gamma = np.zeros(n_sample)

# initialize first sample

rho[0] = 5.16e-5

gamma[0] = 0.218

v_2 = sample_laplacian(siz)

v_rd = rd.normal(0, 1, 256 ** 2).reshape((256, 256))

W = np.sqrt(gamma[0]) * np.conj(PSF) * fft2(v_rd) + np.sqrt(rho[0]) * fft2(v_2)

IMG_STORE = (gamma[0] * IMG_FOUR * np.conj(PSF) + W) / (rho[0] * abs(L) + gamma[0] * abs(PSF) ** 2)

im = ifft2(IMG_STORE).real

res_img[0] = im

plt.imshow(res_img[0], cmap='gray')
plt.show()
norm_res = np.linalg.norm(IMG_STORE * PSF - IMG_FOUR) / 256

norm_L = np.sqrt(sum(sum(abs(IMG_STORE.conj() * L * IMG_STORE)))) / 256

for n in range(1, n_sample):
    norm_L = np.sqrt(sum(sum(abs(IMG_STORE.conj() * L * IMG_STORE)))) / 256
    norm_res = np.linalg.norm(IMG_STORE * PSF - IMG_FOUR) / 256
    shape_gamma, scale_gamma = siz ** 2 / 2 + a_gamma, 1 / (0.5 * norm_res ** 2 + b_gamma)  # 1,1e4 #

    gamma[n] = np.random.default_rng().gamma(shape_gamma, scale_gamma)
    shape_rho, scale_rho = siz ** 2 / 2 + a_rho, 1 / (0.5 * norm_L ** 2 + b_rho)  # 1,1e4 #
    rho[n] = np.random.default_rng().gamma(shape_rho, scale_rho)
    v_2 = sample_laplacian(siz)
    v_rd = rd.normal(0, 1, 256 ** 2).reshape((256, 256))
    W = np.sqrt(gamma[n]) * np.conj(PSF) * fft2(v_rd) + np.sqrt(rho[n]) * fft2(v_2)
    print(rho[n])
    print(gamma[n])
    IMG_STORE = (gamma[n] * IMG_FOUR * np.conj(PSF) + W) / (
            rho[n] * abs(L) + gamma[n] * abs(PSF) ** 2)
    res_img[n] = ifft2(
        (gamma[n] * IMG_FOUR * np.conj(PSF) + W) / (rho[n] * abs(L) + gamma[n] * abs(PSF) ** 2)).real


plt.imshow(res_img[-1], cmap='gray')
plt.show()
