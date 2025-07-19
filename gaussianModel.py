import numpy as np
import pandas as pd
import utils
import os
from scipy.misc import imsave
from math import pi,sqrt,e

#loading Mean and covariance values
data = pd.read_csv('skin.csv')

Mean_skin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['MeanSkin'].values]

covariance_skin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['CovarianceSkin'].values]

Mean_Nonskin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['MeanNonskin'].values]

covariance_Nonskin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['CovarianceNonskin'].values]

#Loading data
listOfExamples = os.listdir('Database/002')
listOfGrundtruths = os.listdir('Database/0002')

listOfExamples.sort()
listOfGrundtruths.sort()

skin_pixels = non_skin_pixels = None

for imageName,maskname in zip(listOfExamples,listOfGrundtruths):
    image = utils.load_image('Database/002/' + imageName) * 255
    desiredResult = utils.load_image('Database/0002/' + maskname)

    # remove saturated pixels

    #image = image * np.all(image < (250 / 255), 2)[..., None]
    Mask = np.all(desiredResult < 250 / 255, 2)

    # gather all skin pixels
    def extract_class_pixels(Mask, image):
        pixels_r = np.expand_dims(np.extract(Mask, image[..., 0]), axis=1)
        pixels_g = np.expand_dims(np.extract(Mask, image[..., 1]), axis=1)
        pixels_b = np.expand_dims(np.extract(Mask, image[..., 2]), axis=1)

        return np.concatenate([pixels_r, pixels_g, pixels_b], axis=1)

    next_skin = extract_class_pixels(Mask, image)
    next_non_skin = extract_class_pixels(Mask == False, image)

    skin_pixels = next_skin if skin_pixels is None else np.concatenate([skin_pixels, next_skin], axis=0)
    non_skin_pixels = next_non_skin if non_skin_pixels is None else np.concatenate([non_skin_pixels, next_non_skin], axis=0)


def gaussian_Mode(x,mu,sigma) -> float:
    return (1/(sqrt((2*pi) ** 3) * sqrt(np.linalg.norm(np.diag(sigma))))) * (e ** (-0.5 * np.dot(np.dot(x - mu, np.linalg.inv(np.diag(sigma))), x - mu)))

gaussian_skin_pixel = sum([gaussian_Mode(pixel, mean, cov) for pixel, mean, cov in zip(skin_pixels, Mean_skin, covariance_skin)])
gaussian_Nonskin_pixel = sum([gaussian_Mode(pixel, mean, cov) for pixel, mean, cov in zip(skin_pixels, Mean_Nonskin, covariance_Nonskin)])


#







