import utils
import os
import numpy as np
import pandas as pd
from math import pi,sqrt,e
from scipy.misc import imsave
import Fmeasure as FM

class skin:
    def __init__(self) :
        data = pd.read_csv('skin.csv')
        self.muSkin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['MeanSkin'].values]

        self.sigmaSkin =[np.array([float(pixel) for pixel in sigma[1:-2].split(',')], dtype=np.float) for sigma in data['CovarianceSkin'].values]
    @staticmethod
    def gaussian_Mode(x,mu,sigma) -> float:
        return (1/(sqrt((2*pi) ** 3) * sqrt(np.linalg.norm(np.diag(sigma))))) * (e ** (-0.5 * np.dot(np.dot(x - mu, np.linalg.inv(np.diag(sigma))), x - mu)))

    def probability(self, x):
        return sum([self.gaussian_Mode(x, mean, cov) for mean, cov in zip(self.muSkin, self.sigmaSkin)])

class Nonskin:
    def __init__(self) :
        data = pd.read_csv('skin.csv')
        self.muNonSkin = [np.array([float(pixel) for pixel in mu[1:-2].split(',')], dtype=np.float) for mu in data['MeanNonskin'].values]

        self.sigmaNonSkin =[np.array([float(pixel) for pixel in sigma[1:-2].split(',')], dtype=np.float) for sigma in data['CovarianceNonskin'].values]
    @staticmethod
    def gaussian_Mode(x,mu,sigma) -> float:
        return (1/(sqrt((2*pi) ** 3) * sqrt(np.linalg.norm(np.diag(sigma))))) * (e ** (-0.5 * np.dot(np.dot(x - mu, np.linalg.inv(np.diag(sigma))), x - mu)))

    def probability(self, x) -> float:
        return sum([self.gaussian_Mode(x, mean, cov) for mean, cov in zip(self.muNonSkin, self.sigmaNonSkin)])

skin_classifier = skin()
Nonskin_classifier = Nonskin()

# Iterate over Evaluation data

evaluation_image_files = os.listdir('testdata/002')
evaluation_mask_files =os.listdir('testdata/0002')

evaluation_image_files.sort()
evaluation_mask_files.sort()

#IOUs = []
F_measure_dump = open('resultTestgaussian.txt', 'w')
average_F_measure = []

for index, (imageName, maskname) in enumerate(zip(evaluation_image_files,evaluation_mask_files)):

    # load images

    image = utils.load_image('testdata/002/' + imageName)*255

    shape = image.shape

    desiredResult = utils.load_image('testdata/0002/' + maskname)*255
    groundTruthMask = np.all(desiredResult < 250 / 255, 2)
    #utils.show(groundTruthMask)


    data = np.reshape(image, [-1, 3])

    result = np.array([skin_classifier.probability(data[i, :]) > Nonskin_classifier.probability(data[i, :]) for i in range(data.shape[0])])

    desiredResult = utils.load_image('testdata/0002/' + maskname)
    # remove saturated pixels
    GT = np.all(desiredResult < 250 / 255, 2)

    result = np.reshape(result, GT.shape)

    F, _ = FM.GetFMeasure(result, GT, 2, ClassNames=['Non-Skin', 'Skin'], DisplyResults=True)
    average_F_measure.append(F)

    F_measure_dump.write('{}\n'.format(F))

    imsave('Label/' + str(index) + '.png', np.reshape(result.astype(np.uint8) * 255, shape[:2]))

print('Average F: {}'.format(sum(average_F_measure) / len(average_F_measure)))