import os
import utils
import numpy as np
from math import sqrt,pi,e

from scipy.misc import imsave
from functools import reduce
import Fmeasure as FM

listOfExamples = os.listdir('Database/002')
listOfGrundtruths = os.listdir('Database/0002')

listOfExamples.sort()
listOfGrundtruths.sort()
# histogram
#arraye khali
skin_pixels = []
#dota loop hamzan loop tashkil midi
for imageName,maskname in zip(listOfExamples,listOfGrundtruths):
    #to image axaro mizare toosh
    image = utils.load_image('Database/002/' + imageName)
    desiredResult = utils.load_image('Database/0002/' + maskname)

    # remove saturated pixels
#np har pixle 0 ta 255 age hrkodom azin pixlha age koochiktartaghsim be hame meshki harja sefid nis mask bezar
    Mask = np.all(desiredResult < 250 / 255, 2)
#image shape ;shomare pixle harkodom azinaro y mishe tool position 0=1 position 1 mishe 4
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # load pixel and its label
            pixel = image[y,x]
            is_skin = Mask[y,x] == True

            # increment histogram

            if is_skin:
                skin_pixels.append(pixel)


skim = np.transpose(np.asarray(skin_pixels))
mean = np.mean(skim,axis=1)
var = np.cov(skim)


def gaussian_Mode(x, mu, sigma) -> float:
    return (1 / (sqrt((2 * pi) ** 3) * sqrt(np.linalg.norm(sigma)))) * (
                e ** (-0.5 * np.dot(np.dot(x - mu, np.linalg.inv(sigma)), x - mu)))

evaluation_image_files = os.listdir('testdata/002')
evaluation_mask_files =os.listdir('testdata/0002')

evaluation_image_files.sort()
evaluation_mask_files.sort()

F_measure_dump = open('result.txt', 'w')
average_F_measure = []

for index, (imageName, maskname) in enumerate(zip(evaluation_image_files,evaluation_mask_files)):

    # load images

    image = utils.load_image('testdata/002/' + imageName)
    image = image * np.all(image < (250 / 255), 2)[..., None]

    desiredResult = utils.load_image('testdata/0002/' + maskname)
    groundTruthMask = np.all(desiredResult < 250 / 255, 2)
    #utils.show(groundTruthMask)

    # iterate over rgb image and use LUT
    mask = np.zeros(groundTruthMask.shape)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y,x]
            mask[y,x] = gaussian_Mode(pixel, mean, var) > 0.1

    # F

    desiredResult = utils.load_image('testdata/0002/' + maskname)
    # remove saturated pixels
    GT = np.all(desiredResult < 250 / 255, 2)

    F, _ = FM.GetFMeasure(mask, GT, 2, ClassNames=['Non-Skin', 'Skin'], DisplyResults=True)
    average_F_measure.append(F)

    F_measure_dump.write('{}\n'.format(F))
    #NameEnd = ''
    imsave('single_gauss/' + str(index) + '.png', mask)
    # mask is done

print('Average F: {}'.format(sum(average_F_measure) / len(average_F_measure)))