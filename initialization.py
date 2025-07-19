import os
import utils
import numpy as np

from scipy.misc import imsave
from functools import reduce
import Fmeasure as FM

listOfExamples = os.listdir('Database/002')
listOfGrundtruths = os.listdir('Database/0002')

listOfExamples.sort()
listOfGrundtruths.sort()
# histogram
skin_frequency = np.zeros([255,255,255])
non_skin_frequency = np.zeros([255,255,255])

for imageName,maskname in zip(listOfExamples,listOfGrundtruths):
    image = utils.load_image('Database/002/' + imageName)
    desiredResult = utils.load_image('Database/0002/' + maskname)

    # remove saturated pixels

    image = image * np.all(image < (250 / 255), 2)[..., None]
    Mask = np.all(desiredResult < 250 / 255, 2)


    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # load pixel and its label
            pixel = (image[y,x] * 255).astype(np.uint32)
            is_skin = Mask[y,x] == True

            # increment histogram

            if is_skin:
                skin_frequency[pixel[0], pixel[1], pixel[2]] += 1
            else:
                non_skin_frequency[pixel[0], pixel[1], pixel[2]] += 1

# Here, algorithm has iterated over all images
#confidence_level = 0.9
skin_prob_bayes = skin_frequency / (skin_frequency + non_skin_frequency + 1e-6)

#Skin_LUT = skin_prob_bayes > confidence_level
Skin_LUT = skin_prob_bayes > skin_frequency
# Iterate over Evaluation data

evaluation_image_files = os.listdir('testdata/002')
evaluation_mask_files =os.listdir('testdata/0002')

evaluation_image_files.sort()
evaluation_mask_files.sort()

F_measure_dump = open('result1.txt', 'w')
average_F_measure = []

IOUs = []

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
            pixel = (image[y,x] * 32).astype(np.uint32)
            mask[y,x] = Skin_LUT[pixel[0], pixel[1], pixel[2]]

    #utils.show(mask)
    #line73,31,15,16
    desiredResult = utils.load_image('testdata/0002/' + maskname)
    # remove saturated pixels
    GT = np.all(desiredResult < 250 / 255, 2)

    F, _ = FM.GetFMeasure(mask, GT, 2, ClassNames=['Non-Skin', 'Skin'], DisplyResults=True)
    average_F_measure.append(F)

    F_measure_dump.write('{}\n'.format(F))
    #NameEnd = ''
    imsave('Label/' + str(index) + '.png', mask)
    # mask is done

print('Average F: {}'.format(sum(average_F_measure) / len(average_F_measure)))





