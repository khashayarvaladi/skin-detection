import os
import utils
import numpy as np

from scipy.misc import imsave
from sklearn import mixture
import IOU

listOfExamples = os.listdir('Database/002')
listOfGrundtruths = os.listdir('Database/0002')

listOfExamples.sort()
listOfGrundtruths.sort()

skin_pixels = non_skin_pixels = None

for imageName,maskname in zip(listOfExamples,listOfGrundtruths):
    image = utils.load_image('Database/002/' + imageName)
    desiredResult = utils.load_image('Database/0002/' + maskname)

    # remove saturated pixels

    image = image * np.all(image < (250 / 255), 2)[..., None]
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


gmm_skin_pixels = mixture.GaussianMixture(n_components=16, n_init=2, max_iter=100)
gmmskin = gmm_skin_pixels.fit(skin_pixels)
gmm_non_skin_pixels = mixture.GaussianMixture(n_components=16, n_init=2, max_iter=100)
gmmNonskin = gmm_non_skin_pixels.fit(non_skin_pixels)


# Iterate over Evaluation data

evaluation_image_files = os.listdir('testdata/002')
evaluation_mask_files =os.listdir('testdata/0002')

evaluation_image_files.sort()
evaluation_mask_files.sort()

IOUs = []

for index, (imageName, maskname) in enumerate(zip(evaluation_image_files,evaluation_mask_files)):

    # load images

    image = utils.load_image('testdata/002/' + imageName)
    image = image * np.all(image < (250 / 255), 2)[..., None]

    shape = image.shape

    desiredResult = utils.load_image('testdata/0002/' + maskname)
    groundTruthMask = np.all(desiredResult < 250 / 255, 2)
    #utils.show(groundTruthMask)


    data = np.reshape(image, [-1, 3])

    skin_proba = gmmskin.predict_proba(data).max(axis=1)
    non_skin_proba = gmmNonskin.predict_proba(data).max(axis=1)

    result = np.where(skin_proba < non_skin_proba, [True], [False])



    imsave('Label/' + str(index) + '.png', np.reshape(result.astype(np.uint8) * 255, shape[:2]))









