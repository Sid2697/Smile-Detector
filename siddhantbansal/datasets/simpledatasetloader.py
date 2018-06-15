import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # Store the image preprocessor
        self.preprocessors = preprocessors

        # If the preprocessors are none, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # Initialize the list of features and labels
        data = []
        labels = []

        # Loop over the input image
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming that our
            # paths has the format /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]  # Returns tuple '(head/tail)' where 'tail' is everything after the final slash.
            # check to see if our preprocessors are none or not
            if self.preprocessors is not None:
                # Loop over the preprocessor and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as feature vector
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))
