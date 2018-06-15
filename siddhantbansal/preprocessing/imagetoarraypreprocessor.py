# importing necessary packages
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        '''
        Apply the kernel utility function that correctly rearranges
        the dimentions of the image
        '''
        return img_to_array(image, data_format=self.dataFormat)
