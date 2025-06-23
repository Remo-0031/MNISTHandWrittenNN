import pandas as pd
import numpy as np
import cv2


def preProcessScreenShot():
    image_files = pd.DataFrame(columns=range(784)).add_prefix('pixels_')
    for i in range(1, 6):
        r_image = cv2.imread(f'ScreenShot.png')
        numpy_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
        numpy_image = 255 - numpy_image
        image = cv2.resize(numpy_image, (28, 28)).astype(np.int32)
        image = image.reshape(-1)
        image_files.loc[f'ScreenShot.png', 'pixels_0':] = image

    image_files = image_files / 255.0
    image = image_files.iloc[0].values.astype(np.float32).reshape(-1, 1)
    return image
