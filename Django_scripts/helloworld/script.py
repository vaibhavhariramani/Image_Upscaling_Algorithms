# import the necessary packages
import argparse
import time
import cv2
import os
import sys
from PIL import Image
image_fullpath = sys.argv[1]
image_name = sys.argv[2]
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# extract the model name and model scale from the file path
model = "FSRCNN_x2.pb"
modelName = "fsrcnn"
modelScale = 2


sr.readModel(model)
sr.setModel(modelName, modelScale)
# load the input image from disk and display its spatial dimensions
image = cv2.imread(str(image_fullpath))


upscaled = sr.upsample(image)


image_save_path = image_fullpath.replace(image_name, "temp.png")
cv2.imwrite(str(image_save_path), upscaled)
print('media/temp.png')
