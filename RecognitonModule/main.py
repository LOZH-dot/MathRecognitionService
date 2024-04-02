from segmentation import segmentation
from predict import predict
import os
import shutil

roisFolder = '.\\rois'
for filename in os.listdir(roisFolder):
    file_path = os.path.join(roisFolder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

segmentation("123432.png")

roisFiles = os.listdir(roisFolder)
roisFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for roi in roisFiles:
    print(predict(roisFolder + "\\" + roi))
