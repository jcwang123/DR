import time
import numpy as np
import cv2
import random
import SimpleITK as sitk

from imgaug import augmenters as iaa
data_path = '/home/data/DR_png/'
def change_to_255(img_arr):
    mn = np.amin(img_arr)
    mx = np.amax(img_arr)
    rg = mx - mn
    img_arr = (img_arr - mn).astype(float)/rg * 255
    return img_arr

def check_pos(image_path):
    pos = True
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    keys = reader.GetMetaDataKeys()
    if '0028|0004' in keys:
        photometric_interpretation = reader.GetMetaData('0028|0004')
        # MONOCHROME1 NEG     MONOCHROME2 POS
        if ('MONOCHROME' in photometric_interpretation and '1' in photometric_interpretation):
            pos = False
    if '2050|0020' in keys:
        presentation_lut_shape = reader.GetMetaData('2050|0020')
        if ('INVERSE' in presentation_lut_shape):
            pos = False
    return pos

def getimg(path, norm):
    if (path.split('.')[-1] == "dcm" or path.split('.')[-1] == "DCM"): 
                pos = check_pos(path)
                img = sitk.ReadImage(path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr  
                img_arr = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2).astype('uint8')
    else:
        img_arr = cv2.imread(path)
    if not norm:
        return img_arr
    result = np.zeros(img_arr.shape, dtype=np.float32)
    cv2.normalize(img_arr, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return result

def DataGenerator(txt_path, batch_size, norm=True, shuf=True):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    X = []
    Y = []
    i = 0
    while True:
        if shuf:
            random.shuffle(lines)
        for l in lines:
            img_path = data_path + l.split(' ')[0]
            try:
                img = getimg(img_path, norm=norm)
            except Exception as e:
                #print(img_path)
                #print(e)
                continue
            assert(int(l.split(' ')[1]) in [0,1])
            label = np.array(list(map(int, l.split(' ')[1:])))
            X.append(img)
            Y.append(label)
            i+=1
            if i==batch_size:
                X=np.array(X)
                Y=np.array(Y)
                yield X,Y
                X = []
                Y = []
                i = 0

if __name__ == "__main__":
    for i,(X,Y) in enumerate(DataGenerator('train.txt', 4, 512)):
        for k in range(X.shape[0]):
            label = np.where(Y[k][0]==1)[0][0]
            cv2.imwrite(f'testgener/{i}_{k}_{label}.jpg', X[k]*255)
 