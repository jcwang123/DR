from keras.preprocessing.image import random_zoom, random_shift
from keras.preprocessing.image import random_rotation, load_img
from keras.preprocessing.image import img_to_array
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
import numpy as np
import pandas
import os, cv2
from PIL import Image
#from PIL import Image
from keras import backend as K
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from tools import threadsafe_generator
import SimpleITK as sitk



#gray = False 
mean = 122.94
scale = 0.017
img_rows = 512
img_cols = 512
feat_map_threshold = 70
# get all CSV file paths
# path: all label's path
# return: each CSV files path
def get_csv_path(path):
    files_1 = os.listdir(path)
    files_1.sort()
    path_list = []
    for file_1 in files_1:
        path_list.append(path+"/"+file_1)
    path_list.sort()
    path_list = np.array(path_list)
    return path_list


# get index array from each CSV file
# label_path: each CSV files path
# return: CSV's information
def get_index_array(label_path):
    dataframe = pandas.read_csv(label_path, header=0)
    dataset = dataframe.values
    label = np.array(dataset)
    #print(label[1,:])
    return label


# split dataset to train validation and test by 7:1:2
def random_split(label):
    train, test = train_test_split(
        label, test_size=0.1, random_state=1)
    train, validation = train_test_split(
        train, test_size=0.1, random_state=1)
    return train, validation, test


# get image and label paths
# label_path: all label's path
# return: each sample's path and 14 dim labels
def get_image_and_label(label_path):
    # Get every csv file path
    label_files_path_list = get_csv_path(label_path)
    train = []
    validation = []
    test = []
    for file in label_files_path_list:
        index_array = get_index_array(file)
    # Randomly(seed = 0) split to train, validation and test 70%,10%,20%
        data = random_split(index_array)
        train.extend(data[0])
        validation.extend(data[1])
        test.extend(data[2])
    train=np.array(train)
    validation=np.array(validation)
    test = np.array(test)
    np.random.shuffle(train)
    np.random.shuffle(validation)
    np.random.shuffle(test)
    #print(train.shape)
    #print(train)
    X_train = train[:, :1]
    X_validation = validation[:, :1]
    Y_train = train[:, 1:]
    Y_validation = validation[:, 1:]
    X_test = test[:, :1]
    Y_test = test[:, 1:]
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


def random_split_train(label):
    train, validation = train_test_split(
        label, test_size=0.1, random_state=1)

    return train, validation


def get_train_image_and_label(label_path):
    # Get every csv file path
    label_files_path_list = get_csv_path(label_path)
    train = []
    validation = []

    for file in label_files_path_list:
        index_array = get_index_array(file)
    # Randomly(seed = 0) split to train, validation and test 70%,10%,20%
        data = random_split_train(index_array)
        train.extend(data[0])
        validation.extend(data[1])
    train=np.array(train)
    validation=np.array(validation)
    np.random.shuffle(train)
    np.random.shuffle(validation)

    print("traing shape",train.shape)
    print("validation shape ", validation.shape)
    X_train = train[:, :1]
    X_validation = validation[:, :1]
    Y_train = train[:, 1:]
    Y_validation = validation[:, 1:]

    return X_train, Y_train, X_validation, Y_validation

def get_test_image_and_label(label_path):
    label_files_path_list = get_csv_path(label_path)
    test = []
    for file in label_files_path_list:
        index_array = get_index_array(file)
        test.extend(index_array)
    test = np.array(test)
    X_test = test[:, :1]
    Y_test = test[:, 1:]
    return X_test, Y_test



def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
  
    # interpolate linearly
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def template_match(template_image_path, img_path):
    filename, file_extension = os.path.splitext(img_path)

    if (file_extension == ".dcm" or file_extension == ".DCM"):
        img = sitk.ReadImage(img_path)   
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = img_arr[0,:,:]
    else:
        img = Image.open(img_path)
        img_arr = np.asarray(img)

    img_temp = Image.open(template_image_path)
    template_arr = np.asarray(img_temp)
    img_arr_matched = hist_match(img_arr, template_arr)

    return img_arr_matched, template_arr


# Data Augmentation:
# Randomly translated in 4 directions by 25 pixels
# Randomly rotated from -15 to 15 degrees
# Randomly scaled between 80% and 120%
def distorted_image(image):
   image1 = random_rotation(
       image, 15, row_axis=0, col_axis=1, channel_axis=2)
   image2 = random_zoom(
       image1, (0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2)
   image3 = random_shift(
       image2, 0.05, 0.05, row_axis=0, col_axis=1, channel_axis=2)
   return image3


# change array to list
def array_to_list(target_array):
    target_list = []
    for i in target_array:
        for a in i:
            target_list.append(a)
    return target_list

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


@threadsafe_generator
def generate_from_source_rescale(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                # img = sitk.ReadImage(image_path)   
                # img_arr = sitk.GetArrayFromImage(img)
                # img_arr = img_arr[0,:,:]
                # img_arr = change_to_255(img_arr)
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr  
                img_arr = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2).astype('uint8')
                image = Image.fromarray(img_arr)
            else :
                image = Image.open(image_path).convert("RGB")
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)          
            image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            label = label_path

            x.append(distorted_image(image_array))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []
                

                
                
def generate_from_source_rescale_resized(image_path_list, label_path_list, batch_size,img_rows, img_cols):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                # img = sitk.ReadImage(image_path)   
                # img_arr = sitk.GetArrayFromImage(img)
                # img_arr = img_arr[0,:,:]
                # img_arr = change_to_255(img_arr)
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:

                    img_arr = 255 - img_arr  
                img_arr = np.repeat(img_arr[:, :, np.newaxis], 3, axis=2).astype('uint8')
                image = Image.fromarray(img_arr)
            else :
                image = Image.open(image_path).convert("RGB")
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)          
            image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            label = label_path
            print("shape ", image_array.shape)
            x.append(distorted_image(image_array))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []
                                
@threadsafe_generator
def generate_from_source_clahe(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)        
            
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr                               
                image_arr_clahe = clahe.apply(img_arr.astype('uint8'))

            else :
                image = Image.open(image_path).convert("RGB")
                image_arr_clahe = clahe.apply(np.array(image))
                
            img_arr = np.repeat(image_arr_clahe[:, :, np.newaxis], 3, axis=2).astype('uint8')
            image = Image.fromarray(img_arr)    
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            # print(image_path)
            # plt.imshow(image)
            # plt.show()
            image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            label = label_path

            x.append(distorted_image(image_array))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []
  

@threadsafe_generator
def generate_from_source_hist(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    template_image_path = "/home2/data/DR/CXR8/images/images_001/00000005_005.png"
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)        
            
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr                               


            else :
                image = Image.open(image_path).convert("RGB")
                img_arr = np.array(image)
            img_temp = Image.open(template_image_path)
            template_arr = np.asarray(img_temp)
            img_arr_matched = hist_match(img_arr, template_arr)            
            
            img_arr = np.repeat(img_arr_matched[:, :, np.newaxis], 3, axis=2).astype('uint8')
            image = Image.fromarray(img_arr)    
            image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            label = label_path

            x.append(distorted_image(image_array))
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []

                
                

def generate_from_source_hist_for_predict(image_path, img_rows, img_cols):
    x = []
    y = []



    template_image_path = "/home2/data/DR/CXR8/images/images_001/00000005_005.png"

    filename, file_extension = os.path.splitext(image_path)

          
#     if not os.path.isfile(image_path):
#         continue

    if (file_extension == ".dcm" or file_extension == ".DCM"):
        pos = check_pos(image_path)
        img = sitk.ReadImage(image_path)   
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = img_arr[0,:,:]
        img_arr = change_to_255(img_arr)
        if not pos:
            img_arr = 255 - img_arr   
        image = Image.fromarray(img_arr) 
        image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
        img_arr = img_to_array(image)
    else :
        image = Image.open(image_path).convert("L")
        image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
        img_arr = img_to_array(image)
    img_temp = Image.open(template_image_path)
    template_arr = np.asarray(img_temp)
    img_arr_matched = hist_match(img_arr, template_arr)            

    image_array = np.repeat(img_arr_matched[:, :, np.newaxis], 3, axis=2).astype('uint8')
    image_array = image_array - mean 
    image_array *= scale

    if (len(image_array.shape) == 4):
        image_array = np.squeeze(image_array)
    image_array = np.expand_dims(image_array, axis = 0)
    
    return image_array
                
@threadsafe_generator
def generate_from_source_hist_resized(image_path_list, label_path_list, batch_size, img_rows, img_cols):
    cnt = 0
    x = []
    y = []
    y_abnormal = []

    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    template_image_path = "/home2/data/DR/CXR8/images/images_001/00000005_005.png"
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)        
            if not os.path.isfile(image_path):
                continue
            
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr                               
                image = Image.fromarray(img_arr) 
                image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
                img_arr = img_to_array(image)
            else :
                image = Image.open(image_path).convert("L")
                image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
                img_arr = img_to_array(image)
            img_temp = Image.open(template_image_path)
            template_arr = np.asarray(img_temp)
            img_arr_matched = hist_match(img_arr, template_arr)            
            
            image_array = np.repeat(img_arr_matched[:, :, np.newaxis], 3, axis=2).astype('uint8')
            # image = Image.fromarray(img_arr)    
            # image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            #image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            label = label_path
#            print("shape ", image_array.shape)
            if (len(image_array.shape) == 4):
                image_array = np.squeeze(image_array)
            x.append(distorted_image(image_array))
            y.append(label[:-1])
            y_abnormal.append(label[-1])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),[np.array(y), np.array(y_abnormal)])
                x = []
                y = []
                y_abnormal = []
 

@threadsafe_generator
def generate_from_source_hist_resized_tb(image_path_list, label_path_list, batch_size, img_rows, img_cols):
    cnt = 0
    x = []
    y = []


    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    template_image_path = "/home2/data/DR/CXR8/images/images_001/00000005_005.png"
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            filename, file_extension = os.path.splitext(image_path)        
            if not os.path.isfile(image_path):
                continue
            
            if (file_extension == ".dcm" or file_extension == ".DCM"):
                pos = check_pos(image_path)
                img = sitk.ReadImage(image_path)   
                img_arr = sitk.GetArrayFromImage(img)
                img_arr = img_arr[0,:,:]
                img_arr = change_to_255(img_arr)
                if not pos:
                    img_arr = 255 - img_arr   
                print("image path ", image_path)
                print("pos ", pos)
                image = Image.fromarray(img_arr) 
                image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
                img_arr = img_to_array(image)
            else :
                image = Image.open(image_path).convert("L")
                image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
                img_arr = img_to_array(image)
            img_temp = Image.open(template_image_path)
            template_arr = np.asarray(img_temp)
            img_arr_matched = hist_match(img_arr, template_arr)            
            
            image_array = np.repeat(img_arr_matched[:, :, np.newaxis], 3, axis=2).astype('uint8')
            # image = Image.fromarray(img_arr)    
            # image = image.resize((img_rows, img_cols),Image.ANTIALIAS)
            #image_array = img_to_array(image)

            image_array = image_array - mean 
            image_array *= scale
            print("mean ", np.mean(image_array))
            label = label_path
#            print("shape ", image_array.shape)
            if (len(image_array.shape) == 4):
                image_array = np.squeeze(image_array)
            # x.append(distorted_image(image_array))
            x.append(image_array)
            y.append(label)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),np.array(y))
                x = []
                y = []

@threadsafe_generator
def generate_from_source(image_path_list, label_path_list, batch_size):
    cnt = 0
    x = []
    y_14 = []
    y_abnormal = []

    batch_size = batch_size
    image_path_list = array_to_list(image_path_list)
    while True:
        for image_path, label_path in zip(image_path_list, label_path_list):
            image_array = img_to_array(
                load_img(image_path, grayscale=gray, target_size=(512, 512)))
            label = label_path
            label_14 = label[:-1]
            label_abnormal = label[-1]

            x.append(distorted_image(image_array))
            y_14.append(label_14)
            y_abnormal.append(label_abnormal)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x),[np.array(y_14),np.array(y_abnormal)])
                x = []
                y_14 = []
                y_abnormal = []