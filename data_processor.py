import tensorflow as tf

import gdal 

from os import listdir, rename, remove
from os.path import join

import numpy as np
import random
import math
import glob

import time
import functools

from subprocess import call

import cv2

from math import ceil
def rint (x):
    return int(ceil(x))

class Data_processor:
    
    """
    Data_processor class handles data related operations such as
    retrieving a batch of patches to feed the network, augmenting patches, etc.
    
    Attributes:
        image_paths (list [str])  : full paths of the images
        label_paths (list [str])  : full paths of the labels maps in the training phase, predicted maps in the test phase
        valid_image_paths (list [str]) : full paths of the validation images (used only in training phase)
        valid_label_paths (list [str]) : full paths of the validation labels maps (used only in training phase)
        patch_size (int)          : used to calculate height and width (they are exactly the same) of the patches that are fed to the network
        num_of_channels (str)     : number of channels in the input images 
        num_of_classes (str)      : number of classes in the input label maps
        mean_list (list [float])  : list containing mean values for the channels
        batch_size (int)          : batch size (number of patches in a batch). 1 for the test phase
        is_training (bool)        : True: training, False: test
        min_num_of_bits(int)      : min number of bits to represent number of classes
        geo_image (gdal object)   : gdal object pointing the current test image (used only in test phase)
        geo_label (gdal object)   : gdal object pointing the predicted label map for the current test image (used only in test phase)
    """
    
    def __init__(self, 
                 train_dir,
                 valid_dir,
                 patch_size,
                 padding,
                 num_of_classes,
                 batch_size,
                 is_training,
                 seg_en,
                 pansh_en,
                 r=4):
        
        self.is_training = is_training
        
        if self.is_training:
            self.train_paths = self.create_image_label_paths(train_dir)
            self.valid_paths = self.create_image_label_paths(valid_dir)
            self.patch_size = self.find_patch_size()

        else:
            self.valid_paths = self.train_paths = self.create_image_label_paths(valid_dir)
            self.patch_size = patch_size
                     
        self.num_of_channels = self.find_num_of_channels()

        self.num_of_classes = num_of_classes
        self.padding = padding
        self.batch_size = batch_size
        self.r = r

        self.seg_en=seg_en
        self.pansh_en=pansh_en
                    
    def batch_generator(self, is_training):
        """
        create a generator, which generates a batch of image patches and their corresponding label maps
        
        Returns:
            iterator (tensorflow iterator object): iterator, which generates batches
        """
       
        #shapes of the outputs that generator produces
        output_shapes = (tf.TensorShape([self.patch_size//self.r, self.patch_size//self.r, self.num_of_channels]),
                        tf.TensorShape([self.patch_size, self.patch_size]),
                        tf.TensorShape([self.patch_size, self.patch_size, self.num_of_channels]),
                        tf.TensorShape([self.patch_size, self.patch_size]))
    
        #data types of the outputs that generator produces
        data_types = (tf.float32, tf.float32, tf.float32, tf.float32)
        
        generator_fun = functools.partial(self.patch_generator_fun, is_training = is_training)
        
        #create a dataset object
        dataset = tf.data.Dataset.from_generator(generator_fun, 
                                                 output_types = data_types,
                                                 output_shapes = output_shapes)
        
        #augment the data 
        #data is augmented in parallel
        dataset = dataset.map(lambda ms, pan,target, mask: self.process_train_val_patches(ms, pan,target, mask, is_training), num_parallel_calls = self.batch_size)
        
        #get a batch
        dataset = dataset.batch(self.batch_size)
        
        #prefetch is used to increase training speed
        #while the data in Nth iteration is being processed, the data for (N + 1)th iteration is getting prepared
        dataset = dataset.prefetch(1)
        
        iterator = dataset.make_one_shot_iterator()
        
        return iterator
    
    def patch_generator_fun(self, is_training = True):
        """
        generator function that yields an image and a label map
        
        Args:
            is_training : True  = training
                          False = validation 
        
        Yields:
            image (matrix): an image patch: [patch_size, 
                                             patch_size, 
                                             # of channels]
            label (matrix): label map     : [patch_size,
                                             patch_size]
        """
        if is_training:
            num_of_images = len(self.train_paths['target'])    
        else:
            num_of_images = len(self.valid_paths['target'])
        
        while True:
            
            #randomly select 1 sample
            image_index = random.randint(0, num_of_images - 1)

            #read and augment the image patch and its label map
            ms, pan, target, mask = self.read_training_patch(image_index,is_training)
            
            yield ms, pan, target, mask
            
    def test_patch_generator(self):
        """
        create a generator, which generates a batch of image patches and their corresponding label maps
        image patches and label maps are randomly cropped and flipped
        
        Returns:
            iterator (tensorflow iterator object): iterator, which generates batches
        """
       
        #shapes of the outputs that generator produces
        output_shapes = (tf.TensorShape([1,
                                         self.num_of_channels,
                                         (self.patch_size + 2 * self.padding)//self.r,
                                         (self.patch_size + 2 * self.padding)//self.r]),
                        tf.TensorShape([1,
                                         1,
                                         self.patch_size + 2 * self.padding,
                                         self.patch_size + 2 * self.padding]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([]))
    
        #data types of the outputs that generator produces
        data_types = (tf.float32,tf.float32, tf.int64, tf.int64, tf.int64, tf.int64)
        
        #create a dataset object
        dataset = tf.data.Dataset.from_generator(self.test_patch_generator_fun, 
                                                 output_types = data_types,
                                                 output_shapes = output_shapes)
                    
        iterator = dataset.make_one_shot_iterator()
        
        return iterator
            
    def test_patch_generator_fun(self):
        """
        TODO: READING A PATCH FROM A BIG TIF FILE USING GDAL IS EXTREMELY INEFFICIENT
        ESPECIALLY IF THE PATCH IS FAR AWAY FROM THE TOP-LEFT CORNER
        IT MAKES INFERENCE TIME SUPER SLOW
        TRY TO FIND ANOTHER SOLUTION TO READ PATCHES FROM A BIG TIF!
            
        generator function that yields patches from the training images
        the function also yields top-left x and y coordinate location of the patches in big the images
        and their actual size (height and width of rightmost and bottommost patches migh be lower than <self.patch_size>)
        
        assume that there is a big tif file consisting of 20 patches.
        this function yields the patches in this order:
         0 -  1 -  2 -  3 -  4
         5 -  6 -  7 -  8 -  9
        10 - 11 - 12 - 13 - 14
        15 - 16 - 17 - 18 - 19
        
        Yields:
            patch_4d (matrix): a normalized image patch:  [1, 
                                                           # of channels,
                                                           patch_size, 
                                                           patch_size]
            y_top_left (int) : y coordinate of top-left location of the patch in the image
            x_top_left (int) : x coordinate of top-left location of the patch in the image
            actual_patch_height (int) : height of the patch
            actual_patch_width (int)  : width of the patch
        """
        
        num_of_images = len(self.valid_paths['target'])  
        
        #traverse on each image
        for image_id in range(num_of_images):
            
            #open the current image and a tif file for its predicted map
            self.open_test_image_label(image_id)
            
            #height and width of the image
            orig_img_h = self.geo_pan.RasterYSize
            orig_img_w = self.geo_pan.RasterXSize
            
            #number of patches horizontally and vertically
            n_patch_horiz = int(math.ceil(orig_img_w / self.patch_size))
            n_patch_vert = int(math.ceil(orig_img_h / self.patch_size))
            
            total_num_of_patches = n_patch_horiz * n_patch_vert
            
            for i in range(n_patch_vert):
                for j in range(n_patch_horiz):
                    
                    #top - left location of the patch
                    y_top_left = i * self.patch_size
                    x_top_left = j * self.patch_size
                    
                    #actual height and width of each patch
                    #size of rightmost and bottommost patches might be lower than <self.patch_size>
                    actual_patch_height = min(self.patch_size, (orig_img_h - y_top_left))
                    actual_patch_width = min(self.patch_size, (orig_img_w - x_top_left))
                    
                    #read a patch 
                    patch_ms,patch_pan = self.read_test_patch(x_top_left, y_top_left, orig_img_w, orig_img_h)
                    
                    #normalize the patch
                    patch_ms_normalized = self.normalize_data(patch_ms.astype(np.float32))
                    patch_pan_normalized = self.normalize_data(patch_pan.astype(np.float32))
                    # patch_normalized = self.normalize_data_01(patch)
                    
                    #convert hwc to chw
                    patch_ms_normalized = np.transpose(patch_ms_normalized, [2, 0, 1])
                    patch_pan_normalized = np.transpose(patch_pan_normalized, [2, 0, 1])
                    
                    #convert <patch_normalized> to 4d matrix
                    patch_ms_4d = np.expand_dims(patch_ms_normalized, axis = 0)
                    patch_pan_4d = np.expand_dims(patch_pan_normalized, axis = 0)
                    
                    time_start = time.time()                                  
                    #generate a patch as well as its location and actual dimensions
                    #location and dimensions are needed to determine where to put the predicted label map
                    yield patch_ms_4d,patch_pan_4d, y_top_left, x_top_left, actual_patch_height, actual_patch_width
                    time_elapsed = time.time() - time_start
                    
                    print('Image %d / %d, patch %d / %d has been classified, elapsed time: %.4f secs' % 
                          (image_id + 1, num_of_images, 
                          i * n_patch_horiz + j + 1, total_num_of_patches,
                          time_elapsed))
            
            #close the current image and its predicted map
            self.close_test_image_label()
            
        
    def read_training_patch(self, image_index, is_training):
        """
        read an image patch and its label map
        
        Args:
            image_path (str) : full path of an image patch
            label_path (str) : full path of an label map
            
        Returns:
            image (tensor) : image patch
            label (tensor) : label patch
        """
        
        #read an image patch
        #convert chw to hwc
        if is_training:
            paths = self.train_paths   
        else:
            paths = self.valid_paths

        geo_image = gdal.Open(paths['target'][image_index])
        target = np.transpose(geo_image.ReadAsArray(), [1, 2, 0])
        
        #read a label map
        geo_label = gdal.Open(paths['masks'][image_index])
        mask = geo_label.ReadAsArray().astype(np.float32)

        geo_image = gdal.Open(paths['ms'][image_index])
        ms = np.transpose(geo_image.ReadAsArray(), [1, 2, 0])

        geo_label = gdal.Open(paths['pan'][image_index])
        pan = geo_label.ReadAsArray()        

        return ms, pan, target, mask
            
    def process_train_val_patches(self,  ms, pan,target, mask, is_training):
        """
        - normalize images
        - one hot encode label maps
        - cast both images and label maps to float32
        - augment data
        if it is validation phase, skip the augmentation part
        
        Args:
            image (matrix)     : image [patch_size, patch_size, # of channels]
            label (matrix)     : label [patch_size, patch_size]
            is_training (bool) : True  = training
                                 False = validation 
            
        Returns:
            image (tensor) : modified image [patch_size, patch_size, # of channels] 
            label (tensor) : modified label [patch_size, patch_size, 1]

        """
        
        #outputs of any tf.py_func have no shape
        #reshape image and label
        target = tf.reshape(target, [self.patch_size, self.patch_size, self.num_of_channels])
        ms = tf.reshape(ms, [self.patch_size//self.r, self.patch_size//self.r, self.num_of_channels])
        pan = tf.reshape(pan, [self.patch_size, self.patch_size,1])
        mask = tf.reshape(mask, [self.patch_size, self.patch_size])
                
        #if it is multi class classification, one hot encode the label map
        if self.num_of_classes > 2:
            mask = tf.one_hot(tf.cast(mask,tf.uint8), depth = self.num_of_classes)
        #if it is a binary classification, the label map is already one hot encoded
        #just expand dimension
        else:
            mask = tf.expand_dims(mask, -1)
            
        #cast both input image and label map to float32
        target = tf.cast(target, tf.float32)
        ms = tf.cast(ms, tf.float32)
        pan = tf.cast(pan, tf.float32)
        mask = tf.cast(mask, tf.float32)

        #augment the data only if training images and label maps are read
        #if validation images and label maps are read, skip the augmentation step
        if is_training:
            
            #apply gamma correction using the randomly generated gamma value  
            # gamma_correction_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            # gamma = tf.random_uniform(shape = [], minval = 0.75, maxval = 1.5, dtype = tf.float32)
            # image = tf.cond(gamma_correction_flag, lambda: self.gamma_correction(image, gamma), lambda: tf.identity(image))
            
            #randomly flip left-right
            left_right_flip_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            ms, pan,target, mask = tf.cond(left_right_flip_flag, lambda: self.flip_left_right(ms, pan, target, mask), 
                                                         lambda: (tf.identity(ms), tf.identity(pan), tf.identity(target), tf.identity(mask)))
                
            #randomly flip up-down
            up_down_flip_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            ms, pan,target, mask = tf.cond(up_down_flip_flag, lambda: self.flip_up_down(ms, pan, target, mask),
                                                      lambda: (tf.identity(ms), tf.identity(pan), tf.identity(target), tf.identity(mask)))
            
            #randomly change contrast
            # contrast_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            # image = tf.cond(contrast_flag, lambda: self.random_contrast(image), lambda: tf.identity(image))
            
            #randomly rotate
            rotate_flag = tf.random_uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
            #rotate_flag:
            #0 - no rotation
            #1 - rotate 90 degrees
            #2 - rotate 180 degrees
            #3 - rotate 270 degrees
            ms = tf.image.rot90(ms, k = rotate_flag)
            pan = tf.image.rot90(pan, k = rotate_flag)
            target = tf.image.rot90(target, k = rotate_flag)
            mask = tf.image.rot90(mask, k = rotate_flag)
    
            #translate the image patch and its label map
            # translate_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            # image, label = tf.cond(translate_flag, lambda: self.translate_patch(image, label), lambda: (tf.identity(image), tf.identity(label)))
         
            #add gaussian noise to the image
            #gaussian_noise_flag = tf.cast(tf.random_uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32), tf.bool)
            #image = tf.cond(gaussian_noise_flag, lambda: self.add_gaussian_noise(image), lambda: tf.identity(image))
        
        #normalize the patch
        ms = self.normalize_data(ms)
        pan = self.normalize_data(pan)
        target = self.normalize_data(target)
        
        return ms,pan,target,mask
    
    def flip_up_down(self, ms, pan, target, mask):
        """
        up-down flip
        
        Args:
            image (matrix) : input image
            label (matrix) : input label map
            
        Returns:
            image (matrix) : flipped image
            label (matrix) : flipped label map
        """
        ms = tf.image.flip_up_down(ms)
        pan = tf.image.flip_up_down(pan)
        target = tf.image.flip_up_down(target)
        mask = tf.image.flip_up_down(mask)
        
        return ms, pan, target, mask
        
    
    def flip_left_right(self, ms, pan, target, mask):
        """
        left-right flip
        
        Args:
            image (matrix) : input image
            label (matrix) : input label map
            
        Returns:
            image (matrix) : flipped image
            label (matrix) : flipped label map
        """
        
        ms = tf.image.flip_left_right(ms)
        pan = tf.image.flip_left_right(pan)
        target = tf.image.flip_left_right(target)
        mask = tf.image.flip_left_right(mask)
        
        return ms, pan, target, mask
    
    def add_gaussian_noise(self, image, std = 1.0):
        """
        add a noise to the input image using the gaussian distribution, where mean is 0.0 and 
        standard deviation is <std>
        
        Args:
            image (matrix) : input image
            str (float)    : standard deviation for the gaussian distribution (optional)
        
        Returns:
            image (matrix) : modified image
        """
        
        noise = tf.random_normal(shape = tf.shape(image), mean = 0.0, stddev = std, dtype = tf.float32) 
        image = tf.add(image, noise)
        
        #constrain value of each pixel in the image between 0 and 255
        #we assume that the image is 8 bit
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)
    
        return image
    
    def random_contrast(self, image, min_val = 0.75, max_val = 1.25):
        """
        randomly change contrast of the image
        
        Args:
            image (matrix)  : input image
            min_val (float) : minimum value for the contrast change (optional)
            max_val (float) : maximum value for the contrast change (optional)
            
        Returns:
            image (matrix)  : modified image
        """
        
        image = tf.image.random_contrast(image, lower = min_val, upper = max_val)
    
        #constrain value of each pixel in the image between 0 and 255
        #we assume that the image is 8 bit
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 255.0)
    
        return image
    
    def translate_patch(self, image, label_map):
        """
        translate the image as well as its label map to left, right, top, and bottom directions
        magnitude of the translation for each direction is selected randomly
        after the image patch and label map are translated, their background pixels are cropped out
        then their remaining parts are resized back to their original sizes
        
        Args:
            image (matrix)     : image patch
            label_map (matrix) : label map
        
        Returns:
            image (matrix)     : translated image patch
            label_map (matrix) : translated label map
        """
                
        #min and max value for the shift
        shift_min = -int(self.patch_size / 5)
        shift_max = int(self.patch_size / 5)
        
        #generate random values for the horizontal and vertical shifts
        vert_shift = tf.random_uniform(shape = [], minval = shift_min, maxval = shift_max, dtype = tf.int32)
        horiz_shift = tf.random_uniform(shape = [], minval = shift_min, maxval = shift_max, dtype = tf.int32)
        
        top_left_x = tf.maximum(horiz_shift, tf.constant(0))
        top_left_y = tf.maximum(vert_shift, tf.constant(0))
        width = tf.subtract(self.patch_size, tf.abs(horiz_shift))
        height = tf.subtract(self.patch_size, tf.abs(vert_shift))
        
        #crop image according to the randomy generated values
        cropped_image_patch = tf.image.crop_to_bounding_box(image, top_left_y, top_left_x, height, width) 
        cropped_label_patch = tf.image.crop_to_bounding_box(label_map, top_left_y, top_left_x, height, width) 
        
        #resize both image and label patches to their original sizes
        resized_image_patch = tf.image.resize_images(images = cropped_image_patch, size = (self.patch_size, self.patch_size))
        resized_label_patch = tf.image.resize_images(images = cropped_label_patch, size = (self.patch_size, self.patch_size))
        
        #convert label map to binary matrix again
        resized_label_patch = tf.cast(resized_label_patch >= 0.5, tf.float32)
    
        return resized_image_patch, resized_label_patch    
    
    def gamma_correction(self, image, gamma):
        """
        gamma correction decribed in: 
        https://en.wikipedia.org/wiki/Gamma_correction
        A is assumed to be 1, it has not been implemented
        we also assume that the input image is 8 bit
        
        Args:
            image (tensor) : input image
            gamma (float)  : gamma value for the correction
        
        Returns:
            image_gamma_corrected (tensor) : gamma corrected image
        """
        
        image_norm = tf.div(image, 255)
        image_gamma_corrected = tf.multiply(tf.pow(image_norm, gamma), 255)
        
        return image_gamma_corrected
        
    def normalize_data(self, image_patch, PERCENT=0.015):
        """
        normalize the data with the following formula
        x_normalized = (x - mean)
        x_normalized is converted to bgr from rgb
        
        Args:
            image_patch: image patch, whose shape is [patch_size,
                                                      patch_size,
                                                      # of channels>]
            
        Returns:
            image_patch_normalized: normalized patch with the same shape
            
        type of image_patch and image_patch_normalized is <tensor> during training phase,
                                                          <numpy array> during test phase
        """
        # if self.is_training:
        #     image_patch_normalized = tf.subtract(image_patch, self.mean_list)
        #     image_patch_normalized = tf.reverse(image_patch_normalized, axis = [-1])
        # else:
        #     image_patch_normalized = (image_patch.astype(np.float) - self.mean_list) 
        #     image_patch_normalized = image_patch_normalized[..., ::-1]
        
        # return image_patch_normalized
        return image_patch/255.#/2047.

        # stacked = None
        # for c in range(image_patch.shape[2].value):
        #     length = image_patch.shape[0].value * image_patch.shape[1].value
        #     channel = tf.reshape(image_patch[:,:,c],[-1])
        #     channel = tf.contrib.framework.sort(channel)
        #     low = channel[tf.cast(length*PERCENT,tf.int32)]
        #     high = channel[tf.cast(length*(1-PERCENT),tf.int32)]
        #     clipped = tf.clip_by_value(image_patch[:,:,c],low,high)
        #     norm = (clipped - low)/(high - low)
        #     if stacked is None:
        #         stacked = tf.expand_dims(norm,2)
        #     else:
        #         stacked = tf.concat([stacked,tf.expand_dims(norm,2)],axis=2)

        # return stacked

    # def normalize_data(self, image_patch, PERCENT=0.015):
    #     img2 = np.array(image_patch,copy=True).astype(np.float32)
    #     for c in range(image_patch.shape[2]):
    #         channel = sorted(img2[:,:,c].reshape(-1))
    #         low = channel[int(len(channel)*PERCENT)]
    #         high = channel[int(len(channel)*(1-PERCENT))]
    #         img2[:,:,c] = np.clip(img2[:,:,c],low,high)
    #         img2[:,:,c] = (img2[:,:,c] - low)/(high - low)
    #     return img2    

    def compress_label_maps(self):
        """
        compress all the predicted maps generated by the network with LZW compression algorithm
        """
        
        for label_path in self.valid_paths['masks']:
            label_path.replace('masks','pred_mask')

            #create a full path for the compressed image
            compressed_label_path = label_path.split('.')[0] + '_c.tif'
            
            #compress the image using LZW compression algorithm
            call(['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'BIGTIFF=YES', label_path, compressed_label_path])
    
            #remove the original image
            remove(label_path)
            
            #rename the compressed image as <label_path>
            rename(compressed_label_path, label_path)

    def read_test_patch(self, x_top_left, y_top_left, width, height):
        """
        read a patch from the data pointed by <self.geo_image>
        the patch is padded if it is needed
        
        Args:
            x_top_left (int)   : top left location (x coordinate) of the patch in the image
            y_top_left (int)   : top left location (y coordinate) of the patch in the image
            width (int)        : width of the image, from which the patch would be read
            height (int)       : height of the image, from which the patch would be read
            
        Returns:
            patch : shape : [# of channels, height, width] 
        """ 
                
        #left padding 
        pad_x_before = abs(min((x_top_left - self.padding), 0))  
        
        #right padding
        pad_x_after = abs(min(width - (x_top_left + self.patch_size + self.padding), 0))
        
        #top padding
        pad_y_before = abs(min(y_top_left - self.padding, 0))
            
        #bottom padding
        pad_y_after = abs(min(height - (y_top_left + self.patch_size + self.padding), 0))
        
        #read a patch from the data pointed by <self.geo_image>
        patch_ms = self.geo_ms.ReadAsArray(int(x_top_left - self.padding + pad_x_before)//self.r, 
                                           int(y_top_left - self.padding + pad_y_before)//self.r,
                                           int(self.patch_size + 2 * self.padding - pad_x_before - pad_x_after)//self.r,
                                           int(self.patch_size + 2 * self.padding - pad_y_before - pad_y_after)//self.r)[[0,1,2],:,:]    

        patch_pan = self.geo_pan.ReadAsArray(int(x_top_left - self.padding + pad_x_before), 
                                           int(y_top_left - self.padding + pad_y_before),
                                           int(self.patch_size + 2 * self.padding - pad_x_before - pad_x_after),
                                           int(self.patch_size + 2 * self.padding - pad_y_before - pad_y_after))

        num_of_channels = patch_ms.shape[0]#self.geo_ms.RasterCount

        #pad the patch if it is needed                                                 
        if num_of_channels == 1:
            patch_ms_padded = np.pad(patch_ms, ((pad_y_before//self.r, pad_y_after//self.r), (pad_x_before//self.r, pad_x_after//self.r)), mode = 'symmetric')
            #transform patch_padded from [height, width] to [1, height, width]
            patch_ms_padded = np.expand_dims(patch_ms_padded, axis = 0)            
        else:
            patch_ms_padded = np.pad(patch_ms, ((0, 0), (rint(pad_y_before/self.r), rint(pad_y_after/self.r)), (rint(pad_x_before/self.r), rint(pad_x_after/self.r))), mode = 'symmetric')
        

        patch_pan_padded = np.pad(patch_pan, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode = 'symmetric')
        #transform patch_padded from [height, width] to [1, height, width]
        patch_pan_padded = np.expand_dims(patch_pan_padded, axis = 0)            

        #convert chw to hwc
        patch_ms_padded = np.transpose(patch_ms_padded, [1, 2, 0])
        patch_pan_padded = np.transpose(patch_pan_padded, [1, 2, 0])
        
        return patch_ms_padded,patch_pan_padded
            
    def create_image_label_paths(self, images_dir):
        """
        get full paths of the images and their corresponding label/predicted maps in a directory
        (label map): training phase, (predicted map): test phase 
        
        Args:
            images_dir (str) : directory, in which images are located
            labels_dir (str) : directory, in which label/predicted maps are/would be located
            
        Returns: 
            image_paths (list [str]): list that keeps full paths of all the images in a directory
            label_paths (list [str]): list that keeps full paths of all the label/pred maps in a directory
            
        """
                
        #get names of all the files under the given directory
        all_image_names = listdir(join(images_dir,'target'))

        target_paths = []
        masks_paths = []
        pan_paths = []
        ms_paths = []
        
        #there can be redundant files (applications like QGIS usually create an ".xml" file when an image is displayed)
        #all the files except ".tif" need to be filtered out
        for image_name in all_image_names:
            if image_name.endswith('.tif'):
                target_paths.append(join(images_dir,'target',image_name))
                masks_paths.append(join(images_dir,'masks', image_name))
                pan_paths.append(join(images_dir,'images','pan', image_name))
                ms_paths.append(join(images_dir,'images','ms', image_name))
     
        return dict(target=target_paths,masks=masks_paths,pan=pan_paths,ms=ms_paths)
        
    def open_test_image_label(self, image_no):
        """
        create a tif file for an output classification map
        georeference the classification map using the input image
        
        Args:
            image_no (int) : index determining for which image the network would produce a label map
        """
        
        #calculate minimum number of bits to represent number of classes
        min_num_of_bits = self.calc_num_of_bits()
        
        

        #open current image
        self.geo_ms = gdal.Open(self.valid_paths['ms'][image_no])
        prj = self.geo_ms.GetProjection()
        geotransform = self.geo_ms.GetGeoTransform()
        
        height = self.geo_ms.RasterYSize
        width = self.geo_ms.RasterXSize
        
        driver = gdal.GetDriverByName("GTiff")

        self.geo_pan = gdal.Open(self.valid_paths['pan'][image_no])
        prj_ = self.geo_pan.GetProjection()
        geotransform_ = self.geo_pan.GetGeoTransform()
        
        height_ = self.geo_pan.RasterYSize
        width_ = self.geo_pan.RasterXSize
                
        #create a tif file for the predicted map
        self.geo_mask = driver.Create(self.valid_paths['target'][image_no].replace('target','pred_mask'), 
                                           width_,
                                           height_,
                                           1,
                                           gdal.GDT_Byte, ['NBITS=' + str(min_num_of_bits)])
                                           
        #georeference the label map using georeference information of the input image
        # self.geo_mask.SetGeoTransform(geotransform_)
        # self.geo_mask.SetProjection(prj_)  

        #create a tif file for the predicted map
        self.geo_p = driver.Create(self.valid_paths['target'][image_no].replace('target','pred_p'), 
                                           width_,
                                           height_,
                                           self.num_of_channels,
                                           gdal.GDT_UInt16, ['NBITS=16'])
                                           
        #georeference the label map using georeference information of the input image
        # self.geo_p.SetGeoTransform(geotransform_)
        # self.geo_p.SetProjection(prj_)
        
    def close_test_image_label(self):
        """
        close the created label map and the image
        """
        self.geo_mask = None
        self.geo_p = None
        self.geo_pan = None
        self.geo_ms = None
        
    def calc_num_of_bits(self):
        """
        calculate min number of bits to represent number of classes
        useful to reduce the space occupied by the generated label map
        
        Returns:
            num_of_bits (int) : min number of bits to represent number of classes
        """
        num_of_bits = np.int(np.floor(np.log2(self.num_of_classes))) + 1
        return num_of_bits
        
    def find_num_of_channels(self):
        """
        find # of channels of the patches using the first patch
        we assume that all of the patches have the same number of channels
        
        Returns:
            num_of_channels (int) : number of channels in each patch
        """
        geo = gdal.Open(self.train_paths['target'][0])
        
        num_of_channels = geo.RasterCount
        
        del geo

        return num_of_channels
    
    def find_patch_size(self):
        """
        find patch size of the patches using the first patch
        we assume that height and width of all the patches are the same and equal to patch size
        """
        geo = gdal.Open(self.train_paths['target'][0])
    
        patch_size = geo.RasterYSize
        
        del geo
        
        return patch_size

