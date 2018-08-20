# Copyright 2016 Vladimir Nekrasov
import numpy as np
import tensorflow as tf


def image_scaling(img, label):
  """Randomly scales the images between 0.5 to 1.5 times the original size.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels], and another
    tensor of size [batch_size, height_out, width_out]
  """
  scale = tf.random_uniform(
      [1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
  h_new = tf.to_int32(tf.to_float(tf.shape(img)[0]) * scale)
  w_new = tf.to_int32(tf.to_float(tf.shape(img)[1]) * scale)
  new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
  img = tf.image.resize_images(img, new_shape)
  # Rescale labels by nearest neighbor sampling.
  label = tf.image.resize_nearest_neighbor(
      tf.expand_dims(label, 0), new_shape)
  label = tf.squeeze(label, squeeze_dims=[0])
   
  return img, label


def image_mirroring(img, label):
  """Randomly horizontally mirrors the images and their labels.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels], and another
    tensor of size [batch_size, height_in, width_in]
  """
  distort_left_right_random = tf.random_uniform(
      [1], 0, 1.0, dtype=tf.float32)
  distort_left_right_random = distort_left_right_random[0]

  mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
  mirror = tf.boolean_mask([0, 1, 2], mirror)
  img = tf.reverse(img, mirror)
  label = tf.reverse(label, mirror)

  return img, label


def crop_and_pad_image_and_labels(image,
                                  label,
                                  crop_h,
                                  crop_w,
                                  ignore_label=255,
                                  random_crop=True):
  """Randomly crops and pads the images and their labels.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]
    crop_h: A number indicating the height of output data.
    crop_w: A number indicating the width of output data.
    ignore_label: A number indicating the indices of ignored label.
    random_crop: enable/disable random_crop for random cropping.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels], and another
    tensor of size [batch_size, height_out, width_out, 1]
  """
  # Needs to be subtracted and later added due to 0 padding.
  label = tf.cast(label, dtype=tf.float32)
  label = label - ignore_label 

  # Concatenate images with labels, which makes random cropping easier.
  combined = tf.concat(axis=2, values=[image, label]) 
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined,
      0,
      0,
      tf.maximum(crop_h, image_shape[0]),
      tf.maximum(crop_w, image_shape[1]))
    
  last_image_dim = tf.shape(image)[-1]
  last_label_dim = tf.shape(label)[-1]

  if random_crop:
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
  else:
    combined_crop = tf.image.resize_image_with_crop_or_pad(
        combined_pad,
        crop_h,
        crop_w)

  img_crop = combined_crop[:, :, :last_image_dim]
  label_crop = combined_crop[:, :, last_image_dim:]
  label_crop = label_crop + ignore_label
  label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
  # Set static shape so that tensorflow knows shape at running. 
  img_crop.set_shape((crop_h, crop_w, 3))
  label_crop.set_shape((crop_h,crop_w, 1))

  return img_crop, label_crop  


def read_labeled_image_list(data_dir, data_list):
  """Reads txt file containing paths to images and ground truth masks.
    
  Args:
    data_dir: A string indicating the path to the root directory of images
      and masks.
    data_list: A string indicating the path to the file with lines of the form
      '/path/to/image /path/to/label'.
       
  Returns:
    Two lists with all file names for images and masks, respectively.
  """
  f = open(data_list, 'r')
  images = []
  masks = []
  for line in f:
    try:
      image, mask = line.strip("\n").split(' ')
    except ValueError: # Adhoc for test.
      image = mask = line.strip("\n")
    images.append(data_dir + image)
    masks.append(data_dir + mask)
  return images, masks


def read_images_from_disk(input_queue,
                          input_size,
                          random_scale,
                          random_mirror,
                          random_crop,
                          ignore_label,
                          img_mean):
  """Reads one image and its corresponding label and perform pre-processing.
    
  Args:
    input_queue: A tensorflow queue with paths to the image and its mask.
    input_size: A tuple with entries of height and width. If None, return
      images of original size.
    random_scale: enable/disable random_scale for randomly scaling images
      and their labels.
    random_mirror: enable/disable random_mirror for randomly and horizontally
      flipping images and their labels.
    ignore_label: A number indicating the index of label to ignore.
    img_mean: A vector indicating the mean colour values of RGB channels.
      
  Returns:
    Two tensors: the decoded image and its mask.
  """

  img_contents = tf.read_file(input_queue[0])
  label_contents = tf.read_file(input_queue[1])
    
  img = tf.image.decode_jpeg(img_contents, channels=3)
  img = tf.cast(img, dtype=tf.float32)
  # Extract mean.
  img -= img_mean

  label = tf.image.decode_png(label_contents, channels=1)

  if input_size is not None:
    h, w = input_size

    # Randomly scale the images and labels.
    if random_scale:
      img, label = image_scaling(img, label)

    # Randomly mirror the images and labels.
    if random_mirror:
      img, label = image_mirroring(img, label)

    # Randomly crops the images and labels.
    img, label = crop_and_pad_image_and_labels(
      img, label, h, w, ignore_label, random_crop
    )

  return img, label


class ImageReader(object):
  """
  Generic ImageReader which reads images and corresponding
  segmentation masks from the disk, and enqueues them into
  a TensorFlow queue.
  """

  def __init__(self, data_dir, data_list, input_size,
               random_scale, random_mirror, random_crop,
               ignore_label, img_mean):
    """
    Initialise an ImageReader.
          
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
                 '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
                  images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      A tensor of size [batch_size, height_out, width_out, channels], and
      another tensor of size [batch_size, height_out, width_out]
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size
          
    self.image_list, self.label_list = read_labeled_image_list(
        self.data_dir, self.data_list)
    self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
    self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
    self.queue = tf.train.slice_input_producer(
        [self.images, self.labels],
        shuffle=input_size is not None) # not shuffling if it is val
    self.image, self.label = read_images_from_disk(
        self.queue,
        self.input_size,
        random_scale,
        random_mirror,
        random_crop,
        ignore_label,
        img_mean) 


  def dequeue(self, num_elements):
    """Packs images and labels into a batch.
        
    Args:
      num_elements: A number indicating the batch size.
          
    Returns:
      A tensor of size [batch_size, height_out, width_out, 3], and
      another tensor of size [batch_size, height_out, width_out, 1]
    """
    image_batch, label_batch = tf.train.batch(
        [self.image, self.label],
        num_elements,
        num_threads=2)
    return image_batch, label_batch
