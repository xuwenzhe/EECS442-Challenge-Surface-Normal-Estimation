import numpy as np
import tensorflow as tf
import skimage.data
import os
from PIL import Image
from hourglass_net import hnet_3stacks_no_incep


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


NUM_IMAGES = 2000

def load_all_images(data_dir):
	data_ = []
	file_order = []
	file_names = [os.path.join(data_dir, f)
		for f in os.listdir(data_dir)]
	file_order =  [ f
		for f in os.listdir(data_dir)]
	for f in file_names:
		data_.append(skimage.data.imread(f))
	data_ = np.array(data_,dtype='f')
	return data_, file_order
	

color_test,order = load_all_images('./test/color')
mask_test,_ = load_all_images('./test/mask/')

images_test = np.zeros((NUM_IMAGES,128,128,2),dtype='f')
images_test[...,0] = color_test[...,2]
images_test[...,1] = mask_test

x = tf.placeholder(tf.float32, shape=(None, 128,128,2)) 

output0, output1, output2 = hnet_3stacks_no_incep(x,True)


with tf.Session() as sess:
	saver = tf.train.Saver()
	saver = tf.train.import_meta_graph('./model_100000/model_100000.ckpt.meta')
	saver.restore(sess,"./model_100000/model_100000.ckpt")
	print "Model restored."
  
	for i in range(NUM_IMAGES):
		X = np.zeros((1,128,128,2),dtype='f')
		X[0,...,0] = images_test[i,...,0]
		X[0,...,1] = mask_test[i]
		X = (X/255.0-0.5)*2
		image_tensor = sess.run([output2], feed_dict={x:X})


		image_tensor  = np.asarray(image_tensor)
		image = image_tensor[0,0,...]
		image_mag = np.expand_dims(np.sqrt(np.square(image).sum(axis=2)),-1)
		image_unit = np.divide(image,image_mag)
		image_out = (image_unit/2+0.5)*255
	  
		mask = mask_test[i]
		image_out = np.multiply(image_out,np.expand_dims(mask/255,-1))

		f = Image.fromarray(image_out.astype(np.uint8))
		f.save(os.path.join('./test/normal/', order[i]))
	print "Complete!"