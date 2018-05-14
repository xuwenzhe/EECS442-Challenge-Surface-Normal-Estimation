# EECS442 Challenge project, 2018-04-04, Team: deepEye
# created by Wenzhe Xu
# Network architecture is based on the following paper:
#
# @inproceedings{newell2016stacked,
#   title={Stacked hourglass networks for human pose estimation},
#   author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
#   booktitle={European Conference on Computer Vision},
#   pages={483--499},
#   year={2016},
#   organization={Springer}
# }

import tensorflow as tf
import numpy as np
import skimage.data
from PIL import Image
from hourglass_net import hnet_3stacks_no_incep


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


NUM_IMAGES = 20000
BATCH_SIZE = 20


def load_batch(idx):
	batch_color = []
	batch_mask = []
	batch_normal = []
	for i in range(len(idx)):
		image_name = str(idx[i]) + ".png"
		# image_name = str(batch_shuffle[batch_idx]*BATCH_SIZE+i)+".png"
		batch_color.append(skimage.data.imread("./train/color/"+image_name))
		batch_mask.append(skimage.data.imread("./train/mask/"+image_name))
		batch_normal.append(skimage.data.imread("./train/normal/"+image_name))

	batch_color = np.array(batch_color,dtype='f')
	batch_mask = np.array(batch_mask,dtype='f')
	batch_normal = np.array(batch_normal,dtype='f')

	return batch_color, batch_mask, batch_normal

def calc_loss(output, y,z):

	# gives mean angle error for given output tensor and its ref y
	output_mask = tf.abs(output) < 1e-5
	output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(output), output)
	output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)),-1)
	output_unit = tf.divide(output_no0,output_mag)

	z_mask = z[...,0]
	a11 = tf.boolean_mask(tf.reduce_sum(tf.square(output_unit),3),z_mask)
	a22 = tf.boolean_mask(tf.reduce_sum(tf.square(y),3),z_mask)
	a12 = tf.boolean_mask(tf.reduce_sum(tf.multiply(output_unit,y),3),z_mask)

	cos_angle = a12/tf.sqrt(tf.multiply(a11,a22))
	cos_angle_clipped = tf.clip_by_value(tf.where(tf.is_nan(cos_angle),-1*tf.ones_like(cos_angle),cos_angle),-1,1)
	# MAE, using tf.acos() is numerically unstable, here use Taylor expansion of "acos" instead
	loss = tf.reduce_mean(3.1415926/2-cos_angle_clipped-tf.pow(cos_angle_clipped,3)/6-tf.pow(cos_angle_clipped,5)*3/40-tf.pow(cos_angle_clipped,7)*5/112-tf.pow(cos_angle_clipped,9)*35/1152)
	return loss

x = tf.placeholder(tf.float32, shape=(None, 128,128,2)) 
y = tf.placeholder(tf.float32, shape=(None, 128,128,3))
z = tf.placeholder(tf.bool, shape=(None, 128,128,1))

output0, output1, output2 = hnet_3stacks_no_incep(x,True)

total_loss = calc_loss(output0,y,z) + calc_loss(output1,y,z) + calc_loss(output2,y,z)

train_step = tf.train.AdamOptimizer().minimize(total_loss)


with tf.Session() as sess:
	saver = tf.train.Saver()
	# saver = tf.train.import_meta_graph('./model_90000/model_90000.ckpt.meta')
	# saver.restore(sess,"./model_90000/model_90000.ckpt")
	# print "Model restored."
	sess.run(tf.global_variables_initializer())
	for i in range(100050):
		idx = np.random.choice(NUM_IMAGES, BATCH_SIZE)
		batch_color, batch_mask, batch_normal = load_batch(idx)
		X = np.zeros((BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,2),dtype='f')
		X[...,0] = batch_color[...,2]
		X[...,1] = batch_mask
		Y = np.zeros((BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
		Y[...] = batch_normal
		Z = np.zeros((BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
		Z[...,0] = batch_mask != 0

		X = (X/255.0-0.5)*2
		Y = (Y/255.0-0.5)*2

		_,loss_val,prediction = sess.run([train_step,total_loss,output2],feed_dict={x:X,y:Y,z:Z})
		
		if i%50 == 0:
			print "step %d, training loss %g"%(i, loss_val)


		if i % 5000 == 0:
			# visually compare the first sample in the batch between predicted and ground truth
			image_tensor  = np.asarray(prediction)
			image = image_tensor[0,...]
			image_mag = np.expand_dims(np.sqrt(np.square(image).sum(axis=2)),-1)
			image_unit = np.divide(image,image_mag)
			
			image_out = (image_unit/2+0.5)*255

			mask = batch_mask[0,...]
			image_out = np.multiply(image_out, np.expand_dims(mask/255,-1))	

			f = Image.fromarray(image_out.astype(np.uint8))
			f.show()
			f = Image.open("./train/normal/"+str(idx[0])+".png")
			f.show()


		if i % 10000 == 0 and i != 0:
			save_path = saver.save(sess,"./model_"+str(i)+"/model_"+str(i)+".ckpt")
		
