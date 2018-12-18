from unet_model_helpers import *
import tensorflow as tf

def build_fusenet(model,ms_patches,pan_patches, reuse = False):
	with tf.variable_scope('model', reuse = reuse):
			
		#X_PAN
		#convolution sequence 1
		conv_seq1 = conv_block_sequence(inputs = pan_patches, filters = 16, kernel_size=13, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq1')
		pool1 = max_pool(conv_seq1, 'pool1')
	
		#convolution sequence 2
		conv_seq2 = conv_block_sequence(inputs = pool1, filters = 32, kernel_size=7, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq2')
		pool2 = max_pool(conv_seq2, 'pool2')


		#X_MS
		conv_seq_ms = conv_block_sequence(inputs = ms_patches, filters = 32, kernel_size=1, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq_ms')
		
		conc = tf.concat([pool2,conv_seq_ms],axis=1)

		#convolution sequence 3
		conv_seq3 = conv_block_sequence(inputs = conc, filters = 64, kernel_size=3, num_of_conv_blocks = 2,
										training = model.is_training, scope_name = 'seq3')
		pool3 = max_pool(conv_seq3, 'pool3')
		 
		#convolution sequence 4		
		conv_seq4 = conv_block_sequence(inputs = pool3, filters = 128, kernel_size=3, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq4')
		pool4 = max_pool(conv_seq4, 'pool4')
			
	
		#upsample - concatenation - convolution 2 

		up2_conv_seq = deconv(inputs = pool4, filters = 128, 
										   training = model.is_training, scope_name = 'up2_seq')
		
		#upsample - concatenation - convolution 3
		in3 = tf.concat([up2_conv_seq,conv_seq4],axis=1)
		up3_conv_seq = deconv(inputs = in3, filters = 64, 
										   training = model.is_training, scope_name = 'up3_seq')
		
		#upsample - concatenation - convolution 4
		in4 = tf.concat([up3_conv_seq,conv_seq3],axis=1)
		up4_conv_seq = deconv(inputs = in4, filters = 32,
										   training = model.is_training, scope_name = 'up4_seq')
		
		#upsample - concatenation - convolution 5
		in5 = tf.concat([up4_conv_seq,conv_seq2],axis=1)
		up5_conv_seq = deconv(inputs = in5, filters = 16,
										   training = model.is_training, scope_name = 'up5_seq')
		
		#seg-head
		seg_conv = conv2d(up5_conv_seq, filters = model.depth, kernel_size = (1, 1), strides = (1, 1), conv_name = 'seg_conv')
		seg = tf.transpose(seg_conv, [0, 2, 3, 1])
 
		return seg,None

def build_fusenet_low(model,ms_patches,pan_patches, reuse = False):
	with tf.variable_scope('model', reuse = reuse):
			
		#X_PAN
		#convolution sequence 1
		conv_seq1 = conv_block_sequence(inputs = pan_patches, filters = 16, kernel_size=13, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq1')
		pool1 = max_pool(conv_seq1, 'pool1')
	
		#convolution sequence 2
		conv_seq2 = conv_block_sequence(inputs = pool1, filters = 32, kernel_size=7, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq2')
		pool2 = max_pool(conv_seq2, 'pool2')


		#X_MS
		conv_seq_ms = conv_block_sequence(inputs = ms_patches, filters = 32, kernel_size=1, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq_ms')
		
		conc = tf.concat([pool2,conv_seq_ms],axis=1)

		#convolution sequence 3
		conv_seq3 = conv_block_sequence(inputs = conc, filters = 64, kernel_size=3, num_of_conv_blocks = 2,
										training = model.is_training, scope_name = 'seq3')
		pool3 = max_pool(conv_seq3, 'pool3')
		 
		#convolution sequence 4		
		conv_seq4 = conv_block_sequence(inputs = pool3, filters = 128, kernel_size=3, num_of_conv_blocks = 2, 
										training = model.is_training, scope_name = 'seq4')
		pool4 = max_pool(conv_seq4, 'pool4')
			
	
		#upsample - concatenation - convolution 2 

		up2_conv_seq = deconv(inputs = pool4, filters = 128, 
										   training = model.is_training, scope_name = 'up2_seq')
		
		#upsample - concatenation - convolution 3
		up3_conv_seq = deconv(inputs = up2_conv_seq, filters = 64, 
										   training = model.is_training, scope_name = 'up3_seq')
		
		#upsample - concatenation - convolution 4
		up4_conv_seq = deconv(inputs = up3_conv_seq, filters = 32,
										   training = model.is_training, scope_name = 'up4_seq')
		
		#upsample - concatenation - convolution 5
		up5_conv_seq = deconv(inputs = up4_conv_seq, filters = 16,
										   training = model.is_training, scope_name = 'up5_seq')
		
		#seg-head
		seg_conv = conv2d(up5_conv_seq, filters = model.depth, kernel_size = (1, 1), strides = (1, 1), conv_name = 'seg_conv')
		seg = tf.transpose(seg_conv, [0, 2, 3, 1])
 
		return seg,None

