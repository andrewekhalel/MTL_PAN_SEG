import tensorflow as tf

import numpy as np
from os.path import join
import gdal
import math

from unet_model_helpers import *
from data_processor import Data_processor

from models import build_fusenet,build_fusenet_low
import numpy as np

import time

class Unet_model:
    """
    Unet model described in this paper: "https://arxiv.org/abs/1505.04597"
    
    Attributes:
    sess (Tensorflow session): Tensorflow session
    pretrained_weights_path (str) : path for the VGG16 pretrained weights
    patch_size (int)         : used to calculate height and width (they are exactly the same) of the patches that are fed to the network
                               label maps dims = [height: patch_size, width: patch_size]            
    num_of_channels (int)    : number of channels in the input images
    batch_size (int)         : batch size (number of patches in a batch). 1 for the test phase
    learning_rate (float)    : learning rate for the adam optimizer (used only in training phase)
    num_of_epochs (int)      : number of epochs (used only in training phase)
    num_of_iterations (int)  : number of iterations in each epoch (used only in training phase)
    decay_epoch (int)        : the parameter that determines in which epoch the learning rate for the adam optimizer would be decreased
                               (used only in training phase)
    decay_rate (float)       : the parameter that determines how much the learning rate would be decreased
                               the learning rate is muliplied by this number in epoch <decay_epoch>
                               (used only in training phase)
    is_training (bool)       : True = training phase / False = test phase
    depth (int)              : multi class classification: # of channels
                               binary classification     : 1
    data_processor           : an instance of Data_processor class
    """
    def __init__(self, sess,
                       train_dir,
                       valid_dir,
                       pretrained_weights_path,
                       patch_size,
                       padding, 
                       num_of_classes,
                       batch_size,
                       learning_rate,
                       num_of_epochs,
                       num_of_iterations,
                       decay_epoch,
                       decay_rate,
                       seg_en,
                       pansh_en,
                       is_training):
        
        self.data_processor = Data_processor(train_dir = train_dir,
                                             valid_dir = valid_dir,
                                             patch_size = patch_size,
                                             padding = padding,
                                             num_of_classes = num_of_classes,
                                             batch_size = batch_size,
                                             is_training = is_training,
                                             seg_en=seg_en,
                                             pansh_en=pansh_en)
        
        self.num_of_channels = self.data_processor.num_of_channels
    
        self.valid_dir=valid_dir

        self.sess = sess
        self.is_training = is_training
        self.seg_en = seg_en
        self.pansh_en = pansh_en
        
        #set parameters for the training phase
        if self.is_training:
            self.pretrained_weights_path = pretrained_weights_path
            self.patch_size = self.data_processor.patch_size
            self.learning_rate = learning_rate
            self.num_of_epochs = num_of_epochs
            self.num_of_iterations = num_of_iterations
            self.decay_epoch = decay_epoch
            self.decay_rate = decay_rate
            self.batch_size = batch_size
            
        #set parameters for the test phase
        else:
            self.patch_size = patch_size
            self.batch_size = 1
            self.padding = padding
        
        #set depth
        if num_of_classes == 2:
            self.depth = 1
        else:
            self.depth = num_of_classes        
        
                                       
    def build_model(self, ms_patches,pan_patches, start_filter_num = 32, reuse = False):
        """
        build the Unet model
        
        Args:
            input_patches (4d tensor [float]) : inputs image patches
            start_filter_num (int)            : number of output filters for the first convolution. Optional (64 by default)
            reuse (bool)                      : 
        
        Returns:
            pred (4d tensor [float]) : unscaled predictions
        """
        # return build_fusenet(self,ms_patches,pan_patches, reuse = reuse)

        with tf.variable_scope('model', reuse = reuse):
            
            #contraction part
            #convolution sequence 1
            conv_seq1 = conv_block_sequence(inputs = pan_patches, filters = start_filter_num, num_of_conv_blocks = 2, 
                                            training = self.is_training, scope_name = 'seq1')
            pool1 = max_pool(conv_seq1, 'pool1')
        
            #convolution sequence 2
            conv_seq2 = conv_block_sequence(inputs = pool1, filters = start_filter_num * 2, num_of_conv_blocks = 2, 
                                            training = self.is_training, scope_name = 'seq2')
            pool2 = max_pool(conv_seq2, 'pool2')

            conv_seq_ms = conv_block_sequence(inputs = ms_patches, filters = start_filter_num * 2, num_of_conv_blocks = 2, 
                                            training = self.is_training, scope_name = 'seq_ms')
            
            conc = tf.concat([pool2,conv_seq_ms],axis=1)

            #convolution sequence 3
            conv_seq3 = conv_block_sequence(inputs = conc, filters = start_filter_num * 4, num_of_conv_blocks = 2,
                                            training = self.is_training, scope_name = 'seq3')
            pool3 = max_pool(conv_seq3, 'pool3')
             
            #convolution sequence 4
            conv_seq4 = conv_block_sequence(inputs = pool3, filters = start_filter_num * 8, num_of_conv_blocks = 2, 
                                            training = self.is_training, scope_name = 'seq4')
            pool4 = max_pool(conv_seq4, 'pool4')
            
            # #convolution sequence 5
            # conv_seq5 = conv_block_sequence(inputs = pool4, filters = start_filter_num * 8, num_of_conv_blocks = 2, 
            #                                 training = self.is_training, scope_name = 'seq5')     
           
            # #center
            # pool5 = max_pool(conv_seq5, 'pool5')
            # center = conv2d(inputs = pool5, filters = start_filter_num * 8, kernel_size = (3, 3), strides = (1, 1), conv_name = 'center')
            
            center = conv_block_sequence(inputs = pool4, filters = start_filter_num * 16, num_of_conv_blocks = 2, 
                                            training = self.is_training, scope_name = 'seq5')     

            if self.seg_en:
                #expansion part 1
                # #upsample - concatenation - convolution 1
                # up1 = upsample_concat(inputs1 = conv_seq5, inputs2 = center, num_of_channels_reduce_factor = 2, 
                #                       training = self.is_training, scope_name = 'up1')
                # up1_conv_seq = conv_block_sequence(inputs = up1, filters = conv_seq5.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                #                                    training = self.is_training, scope_name = 'up1_seq')
                
                #upsample - concatenation - convolution 2 
                up2 = upsample_concat(inputs1 = conv_seq4, inputs2 = center, num_of_channels_reduce_factor = 2,
                                      training = self.is_training, scope_name = 'up2')
                up2_conv_seq = conv_block_sequence(inputs = up2, filters = start_filter_num * 8, num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up2_seq')
                
                #upsample - concatenation - convolution 3
                up3 = upsample_concat(inputs1 = conv_seq3, inputs2 = up2_conv_seq, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up3')
                up3_conv_seq = conv_block_sequence(inputs = up3, filters = start_filter_num * 4, num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up3_seq')
                
                #upsample - concatenation - convolution 4
                up4 = upsample_concat(inputs1 = conv_seq2, inputs2 = up3_conv_seq, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up4')
                up4_conv_seq = conv_block_sequence(inputs = up4, filters = start_filter_num * 2, num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up4_seq')
                
                #upsample - concatenation - convolution 5
                up5 = upsample_concat(inputs1 = conv_seq1, inputs2 = up4_conv_seq, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up5')
                up5_conv_seq = conv_block_sequence(inputs = up5, filters = start_filter_num, num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up5_seq')
                
                #seg-head
                seg_conv = conv2d(up5_conv_seq, filters = self.depth, kernel_size = (1, 1), strides = (1, 1), conv_name = 'seg_conv')
                seg = tf.transpose(seg_conv, [0, 2, 3, 1])

            if self.pansh_en:
                #expansion part 2
                # #upsample - concatenation - convolution 1
                # up1_2 = upsample_concat(inputs1 = conv_seq5, inputs2 = center, num_of_channels_reduce_factor = 2, 
                #                       training = self.is_training, scope_name = 'up1_2' )
                # up1_conv_seq_2 = conv_block_sequence(inputs = up1_2, filters = conv_seq5.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                #                                    training = self.is_training, scope_name = 'up1_seq_2')
                
                #upsample - concatenation - convolution 2 
                up2_2 = upsample_concat(inputs1 = conv_seq4, inputs2 = center, num_of_channels_reduce_factor = 2,
                                      training = self.is_training, scope_name = 'up2_2')
                up2_conv_seq_2 = conv_block_sequence(inputs = up2_2, filters = conv_seq4.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up2_seq_2')
                
                #upsample - concatenation - convolution 3
                up3_2 = upsample_concat(inputs1 = conv_seq3, inputs2 = up2_conv_seq_2, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up3_2')
                up3_conv_seq_2 = conv_block_sequence(inputs = up3_2, filters = conv_seq3.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up3_seq_2')
                
                #upsample - concatenation - convolution 4
                up4_2 = upsample_concat(inputs1 = conv_seq2, inputs2 = up3_conv_seq_2, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up4_2')
                up4_conv_seq_2 = conv_block_sequence(inputs = up4_2, filters = conv_seq2.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up4_seq_2')
                
                #upsample - concatenation - convolution 5
                up5_2 = upsample_concat(inputs1 = conv_seq1, inputs2 = up4_conv_seq_2, num_of_channels_reduce_factor = 4,
                                      training = self.is_training, scope_name = 'up5_2')
                up5_conv_seq_2 = conv_block_sequence(inputs = up5_2, filters = conv_seq1.get_shape().as_list()[1], num_of_conv_blocks = 2, 
                                                   training = self.is_training, scope_name = 'up5_seq_2')
                
                # pansharpen-head
                pansharpen_conv = conv2d(up5_conv_seq_2, filters = self.num_of_channels, kernel_size = (1, 1), strides = (1, 1), conv_name = 'pansharpen_conv')
                pansharpen = tf.transpose(pansharpen_conv, [0, 2, 3, 1])
        
        if self.seg_en and self.pansh_en: 
            return seg,pansharpen
        elif self.seg_en: 
            return seg,None
        else: 
            return None,pansharpen

    def train_model(self, snap_dir, snap_freq, log_dir, fine_tuning):
       
        """
        train the neural network and save weights of the trained network to the disk
        if fine_tuning mode is on, the pretrained model is restored and continued training
        if it is off, the model is trained from scratch
        
        Args:
            snap_dir (str)     : directory, where the trained network would be saved
            snap_freq (int)    : parameter determining how often the trained model would be saved
            log_dir (str)      : directory, where the loss over the time would be saved
            fine_tuning (bool) : True: yes, False: no
        """
        
        #create iterators for training and validation data
        training_generator = self.data_processor.batch_generator(is_training = True)
        validation_generator = self.data_processor.batch_generator(is_training = False)
        
        #get a training batch and unpack it
        training_batch = training_generator.get_next()
        training_ms_batch = training_batch[0]
        training_pan_batch = training_batch[1]
        training_target_batch = training_batch[2]
        training_mask_batch = training_batch[3]

        # print(self.sess.run(tf.reduce_mean(training_mask_batch)))
        
        #get a validation batch and unpack it 
        validation_batch = validation_generator.get_next()
        validation_ms_batch = validation_batch[0]
        validation_pan_batch = validation_batch[1]
        validation_target_batch = validation_batch[2]
        validation_mask_batch = validation_batch[3]
        
        #[batch_size, height, width, num_of_channels] to [batch_size, num_of_channels, height, width]
        training_ms_batch = tf.transpose(training_ms_batch, [0, 3, 1, 2])
        training_pan_batch = tf.transpose(training_pan_batch, [0, 3, 1, 2])
        validation_ms_batch = tf.transpose(validation_ms_batch, [0, 3, 1, 2])
        validation_pan_batch = tf.transpose(validation_pan_batch, [0, 3, 1, 2])
        
        #calculate training loss
        training_seg,training_p = self.build_model(training_ms_batch,training_pan_batch, reuse = False)
        if self.seg_en:
            training_seg_loss = calc_loss(training_seg, training_mask_batch, self.depth)
        if self.pansh_en:
            training_p_loss = calc_loss_p(training_p, training_target_batch, self.batch_size,self.num_of_channels)
        
        alpha = 0.2
        if self.seg_en and self.pansh_en:
            training_loss =  (alpha*training_seg_loss) + ((1-alpha)*training_p_loss)
        elif self.seg_en:
            training_loss =  training_seg_loss
        else:
            training_loss =  training_p_loss

        #optimization step
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training_step = tf.train.AdamOptimizer(self.learning_rate).minimize(training_loss)

        #initialize all the variables
        self.sess.run(tf.global_variables_initializer())

        #calculate validation loss
        validation_seg,validation_p = self.build_model(validation_ms_batch,validation_pan_batch, reuse = True)
        if self.seg_en:
            validation_seg_loss = calc_loss(validation_seg, validation_mask_batch, self.depth)
        if self.pansh_en:
            validation_p_loss = calc_loss_p(validation_p, validation_target_batch, self.batch_size,self.num_of_channels)

        if self.seg_en and self.pansh_en:
            validation_loss =  (alpha*validation_seg_loss) + ((1-alpha)*validation_p_loss)
        elif self.seg_en:
            validation_loss =  validation_seg_loss
        else:
            validation_loss =  validation_p_loss

        
        if self.pansh_en:
            sig_vl_p = tf.nn.sigmoid(validation_p)
            uqi_val = Q(sig_vl_p, validation_target_batch,batch_size=self.batch_size,ms_channels=self.num_of_channels)
            psnr_val = psnr(sig_vl_p, validation_target_batch)
        if self.seg_en:
            sig_vl_seg = output_layer(validation_seg,self.depth)
            iou_val = iou(sig_vl_seg, validation_mask_batch,self.batch_size,self.depth)

        self.sess.run(tf.local_variables_initializer())

        # import time
        # time.sleep(10)

        #if fine_tuning mode is on, the pretrained model is restored
        #curr_epoch is equal to the (number of epochs + 1) used in training phase for the pretrained model
        if fine_tuning:
            self.restore_model(snap_dir)
            curr_epoch = self.find_model_epoch(snap_dir) + 1
            
        #if fine_tuning mode is off, the model is trained from scratch
        #curr_epoch starts from 1
        else:
            curr_epoch = 1
            
            #if directory for the pretrained weights is not None, load pretrained VGG16 weights
            if (self.pretrained_weights_path != None):
                self.load_weights()
                       
        #summary for the loss
        if self.seg_en:
            mult_value = int(255./self.depth)
            tf.summary.image('mask_gt',tf.expand_dims(tf.cast(tf.argmax(validation_mask_batch, axis = -1),tf.uint8), -1)*mult_value)
            tf.summary.image('mask_p', tf.cast(sig_vl_seg,tf.uint8)*mult_value)

        if self.pansh_en:
            tf.summary.image('p_gt',validation_target_batch)
            tf.summary.image('p_pred', sig_vl_p)

        tf.summary.scalar('training_loss', training_loss)
        tf.summary.scalar('validation_loss', validation_loss)
        merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep = (self.num_of_epochs - curr_epoch + 1) // snap_freq)                
        train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)  
        
        #each epoch
        while curr_epoch <= self.num_of_epochs:
                     
            if curr_epoch == self.decay_epoch:
                self.learning_rate = self.learning_rate * self.decay_rate

            #each iteration
            for curr_iter in range(1, self.num_of_iterations + 1):
                
                time_start = time.time()
                # print(self.sess.run([training_seg_loss,training_p_loss,training_loss]))
                # print (self.sess.run(tf.cast(tr_sig>=0.5,tf.uint8)).mean())
                # _,q,ps,iu=self.sess.run([training_step,uqi_val,psnr_val,iou_val])
                _ =self.sess.run(training_step)
                
                #save the current loss to a log file
                if ((curr_iter - 1) % 100) == 0:
                    summary = self.sess.run(merged)
                    train_writer.add_summary(summary, (curr_epoch - 1) * self.num_of_iterations + curr_iter)
                
                elapsed_time = time.time() - time_start
                
                # print('Epoch: %d / %d, Iter : %d / %d, Elapsed_time : %.4f secs, Q: %.5f, PSNR: %.5f, IOU: %.5f' % 
                #       (curr_epoch, self.num_of_epochs, curr_iter, self.num_of_iterations, elapsed_time,q,ps,iu))

                report  = 'Epoch: %d / %d, Iter : %d / %d, Elapsed_time : %.4f secs' % \
                      (curr_epoch, self.num_of_epochs, curr_iter, self.num_of_iterations, elapsed_time)

                if self.seg_en:
                    iv =self.sess.run(iou_val)
                    report +=' Validation IoU: %.6f'%iv

                if self.pansh_en:
                    q,ps=self.sess.run([uqi_val,psnr_val])
                    report +=' Validation Q: %.6f Validation psnr: %.6f'%(q,ps)

                print(report)

                
                
            if curr_epoch % snap_freq == 0:
                saver.save(self.sess, snap_dir + '/model', global_step = curr_epoch)
            
            curr_epoch += 1
           
    def find_model_epoch(self, snap_dir):
        """
        find out how many epochs have been used in the training phase for the model that would be restored
        Note: when calculating the number of epochs, only the model indicated by the latest checkpoint is considered
        the others are ignored
        
        Args:
            snap_dir (str)    : directory, where parameters for the trained network are located 
        
        Returns:
            model_epoch (int) : number of epochs have been used in the training phase for the model that would be restored
        """

        f = open(join(snap_dir, 'checkpoint'),'r')
        lines = f.readlines()
        model_id = lines[0].split('"')
        model_epoch = np.int(model_id[1].split('-')[1])
        f.close()
        return model_epoch

    def restore_model(self, snap_dir):
        """
        restore parameters of the pretrained network using the last checkpoint in the snapshot directory
        
        Args:
            snap_dir (str) : directory, where the trained network is located 
        """
        saver = tf.train.Saver()
        
        latest_check_point = tf.train.latest_checkpoint(snap_dir)
        saver.restore(self.sess, latest_check_point)    
    
    def classify(self, snap_dir):
       
        """
        load the learned parameters to classify each test image
        since the test images might be big, perform classification patch by patch
        in order to get rid of border effects, pad each patch by (padding)
        
        Args:
            snap_dir (str) : directory, where parameters for the trained network are located
                             learned parameters are loaded from the latest checkpoint under this directory
        """
        if self.padding == 0:
            patch_ms = tf.placeholder(tf.float32,shape=(1,
                                            self.num_of_channels,
                                             (self.patch_size + 2 * self.padding)//4,
                                             (self.patch_size + 2 * self.padding)//4),name='ms_in')
            patch_pan = tf.placeholder(tf.float32,shape=(1,
                                            1,
                                             (self.patch_size + 2 * self.padding),
                                             (self.patch_size + 2 * self.padding)),name='pan_in')
            pred_seg,pred_p = self.build_model(patch_ms,patch_pan)
            
            self.restore_model(snap_dir)        
            
            if self.seg_en:
                output = output_layer(pred_seg, self.depth)
                # output = tf.nn.sigmoid(pred_seg)
            else:
                output=None

            
            print('Switching to another implementation ...')
            from pred import predictor
            predictor(self,patch_ms,patch_pan,output,pred_p)
            exit()


        #create an iterator
        generator = self.data_processor.test_patch_generator()
        
        #get a patch, its top-left x and y coordinate location in the actual image
        #and its actual height and width
        next_element = generator.get_next()
        patch_ms = next_element[0]
        patch_pan = next_element[1]
        y_top_left_tensor = next_element[2]
        x_top_left_tensor = next_element[3]
        patch_height_tensor = next_element[4]
        patch_width_tensor = next_element[5]
        
        pred_seg,pred_p = self.build_model(patch_ms,patch_pan)
        
        self.restore_model(snap_dir)        
        
        if self.seg_en:
            output = output_layer(pred_seg, self.depth)
        
        while True:
            try:
                #get the patch produced by the generator
                # seg_pred,p_pred, y_top_left, x_top_left, patch_height, patch_width = self.sess.run([output,
                #                                                                                tf.nn.sigmoid(pred_p),
                #                                                                                y_top_left_tensor, 
                #                                                                                x_top_left_tensor,
                #                                                                                patch_height_tensor, 
                #                                                                                patch_width_tensor])

                    #classify the patch
                if self.seg_en:
                    seg_pred, y_top_left, x_top_left, patch_height, patch_width = self.sess.run([output,
                                                                                                   y_top_left_tensor, 
                                                                                                   x_top_left_tensor,
                                                                                                   patch_height_tensor, 
                                                                                                   patch_width_tensor])
                    seg_for_actual_patch = seg_pred[0, 
                                                       self.padding:(self.padding + patch_height), 
                                                       self.padding:(self.padding + patch_width), 
                                                       0]

                    #write the patch to a file
                    self.data_processor.geo_mask.GetRasterBand(1).WriteArray(seg_for_actual_patch, int(x_top_left), int(y_top_left))

                if self.pansh_en:
                    p_pred, y_top_left, x_top_left, patch_height, patch_width = self.sess.run([tf.nn.sigmoid(pred_p),
                                                                                                   y_top_left_tensor, 
                                                                                                   x_top_left_tensor,
                                                                                                   patch_height_tensor, 
                                                                                                   patch_width_tensor])
                    p_for_actual_patch = p_pred[0, 
                                                       self.padding:(self.padding + patch_height), 
                                                       self.padding:(self.padding + patch_width), 
                                                       :] 
                
                    self.data_processor.geo_p.GetRasterBand(1).WriteArray((p_for_actual_patch[:,:,0]*2047.).astype(np.uint16), int(x_top_left), int(y_top_left))
                    self.data_processor.geo_p.GetRasterBand(2).WriteArray((p_for_actual_patch[:,:,1]*2047.).astype(np.uint16), int(x_top_left), int(y_top_left))
                    self.data_processor.geo_p.GetRasterBand(3).WriteArray((p_for_actual_patch[:,:,2]*2047.).astype(np.uint16), int(x_top_left), int(y_top_left))
        
            #when all the patches are read, exit
            except tf.errors.OutOfRangeError:
                # self.data_processor.close_test_image_label()
                break
        
        #compress all the predicted label maps
        self.data_processor.compress_label_maps()
        
    def load_weights(self):
        """
        load VGG16 pretrained weights
        """
        
        #load the pretrained weights
        weights = np.load(self.pretrained_weights_path)
        
        #name of each variable
        keys = sorted(weights.keys())
        
        #all the trainable weights in the network
        vars = {v.name:v for v in tf.trainable_variables()}
        
        #traverse on each variable in the pretrained weights
        for key in keys:
            
            #ignore the fully connected layers
            if key.startswith('c'):
                
                #which sequence?
                seq_id = key[4]
                
                #which convolution block?
                conv_block_id = key[6]
                
                #variable name
                if key[-1] == 'W':
                    var_name = 'model/seq' + seq_id + '/conv_' + conv_block_id + '/conv2d/kernel:0'
                else:
                    var_name = 'model/seq' + seq_id + '/conv_' + conv_block_id + '/conv2d/bias:0'
                    
                #assign the pretrained weights to the variable in the network
                self.sess.run(tf.assign(vars[var_name], weights[key]))

