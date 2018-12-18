import sys
import numpy as np
import tensorflow as tf

from unet_model import Unet_model

flags = tf.app.flags
#parameters for both phases
flags.DEFINE_boolean("is_training", None, "True: training phase, False: test phase")
flags.DEFINE_string("train_dir", None, "directory, where training images are located")
flags.DEFINE_string("valid_dir", None, "directory, where valid/test images are located. " + 
                                        "(label maps during training phase, predicted maps during test phase)")
flags.DEFINE_integer("num_of_classes", None, "number of classes")
flags.DEFINE_string("snap_dir", None, "snapshot directory, where weights are saved regularly as the training continues. " + 
                                      "during the test phase, weights are restored from the last checkpoint under this directory")
#parameters for the training phase only
flags.DEFINE_integer("batch_size", None, "number of patches in a batch during the training phase")
flags.DEFINE_float("learning_rate", None, "learning rate for the adam optimizer")
flags.DEFINE_integer("num_of_epochs", None, "number of epochs")
flags.DEFINE_integer("num_of_iterations", None, "number of iterations in each epoch")
flags.DEFINE_integer("decay_epoch", None, "the parameter to determine in which epoch the learning rate for the adam optimizer would be decreased")
flags.DEFINE_float("decay_rate", None, "the parameter to determine how much the learning rate would be decreased")
flags.DEFINE_string("log_dir", None, "log directory, where logs are saved")
flags.DEFINE_boolean("fine_tuning", None, "True  : fine tuning mode on. Pretrained model is restored and continued training." + 
                                          "False : fine tuning model off. The model is trained from scratch")
flags.DEFINE_string("pretrained_weights_path", None, "path of the pretrained weights. This parameter can be None. " + 
                                                     "If it is None, the weights are initialized randomly")
flags.DEFINE_integer("snap_freq", None, "parameter detemining how often the trained model would be saved")
flags.DEFINE_boolean("seg_enable", None, "Whether to enable segmetation task or not.")
flags.DEFINE_boolean("pansharp_enable", None, "Whether to enable Pansharpening task or not.")

#parameters for the test phase only
flags.DEFINE_integer("patch_size", None, "since the test image might be very big, it is segmented patch by patch." + 
                                         "This parameter sets height and width of each patch")
flags.DEFINE_integer("padding", None, "padding is used to get rid of border effect during the test phase. " +
                                      "This parameter determines overlapping amount between the patches that are read from the big test image")


FLAGS = flags.FLAGS

def check_parameters():
    """
    Check of all the required parameters are set
    """    
    if FLAGS.is_training == None:
        sys.exit('--is_training parameter has to be set!')
    if FLAGS.train_dir == None:
        sys.exit('--train_dir parameter has to be set!') 
    if FLAGS.valid_dir == None:
        sys.exit('--valid_dir parameter has to be set!')
    if FLAGS.num_of_classes == None:
        sys.exit('--num_of_classes parameter has to be set!')
    if FLAGS.snap_dir == None:
        sys.exit('--snap_dir parameter has to be set!')
        
    if FLAGS.is_training:
        if FLAGS.batch_size == None:
            sys.exit('--batch_size parameter has to be set!')
        if FLAGS.learning_rate == None:
            sys.exit('--learning_rate parameter has to be set!')
        if FLAGS.num_of_epochs == None:
            sys.exit('--num_of_epochs parameter has to be set!')
        if FLAGS.decay_epoch == None:
            sys.exit('--decay_epoch parameter has to be set!')
        if FLAGS.num_of_iterations == None:
            sys.exit('--num_of_iterations parameter has to be set!')
        if FLAGS.decay_rate == None:
            sys.exit('--decay_rate parameter has to be set!')
        if FLAGS.log_dir == None:
            sys.exit('--log_dir parameter has to be set!')
        if FLAGS.fine_tuning == None:
            sys.exit('--fine_tuning parameter has to be set!')
        if FLAGS.snap_freq == None:
            sys.exit('--snap_freq parameter has to be set!')
        if FLAGS.seg_enable == None:
            sys.exit('--seg_enable parameter has to be set!')
        if FLAGS.pansharp_enable == None:
            sys.exit('--pansharp_enable parameter has to be set!')

    else:
        if FLAGS.patch_size == None:
            sys.exit('--patch_size parameter has to be set!')
        if FLAGS.padding == None:
            sys.exit('--padding parameter has to be set!')
    
def parse_mean_list():
    """
    parse FLAGS.mean_list according to comma and convert it to a float list
    
    Returns:
        mean_list (list [float]) : list containing mean value for each channel
    """
    try:
        mean_list = np.array(FLAGS.mean_list.split(','), np.float32)
    except:
        return None
    
    return mean_list
   
def main(_):
        
    check_parameters()
    # mean_list = parse_mean_list()   

    with tf.Session() as sess:
        unet_model = Unet_model(sess,
                                train_dir = FLAGS.train_dir,
                                valid_dir = FLAGS.valid_dir,
                                pretrained_weights_path = FLAGS.pretrained_weights_path,
                                patch_size = FLAGS.patch_size,
                                padding = FLAGS.padding,
                                num_of_classes = FLAGS.num_of_classes,
                                batch_size = FLAGS.batch_size,
                                learning_rate = FLAGS.learning_rate,
                                num_of_epochs = FLAGS.num_of_epochs,
                                num_of_iterations = FLAGS.num_of_iterations,
                                decay_epoch = FLAGS.decay_epoch,
                                decay_rate = FLAGS.decay_rate,
                                seg_en = FLAGS.seg_enable,
                                pansh_en = FLAGS.pansharp_enable,
                                is_training = FLAGS.is_training)
        if FLAGS.is_training:
            unet_model.train_model(FLAGS.snap_dir, FLAGS.snap_freq, FLAGS.log_dir, FLAGS.fine_tuning)
        else:
            unet_model.classify(FLAGS.snap_dir)
        sess.close()
                                 
if __name__ == '__main__':
    tf.app.run()

