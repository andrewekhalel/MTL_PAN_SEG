import os
import tifffile as tif
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf

r=4 
ms_channels=3
def predictor(unet,ms_plchldr,pan_plchldr,seg_out,pansh_out):
	val_dir = unet.valid_dir

	for f in tqdm(os.listdir(os.path.join(val_dir,'masks'))):
		ms = np.transpose(tif.imread(os.path.join(val_dir,'images','ms',f)),(1, 2, 0))[:,:,[0,1,2]] /2047.
		pan = tif.imread(os.path.join(val_dir,'images','pan',f)) /2047.
		pred_seg,pred_pansh = predict_from_patches(unet,ms,pan,ms_plchldr,pan_plchldr,seg_out,pansh_out,overlap = 256)
		if unet.seg_en:
			if unet.depth <= 2:
				tif.imsave(os.path.join(val_dir,'pred_mask',f),(pred_seg>=0.5).astype(np.uint8))
			else:
				tif.imsave(os.path.join(val_dir,'pred_mask',f),pred_seg.astype(np.uint8))
		if unet.pansh_en:
			tif.imsave(os.path.join(val_dir,'pred_p',f),(pred_pansh*2047.).astype(np.uint16))

def predict_from_patches(unet,ms,pan,ms_plchldr,pan_plchldr,seg_out,pansh_out,overlap = 0):
	"""
	predict full image by overlapping patches

	:param model: trained model for prediction
	:param ms: input ms image
	:param pan: input ms image
	:param patch_shape: patch input shape (height,width,channels)
	:param overlap: overlapped pixels between patches
	"""	
	patch_shape = (unet.patch_size,unet.patch_size)
	outer_margin = unet.padding//2

	# add reflection
	pan_with_reflection = cv2.copyMakeBorder(pan,\
		outer_margin,patch_shape[0],outer_margin,patch_shape[1],cv2.BORDER_REFLECT_101)
	ms_with_reflection = cv2.copyMakeBorder(ms,\
		outer_margin//r,patch_shape[0]//r,outer_margin//r,patch_shape[1]//r,cv2.BORDER_REFLECT_101)

	if unet.seg_en:
		output_seg = np.zeros((pan.shape[0],pan.shape[1]))
	if unet.pansh_en:
		output_pansh = np.zeros((pan.shape[0],pan.shape[1],ms_channels))
	times = np.zeros((pan.shape[0],pan.shape[1]))

	for h in range(0,pan_with_reflection.shape[0]-patch_shape[0]+1,patch_shape[0]-overlap):
		for w in range(0,pan_with_reflection.shape[1]-patch_shape[1]+1,patch_shape[1]-overlap):
			# use model to predict
			pp = pan_with_reflection[h:(h+patch_shape[0]),w:(w+patch_shape[1])]
			mp = ms_with_reflection[h//r:((h+patch_shape[0])//r),w//r:((w+patch_shape[1])//r),:]

			# test time augmentation
			if unet.seg_en:
				shp = output_seg[(h):(h+patch_shape[0]),(w):(w+patch_shape[1])].shape
				rots = 4
				if shp[0] != shp[1]:
					rots=1
				aug_seg = np.zeros(shp)
				for k in range(rots):
					mp_r = np.rot90(mp,k=k)
					pp_r = np.rot90(pp,k=k)
					pred_seg = unet.sess.run(seg_out, feed_dict={ms_plchldr:np.transpose(mp_r[np.newaxis,:,:,:],(0,3,1,2)),
															pan_plchldr:pp_r[np.newaxis,np.newaxis,:,:]})
					aug_seg += np.rot90(pred_seg[0,:shp[0],:shp[1],0],k=-k)
				
				output_seg[(h):(h+patch_shape[0]),(w):(w+patch_shape[1])] += aug_seg/rots
			if unet.pansh_en:
				shp = output_pansh[(h):(h+patch_shape[0]),(w):(w+patch_shape[1]),:].shape
				rots = 4
				if shp[0] != shp[1]:
					rots=1
				aug_pansh = np.zeros(shp)
				for k in range(rots):
					mp_r = np.rot90(mp,k=k)
					pp_r = np.rot90(pp,k=k)
					pred_pansh = unet.sess.run(tf.nn.sigmoid(pansh_out) , feed_dict={ms_plchldr:np.transpose(mp_r[np.newaxis,:,:,:],(0,3,1,2)),
															pan_plchldr:pp_r[np.newaxis,np.newaxis,:,:]})
					aug_pansh += np.rot90(pred_pansh[0,:shp[0],:shp[1],:],k=-k)

				output_pansh[(h):(h+patch_shape[0]),(w):(w+patch_shape[1]),:] += aug_pansh/rots
			times[(h):(h+patch_shape[0]),(w):(w+patch_shape[1])] += 1

			# ensure all pixels are covered
			# assert np.sum(times == 0) == 0
	if unet.seg_en and unet.pansh_en:
		return output_seg/times,output_pansh/times[:, :, np.newaxis]
	elif unet.seg_en:
		return output_seg/times,None
	else:
		return None,output_pansh/times[:, :, np.newaxis]

