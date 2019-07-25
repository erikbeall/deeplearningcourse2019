
# ~/RawData/fMRI/afni/to3d -prefix fMRI_S10vol.nii  -time:zt 31 160 2800 alt+z short/.MR..7.?.2008* short/.MR..7.??.2008* short/.MR..7.???.2008*
# ~/RawData/fMRI/afni/to3d -prefix fMRI_S20vol.nii  -time:zt 31 160 2800 alt+z short/.MR..9.?.2008* short/.MR..9.??.2008* short/.MR..9.???.2008*
# ~/RawData/fMRI/afni/to3d -prefix fMRI_S30vol.nii  -time:zt 31 160 2800 alt+z short/.MR..11.?.2008* short/.MR..11.??.2008* short/.MR..11.???.2008*
# ~/RawData/fMRI/afni/to3d -prefix brain_mprage.nii short/.MR..4.?.2008* short/.MR..4.??.2008* short/.MR..4.???.2008*
# ~/RawData/fMRI/afni/3dmerge -1blur_fwhm 4.0 -doall -prefix fMRI_S30vol.blur.nii fMRI_S30vol.nii
import numpy as np
import nibabel as nib
ima=nib.load('fMRI_S30vol.blur.nii')
img=ima.get_fdata()

motor_slice=img[:,:,24,:]
np.save('motor_slice.npy',motor_slice)
brain=nib.load('brain_mprage.nii')
mprage=brain.get_data()
mprage_slice=mprage[:,:,90]
np.save('brain_slice.npy', mprage_slice)
'''
'''
motor_slice=np.load('motor_slice.npy')
mprage_slice=np.load('brain_slice.npy')
import matplotlib.pylab as plt
plt.ion()

# switch to voxel indices in top left of main controller
tseries=motor_slice[78,64,:]
tseries=(tseries-np.min(tseries))/np.ptp(tseries)

# regressors
regressor_meanval = np.ones((160,))
regressor_trend = (np.arange(160)/159.5)-0.5
regressor_func = np.concatenate([np.zeros((24,)), np.ones((16,)), np.zeros((16,)), np.ones((16,)), np.zeros((16,)), np.ones((16,)),np.zeros((16,)), np.ones((16,)), np.zeros((24,))])-0.5
regressors=np.asarray([ regressor_meanval, regressor_trend, regressor_func]).transpose()

# remove first four for saturation
tseries=tseries[4:]
# actually shifting the regressors for hemodynamic delay so remove first two and last two
regressors=regressors[2:-2,:]

# least-squares fitting
fits,res,rnk,s = np.linalg.lstsq(regressors, tseries)
fitted_regs = np.sum(regressors*fits.reshape((1,3)), axis=1)
pooled_var = np.std(tseries-fitted_regs)
tstat = fits[-1]/pooled_var

# do for entire slice and create t-stat map
slice_data = motor_slice[:,:,4:].reshape((128*128,156)).transpose()
fits = np.linalg.lstsq(regressors, slice_data)
fitted_regs = np.asarray([np.sum(regressors*fit.reshape((1,3)), axis=1) for fit in fits[0].transpose()])

pooled_vars = np.std(slice_data-fitted_regs.transpose(), axis=0)
amplitudes=fits[0][2]

tstats=amplitudes/pooled_vars
tstats[np.isnan(tstats)]=0.0
tstats = tstats.reshape((128,128))

# set thresholds and combine images
mask=mprage_slice
# upsample tstat by simple nn-up
tstats_matched = tstats.repeat(2, axis=0).repeat(2, axis=1)
#mask[abs(tstats_matched)>2.0] = tstats_matched[abs(tstats_matched)>2.0]
mask[abs(tstats_matched)>2.0] = 200

mprage_slice=np.expand_dims(mprage_slice,-1)
mask=np.expand_dims(mask,-1)
brain=np.concatenate([mask, mprage_slice,mprage_slice], axis=2)

plt.imshow(brain)
plt.show()
plt.pause(2)

