import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from nipype.interfaces import fsl
from nipype.interfaces.semtools.registration import brainsresample

# Handle brain imaging data.
# Preporcessng is done by fMRIPrep.

class fMRIimage:
    def __init__(self, img_path):
        '''
        Load nifti file and return a fMRIimage object.
        Input
        - img_path: the path to the nifti image
        Output
        - self.img = nibabel loaded image of the nifti file
        - self.img_path = the path to the nifti file
        - self.img_data = the array of the nifti image data
        '''
        self.img = nib.load(img_path)
        self.img_path = img_path
        self.img_data = self.img.get_data()

    def __str__(self):
        '''
        To print the fMRIimage object
        '''
        return str(self.img.header) + '\npath = ' + self.img_path

    def mask_non_brain_region(self):
        # USE fmriprep!
        '''
        Remove non barin regions using FSL
        '''
        filename, file_extension1 = os.path.splitext(self.img_path)
        filename, file_extension2 = os.path.splitext(filename)
        out_path = filename + '_mask_non_brain_region' + file_extension2 + file_extension1
        if not os.path.isfile(out_path):
            btr = fsl.BET()
            btr.inputs.in_file = self.img_path
            btr.inputs.out_file = out_path
            btr.inputs.frac = 0.25
            btr.inputs.functional = True
            btr.inputs.output_type = 'NIFTI_GZ'
            btr.cmdline
            btr.run()
        self.img = nib.load(out_path)
        self.img_path = out_path
        self.img_data = self.img.get_data()

    def get_neighbor_voxels_of_a_point(self, x, y, z, neighbor_size):
        '''
        Get the adjacent voxels around the input point.
        Input
        - x: x coordinate
        - y: y coordinate
        - z: z coordinate
        - neighbor_size: the maximum distance between the point and the voxels that we want to get.
        Output
        - out: the adjacent voxels in the shape of 3d cube with the edge length of 2 x neighber_size + 1.
        '''
        max_x = x + neighbor_size + 1
        max_y = y + neighbor_size + 1
        max_z = z + neighbor_size + 1
        min_x = x - neighbor_size
        min_y = y - neighbor_size
        min_z = z - neighbor_size
        out = self.img_data[min_x:max_x, min_y:max_y, min_z:max_z,:]
        assert out.shape[:3] == (2 * neighbor_size + 1, 2 * neighbor_size + 1, 2 * neighbor_size + 1)
        return out

    def get_all_neigbor_voxels_with_size(self, neighbor_size):
        '''
        Run get_neigbor_voxels_of_a_point on a whole brain.
        Will be used to do the search light method.
        Input
        - neighbor_size: the maximum distance between the cosidered points and the voxels that we want to get.
        Output
        - out: the list of the adjacent voxels. Each element is the output from get_neigbor_voxels_of_a_point
        '''
        shape = self.img_data.shape
        out = []
        for x in range(0 + neighbor_size, shape[0] - neighbor_size):
            for y in range(0 + neighbor_size, shape[1] - neighbor_size):
                for z in range(0 + neighbor_size, shape[2] - neighbor_size):
                    out.append = get_neighbor_voxels_of_a_point(self, x, y, z, neighbor_size)
        return out

    def show_slices(self, x, y, z, v):
        '''
        Show the slices of the image.
        Input
        - x: x coordinate
        - y: y coordinate
        - z: z coordinate
        - v: the index of the volume
        '''
        fig, axes = plt.subplots(1, 3)
        slices = [self.img_data[x, :, :, v], self.img_data[:, y, :, v], self.img_data[:, :, z, v]]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()

def concat_sessions(img_list):
    '''
    Concatenate the fMRIimage from multiple sessions to a single array
    Input
    - img_list: list of the fMRIimage objects. The order should be aligned with the sessions
    Output
    - img_data: an array which is the concatenation of the fMRIimage from multiple sessions.
    '''
    out_list = []
    for img in img_list:
        array = img.img_data
        out_list.append(array[:, :, :, 3:-5]) #Refer to Hanke et al., 2014. I removed the overlapping volumes.
    img_data = np.concatenate(out_list, axis=-1)
    return img_data

def full_preproc(data_path, n_of_sessions):
    '''
    Do all the preprocessing in brain.py module
    Input
    - data_path: the path to directory where the nifti files of a subject are stored.
    - n_of_sessions: the number of sessions
    Output
    - all_imgs: a single array that contains all the volumes
    '''
    img_list = []
    for i in range(1, n_of_sessions+1):
        img = fMRIimage(os.path.join(data_path, 'sub-03_ses-movie_task-movie_run-{}_bold.nii.gz'.format(i)))
        # Change the file name as you want.
        img.mask_non_brain_region()
        img_list.append(img)
    all_imgs = concat_sessions(img_list)
    return all_imgs

if __name__ == "__main__":
    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-movie/func'
    img_list = []
    for i in range(1,9):
        img = fMRIimage(os.path.join(data_path, 'sub-03_ses-movie_task-movie_run-{}_bold.nii.gz'.format(i)))
        img.mask_non_brain_region()
        print(str(i) + ' masking done')
        img_list.append(img)
    #img.show_slices(40,40,20,3)

    #img.show_slices(40,40,20,3)

    #img_data = img.get_data()
    all_imgs = concat_sessions(img_list)
    with open('../data/brain.pkl', 'wb') as f:
        pkl.dump(all_imgs, f)

