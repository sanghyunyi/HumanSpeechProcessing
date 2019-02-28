import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from nipype.interfaces import fsl
from nipype.interfaces.semtools.registration import brainsresample

# Handle brain imaging data.
# Preporcess is also done here.

class fMRIimage:
    def __init__(self, img_path):
        self.img = nib.load(img_path)
        self.img_path = img_path
        self.img_data = self.img.get_data()

    def __str__(self):
        return str(self.img.header) + '\npath = ' + self.img_path

    def mask_non_brain_region(self):
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

    def resample(self, interpolation_mode):
        # It seems like it only works for 3D data.
        # What we need is volume wise resampling
        # Use fmriprep!
        filename, file_extension1 = os.path.splitext(self.img_path)
        filename, file_extension2 = os.path.splitext(filename)
        out_path = filename + '_resample_' + interpolation_mode + file_extension2 + file_extension1
        resampler = brainsresample.BRAINSResample()
        resampler.inputs.inputVolume = self.img_path
        resampler.inputs.outputVolume = out_path
        resampler.inputs.interpolationMode = interpolation_mode
        resampler.inputs.environ = {'PATH': '/Applications/Slicer.app/Contents/lib/Slicer-4.10/cli-modules'}
        resampler.inputs.gridSpacing = 0
        resampler.cmdline
        resampler.run()
        self.img = nib.load(out_path)
        self.img_path = out_path
        self.img_data = self.img.get_data()


    def get_neighbor_voxels_of_a_point(self, x, y, z, neighbor_size):
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
        shape = self.img_data.shape
        out = []
        for x in range(0 + neighbor_size, shape[0] - neighbor_size):
            for y in range(0 + neighbor_size, shape[1] - neighbor_size):
                for z in range(0 + neighbor_size, shape[2] - neighbor_size):
                    out.append = get_neighbor_voxels_of_a_point(self, x, y, z, neighbor_size)
        return np.array(out)

    def show_slices(self, x, y, z, v):
        fig, axes = plt.subplots(1, 3)
        slices = [self.img_data[x, :, :, v], self.img_data[:, y, :, v], self.img_data[:, :, z, v]]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()

if __name__ == "__main__":
    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-movie/func'
    img = fMRIimage(os.path.join(data_path, 'sub-03_ses-movie_task-movie_run-1_bold.nii.gz'))
    #img.show_slices(40,40,20,3)
    img.mask_non_brain_region()
    img.resample("Lanczos")
    img.show_slices(40,40,20,3)

    #img_data = img.get_data()

