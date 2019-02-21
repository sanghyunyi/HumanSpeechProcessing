import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from nipype.interfaces import fsl


# Handle brain imaging data.
# Preporcess is also done here.

class fMRIimage:
    def __init__(self, img_path):
        self.img = nib.load(img_path)
        self.img_path = img_path
        self.img_data = self.img.get_data()

    def mask_non_brain_region(self):
        btr = fsl.BET()
        btr.inputs.in_file = self.img_path
        filename, file_extension = os.path.splitext(self.img_path)
        out_path =  filename + '_mask_non_brain_region' + file_extension
        btr.inputs.out_file = out_path
        btr.inputs.frac = 0.7
        btr.functional = True
        btr.output_type = 'NIFTI_GZ'
        btr.cmdline
        res = btr.run()
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
                    out.append = get_neighbor_voxels_of_a_point(self x, y, z, neighbor_size)
        return np.array(out)

    def show_slices(self, x, y, z, v):
        fig, axes = plt.subplots(1, 3)
        slices = [self.img_data[x, :, :, v], self.img_data[:, y, :, v], self.img_data[:, :, z, v]]
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()

if __name__ == "__main__":
    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-forrestgump/func'
    img = nib.load(os.path.join(data_path, 'sub-03_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz'))
    #img_data = img.get_data()

    print(img.header)
