import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Handle brain imaging data.
# img_data is an array from get_data method

def get_neighbor_voxels_of_a_point(img_data, x, y, z, neighbor_size):
    max_x = x + neighbor_size + 1
    max_y = y + neighbor_size + 1
    max_z = z + neighbor_size + 1
    min_x = x - neighbor_size
    min_y = y - neighbor_size
    min_z = z - neighbor_size
    out = img_data[min_x:max_x, min_y:max_y, min_z:max_z,:]
    assert out.shape[:3] == (2 * neighbor_size + 1, 2 * neighbor_size + 1, 2 * neighbor_size + 1)
    return out

def get_all_neigbor_voxels_with_size(img_data, neighbor_size):
    shape = img_data.shape
    out = []
    for x in range(0 + neighbor_size, shape[0] - neighbor_size):
        for y in range(0 + neighbor_size, shape[1] - neighbor_size):
            for z in range(0 + neighbor_size, shape[2] - neighbor_size):
                out.append = get_neighbor_voxels_of_a_point(img_data, x, y, z, neighbor_size)
    return np.array(out)

def make_background_zero(img_data):
    return None

def show_slices(img_data, x, y, z, v):
    fig, axes = plt.subplots(1, 3)
    slices = [img_data[x, :, :, v], img_data[:, y, :, v], img_data[:, :, z, v]]
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

if __name__ == "__main__":
    data_path = '/Users/YiSangHyun/ds000113-download/sub-03/ses-forrestgump/func'
    img = nib.load(os.path.join(data_path, 'sub-03_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz'))
    #img_data = img.get_data()

    print(img.header)
