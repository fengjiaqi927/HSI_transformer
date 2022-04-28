from scipy import io
import numpy as np
import random
from tqdm import tqdm

def sample_data_to_form_patch(gt,img,sr,patch_size=7):
    label = gt - 1
    x_pos, y_pos = np.nonzero(gt)
    indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
    labels = [label[x, y] for x, y in indices]
    data_of_all = []
    

    for index,(center_x,center_y) in enumerate(tqdm(indices)):
        # print(index)
        ### get patch from SLIC segment_results
        # step_1 find the superpixel that the pixel belongs
        temp = sr.copy()
        superpixel_id = temp[center_x,center_y]
        temp[center_x,center_y] = -1

        # step_2.2 if the num of superpixel is less then patch_size, reget from superpixel
        super_x, super_y=np.where(temp==superpixel_id)
        num_of_superpixel = len(super_x)
        if num_of_superpixel == 0:
            data = np.reshape([img[center_x,center_y,:] for i in range(patch_size*patch_size)],[patch_size,patch_size,-1])
        else:
            num_of_samples = patch_size*patch_size-1
            times = num_of_samples//num_of_superpixel
            remainder = num_of_samples%num_of_superpixel
            superpixel_group = [(x, y) for x, y in zip(super_x, super_y)]
            if times == 0:
                sample_group = random.sample(superpixel_group, remainder)
            else:
                sample_group = [(x, y) for i in range(times) for x, y in zip(super_x, super_y)]
                sample_group.extend(random.sample(superpixel_group, remainder))
            sample_group.insert(num_of_samples//2,(center_x, center_y))
            # print(sample_group)
            # print(num_of_superpixel)
            # if num_of_superpixel==47:
            #     print(1)
            data = np.reshape([img[x,y,:] for x,y in sample_group],[patch_size,patch_size,-1])
        data_of_all.append(data)


    return data_of_all, indices

if __name__=="__main__":
    datasets = {
        'PaviaC': {
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'img': 'IndianPines.mat',
            'gt': 'IndianPines_GT.mat',
            'sr':'segmentation_results_200'
            # 'sr':'IP_segment_results_100'
            # 'sr':'segmentation_results'
            }}
    dataset_name = 'IndianPines'
    dataset = datasets[dataset_name]
    folder = './Datasets/' +  dataset_name + '/'
    img = io.loadmat(folder+dataset['img'])['imageCube']# Load image to numpy.ndarray
    # ['indian_pines_corrected']
    img = (img - np.min(img))/(np.max(img)-np.min(img))     # Normalization
    print(type(img),img.shape)
    gt = io.loadmat(folder + dataset['gt'])['groundTruth']
    sr = io.loadmat(folder + dataset['sr'])['segmentation_results']
    data_of_all, indices = sample_data_to_form_patch(gt,img,sr,patch_size=15)
    np.save('SLIC_samples.npy',np.array(data_of_all))
    np.save('indices.npy',np.array(indices))
    print("done!")
