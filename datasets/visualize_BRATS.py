import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import matplotlib.pyplot as plt
from PIL import Image

'''flair/t1/t1ce/t2相当于不同的模态，seg是分割'''

np.set_printoptions(threshold=np.inf)
father_dir_path = r'F:\AI_projects\Data\BRATS2020\MICCAI_BraTS2020_TrainingData'
origin_path = os.path.join(father_dir_path, 'BraTS20_Training_001\BraTS20_Training_001_flair.nii.gz')
mask_path = os.path.join(father_dir_path, 'BraTS20_Training_001\BraTS20_Training_001_seg.nii.gz')
img = nib.load(origin_path)
mask = nib.load(mask_path)
print(img.shape == mask.shape)

img_fdata = img.get_fdata()
mask_fdata = mask.get_fdata()
dir_path = r'/datasets/BRATS_image'

i = 120
slice_img = img_fdata[i, :, :]
slice_mask = mask_fdata[i, :, :]
plt.imshow(slice_img)
plt.show()
plt.imshow(slice_mask)
plt.show()
