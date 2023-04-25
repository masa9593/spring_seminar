import numpy as np
import os
import cv2
import h5py
import matplotlib.pyplot as plt
import skimage.io, skimage.util
from sklearn.model_selection import train_test_split

from tools import hio, util

conf = util.parse_config()

class HSIDataset():
    def __init__(self) -> None:
        pass
        
    def decompose(self, num_of_components):
        num_decom = self.image.shape[0] - num_of_components
        self.image = self.image[num_decom//2:num_decom//2+num_of_components, :, :]
        self.wavelength_information = self.wavelength_information[num_decom//2:num_decom//2+num_of_components]
        return self.image
    
    def mask(self):
        self.image = self.image * self.mask_image
        self.original_image = self.original_image * self.mask_image
        self.show_hsi(save_name='after_mask.jpg')
        return self.image
    
    def split(self, patch_size, class_correspondence_table):
        image_size = np.array(self.image.shape)
        num_cut = np.floor(image_size[1:3] / patch_size).astype(int)
        
        cropped_image = self.__center_crop_hsi(self.image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        cropped_original_image = self.__center_crop_hsi(self.original_image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        cropped_label_image = self.__center_crop_hsi(self.label_image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        # patchIndex = np.meshgrid(np.arange(0,st[0], dtype=np.int32), np.arange(0,st[1], dtype=np.int32))
        
        self.split_dataset = []
        
        for x in range(num_cut[0]):
            for y in range(num_cut[1]):
                dataset = HSIDataset()
                dataset.image = cropped_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.original_image = cropped_original_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.label_image = cropped_label_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.wavelength_information = self.wavelength_information
                dataset.ID = self.ID
                
                if np.sum(dataset.label_image) == 0:
                    if 'Non Cancer' in class_correspondence_table.keys():
                        dataset.label = class_correspondence_table['Non Cancer']
                    else:
                        continue
                else:
                    dataset.label = self.label
                
                if np.sum(dataset.image) > patch_size:
                    self.split_dataset.append(dataset)
                    
                self.show_hsi(save_name=f'{str(self.ID)}{str(x)}{str(y)}.jpg')
                    
        return self.split_dataset
    
    def show_hsi(self, save_directory=None, save_name='hsi.jpg'):
        rgb_image = util.get_display_image(self.original_image, channel = 401)
        if rgb_image.dtype == 'float':
            rgb_image = (rgb_image * 255).astype('uint8')
        
        if save_directory is None:
            save_directory = os.path.join(conf['Directories']['outputDir'], 'T20230411')
        save_path = os.path.join(save_directory, save_name)
        
        skimage.io.imsave(save_path, rgb_image)
        
        return 1

    
    def set_ID(self, ID, used_IDs):
        if type(ID) is int:
            ID = self.__change_three_digit_number_to_str(ID)
        elif type(ID) is str:
            pass
        else:
            raise ValueError('Unknown type')
        
        if ID in used_IDs:
            self.ID = ID
            return 1
        
        return 0
    
    def set_sample_ID(self, sample_ID):
        if type(sample_ID) is int:
            sample_ID = self.__change_three_digit_number_to_str(sample_ID)
        elif type(sample_ID) is str:
            pass
        else:
            raise ValueError('Unknown type')
        
        self.sample_ID = sample_ID
        return 1
    
    def set_image_and_wavelength_information(self, file_name):  # image shape: (channel, width, height)
        if file_name[-3:] != 'raw':
            return 0
        
        file_path = os.path.join(conf['Directories']['dataDir'], file_name + '.h5')
        f = self.__load_from_h5(file_path)
        self.image = f['SpectralImage']
        self.original_image = self.image
        self.wavelength_information = f['Wavelengths']
        
        if self.image.shape[0] != 401:
            raise ValueError('wrong image shape')
        
        if self.image is None:
            return 0
        
        return 1
    
    def set_label(self, label_data_file, class_correspondence_table):
        for j, label_sample_ID in enumerate(label_data_file['SampleID']):
            label_sample_ID = self.__change_three_digit_number_to_str(label_sample_ID)
            if self.sample_ID == label_sample_ID:
                label = label_data_file['Type'][j]
                if label in class_correspondence_table.keys():
                    self.label = class_correspondence_table[label]
                    return 1
        return 0
    
    def set_label_image(self):  # image shape: (channel(1), width, height)
        file_path = os.path.join(conf['Directories']['labelDir'], self.ID + '.png')
        image = self.__load_color_image_as_gray_image(file_path)
        self.label_image = self.__get_binary_image_from_image(image)
        self.label_image = self.__cut_image_to_same_size(self.label_image, self.image)
        
        if self.label_image is None:
            return 0
        
        return 1
    
    def set_mask_image(self):  # image shape: (channel(1), width, height)
        file_path = os.path.join(conf['Directories']['maskDir'], self.ID + '.png')
        image = self.__load_color_image_as_gray_image(file_path)
        self.mask_image = self.__get_binary_image_from_image(image)
        self.mask_image = self.__cut_image_to_same_size(self.mask_image, self.image)
        
        if self.mask_image is None:
            return 0
        
        return 1
    
    def __load_from_h5(self, file_name):
        val = h5py.File(file_name, 'r')
        return val

    def __load_color_image_as_gray_image(self, file_path): # -> (channel * width * height)
        label_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        rot_img = label_img.transpose(1, 0)
        rot_img = rot_img[np.newaxis,:,:]
        return rot_img

    def __get_binary_image_from_image(self, image): 
        if image.shape[0] == 3:
            gray_img = (image[0:1, :, :] + image[1:2, :, :] + image[2:3, :, :]) / 3
        elif image.shape[0] == 1:
            gray_img = image
        else:
            raise ValueError('Unsupposed mask channel number')
        
        binary_img = np.where(gray_img==0, 0, 1)
        return binary_img
    
    def __cut_image_to_same_size(self, cut_image, image):
        return cut_image[:image.shape[0], :image.shape[1], :image.shape[2]]
    
    def __change_three_digit_number_to_str(self, three_digit_number):
        return str(three_digit_number + 1000)[1:]

    def __center_crop_hsi(self, hsi, target_width, target_height):        
        width = hsi.shape[1]
        height = hsi.shape[2]
        
        if target_width is None:
            target_width = min(width, height)

        if target_height is None:
            target_height = min(width, height)

        left = int(np.ceil((width - target_width) / 2))
        right = width - int(np.floor((width - target_width) / 2))

        top = int(np.ceil((height - target_height) / 2))
        bottom = height - int(np.floor((height - target_height) / 2))

        if np.ndim(hsi) > 2:
            croppedImg = hsi[:, left:right, top:bottom]
        else:
            croppedImg = hsi[left:right, top:bottom]
            
        return croppedImg    


class HSIDatasetList():
    def __init__(self, class_correspondence_table, used_IDs) -> None:
        self.class_correspondence_table = class_correspondence_table
        self.used_IDs = used_IDs
        
    def load_dataset(self, data_file, label_data_file):
        self.hsi_dataset_list = []
        for i, ID in enumerate(data_file['ID']):
            hsi_dataset = HSIDataset()
            
            flag = hsi_dataset.set_ID(ID, self.used_IDs)
            if flag != 1:
                continue
            
            sample_ID = data_file['SampleID'][i][:3]
            flag = hsi_dataset.set_sample_ID(sample_ID)
            if flag != 1:
                continue
                    
            image_file_name = data_file['Filename'][i]
            flag = hsi_dataset.set_image_and_wavelength_information(image_file_name)
            if flag != 1:
                continue
            
            flag = hsi_dataset.set_label(label_data_file, self.class_correspondence_table)
            if flag != 1:
                continue
            
            flag = hsi_dataset.set_label_image()
            if flag != 1:
                continue
            
            flag = hsi_dataset.set_mask_image()
            if flag != 1:
                continue
            
            self.hsi_dataset_list.append(hsi_dataset)
        
        return 1
    
    def preprocess(self, num_of_components, patch_size, test_size, random_state):
        train_IDs, test_IDs = train_test_split(self.used_IDs, test_size=test_size, random_state=random_state)
        self.train_preprocessed_hsi_dataset_list = []
        self.test_preprocessed_hsi_dataset_list = []
        self.preprocessed_hsi_dataset_list = []
        for hsi_dataset in self.hsi_dataset_list:
            hsi_dataset.decompose(num_of_components)
            hsi_dataset.mask()
            hsi_dataset.split(patch_size, self.class_correspondence_table)
            
            self.preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
            if hsi_dataset.ID in train_IDs:
                self.train_preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
            elif hsi_dataset.ID in test_IDs:
                self.test_preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
            else:
                print(f'No ID: {hsi_dataset.ID}')
                continue
        
        return 1
    
    def show_hsi_montage(self, phase=None, save_directory=None, save_name='montage.jpg'):
        if phase == 'train':
            hsi_list = self.train_preprocessed_hsi_dataset_list
        elif phase == 'test':
            hsi_list = self.test_preprocessed_hsi_dataset_list
        else:
            hsi_list = self.preprocessed_hsi_dataset_list
        
        rgb_image_list = []
        
        for hsi_dataset in hsi_list:
            rgb_image = util.get_display_image(hsi_dataset.original_image, channel=401)
            if rgb_image.shape[0] != 64 or rgb_image.shape[1] != 64 or rgb_image.shape[2] != 3:
                print(hsi_dataset.ID)
                print(hsi_dataset.ID)
                print(hsi_dataset.ID)
                print(hsi_dataset.ID)
                print(hsi_dataset.ID)
                print(rgb_image.shape)
            if rgb_image.dtype == 'float':
                rgb_image = (rgb_image * 255).astype('uint8')
            rgb_image_list.append(rgb_image)
        rgb_image_list = np.array(rgb_image_list, dtype='uint8')
        
        montage = skimage.util.montage(rgb_image_list, multichannel=True)
        
        if save_directory is None:
            save_directory = os.path.join(conf['Directories']['outputDir'], 'T20230411')
        
        save_path = os.path.join(save_directory, save_name)
        
        skimage.io.imsave(save_path, montage)
