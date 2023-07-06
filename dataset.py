import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

from remove_files import empty_directory

PARTITION = '_______________'

class HSIDataset():
    def __init__(self) -> None:
        pass
    
    def show_spectral_of_image(self, hsi, save_directory=None, save_name=None):
        width = hsi.shape[1]
        height = hsi.shape[2]
        
        pixel_spectral_list = []
        label_list = []
        pixel_spectral_list.append(hsi[:, width//2, height//2])
        label_list.append("central")
        pixel_spectral_list.append(self.__search_max_spectral(hsi))
        label_list.append("max")
        
        self.__show_spectral_of_pixel(pixel_spectral_list, label_list, save_directory, save_name)
        
        return 1
    
    def __search_max_spectral(self, hsi):
        sum_hsi = np.sum(hsi, axis=0)

        # 合計値が最大となる1次元配列のインデックスを取得
        max_index = np.unravel_index(np.argmax(sum_hsi), sum_hsi.shape)

        # 最大値を持つ1次元配列とその要素を表示
        max_spectral = hsi[:, max_index[0], max_index[1]]
        
        return max_spectral
    
    def __show_spectral_of_pixel(self, pixel_spectral_list, label_list, save_directory=None, save_name=None):
        num_of_spectral = len(pixel_spectral_list[0])
        if num_of_spectral == 401:
            x = self.wavelength_information
        else:
            x = range(len(pixel_spectral_list[0]))
        
        fig, ax = plt.subplots()
        for pixel_spectral, label in zip(pixel_spectral_list, label_list):
            ax.plot(x, pixel_spectral, label=label)
        
        ax.legend()
        
        if save_directory is None:
            save_directory = os.path.join('output-python', 'T20230411')
        if save_name is None:
            save_name = 'image' + str(self.ID)
        save_path = os.path.join(save_directory, save_name)
        plt.savefig(save_path)
        plt.close(fig)
    
    def denoise_hsi(self):
        #  ここに処理を書く
        return self.image
    
    def set_image(self, file_name):  # image shape: (channel, width, height)
        if file_name[-3:] != 'raw':
            return 0
        
        file_path = os.path.join('dataset', file_name + '.h5')
        with h5py.File(file_path, 'r') as f:
            self.image = np.array(f['NGMeet'])
            self.noisy_image = np.array(f['SpectralImage'])
            self.original_image = self.image
            self.wavelength_information = np.array(f['Wavelengths'])
        if self.image.shape[0] != 401:
            raise ValueError('wrong image shape')
        
        if self.image is None:
            return 0
        
        return 1
    
    
def load_dataset():
    hsi_dataset = HSIDataset()
    
    image_file_name = "20210824_171211_002_leftarm_raw"
    hsi_dataset.set_image(image_file_name)
    
    print('finished loading dataset')
    print(PARTITION)
    
    return hsi_dataset

def preprocess(hsi_dataset):
    save_directory = os.path.join('output-python', 'T20230411')
    save_name = 'NGMeet'
    hsi_dataset.show_spectral_of_image(hsi_dataset.image, save_directory=save_directory, save_name=save_name)
    
    hsi_dataset.image = hsi_dataset.denoise_hsi()
    save_name = 'your_denoise'
    hsi_dataset.show_spectral_of_image(hsi_dataset.image, save_directory=save_directory, save_name=save_name)
    
    save_name = 'original'
    hsi_dataset.show_spectral_of_image(hsi_dataset.noisy_image, save_directory=save_directory, save_name=save_name)
    
    print(PARTITION)
    
    return 1

if __name__ == '__main__':
    remove_directries = [
        'output-python/T20230411'
        ]
    empty_directory(remove_directries)
    
    '''
    class_correspondence_table = {
        'Malignant': 0,
        'Benign': 1,
        'Non Cancer': 2
        }
    '''
    
    hsi_dataset = load_dataset()
    preprocess(hsi_dataset)