import torch
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import skimage.io

from tools import util
from learning.LearnModel import Learn_Model, valid
from models.ChoiceModel import Choice_Model
from dataset import HSIDataset, HSIDatasetList

conf = util.parse_config()

class MedHSIDataset(Dataset):
    def __init__(self, dataset_list) -> None:
        super().__init__()
        self.dataset_list = dataset_list
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, index):
        dataset = {
            'image': self.dataset_list[index].image[np.newaxis, :, :, :],
            'label': self.dataset_list[index].label,
            'original_image': self.dataset_list[index].original_image
        }
        return dataset

def class_count(dataset_list, target_class):
    count = np.zeros(len(target_class))
    for dataset in dataset_list:
        if dataset.label in target_class:
            count[dataset.label] += 1
    return count

def show_hsi_montage(hsi_list, channel=401, save_directory=None, save_name='montage.jpg'):
    rgb_image_list = []
    
    for hsi in hsi_list:
        rgb_image = util.get_display_image(hsi, channel=channel)
        if rgb_image.dtype == 'float':
            rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image_list.append(rgb_image)
    rgb_image_list = np.array(rgb_image_list, dtype='uint8')
    
    montage = skimage.util.montage(rgb_image_list, multichannel=True)
    
    if save_directory is None:
        save_directory = os.path.join(conf['Directories']['outputDir'], 'T20230411')
    
    save_path = os.path.join(save_directory, save_name)
    
    skimage.io.imsave(save_path, montage)

if __name__ == '__main__':
    num_of_components = 64 * 6
    patch_size = 64
    test_size = 0.1
    random_state = 1
    
    class_correspondence_table = {
        'Malignant': 0,
        'Benign': 1,
        'Non Cancer': 2
        }
    '''
    class_correspondence_table = {
        'Malignant': 0,
        'Benign': 1
        }
    '''
    
    target_class = list(class_correspondence_table.values())

    batch_size = 4
    model_name = 'vggLike_3DCNN'
    # model_name = 'ResNetLike_3DCNN'
    # model_name = 'my_3DCNN'
    # model_name = 'sample_3DCNN'
    learningRate = 0.001
    num_of_epoch = 300

    num_of_attempts = 1
    history = []
    
    data_info_path = os.path.join(conf['Directories']['importDir'], conf['File Names']['dataInfoTableName'])
    data_file = pd.read_excel(data_info_path)
    
    label_data_info_path = os.path.join(conf['Directories']['importDir'], conf['File Names']['diagnosisInfoTableName'])
    label_data_file = pd.read_excel(label_data_info_path)
    
    '''
    used_IDs = [
        '150', '157', '160', '163', '175', '181', '187', '193', '196', '199', '205', '212', '215', '218', '227', '230', 
        '233', '236', '251', '260', '263', '266', '284', '290', '296', '308', '321', '324', '333', '342', '348', '352', '361'
        ]
    '''
    
    used_IDs = ['150', '157']
    
        
    hsi_dataset = HSIDatasetList(class_correspondence_table, used_IDs)
    hsi_dataset.load_dataset(data_file, label_data_file)    
    hsi_dataset.preprocess(num_of_components, patch_size, test_size, random_state)
    train_dataset_list = hsi_dataset.train_preprocessed_hsi_dataset_list
    test_dataset_list = hsi_dataset.test_preprocessed_hsi_dataset_list
    print(f'num of train: {len(train_dataset_list)}, num of test: {len(test_dataset_list)}')
    print(f'num of train per class: {class_count(train_dataset_list, target_class)}')
    print(f'num of test per class: {class_count(test_dataset_list, target_class)}')
    
    hsi_dataset.show_hsi_montage(phase='train', save_name='train_montage.jpg')
    hsi_dataset.show_hsi_montage(phase='test', save_name='test_montage.jpg')
    hsi_dataset.show_hsi_montage(save_name='all_montage.jpg')
    
    train_dataset = MedHSIDataset(train_dataset_list)
    test_dataset = MedHSIDataset(test_dataset_list)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    
    for i in range(num_of_attempts):
        model = Choice_Model(model_name, train_dataset[0]['image'], target_class).to(torch.float).to(device)
        if device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
            torch.backends.cudnn.benchmark = True
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate, momentum=0.9)
        
        model, result = Learn_Model(train_dataloader, test_dataloader, model, num_of_epoch, criterion, optimizer, device)
        
        history.append([result['train']['loss'][-1], result['train']['acc'][-1], result['test']['loss'][-1], result['test']['acc'][-1]])
    
    for i in range(num_of_attempts):
        print(history[i])
    
    val_loss, val_acc, Correct_Data, False_Data = valid(model, test_dataloader, criterion, device, is_test=True)

    print(len(Correct_Data))
    if len(Correct_Data) != 0:
        print(Correct_Data[0].shape)
        
    print(len(False_Data))
    if len(False_Data) != 0:
        print(False_Data[0].shape)

    if len(Correct_Data) != 0:
        show_hsi_montage(Correct_Data, channel=401, save_name='Correct_Data_montage.jpg')

    if len(False_Data) != 0:
        show_hsi_montage(False_Data, channel=401, save_name='False_Data_montage.jpg')

    x = np.arange(num_of_epoch)

    loss_max = max(max(result['train']['loss'], result['test']['loss'], key=max))

    plt.subplot(1, 2, 1, title='Loss')
    plt.plot(x, result['train']['loss'], color='red', label='train')
    plt.plot(x, result['test']['loss'], color='blue', label='test')
    plt.ylim(0, loss_max+0.3)
    plt.legend()

    plt.subplot(1, 2, 2, title='Acc')
    plt.plot(x, result['train']['acc'], color='red', label='train')
    plt.plot(x, result['test']['acc'], color='blue', label='test')
    plt.ylim(0, 1)

    plt.legend()

    filename = os.path.join(conf['Directories']['outputDir'], 'T20230411', 'learning_curve.png')
    plt.savefig(filename)

    plt.show()

    from torchinfo import summary
    
    num_of_components, width, height = train_dataset_list[0].image.shape
    summary(model=model, input_size=(batch_size, 1, num_of_components, width, height))

