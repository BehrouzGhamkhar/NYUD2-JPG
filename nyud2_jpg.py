import h5py
import scipy.io
import os
import numpy
import matplotlib.pyplot as plt

required_classes = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser', 'garbage bin', \
                    'lamp', 'monitor', 'night stand', 'pillow', 'sink', 'sofa', 'table', 'television', 'toilet']

def get_names(dataset):

    label_dict = {}

    for index, ref_name in enumerate(dataset['names'][0]):
        name = dataset[ref_name]
        m_string = ''
        for letter in name:
            m_string = m_string + chr(letter[0])
        label_dict[index] = m_string

    return label_dict

def load_dataset_file(dataset_path, splits_path):
    dataset = h5py.File(dataset_path, 'r')
    splits = scipy.io.loadmat(splits_path)
    output_dict = dict()
    output_dict['images'] = dataset['images']
    output_dict['depths'] = dataset['depths']
    output_dict['labels'] = dataset['labels']
    output_dict['names'] = get_names(dataset)
    output_dict['train_split'] = splits['trainNdxs']
    output_dict['test_split'] = splits['testNdxs']

    return output_dict

def make_required_class_directories():
    for required_class in required_classes:
        required_class_path = os.path.join('Dataset', required_class)
        if not os.path.exists(required_class_path):
            os.makedirs(required_class_path)

def extract_data_from_split(dataset, split):
    split_size = dataset[split].shape[0]
    images = numpy.zeros((split_size, 3, 640, 480), dtype=numpy.uint8)
    depths = numpy.zeros((split_size, 3, 640, 480), dtype=numpy.uint8)
    labels = numpy.zeros((split_size, 640, 480), dtype=numpy.uint16)

    for index in range(split_size):
        access_index = dataset[split][index][0] - 1
        image = dataset['images'][access_index]
        images[index] = image

        depth = dataset['depths'][access_index]
        depth = (depth - numpy.min(depth)) * 255 / (numpy.max(depth) - numpy.min(depth)) # convert to [0, 255]
        depths[index][0] = depth
        depths[index][1] = depth
        depths[index][2] = depth
        
        label = dataset['labels'][access_index]
        labels[index] = label

        print(f'Extracted {index}/{split_size} from {split}')

    images = numpy.moveaxis(images, 1, -1)
    images = numpy.moveaxis(images, 1, 2)
    depths = numpy.moveaxis(depths, 1, -1)
    depths = numpy.moveaxis(depths, 1, 2)
    labels = numpy.moveaxis(labels, 1, 2)

    return images, depths, labels

def extract_dataset(split):
    make_required_class_directories()

dataset = load_dataset_file('nyud2.mat', 'splits.mat')
images, depths, labels = extract_data_from_split(dataset, 'test_split')
print(images.shape, depths.shape, labels.shape)

while True:
    index = int(input("Enter index: "))
    print(numpy.unique(depths[index]))
    plt.imshow(images[index])
    plt.show()
    plt.imshow(depths[index])
    plt.show()
    plt.imshow(labels[index])
    plt.show()