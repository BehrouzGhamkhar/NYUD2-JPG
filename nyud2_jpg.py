import h5py
import scipy.io
import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image

required_classes = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser', 'garbage bin', \
                    'lamp', 'monitor', 'night stand', 'pillow', 'sink', 'sofa', 'table', 'television', 'toilet']

def get_names(dataset):
    names = list()
    for ref_name in dataset['names'][0]:
        name = dataset[ref_name]
        m_string = ''
        for letter in name:
            m_string = m_string + chr(letter[0])
        names.append(m_string)
    return names

def load_dataset_file(dataset_path, splits_path):
    dataset = h5py.File(dataset_path, 'r')
    splits = scipy.io.loadmat(splits_path)
    output_dict = dict()
    output_dict['images'] = dataset['images']
    output_dict['depths'] = dataset['depths']
    output_dict['labels'] = dataset['labels']
    output_dict['names'] = get_names(dataset)
    output_dict['training_split'] = splits['trainNdxs']
    output_dict['testing_split'] = splits['testNdxs']

    return output_dict

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

        print(f'Extracted {index + 1}/{split_size} from {split}.')

    images = numpy.moveaxis(images, 1, -1)
    images = numpy.moveaxis(images, 1, 2)
    depths = numpy.moveaxis(depths, 1, -1)
    depths = numpy.moveaxis(depths, 1, 2)
    labels = numpy.moveaxis(labels, 1, 2)

    return images, depths, labels

def crop_data(images, depths, labels, names, split):
    counter = 0
    for index in range(images.shape[0]):
        image = images[index]
        depth = depths[index]
        label = labels[index]

        for obj in numpy.unique(label):
            if obj > 0 and names[obj - 1] in required_classes:
                label_obj = numpy.argwhere(label == obj)
                minx, miny = numpy.min(label_obj, axis=0)
                maxx, maxy = numpy.max(label_obj, axis=0)

                image_obj = image[minx:maxx + 1, miny:maxy + 1, :]
                depth_obj = depth[minx:maxx + 1, miny:maxy + 1, :]

                image_obj_pil = Image.fromarray(image_obj)
                depth_obj_pil = Image.fromarray(depth_obj)

                image_obj_pil.save(os.path.join('Dataset', split[:-6], 'RGB', names[obj - 1], str(counter) + '.jpg'))
                depth_obj_pil.save(os.path.join('Dataset', split[:-6], 'depth', names[obj - 1], str(counter) + '.jpg'))
                
                print(f'Saved {counter}.jpg.')
                counter += 1

        print(f'Processed {index + 1}/{images.shape[0]} from {split}.')


def make_required_class_directories(split, modality):
    for required_class in required_classes:
        required_class_path = os.path.join('Dataset', split[:-6], modality, required_class)
        if not os.path.exists(required_class_path):
            os.makedirs(required_class_path)

def extract_dataset(split):
    make_required_class_directories(split, 'RGB')
    make_required_class_directories(split, 'depth')
    dataset = load_dataset_file('nyud2.mat', 'splits.mat')
    images, depths, labels = extract_data_from_split(dataset, split)
    crop_data(images, depths, labels, dataset['names'], split)

extract_dataset('training_split')
extract_dataset('testing_split')