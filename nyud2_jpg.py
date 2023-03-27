import h5py
import scipy.io

def get_names(dataset):

    label_dict = {}

    for index, ref_name in enumerate(dataset['names'][0]):
        name = dataset[ref_name]
        m_string = ''
        for letter in name:
            m_string = m_string + chr(letter[0])
        label_dict[index] = m_string

    return label_dict

def load_dataset(dataset_path, splits_path):
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

path = './nyu_depth_v2_labeled.mat'

# names = load_dataset_class_names(path)
# label_dict = get_all_labels(names)

# print(label_dict)

a = load_dataset('nyud2.mat', 'splits.mat')
print(a['names'])