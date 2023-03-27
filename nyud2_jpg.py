import h5py

def load_dataset_class_names(path_to_depth):
    
    image_dataset = h5py.File(path_to_depth)
    return image_dataset['names'][()]

def get_all_labels(names):

    label_dict = {}

    for index, ref_name in enumerate(names[0]):
        name = image_dataset[ref_name]
        m_string = ''
        for letter in name:
            m_string = m_string + chr(letter[0])
        label_dict[index] = m_string

    return label_dict

path = './nyu_depth_v2_labeled.mat'

names = load_dataset_class_names(path)
label_dict = get_all_labels(names)

print(label_dict)
