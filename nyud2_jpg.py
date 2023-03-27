import h5py

# data path
path_to_depth = './nyu_depth_v2_labeled.mat'

# read mat file
image_dataset = h5py.File(path_to_depth)
temp1 = image_dataset['names'][()]
for i in temp1[0]:
  m_reference = i
  m_object = image_dataset[m_reference]
  m_string = ''
  for i in m_object:
    m_string = m_string + chr(i[0])
  print(m_string)