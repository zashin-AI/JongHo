import numpy as np

x1 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f_data.npy')
y1 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f_label.npy')

x2 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_data.npy')
y2 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_label.npy')

x = np.concatenate([x1, x2], 0)
y = np.concatenate([y1, y2], 0)

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

np.save('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy', arr=x)
np.save('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy', arr=y)

x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)


