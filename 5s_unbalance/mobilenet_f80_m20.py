import numpy as np

x1 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f_data.npy')
y1 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f_label.npy')

# print(x1.shape, y1.shape) # (2331, 128, 862) (2331,)

x1 = x1[:1864]
y1 = y1[:1864]

print(x1.shape, y1.shape) # (1864, 128, 862) (1864,)

x2 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_data.npy')
y2 = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_m_label.npy')

print(x2.shape, y2.shape) # (2205, 128, 862) (2205,)

x2 = x2[:441]
y2 = y2[:441]

print(x2.shape, y2.shape) # (441, 128, 862) (441,)

x = np.concatenate([x1, x2], 0)
y = np.concatenate([y1, y2], 0)

np.save('C:/nmb/nmb_data/npy/project_total_npy/total_f80_m20_data.npy', arr=x)
np.save('C:/nmb/nmb_data/npy/project_total_npy/total_f80_m20_label.npy', arr=y)

x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f80_m20_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_f80_m20_label.npy')

print(x.shape, y.shape) # (2305, 128, 862) (2305,) 

