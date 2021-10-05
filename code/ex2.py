import numpy as np

head = np.array([4.03035065e+02, 4.95879211e+01])
shoulder1 = np.array([3.58295135e+02, 1.43587021e+02])
shoulder2 = np.array([4.11967651e+02, 1.43505188e+02])
hip1 = np.array([3.80655579e+02, 2.59821228e+02])
hip2 = np.array([4.20863770e+02, 2.55354691e+02])
chest = np.array([3.98500092e+02, 1.97244690e+02])

hip = (hip1 + hip2) / 2
shoulder = (shoulder1 + shoulder2) / 2

len_head = ((head[0] - shoulder[0]) ** 2 + (head[1] - shoulder[1]) ** 2) ** 0.5
len_body = ((shoulder[0] - chest[0]) ** 2 + (shoulder[1] - chest[1]) ** 2) ** 0.5 + \
           ((chest[0] - hip[0]) ** 2 + (chest[1] - hip[1]) ** 2) ** 0.5

print(len_head / (len_head + len_body))
