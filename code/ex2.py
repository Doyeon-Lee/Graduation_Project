import numpy as np

head = np.array([4.0853763e+02, 2.1779082e+02])
shoulder1 = np.array([5.1485962e+02, 2.4882353e+02])
shoulder2 = np.array([5.0144388e+02, 2.7095911e+02])
hip1 = np.array([6.5656818e+02, 3.3316537e+02])
hip2 = np.array([6.3884631e+02, 3.4626001e+02])
chest = np.array([5.8111829e+02, 2.9763773e+02])

hip = (hip1 + hip2) / 2
shoulder = (shoulder1 + shoulder2) / 2

len_head = ((head[0] - shoulder[0]) ** 2 + (head[1] - shoulder[1]) ** 2) ** 0.5
len_body = ((shoulder[0] - chest[0]) ** 2 + (shoulder[1] - chest[1]) ** 2) ** 0.5 + \
           ((chest[0] - hip[0]) ** 2 + (chest[1] - hip[1]) ** 2) ** 0.5

print(len_head / (len_head + len_body))
