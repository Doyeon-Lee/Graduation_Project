import numpy as np

head = np.array([1.19900183e+03, 3.00003937e+02])
shoulder1 = np.array([1.15889636e+03, 4.03024048e+02])
shoulder2 = np.array([1.22136572e+03, 4.34262207e+02])
hip1 = np.array([1.08720959e+03, 5.23684998e+02])
hip2 = np.array([1.13186621e+03, 5.59552490e+02])
chest = np.array([1.14540491e+03, 4.83424835e+02])

# hip = (hip1 + hip2) / 2
hip = hip1
shoulder = (shoulder1 + shoulder2) / 2

len_head = ((head[0] - shoulder[0]) ** 2 + (head[1] - shoulder[1]) ** 2) ** 0.5
len_body = ((shoulder[0] - chest[0]) ** 2 + (shoulder[1] - chest[1]) ** 2) ** 0.5 + \
           ((chest[0] - hip[0]) ** 2 + (chest[1] - hip[1]) ** 2) ** 0.5

print(len_head / (len_head + len_body))
