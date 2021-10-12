import os
import re

violence_list = os.listdir('../media/violence')
violence_list = [re.sub('.mp4', '', i) for i in violence_list]

non_violence_list = os.listdir('../media/non-violence')
non_violence_list = [re.sub('.mp4', '', i) for i in non_violence_list]

# os.system("python main.py 303 n")

# file_list = ["304", "b24"]
#
# for i in file_list:
#     if i == "304":
#         os.system(f"python main.py {i} n")
#     else:
#         os.system(f"python main.py {i} v")

# for file_name in non_violence_list:
#     os.system(f"python main.py {file_name} n")
#
# for file_name in violence_list:
#     os.system(f"python main.py {file_name} v")
