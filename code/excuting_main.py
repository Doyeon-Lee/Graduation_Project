import os
import re

# violence_list = os.listdir('../media/violence')
# violence_list = [re.sub('.mp4', '', i) for i in violence_list]
#
# non_violence_list = os.listdir('../media/non-violence')
# non_violence_list = [re.sub('.mp4', '', i) for i in non_violence_list]

os.system("python main.py s1 v")

# file_list = ["602", "902", "f2", "b29"]
#
# for i in file_list:
#     if i == "b29":
#         os.system(f"python main.py {i} v")
#     else:
#         os.system(f"python main.py {i} n")

# for file_name in non_violence_list:
#     os.system(f"python main.py {file_name} n")
#
# for file_name in violence_list:
#     os.system(f"python main.py {file_name} v")
