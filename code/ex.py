import os

FILENAME = "../output/json/violent/cam2/"
OUTPUTNAME = "../output/json/violent/cam2/output"

for i in range(1, 116):
    src = FILENAME + str(i) + ".json"
    dst = OUTPUTNAME + str(i) + ".json"
    os.rename(src, dst)