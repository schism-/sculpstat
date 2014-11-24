__author__ = 'christian'

import pickle
import os

model_name = "elder"
diff_path = "/Volumes/PART FAT/diff_new/" + model_name + "/step_1/"
diff_no = 0

if os.path.exists(diff_path + "diff_head"):
    print("SERIALIZED DIFF")
    fh = open(diff_path + "diff_head", 'rb')
    data = pickle.load(fh)
    print(data)
else:
    print("NORMAL DIFF")
    fh = open(diff_path + "diff_" + str(diff_no), 'rb')
    data = pickle.load(fh)



