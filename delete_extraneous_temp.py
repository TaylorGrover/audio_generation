"""
Remove duplicates in the temp folder
"""
import numpy as np
import os
import soundfile as sf

TMP_DIR = "audio/tmp/"

def get_tmp_files():
    files = os.listdir(TMP_DIR)
    return list(map(lambda f: TMP_DIR + f, files))

def get_duplicates(files:list[str]):
    array_dict = {} # [np.ndarray, sample_rate]
    duplicates_list = []
    key_recency = {}
    for i, file in enumerate(files):
        try:
            array, sr = sf.read(file)
            found = False
            for key in array_dict:
                if np.array_equal(array, array_dict[key][0]) and sr == array_dict[key][1]:
                    duplicates_list.append(file)
                    found = True
                    break
                else:
                    key_recency.setdefault(key, 0)
                    key_recency[key] += 1
            for key in key_recency:
                if key_recency[key] >= 10:
                    del array_dict[key]
                    key_recency[key] = 0
            if not found:
                array_dict[i] = [array, sr]
        except sf.LibsndfileError:
            duplicates_list.append(file)
    return duplicates_list


def delete_duplicates(duplicates):
    for item in duplicates:
        print("removed: {}".format(item))
        os.remove(item)


if __name__ == "__main__":
    files = get_tmp_files() 
    print(files[:10])

    duplicates = get_duplicates(files)

    delete_duplicates(duplicates)
