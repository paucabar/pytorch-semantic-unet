import os
from glob import glob
from shutil import copyfile
import random

def shuffle_tuples_in_list(list1, list2):
    assert len(list1) == len(list2)
    list_of_tuples = list(zip(list1, list2))
    random.shuffle(list_of_tuples)
    list1, list2 = zip(*list_of_tuples)
    return list1, list2

def create_train_val_test_split(in_folder, out_folder):
    print(in_folder)
    image_paths = glob(os.path.join(in_folder, "images", "*.tif"))
    image_paths.sort()
    mask_paths = glob(os.path.join(in_folder, "masks", "*.tif"))
    mask_paths.sort()
    assert len(image_paths) == len(mask_paths)

    # create out folder
    os.makedirs(out_folder, exist_ok=True)

    image_paths, mask_paths = shuffle_tuples_in_list(image_paths, mask_paths)
    print(list(zip(image_paths, mask_paths)))

    # % to split dataset
    perc_train = 0.8
    perc_val = 0.1
    n_train = round(len(image_paths) * perc_train)
    n_val = round(len(image_paths) * perc_val)

    for ii, (image, mask) in enumerate(zip(image_paths, mask_paths)):
        if ii < n_train:
            split = "train"
        elif ii < (n_train + n_val):
            split = "val"
        else:
            split = "test"      
        image_out = os.path.join(out_folder, split + "_images")
        mask_out = os.path.join(out_folder, split + "_masks")
        os.makedirs(image_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        copyfile(image, os.path.join(image_out, os.path.split(image)[1]))
        copyfile(mask, os.path.join(mask_out, os.path.split(mask)[1]))

def main():
    create_train_val_test_split("data_pre", "data")


if __name__ == "__main__":
    main()