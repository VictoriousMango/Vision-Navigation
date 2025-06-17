import os
import numpy as np
def remove_calib():
    """
    Removes the calibration files from the KITTI dataset sequences.
    This is useful for cleaning up the dataset directory.
    """
    os.chdir("D:/coding/Temp_Download/data_odometry_color/dataset/sequences") # Change to directory with sequences
    for sequence in os.listdir():
        if os.path.isdir(sequence):
            for i in [3, 4]:
                calib_file = os.path.join(sequence, f"calib{i}.txt")
                if os.path.exists(calib_file):
                    os.remove(calib_file)
                    print(f"Removed {calib_file}")

os.chdir("D:/coding/Temp_Download/data_odometry_color/dataset/sequences") # Change to directory with sequences
for sequence in os.listdir():
    print(f"{sequence}" + "#"*10)
    projections= dict()
    with open(os.path.join(sequence, "calib.txt"), 'r') as file:
        params = file.read().strip().split("\n")
        for projection in params:
            projections[projection.split(":")[0]] = np.array([x.strip().split(" ") for x in projection.split(":")[1:]])
    file.close()
    for index in [2, 3]:
        with open(os.path.join(sequence, f"calib{index}.txt"), 'w') as file:
            file.write(" ".join(projections[f"P{index}"].flatten()))
        file.close()
        
    
    