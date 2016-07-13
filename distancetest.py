import numpy as np

def dist_cal(path1, path2):
    c1 = np.load(path1)
    c2 = np.load(path2)
    dist = np.linalg.norm(c1-c2)
    print dist


path1 = "/home/peth/places365/feature/test1.npy"
path2 = "/home/peth/places365/feature/test2.npy"

dist_cal(path1,path2)