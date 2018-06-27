import skimage.transform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)


def cluster_distance(a, b, means, i1, i2):
    # a, b = np.array(a), np.array(b)
    l = len(means)
    if l <= i1:
        means.append(a.mean(axis=0))
        l = len(means)
    if l <= i2:
        means.append(b.mean(axis=0))
    return np.linalg.norm(means[i1] - means[i2]), means


def main(n, k):
    img = plt.imread(f'{n}.jpg')
    # img = rgb2gray(img)
    s = img.shape

    img = skimage.transform.resize(img, tuple((np.array(s) / 5).tolist()))
    # img = img[::5, ::5]
    shape = img.shape
    img = img * (shape[0] + shape[1])
    values = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            values.append([x, y, img[x, y]])
    values = np.array(values)
    values = values[:, None]
    values = [np.array(v) for v in values.tolist()]
    while len(values) > k:
        closest_clusters_1 = None
        closest_clusters_2 = None
        closest_value = 10e9
        means = []
        for i1, c1 in enumerate(values):
            for i2, c2 in enumerate(values):
                if i1 == i2:
                    continue
                ve, means = cluster_distance(c1, c2, means, i1, i2)
                if ve < closest_value:
                    closest_clusters_1, closest_clusters_2 = i1, i2
                    closest_value = ve
        values[closest_clusters_1] = np.insert(values[closest_clusters_1], 0, values[closest_clusters_2], axis=0)
        values.pop(closest_clusters_2)
        print (len(values))
    clustered_img = np.zeros(img.shape)
    for i, b in enumerate(values):
        for p in b:
            clustered_img[int(p[0]), int(p[1])] = i
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap=cm.gray)
    axs[1].imshow(clustered_img)
    plt.show()


if __name__ == '__main__':
    main('4', 6)
