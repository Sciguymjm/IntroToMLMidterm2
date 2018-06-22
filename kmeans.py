import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from matplotlib import cm


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)


def main(n, k):
    img = plt.imread(f'{n}.jpg')
    print (img)
    # img = rgb2gray(img)
    img = img[::3, ::3]
    shape = img.shape
    img = img * (shape[0] + shape[1]) / (2 * 255)
    values = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            values.append([x, y, img[x, y]])
    values = np.array(values)
    means = values[np.random.choice(np.array(values.shape[0]), k)]
    iterations = 15
    clustered_img = np.zeros(shape)
    for n in range(iterations):
        buckets = [[] for b in range(k)]
        for v in values:
            lowest_value = 10e9
            lowest_index = -1
            for i, m in enumerate(means):
                val = distance(v, m)
                if val < lowest_value:
                    lowest_value = val
                    lowest_index = i
            buckets[lowest_index].append(v)
            clustered_img[int(v[0]), int(v[1])] = lowest_index * 10
        means = [np.array(a).mean(axis=0) for a in buckets]
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap=cm.gray)
    axs[1].imshow(clustered_img)
    plt.show()


if __name__ == '__main__':
    main('2', 10)
