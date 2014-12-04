__author__ = 'christian'

from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import manifold, datasets, decomposition
from sklearn.preprocessing import normalize

from sklearn.cluster import AgglomerativeClustering

from utility import common

#----------------------------------------------------------------------
def plot_clustering_scatter(X_red, labels, raw_data, title=None, force_idx=(0,1,2), fig=None, subplot_idx=321):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    ax = fig.add_subplot(subplot_idx, projection='3d')
    #ax = Axes3D(ax1, rect=[0, 0, 0.95, 1], elev=0, azim=-90)

    d = raw_data[:,0]
    c = normalize(d[:,np.newaxis], axis=0).ravel()

    ax.scatter(X_red[:, force_idx[0]], -1.0 * X_red[:, force_idx[2]],  X_red[:, force_idx[1]],
               c=labels, cmap=plt.cm.cool, s=c * 1000)
    ax.view_init(elev=0, azim=-90)
    plt.axis('equal')
    if title is not None:
        plt.title(title, size=17)

# Loading brush data
model_name = "sage"
json_array = common.load_json("../steps/" + model_name + "/feature_vector.json")

modes = {}
for idx_l in range(len(json_array)):
    for idx_e in range(len(json_array[idx_l])):
        if idx_e not in [2, 21]:
            json_array[idx_l][idx_e] = float(json_array[idx_l][idx_e])
        elif idx_e == 2:
            if json_array[idx_l][idx_e] in modes:
                json_array[idx_l][idx_e] = modes[json_array[idx_l][idx_e]]
            else:
                print("Adding %d for %s" % (len(modes), json_array[idx_l][idx_e]))
                modes[json_array[idx_l][idx_e]] = len(modes)
                json_array[idx_l][idx_e] = modes[json_array[idx_l][idx_e]]
        else:
            del json_array[idx_l][idx_e]


labels_to_idx = {
    0: "size",
    1: "unp_size",
    2: "mode",
    3: "lenght",
    4: "centroid_x",
    5: "centroid_y",
    6: "centroid_z",
    7: "obb_cen_x",
    8: "obb_cen_y",
    9: "obb_cen_z",
    10: "obb_dim_x",
    11: "obb_dim_y",
    12: "obb_dim_z",
    13: "pressure_mean",
    14: "pressure_variance",
    15: "pressure_skewness",
    16: "pressure_curtosis",
    17: "path_mean",
    18: "path_variance",
    19: "path_skewness",
    20: "path_curtosis",
    21: "step"
}

'''
    "size", "unp_size", "mode", "lenght"
    "centroid_x", "centroid_y", "centroid_z"
    "obb_cen_x", "obb_cen_y", "obb_cen_z"
    "obb_dim_x", "obb_dim_y", "obb_dim_z"
    "pressure_mean", "pressure_variance", "pressure_skewness", "pressure_curtosis"
    "path_mean", "path_variance", "path_skewness", "path_curtosis"
    "step
'''

to_del = ["centroid_x", "centroid_y", "centroid_z",
          "obb_cen_x", "obb_cen_x", "obb_cen_x",
          "step"]

centroid_pos = np.array(json_array, dtype='f')[:,7:10]

for idx_l in range(len(json_array)):
    for idx_e in range(len(json_array[idx_l]) - 1, -1, -1):
        if labels_to_idx[idx_e] in to_del:
            del json_array[idx_l][idx_e]


np_array = np.array(json_array, dtype='f')
n_samples, n_features = np_array.shape

print(np_array[13])
print(n_samples, n_features)

np.random.seed(42)

#----------------------------------------------------------------------

# 3D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=3).fit_transform(np_array)
print("Done.")

print("Computing PCA")
pca = decomposition.PCA(n_components=3)
pca.fit(np_array)
X_pca = pca.transform(np_array)
print("Done.")

for linkage in ['average', 'ward', 'complete']:
    cluster_no = 4

    clustering_pure = AgglomerativeClustering(linkage=linkage, n_clusters=cluster_no)
    t0 = time()
    clustering_pure.fit(np_array)
    print("%s : %.2fs" % (linkage, time() - t0))

    clustering_spec = AgglomerativeClustering(linkage=linkage, n_clusters=cluster_no)
    t0 = time()
    clustering_spec.fit(X_red)
    print("%s : %.2fs" % (linkage, time() - t0))

    clustering_pca = AgglomerativeClustering(linkage=linkage, n_clusters=cluster_no)
    t0 = time()
    clustering_pca.fit(X_pca)
    print("%s : %.2fs" % (linkage, time() - t0))

    fig = plt.figure(figsize=(16, 9))

    #plot_clustering_scatter(np_array, clustering_pure.labels_, np_array, "no dim reduction - %s linkage" % linkage, fig=fig, subplot_idx=231)

    #plot_clustering_scatter(X_red, clustering_spec.labels_, np_array, "spectral - %s linkage" % linkage, fig=fig, subplot_idx=232)

    #plot_clustering_scatter(X_pca, clustering_pca.labels_, np_array, "PCA - %s linkage" % linkage, fig=fig, subplot_idx=233)


    plot_clustering_scatter(centroid_pos, clustering_pure.labels_, np_array, "%s pos with no dim reduction" % linkage, fig=fig, subplot_idx=131)

    plot_clustering_scatter(centroid_pos, clustering_spec.labels_, np_array, "%s pos with spectral labels" % linkage, fig=fig, subplot_idx=132)

    plot_clustering_scatter(centroid_pos, clustering_pca.labels_, np_array, "%s pos with PCA labels" % linkage, fig=fig, subplot_idx=133)

    plt.tight_layout()
plt.show()