__author__ = 'christian'

import os
import os.path
import math
import numpy as np
from utility import common

from sklearn import manifold, datasets, decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Data:
    def __init__(self, id, time, type, features):
        self.id = id
        self.time = time
        self.type = type
        self.x, self.y, self.z, self.data = self.unpack(features)
        if not self.data:
            raise TypeError("Error in unpacking data")
        self.clabel = 99999 # unclustered

    def __str__(self):
        return "%d) %.2f %.2f %.2f %s" % (self.id, self.x, self.y, self.z, self.data)

    def __repr__(self):
        return "(id=%d x=%.2f y=%.2f z=%.2f data=%s)" % (self.id, self.x, self.y, self.z, self.data)

    def unpack(self, features):
        if self.type == "b":
            if features["valid"]:
                data = []
                modes = {}

                # path centroid
                pos = features["centroids"][0]

                pos = features["obboxes"][0][0]

                # 0 - path lenght
                data.append(features["lenghts"][0])

                # brush mode
                '''
                if features["mode"][0] in modes:
                    data.append(modes[features["mode"][0]])
                    modes[features["mode"][0]] += 1
                else:
                    modes[features["mode"][0]] = 1
                    data.append(modes[features["mode"][0]])
                '''

                # 1,2,3 - obb extents
                obb_ext = features["obboxes"][0][1]
                data += obb_ext

                # 4,5 - size and unp size
                data += features["size"][0]

                # 6,7,8,9 - path stats
                data.append(features["path_mean"] if features["path_mean"] else 0.0)
                data.append(features["path_variance"] if features["path_variance"] else 0.0)
                data.append(features["path_skewness"] if features["path_skewness"] else 0.0)
                data.append(features["path_curtosis"] if features["path_curtosis"] else 0.0)

                # 10, 11, 12, 13 - pressure stats
                data.append(features["pressure_mean"] if features["pressure_mean"] else 0.0)
                data.append(features["pressure_variance"] if features["pressure_variance"] else 0.0)
                data.append(features["pressure_skewness"] if features["pressure_skewness"] else 0.0)
                data.append(features["pressure_curtosis"] if features["pressure_curtosis"] else 0.0)

                return pos[0], pos[1], pos[2], data
            else:
                return -1, -1, -1, None
        else:
            if features["diff_added_centroids"]:
                data = []
                # path centroid
                pos = features["diff_added_centroids"]

                # added data
                data.append(features["added_vertices"])
                data.append(features["added_normals"])
                data.append(features["added_faces"])

                # bbox extents
                data += features["diff_added_bbox"][1]

                # added stats
                for name in ["added_mean", "added_variance", "added_skewness", "added_curtosis"]:
                    if (features[name][0] and features[name][1] and features[name][2]):
                        data += features[name]

                # deleted data
                data.append(features["deleted_vertices"])
                data.append(features["deleted_normals"])
                data.append(features["deleted_faces"])

                # bbox extents
                data += features["diff_deleted_bbox"][1]

                # deleted stats
                for name in ["deleted_mean", "deleted_variance", "deleted_skewness", "deleted_curtosis"]:
                    if (features[name][0] and features[name][1] and features[name][2]):
                        data += features[name]

                return pos[0], pos[1], pos[2], data
            else:
                return -1, -1, -1, None


    def spatial_distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


    def non_spatial_distance(self, y):
        return np.linalg.norm(np.array(self.data) - np.array(y.data))


    def temporal_distance(self, y):
        return abs(self.time - y.time)


    @staticmethod
    def load_data(filename, is_brush=True):
        final_data = common.load_json(filename)
        id = 0
        loaded_data = [None, ] * len(final_data)
        for step in final_data:
            step_data = final_data[step]
            try:
                loaded_data[int(step_data["step_number"])] = Data(id,
                                                                  int(step_data["step_number"]),
                                                                  "b" if is_brush else "d",
                                                                  step_data["brush_data" if is_brush else "diff_data"])
                id += 1
            except TypeError:
                continue
        loaded_data = [el for el in loaded_data if el]
        return loaded_data



class STDBSCAN(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.data_filename = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/"+self.model_name+"/final_data.json"
        self.noise = 999999
        self.unclustered = 99999
        self.cluster_label = 0
        self.data = Data.load_data(self.data_filename)
        self.min_pts = self.estimate_min_pts()
        self.max_eps1 = self.estimate_eps1()
        self.max_eps2 = self.estimate_eps2()
        self.time_window = self.estimate_time_window()

        print("Instances: %d --- MIN PTS = %d --- MAX EPS1 %f --- MAX EPS2 %f --- time window %d" % (len(self.data), self.min_pts, self.max_eps1, self.max_eps2, self.time_window))


    def estimate_time_window(self):
        #return int(round(math.log(abs(self.data[-1].time - self.data[0].time))))
        return int(round(math.sqrt(abs(self.data[-1].time - self.data[0].time))))


    def estimate_min_pts(self):
        return int(round(math.log(len(self.data))))


    def _nearest_neighbours(self, is_spatial = True):
        X = []
        if is_spatial:
            for el in self.data:
                X.append(np.array((el.x, el.y, el.z)))
        else:
            for el in self.data:
                X.append(np.array(el.data))
        X = np.array(X)
        nbrs = NearestNeighbors(n_neighbors=self.min_pts, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        k_distances = distances[:,-1]
        k_distances.sort()
        k_distances = k_distances[::-1]

        return k_distances


    def estimate_eps1(self):
        k_distances = self._nearest_neighbours()
        elbow, elbow_idx, distToLine = self.find_elbow(k_distances)

        '''
        fig = plt.figure()
        graph = fig.add_subplot(1, 2, 1)
        graph.plot(range(len(k_distances)), k_distances, 'g')
        plt.axvline(elbow_idx)

        graph2 = fig.add_subplot(1, 2, 2)
        graph2.plot(range(len(distToLine)), distToLine, 'g')
        plt.axvline(elbow_idx)
        plt.show()
        '''

        return elbow


    def estimate_eps2(self):
        k_distances = self._nearest_neighbours(False)
        elbow, elbow_idx, distToLine = self.find_elbow(k_distances)

        '''
        fig = plt.figure()
        graph = fig.add_subplot(1, 2, 1)
        graph.plot(range(len(k_distances)), k_distances, 'g')
        plt.axvline(elbow_idx)

        graph2 = fig.add_subplot(1, 2, 2)
        graph2.plot(range(len(distToLine)), distToLine, 'g')
        plt.axvline(elbow_idx)
        plt.show()
        '''

        return elbow


    def find_elbow(self, curve):
        firstPoint = np.array((0, curve[0]))
        lastPoint = np.array((len(curve) - 1, curve[-1]))

        lineVec = lastPoint - firstPoint

        lineVecN = lineVec / np.linalg.norm(lineVec)

        vecFromFirst = []
        for idx, point in enumerate(curve):
            vecFromFirst.append(np.array((idx, point)) - firstPoint)
        vecFromFirst = np.array(vecFromFirst)

        vecFromFirstParallel = []
        for point in vecFromFirst:
            temp = np.dot(point, lineVecN)
            vecFromFirstParallel.append(temp * lineVecN)
        vecFromFirstParallel = np.array(vecFromFirstParallel)

        vecToLine = vecFromFirst - vecFromFirstParallel

        distToLine = []
        for point in vecToLine:
            distToLine.append(np.linalg.norm(point))
        distToLine = np.array(distToLine)

        return np.max(distToLine), np.argmax(distToLine), distToLine


    def eps1(self, x, y):
        return x.spatial_distance(y)


    def eps2(self, x, y):
        return x.non_spatial_distance(y)


    def retrieve_neighbours(self, x):
        Y = []
        for obj in self.data:
            if obj.id == x.id:
                continue
            else:
                e1 = self.eps1(x, obj)
                e2 = self.eps2(x, obj)
                # print(e1, e2)
                if e1 < self.max_eps1 and e2 < self.max_eps2 and abs(obj.time - x.time) <= self.time_window:
                    Y.append(obj)
        return Y


    def clusterize(self):
        clusters = {}
        for obj in self.data:
            if obj.clabel == self.unclustered:
                X = self.retrieve_neighbours(obj)
                if len(X) < self.min_pts:
                    obj.clabel = self.noise
                else:
                    self.cluster_label += 1
                    print("found a new cluster (%d)" % self.cluster_label)

                    obj.clabel = self.cluster_label

                    clusters[self.cluster_label] = [obj.id]

                    for X_obj in X:
                        if X_obj.clabel != self.noise and X_obj.clabel != self.unclustered:
                            pass
                        else:
                            X_obj.clabel = self.cluster_label
                            clusters[self.cluster_label].append(X_obj.id)
                    stack = X

                    while len(stack) > 0:
                        stack_obj = stack.pop(0)

                        stack_X = self.retrieve_neighbours(stack_obj)

                        if len(stack_X) >= self.min_pts:
                            for stack_X_obj in stack_X:
                                if (stack_X_obj.clabel != self.noise or stack_X_obj.clabel == self.unclustered) \
                                        and stack_X_obj.clabel != self.cluster_label:

                                    if stack_X_obj.clabel != self.noise and stack_X_obj.clabel != self.unclustered:
                                        continue

                                    stack_X_obj.clabel = self.cluster_label

                                    clusters[self.cluster_label].append(stack_X_obj.id)

                                    if stack_X_obj not in stack:
                                        stack.append(stack_X_obj)

        acc = 0
        labels = [None, ] * len(self.data)
        centroid_pos = [None, ] * len(self.data)
        raw_data = [None, ] * len(self.data)
        for c in clusters:
            print("Cluster no %s [len = %d]" % (c, len(clusters[c])))
            acc += len(clusters[c])
            #print(clusters[c])
            for el in clusters[c]:
                labels[el] = c

        noise = [obj for obj in self.data if obj.clabel == self.noise]

        print("Noised [len = %d]" % (len(noise)))
        acc += len(noise)
        print("Total instances: %d" % acc)

        for idx in range(len(labels)):
            if not labels[idx]:
                labels[idx] = len(clusters) + 1

        for obj in self.data:
            centroid_pos[obj.id] = [obj.x, obj.y, obj.z]
            raw_data[obj.id] = obj.data[:]

        centroid_pos = np.array(centroid_pos)
        labels = np.array(labels)
        raw_data = np.array(raw_data)

        # saving clustering data
        save_dir = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/clustering/" + self.model_name + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fh = open(save_dir + "bb_centroid_pos", 'wb')
        np.save(fh, centroid_pos)
        fh.close()
        fh = open(save_dir + "bb_labels", 'wb')
        np.save(fh, labels)
        fh.close()
        fh = open(save_dir + "bb_raw_data", 'wb')
        np.save(fh, raw_data)
        fh.close()

        #fig = plt.figure(figsize=(9, 9))
        #plot_clustering_scatter(centroid_pos, labels, raw_data, title="[meanshift] pos with no dim reduction", fig=fig, subplot_idx=111)
        #plt.show()

    def visualize(self):
        load_dir = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/clustering/" + self.model_name + "/"

        fh = open(load_dir + "bb_centroid_pos", 'rb')
        centroid_pos = np.load(fh)
        fh.close()

        fh = open(load_dir + "bb_labels", 'rb')
        labels = np.load(fh)
        fh.close()

        fh = open(load_dir + "bb_raw_data", 'rb')
        raw_data = np.load(fh)
        fh.close()

        fig = plt.figure(figsize=(9, 9))
        plot_clustering_scatter(centroid_pos, labels, raw_data, title="[meanshift] pos with no dim reduction", fig=fig, subplot_idx=111)
        plt.show()

def plot_clustering_scatter(X_red, labels, raw_data, title=None, force_idx=(0,1,2), fig=None, subplot_idx=321):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    ax = fig.add_subplot(subplot_idx, projection='3d')
    #ax = Axes3D(ax1, rect=[0, 0, 0.95, 1], elev=0, azim=-90)

    d = raw_data[:,5]
    c = normalize(d[:,np.newaxis], axis=0).ravel()

    ax.scatter(X_red[:, force_idx[0]], -1.0 * X_red[:, force_idx[2]],  X_red[:, force_idx[1]],
               c=labels, cmap=plt.get_cmap("Paired"), s=c * 1500.0)

    ax.view_init(elev=0, azim=-90)

    ax.set_autoscale_on(False)

    if title is not None:
        plt.title(title, size=17)

if __name__ == "__main__":
    model_names = ["elder", "engineer", "explorer", "fighter", "gargoyle", "gorilla", "merman", "monster", "ogre", "sage", "man"]

    for model in model_names:
        stdbs = STDBSCAN(model)
        #stdbs.clusterize()
        stdbs.visualize()