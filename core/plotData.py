__author__ = 'christian'

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pickle

def plot_mean_times():
    in_file = open("../test.txt", "r")
    meshgit_time = []
    meshhisto_time = []
    n_faces = []
    for l in in_file:
        val = l.split(' ')
        n_faces.append(val[1].strip())
        meshgit_time.append(val[2].strip())
        meshhisto_time.append(val[3].strip())
    int_n_faces = [int(l) for l in range(len(n_faces))]
    fig = plt.figure()
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(int_n_faces, meshgit_time, 'b', int_n_faces, meshhisto_time, 'r-o')

    x_scale = "linear"
    y_scale = "log"
    graph.set_xscale(x_scale)
    graph.set_yscale(y_scale)
    graph.set_xticklabels(n_faces)

    plt.savefig("./grafici/mean_times_" + x_scale + "_" + y_scale + "2.pdf")

    plt.show()


def plot_all_times():
    in_file = open("../dati_comp/all_data_2", "r")

    bins = [0, 2000, 10000, 16000, 'over']

    meshgit_time = {}
    meshhisto_time = {}
    n_faces = {}

    for b in bins:
        meshgit_time[b] = []
        meshhisto_time[b] = []
        n_faces[b] = []

    for l in in_file:
        val = l.split(' ')
        f = int(val[0].strip())
        mg = float(val[1].strip())
        mh = float(val[2].strip())
        found = False
        for k in range(1, len(bins) - 1):
            if (bins[k-1] < int(f) < bins[k]) and ((float(mg) != 0.0) and (float(mh) != 0.0)):
                found = True
                n_faces[bins[k]].append(f)
                meshgit_time[bins[k]].append(mg)
                meshhisto_time[bins[k]].append(mh)
        if not found and ((float(mg) != 0.0) and (float(mh) != 0.0)):
            n_faces['over'].append(f)
            meshgit_time['over'].append(mg)
            meshhisto_time['over'].append(mh)

    fig = plt.figure(figsize=(18, 10))
    int_n_faces = {}
    x_scale = "linear"
    y_scale = "linear"

    for k in range(1, len(bins)):
        int_n_faces[bins[k]] = [int(l) for l in range(len(n_faces[bins[k]]))]

        graph = fig.add_subplot(1, len(bins) - 1, k)

        #graph.plot(int_n_faces[bins[k]], meshgit_time[bins[k]], 'b.',
        #           int_n_faces[bins[k]], meshhisto_time[bins[k]], 'r.')


        graph.plot(n_faces[bins[k]], meshgit_time[bins[k]], 'b.',
                   n_faces[bins[k]], meshhisto_time[bins[k]], 'r.')

        graph.set_xscale(x_scale)
        graph.set_yscale(y_scale)

        graph.xaxis.set_major_locator(tck.FixedLocator(locs=[n_faces[bins[k]][0],
                                                             n_faces[bins[k]][len(n_faces[bins[k]]) - 1]]))

        formatter = tck.FormatStrFormatter('%d')
        #formatter.set_scientific(False)
        graph.xaxis.set_major_formatter(formatter)

        #graph.set_xticklabels(n_faces[bins[k]])

    plt.savefig("./grafici/mean_all_times_" + x_scale + "_" + y_scale + "2.pdf")
    plt.show()


def plot_single_model(model_name, dist, blanks, threshold):
    in_file = open("../dati_comp/" + model_name, "r")
    meshgit_time = []
    meshhisto_time = []
    n_faces = []
    for l in in_file:
        val = l.split(' ')

        f = val[0].strip()
        mg = val[1].strip()
        mh = val[2].strip()

        if (int(f) != 0) and ((float(mg) != 0.0) and (float(mh) != 0.0)):
            n_faces.append(f)
            meshgit_time.append(mg)
            meshhisto_time.append(mh)

    int_n_faces = [int(l) for l in range(len(n_faces))]

    label_n_faces = [n_faces[0]]
    blank = 0
    for k in xrange(1, len(n_faces)):
        if (int(n_faces[k]) == int(n_faces[k-1])) or (int(n_faces[k]) <= (int(n_faces[k-1])) + dist):
            if (int(n_faces[k]) < threshold) and (blank < blanks):
                label_n_faces.append(' ')
                blank += 1
            elif (int(n_faces[k]) >= threshold) and (blank < blanks * 2):
                label_n_faces.append(' ')
                blank += 1
            else:
                label_n_faces.append(n_faces[k])
                blank = 0
        else:
            label_n_faces.append(n_faces[k])

    fig = plt.figure(figsize=(18, 10))
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(int_n_faces, meshgit_time, 'b', int_n_faces, meshhisto_time, 'r--')

    x_scale = "linear"
    y_scale = "log"
    graph.set_xscale(x_scale)
    graph.set_yscale(y_scale)

    xmajorLocator = tck.FixedLocator(int_n_faces)
    graph.xaxis.set_major_locator(xmajorLocator)

    xmajorFormatter = tck.ScalarFormatter()
    graph.xaxis.set_major_formatter(xmajorFormatter)

    graph.set_xticklabels(label_n_faces)

    plt.savefig("./grafici/" + model_name + "_" + x_scale + "_" + y_scale + ".pdf")

    plt.show()

def plot_brush_data(model_name):
    in_file = open("../steps/" + model_name + "/b_data", "rb")
    b_data = pickle.load(in_file)

    in_file2 = open("../steps/" + model_name + "/b_size", "rb")
    b_size = pickle.load(in_file2)

    bbox_points = []
    lenghts = []
    volumes = []
    int_labels = []
    k = 0
    for bbox_p, l, vol in b_data:
        if vol > 0.00001:
            bbox_points.append(bbox_p)
            lenghts.append(l)
            volumes.append(vol)
        else:
            if k > 0:
                bbox_points.append(bbox_points[-1])
                lenghts.append(lenghts[-1])
                volumes.append(volumes[-1])
            else:
                bbox_points.append([])
                lenghts.append(0.0)
                volumes.append(0.0)
        int_labels.append(k)
        k += 1

    sizes = []
    unp_sizes = []
    for s in b_size:
        sizes.append(s[0])
        unp_sizes.append(s[1])

    fig = plt.figure()
    fig.suptitle("LENGHT / VOLUMES")
    graph = fig.add_subplot(1, 2, 1)
    graph.plot(int_labels, lenghts, 'b', label='lenght')
    graph.legend()
    graph = fig.add_subplot(1, 2, 2)
    graph.plot(int_labels, volumes, 'r', label='volume')
    graph.legend()

    fig2 = plt.figure()
    fig2.suptitle("BRUSH SIZES")
    graph2 = fig2.add_subplot(1, 2, 1)
    graph2.plot(int_labels, sizes, 'g', label='b_size')
    graph2.legend()
    graph2 = fig2.add_subplot(1, 2, 2)
    graph2.plot(int_labels, unp_sizes, 'k', label='b_unp_size')
    graph2.legend()

    x_scale = "linear"
    y_scale = "linear"
    graph.set_xscale(x_scale)
    graph.set_yscale(y_scale)
    graph2.set_xscale(x_scale)
    graph2.set_yscale(y_scale)
    #graph.set_xticklabels(n_faces)
    #plt.savefig("./grafici/mean_times_" + x_scale + "_" + y_scale + "2.pdf")

    plt.show()

if __name__ == '__main__':
    #plot_brush_data("task01")
    plot_brush_data("task02")
    #plot_brush_data("task06")
    #plot_brush_data("monster")
    #plot_brush_data("gargoyle2")
