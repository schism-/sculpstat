__author__ = 'christian'

from time import time

import numpy as np

import os.path

from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

from utility import common

import statsmodels.tsa as tsa
import statsmodels.api as sm
import statsmodels.graphics as sg

def filter_diff_attribute(final_data, attribute, time_serialized, filter_valid=True):
  ret_list = []
  labels = []
  k = 0
  for step_idx in range(len(final_data)):
    if filter_valid and not final_data[str(step_idx)]["valid"]:
      k += 1
      continue
    if time_serialized:
      ret_list.append([step_idx, final_data[str(step_idx)]["diff_data"][attribute]])
    else:
      ret_list.append(final_data[str(step_idx)]["diff_data"][attribute])
    labels.append(k)
    k += 1
  return ret_list, labels


def filter_brush_attribute(final_data, attribute, time_serialized):
  ret_list = []
  labels = []
  k = 0
  for step_idx in range(len(final_data)):
    if final_data[str(step_idx)]["valid"]:
      for i in range(final_data[str(step_idx)]["brush_data"]["brush_number"]):
        if time_serialized:
          ret_list.append([step_idx, final_data[str(step_idx)]["brush_data"][attribute][i]])
        else:
          ret_list.append(final_data[str(step_idx)]["brush_data"][attribute][i])
        labels.append(k)
    k += 1
  return ret_list, labels

# Loading brush data
model_name = "ogre"
json_array = common.load_json("../steps/" + model_name + "/final_data.json")

centroid, centroid_label = filter_brush_attribute(json_array, "centroids", False)
centroid_dist = []
for point_idx in range(1, len(centroid)):
    dist = np.linalg.norm(np.array(centroid[point_idx]) - np.array(centroid[point_idx - 1]))
    centroid_dist.append(dist)

for attr_name in ["lenghts", "size", "unp_size", "centroid_dist"]:
    if attr_name == "size" or attr_name == "unp_size":
        attr_data, labels = filter_brush_attribute(json_array, "size", False)
    elif attr_name == "centroid_dist":
        attr_data, labels = centroid_dist, centroid_label
    else:
        attr_data, labels = filter_brush_attribute(json_array, attr_name, False)

    if attr_name == "size":
        attr_data = [el[0] for el in attr_data]
    elif attr_name == "unp_size":
        attr_data = [el[1] for el in attr_data]

    fig1 = plt.figure(figsize=(16,9))

    ax1 = fig1.add_subplot(321)
    plt.plot(attr_data)
    plt.title("Plot for %s" % attr_name)

    ax2 = fig1.add_subplot(322)
    per = tsa.stattools.periodogram(attr_data)
    plt.plot(per)
    plt.title("Periodogram for %s" % attr_name)

    ax3 = fig1.add_subplot(323)
    sm.graphics.tsa.plot_acf(attr_data, lags=60, ax=ax3)

    plt.title("Autocorrelation for %s" % attr_name)

    ax4 = fig1.add_subplot(324)
    sm.graphics.tsa.plot_pacf(attr_data, lags=60, ax=ax4)
    plt.title("Partial Autocorrelation for %s" % attr_name)

    ax5 = fig1.add_subplot(325)
    plt.psd(attr_data, NFFT=256, Fs=2, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None)
    plt.tight_layout()
    #----------------------------------------------------------------------
    #                             saving image
    #----------------------------------------------------------------------
    root_images = "../images/" + model_name + "/"
    if not os.path.exists(root_images):
        os.makedirs(root_images)
    file_name = "d_clust"
    file_name += "basic_"
    file_name += attr_name + ".pdf"
    plt.savefig(root_images + file_name)


lenght_data, _ = filter_brush_attribute(json_array, "lenghts", False)

all_size_data, _ = filter_brush_attribute(json_array, "size", False)
size_data = [el[0] for el in all_size_data]
unp_size_data = [el[1] for el in all_size_data]

fig2 = plt.figure(figsize=(16,9))
k = 1
for stat1, lab1 in [(lenght_data, "lenght"), (size_data, "size"), (unp_size_data, "unprojected size")]:
    for stat2, lab2 in [(lenght_data, "lenght"), (size_data, "size"), (unp_size_data, "unprojected size")]:
        ax4 = fig2.add_subplot(int("33" + str(k)))
        ccf = tsa.stattools.ccf(np.array(stat1), np.array(stat2))
        plt.plot(ccf)
        plt.title("Cross-correlation for %s and %s" % (lab1, lab2))
        k += 1
plt.tight_layout()
#----------------------------------------------------------------------
#                             saving image
#----------------------------------------------------------------------
root_images = "../images/" + model_name + "/"
if not os.path.exists(root_images):
    os.makedirs(root_images)
file_name = "d_clust"
file_name += "crosscorr_.pdf"
plt.savefig(root_images + file_name)

plt.show()

