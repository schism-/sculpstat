import json
import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac

def draw_plot(fig,
              values, 
              labels, 
              model_names, model_tech, model_name,
              pos_idx, gpi, 
              bar_color="g", 
              y_limit = None):
    graph = fig.add_subplot(len(model_names), gpi, pos_idx, title=model_name + model_tech[model_name])
    if y_limit:
      graph.set_ylim(y_limit)
    ret = graph.plot(labels, values, bar_color)
    _ = graph.legend()
    return graph

def draw_hist(fig,
              values, 
              bins, 
              model_names, model_tech, model_name,
              pos_idx, 
              gpi, 
              hist_type="stepfilled", 
              color="blue", 
              is_normed=False):
    graph2 = fig.add_subplot(len(model_names), gpi, pos_idx, title=model_name + model_tech[model_name])
    ret = graph2.hist(values, bins, histtype=hist_type,normed=is_normed, color=color)
    _ = graph2.legend()
    return ret


def draw_acorr(fig,
               values, 
               model_names, model_tech, model_name,
               pos_idx, 
               gpi, 
               line_width=1, 
               max_lags=None, 
               detrend_fun=mlab.detrend_linear):    
    graph2 = fig.add_subplot(len(model_names), gpi, pos_idx, title=model_name + model_tech[model_name])
    _ = graph2.acorr(values, usevlines=True, normed=True, maxlags=max_lags, lw=line_width, detrend=detrend_fun)
    _ = graph2.legend()    


def draw_acorr2(fig,
                values, 
                model_names, model_tech, model_name,
                pos_idx, 
                gpi, 
                line_width=1, 
                max_lags=None, 
                detrend_fun=mlab.detrend_none, 
                show_grid=False):
    graph2 = fig.add_subplot(len(model_names), gpi, pos_idx, title=model_name + model_tech[model_name])
    lags, c, line, b = graph2.acorr( values, 
                                    usevlines=True, 
                                    maxlags=max_lags, 
                                    normed=True, 
                                    lw=line_width, 
                                    detrend=detrend_fun)
    graph2.set_xlim([-10.0, float(len(values) + 10.0)])
    
    if show_grid:
      corr_zeroes = []
      for k in range(len(c) - 1):
        if ((c[k] <= 0 and c[k+1] >= 0) or (c[k] >= 0 and c[k+1] <= 0)) and lags[k] > 0:
          corr_zeroes.append(lags[k])
      graph2.set_xticks(corr_zeroes, minor=False)
      graph2.xaxis.grid(True, which='major')

      print(model_name)
      print(corr_zeroes)
    
    _ = graph2.legend()


def load_data(model_names, root_directory):
  final_data = {}
  for model_name in model_names:
    final_data_file = open(root_directory + "steps/" + model_name + "/final_data.json", "r")
    final_data[model_name] = json.load(final_data_file)
    final_data_file.close()
  return final_data


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


