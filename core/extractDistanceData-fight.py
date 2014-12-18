__author__ = 'christian'

import subprocess
import struct
import numpy
from plyfile import PlyData, PlyElement
from utility import common
import time
import os

class DistanceData(object):

    def __init__(self, model_name, max_step, output_folder, start=1):
        self.obj_path = "/Volumes/Part\ Mac/obj_smooth_normals_files/" + model_name
        self.model_name = model_name
        self.max_step = max_step
        self.output_folder = output_folder
        self.path_to_mlserver = "/Applications/meshlab.app/Contents/MacOS"

        self.output_ply = self.output_folder + self.model_name + "/temp.ply"
        self.script_path = self.output_folder + self.model_name + "/script.mlx"
        self.log_filepath = self.output_folder + self.model_name + "/log.txt"
        self.out_filepath = self.output_folder + self.model_name + "/out.txt"
        self.err_filepath = self.output_folder + self.model_name + "/err.txt"

        self.final_data = {}

        self.sample_perc = 1.0

        self.start = start

    def extract_data(self):
        for k in range(self.start, self.max_step):
            print("%s step %d / %d" % (self.model_name, k, self.max_step))
            obj_1_path = self.obj_path + "/snap" + str(k-1).zfill(6) + ".obj"
            obj_2_path = self.obj_path + "/snap" + str(k).zfill(6) + ".obj"

            self.generate_script(k)

            fh_out = open(self.out_filepath, "a+")
            fh_err = open(self.err_filepath, "a+")

            cmd = "./meshlabserver -i " + obj_1_path + " " + obj_2_path + " -o " + self.output_ply + " -s " + self.script_path + " -l " + self.log_filepath+ " -om vq"
            p = subprocess.Popen(cmd , shell=True, stdout=fh_out, stderr=fh_err, cwd=self.path_to_mlserver)
            p.wait()

            plydata = PlyData.read(open(self.output_ply, 'rb'))
            temp = []
            for idx, value in enumerate(plydata['vertex'].data['quality']):
                if value:
                    print(idx, value)
                    temp.append((idx, value))

            if not os.path.exists(self.output_folder + self.model_name + "/dist_data/"):
                os.makedirs(self.output_folder + self.model_name + "/dist_data/")

            fh_dist = open(self.output_folder + self.model_name + "/dist_data/step"+ str(k), "w")
            for idx, value in temp:
                fh_dist.write("%d %.10f\n" % (idx, value))
            fh_dist.close()

    def generate_script(self, diff_no):
        # .: diff_no = 10      obj_1 = 9       obj_2 = 10
        bbox_diag = {
            "alien":           4.88411,
            "elder":           18.1198,
            "elf":             5.62413,
            "engineer":        2.03795,
            "explorer":        17.2768,
            "fighter":         2.0678,
            "gargoyle":        10.5912,
            "gorilla":         12.3074,
            "man":             4.38392,
            "merman":          4.90317,
            "monster":         6.87775,
            "ogre":            20.0907,
            "sage":            4.91853
        }

        #open mesh data
        md = common.load_json("../steps/" + self.model_name + "/mesh_data.json")
        vertices_no = int(md[str(diff_no)]['vertices_no'])

        file_content = "<!DOCTYPE FilterScript>\n"  + \
                       "<FilterScript>\n"  + \
                       "<filter name=\"Hausdorff Distance\">\n"  + \
                       "<Param type=\"RichMesh\"    name=\"SampledMesh\"    value=\"1\" />\n"  + \
                       "<Param type=\"RichMesh\"    name=\"TargetMesh\"     value=\"0\" />\n"  + \
                       "<Param type=\"RichBool\"    name=\"SaveSample\"     value=\"true\" />\n"  + \
                       "<Param type=\"RichBool\"    name=\"SampleVert\"     value=\"true\" />\n"  + \
                       "<Param type=\"RichBool\"    name=\"SampleEdge\"     value=\"true\" />\n"  + \
                       "<Param type=\"RichBool\"    name=\"SampleFauxEdge\" value=\"false\"/>\n"  + \
                       "<Param type=\"RichBool\"    name=\"SampleFace\"     value=\"true\" />\n"  + \
                       "<Param type=\"RichInt\"     name=\"SampleNum\"      value=\"" + str(int(vertices_no * self.sample_perc)) + "\" />\n"  + \
                       "<Param type=\"RichAbsPerc\" name=\"MaxDist\"        value=\"" + str((bbox_diag[self.model_name] / 100.0) * 5.0) + "\" min=\"0\" max=\"" + str(bbox_diag[self.model_name]) + "\"/>\n"  + \
                       "</filter>\n"  + \
                       "<filter name=\"Select None\">\n"  + \
                       "<Param type=\"RichBool\"    name=\"allFaces\"       value=\"true\" />\n"  + \
                       "<Param type=\"RichBool\"    name=\"allVerts\"       value=\"true\" />\n"  + \
                       "</filter>\n"  + \
                       "<filter name=\"Change the current layer\">\n"  + \
                       "<Param type=\"RichMesh\"    name=\"mesh\"           value=\"1\" />\n"  + \
                       "</filter>\n"  + \
                       "</FilterScript>\n"

        script_fh = open(self.script_path, 'w')
        script_fh.write(file_content)
        script_fh.close()

'''
models = [
    ["alien",       2216],
    ["elder",       3119],
    ["elf",         4307],
    ["engineer",     987],
    ["explorer",    1858],
    ["fighter",     1608],
    ["gargoyle",    1058],
    ["gorilla",     2719],
    ["man",         1580],
    ["merman",      2619],
    ["monster",      967],
    ["ogre",        1720],
    ["sage",        2136]
]
'''

models = [["fighter",     1608]] #684

start = time.time()
for model_name, max_step in models:
    dd = DistanceData(model_name, max_step, "/Users/christian/Desktop/Ph.D./sculptAnalysis/steps/")
    dd.extract_data()
print("took: %f" % (time.time() - start))