import bpy

from os import listdir
from utility import common
from os.path import isfile, join


class BlendConverter(object):

    def __init__(self, blend_root_dir, obj_root_dir, model_name, start, stop, basemesh_name):
        self.blend_files_path = blend_root_dir + model_name
        self.obj_files_path = obj_root_dir + model_name
        self.start = start
        self.stop = stop
        self.basemesh_name = basemesh_name

        self.blend_files = common.get_files_from_directory(self.blend_files_path, ['blend'], "snap")

    def convert_to_obj(self):
        print(self.blend_files)
        for file in self.blend_files:
            try:
                idx = int((file[0].split('/')[-1]).split('.')[0][4:])
            except ValueError:
                idx = 0

            if idx > self.stop:
                print("CONVERSION TO OBJ COMPLETED")
                return

            if idx >= self.start:
                bpy.ops.wm.open_mainfile(filepath=file[0],
                                         filter_blender=True,
                                         filemode=8,
                                         display_type='FILE_DEFAULTDISPLAY',
                                         load_ui=False,
                                         use_scripts=True)
                try:
                    bpy.data.objects[self.basemesh_name].modifiers["Multires"].levels = \
                        bpy.data.objects[self.basemesh_name].modifiers["Multires"].sculpt_levels
                except KeyError as e:
                    print('ERROR @ file %s : modifier not found' % file[1])

                bpy.ops.export_scene.obj(filepath= self.obj_files_path + "/" + file[1].split('.')[0] + ".obj",
                                         axis_forward='-Z',
                                         axis_up='Y',
                                         use_normals=True,
                                         use_materials=False,
                                         keep_vertex_order=True)

if __name__ == "__main__":
    names = [["gargoyle2", "Cube"], ["monster", "Cube"]] # ["task02", "Cube"], ["task06", "Basemesh_FullbodyMale"],
    names = [["task02-alien", "Cube"]]

    names = [["elder", "Plane"], ["elf", "Cube"], ["explorer", "Cube"], ["gorilla", "Cube"], ["sage", "Bust"]]

    for model_name, basemesh in names:
        #blend_dir = "../blend_files/"
        #obj_dir = "../obj2_files/"
        blend_dir = "/Volumes/PART FAT/3ddata/"
        obj_dir = "/Volumes/Part Mac/obj2_files/"

        # arrivato a snap001607
        bc = BlendConverter(blend_dir, obj_dir, model_name, 0, 1000000, basemesh)
        bc.convert_to_obj()