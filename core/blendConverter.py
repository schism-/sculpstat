import bpy

from os import listdir, makedirs
from utility import common
from os.path import isfile, join, exists


class BlendConverter(object):

    def __init__(self, blend_root_dir, obj_root_dir, model_name, start, stop, basemesh_name, mode="obj"):
        self.blend_files_path = blend_root_dir + model_name
        self.obj_files_path = obj_root_dir + model_name
        self.start = start
        self.stop = stop
        self.basemesh_name = basemesh_name
        self.mode = mode

        self.blend_files = common.get_files_from_directory(self.blend_files_path, ['blend'], "snap")
        self.blend_files.sort()

    def convert_to_obj(self):
        if not exists(self.obj_files_path + "/"):
            makedirs(self.obj_files_path + "/")

        for file in self.blend_files:
            try:
                idx = int((file[0].split('/')[-1]).split('.')[0][4:])
            except ValueError:
                idx = 0

            if idx > self.stop:
                print("CONVERSION to %s COMPLETED" % self.mode)
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
                except KeyError:
                    print('modifier not found')

                if self.mode == "obj":
                    bpy.ops.export_scene.obj(filepath= self.obj_files_path + "/" + file[1].split('.')[0] + ".obj",
                                             axis_forward='-Z',
                                             axis_up='Y',
                                             use_mesh_modifiers=True,
                                             use_normals=True,
                                             use_materials=False,
                                             keep_vertex_order=True)
                else:
                    bpy.ops.export_mesh.ply(filepath= self.obj_files_path + "/" + file[1].split('.')[0] + ".ply",
                                            use_mesh_modifiers=True,
                                            axis_forward='-Z',
                                            axis_up='Y',
                                            use_normals=False)


if __name__ == "__main__":

    '''
        ["gargoyle2", "Cube"], ["monster", "Cube"],
        ["task02", "Cube"], ["task06", "Basemesh_FullbodyMale"],
        ["task02-alien", "Cube"], ["elder", "Plane"],
        ["elf", "Cube"], ["explorer", "Cube"],
        ["gorilla", "Cube"], ["sage", "Bust"]
    '''
    names = [["monster", "Cube"], ["ogre", "Cube"]]

    for model_name, basemesh in names:
        #blend_dir = "../blend_files/"
        #obj_dir = "../obj2_files/"
        blend_dir = "/Volumes/PART FAT/3ddata/"
        obj_dir = "/Volumes/Part Mac/obj_true_files/"

        # arrivato a snap001607
        bc = BlendConverter(blend_dir, obj_dir, model_name, 0, 50, basemesh, "obj")
        bc.convert_to_obj()
