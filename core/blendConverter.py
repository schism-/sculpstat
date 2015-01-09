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

    def subd(self):
        current_subd = -1
        changes = {}
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
                    if current_subd != bpy.data.objects[self.basemesh_name].modifiers["Multires"].sculpt_levels:
                        print("%d Changed level from %s to %s " % (idx,
                                                                   current_subd,
                                                                   bpy.data.objects[self.basemesh_name].modifiers["Multires"].sculpt_levels))
                        changes[idx] = [current_subd, bpy.data.objects[self.basemesh_name].modifiers["Multires"].sculpt_levels]
                        current_subd = bpy.data.objects[self.basemesh_name].modifiers["Multires"].sculpt_levels
                except KeyError:
                    print('MULTIRES NOT FOUND. IS IT SUBD?')
                    return None

                print()
        print(changes)
        print('---')
        return changes

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
                    print('MULTIRES NOT FOUND. IS IT SUBD?')
                    return None

                try:
                    if bpy.ops.object.mode_set.poll():
                        bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.faces_shade_smooth()
                except KeyError:
                    print('UNABLE TO SHADE SMOOTH')

                if self.mode == "obj":
                    bpy.ops.export_scene.obj(filepath= self.obj_files_path + "/" + file[1].split('.')[0] + ".obj",
                                             axis_forward='-Z',
                                             axis_up='Y',
                                             use_mesh_modifiers=True,
                                             use_normals=True,
                                             use_uvs=False,
                                             use_materials=False,
                                             keep_vertex_order=True)
                else:
                    bpy.ops.export_mesh.ply(filepath= self.obj_files_path + "/" + file[1].split('.')[0] + ".ply",
                                            use_mesh_modifiers=True,
                                            axis_forward='-Z',
                                            axis_up='Y',
                                            use_normals=False)
                print()
        print('---')


if __name__ == "__main__":
    import time

    names = [["alien", "Cube"],
             ["elder", "Plane"],
             ["elf", "Cube"],
             ["engineer", "Basemesh_FullbodyMale"],
             ["explorer", "Cube"],
             ["fighter", "Basemesh_FullbodyMale"],
             ["gargoyle", "Cube"],
             ["gorilla", "Cube"],
             ["monster", "Cube"],
             ["man", "Bust"],
             ["merman", "Cube"],
             ["monster", "Cube"],
             ["ogre", "Plane"],
             ["sage", "Bust"]]

    names = [
        ["explorer", "Cube"],
        ["fighter", "Basemesh_FullbodyMale"],
        ["gargoyle", "Cube"],
        ["gorilla", "Cube"],
        ["monster", "Cube"],
        ["man", "Bust"],
        ["merman", "Cube"],
        ["monster", "Cube"],
        ["ogre", "Plane"],
        ["sage", "Bust"]
    ]

    all_changes = {}
    for model_name, basemesh in names:
        blend_dir = "/Users/christian/Desktop/"
        blend_dir = "/Volumes/PART FAT/3ddata/"
        obj_dir = "/Users/christian/Desktop/obj_smooth_normals_files/"

        start = time.time()
        bc = BlendConverter(blend_dir, obj_dir, model_name, 0, 5000, basemesh, "obj")
        all_changes[model_name] = bc.subd()
        print("Conversion took %f seconds" % (time.time() - start))

    print(all_changes)


# alien       {0: [-1, 0], 49: [3, 4], 50: [4, 5], 4: [0, 1], 5: [1, 2], 6: [2, 3], 121: [5, 6], 908: [7, 8], 325: [6, 7]}
# elder       {0: [-1, 0], 32: [0, 1], 1395: [3, 4], 192: [1, 2], 516: [2, 3]}
# engi        {0: [-1, 0], 97: [2, 3], 4: [0, 1], 5: [1, 2], 430: [3, 4]}
# fighter:    {0: [-1, 0], 1543: [3, 4], 5: [0, 1], 6: [1, 2], 687: [2, 3]},
# merman:     {0: [-1, 0], 4: [0, 1], 5: [1, 2], 214: [4, 5], 1929: [7, 8], 412: [5, 6], 683: [6, 7], 140: [3, 4], 79: [2, 3]},
# ogre:       {0: [-1, 0], 2: [0, 1], 218: [1, 2], 463: [2, 3]},
# sage:       {0: [-1, 0], 80: [0, 1], 1267: [2, 3], 1525: [3, 4], 1798: [4, 3], 1802: [3, 4], 558: [1, 2]},
# man:        {0: [-1, 0], 16: [3, 2], 658: [2, 3], 1174: [3, 4], 9: [0, 1], 10: [1, 2], 11: [2, 3]},


