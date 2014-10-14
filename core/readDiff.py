__author__ = 'christian'

import pickle


model_name = "gargoyle2"
diff_path = "../diff_new/" + model_name + "/"

diff_no = 15

fh = open(diff_path + "diff_" + str(diff_no), 'rb')

data = pickle.load(fh)

if "mod_normals" in data:
    data['mod_normals'].sort(key=lambda x: x[1], reverse=True)
    for mn in data['mod_normals']:
        print(mn)

if "new_normals" in data:
    data['new_normals'].sort()
    for nn in data['new_normals']:
        print(nn)

print("\t--Diff stats--")
print("\t\t\t\t\t\tMOD \t\tNEW \t\tNUMBER")
print("\t\tVerts stats: \t%d, \t\t%d, \t\t%d" % (len(data["mod_vertices"]) if "mod_vertices" in data else 0,
                                                 len(data["new_vertices"]) if "new_vertices" in data else 0,
                                                 data["verts_no"] if "verts_no" in data else 0))

print("\t\tNorms stats: \t%d, \t\t%d, \t\t%d" % (len(data["mod_normals"]) if "mod_normals" in data else 0,
                                                 len(data["new_normals"]) if "new_normals" in data else 0,
                                                 data["normals_no"] if "normals_no" in data else 0))

print("\t\tFaces stats: \t%d, \t\t%d, \t\t%d" % (len(data["mod_faces"]) if "mod_faces" in data else 0,
                                                 len(data["new_faces"]) if "new_faces" in data else 0,
                                                 data["faces_no"] if "faces_no" in data else 0))

'''
    --Diff stats--
                    MOD 		NEW 		NUMBER
    Verts stats: 	0, 		1235, 		3689
    Norms stats: 	1653, 		2908, 		3701
    Faces stats: 	0, 		2596, 		3625

'''