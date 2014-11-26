import bpy
import os
import sys
import struct
import shutil
from mathutils import *
from math import *

"""
must be run within blender
exports all mesh data after applying modifiers to ply with extra comments for
- camera's orientation
- sculpting brush properties (mirroring, name)
"""


export_all = True
export_subdiv = True


def deselectall():
    set_object_mode()
    for o in bpy.data.objects:
        o.select = False
    #bpy.ops.object.select_all( action='DESELECT' )


def selectall():
    set_object_mode()
    bpy.ops.object.select_all(action='SELECT')


def selectobj(obj):
    set_object_mode()
    if type(obj) is str:
        obj = bpy.data.objects[obj]
    obj.select = True
    bpy.context.scene.objects.active = obj
    return obj


def apply_mods(obj):
    global export_subdiv
    deselectall()
    selectobj(obj)
    if len(obj.modifiers) == 0:
        return
    # make sure Multires modifier has levels set properly before applying
    for mod in obj.modifiers:
        if mod.name == 'Multires':
            if export_subdiv:
                mod.levels = mod.sculpt_levels
            else:
                mod.levels = 0
        elif mod.name == 'Subsurf':
            mod.levels = 0
            mod.render_levels = 0
    bpy.ops.object.convert(target='MESH', keep_original=False)


def triangulate(f_ngon):
    l = len(f_ngon)
    if l < 3:
        return
    elif l == 3:
        yield f_ngon
    else:
        i0 = f_ngon[0]
        for i1, i2 in zip(f_ngon[1:-1], f_ngon[2:]):
            yield (i0, i1, i2)


def writebin_f(fn, l):
    open(fn, 'wb').write(struct.pack('i%df' % len(l), len(l), *l))


def writebin_i(fn, l):
    open(fn, 'wb').write(struct.pack('i%di' % len(l), len(l), *l))


def flatten(ll):
    return [e for l in ll for e in l]


def get_camera():
    a_viewport = [a for a in bpy.data.window_managers[0].windows[0].screen.areas if a.type == 'VIEW_3D']
    if not a_viewport:
        # likely a snapshot while trying to save...  hack!
        try:
            bpy.ops.screen.back_to_previous()
        except Exception:
            try:
                bpy.context.area.type = 'VIEW_3D'
            except Exception:
                return None
        a_viewport = [a for a in bpy.data.window_managers[0].windows[0].screen.areas if a.type == 'VIEW_3D']
    a_viewport = a_viewport[0]
    r3d_viewport = a_viewport.spaces[0].region_3d

    o = r3d_viewport.view_location
    x, y, z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    x.rotate(r3d_viewport.view_rotation)
    y.rotate(r3d_viewport.view_rotation)
    z.rotate(r3d_viewport.view_rotation)

    return {
        'o': list(o),
        'x': list(x), 'y': list(y), 'z': list(z),
        'd': r3d_viewport.view_distance,
        'p': r3d_viewport.view_perspective,
        'q': r3d_viewport.view_rotation,
    }


def set_object_mode():
    while bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    return


def triple_to_string(t):
    return ','.join('%f' % e for e in t)


set_object_mode()

# turn on all layers so object xforms are applied correctly
for i in range(20):
    bpy.data.scenes[0].layers[i] = True

fn_blend = bpy.data.filepath
fn_base = fn_blend   # os.path.splitext( fn_blend )[0]
fn_ply = '%s.ply' % os.path.splitext(fn_base)[0]


if '--' in sys.argv:
    comments = open(sys.argv[sys.argv.index('--')+1],'rt').read().split('\n')
else:
    comments = []


# make all multi-user objects single
bpy.ops.object.make_single_user(type='ALL',object=True,obdata=True)

cam = get_camera()
if not cam:
    open(fn_ply, 'wt').write('skip\ncould not find camera\n')
    bpy.ops.wm.quit_blender()
    sys.exit(0)

cam = '{"o":[%s], "x":[%s], "y":[%s], "z":[%s], "d":%f, "p":%d, "q":[%s]}' % (
    triple_to_string(cam['o']),
    triple_to_string(cam['x']),
    triple_to_string(cam['y']),
    triple_to_string(cam['z']),
    cam['d'],
    0 if cam['p'] == 'ORTHO' else 1,
    triple_to_string(cam['q']),
)

aobj = bpy.context.active_object
if not aobj:
    aobj = '{"o":[0,0,0], "r":[0,0,0], "s":[1,1,1]}'
else:
    aobj = '{"o":[%s], "r":[%s], "s":[%s]}' % (
        triple_to_string(aobj.location),
        triple_to_string(list(aobj.rotation_euler)),
        triple_to_string(aobj.scale),
    )

#lobjs = bpy.data.objects
lobjs = bpy.data.scenes[0].objects

lv, lg = [], []
for i_obj, obj in enumerate(lobjs):
    if obj.type != 'MESH':
        # not a mesh
        continue

    if len(obj.data.polygons) == 0:
        # only contains edges?
        continue

    if not export_all and obj.draw_type not in ['SOLID','TEXTURED']:
        # if user isn't showing the faces, maybe we shouldn't export them either
        continue

    print(obj.name)
    obj.hide = False
    apply_mods(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh = obj.data

    lv_sz = len(lv)
    lv += [list(v.co) for v in mesh.vertices]
    #lg += [[iv + lv_sz for iv in e.vertices] for e in mesh.edges]
    lg += [[iv + lv_sz for iv in f.vertices] for f in mesh.polygons]

if bpy.context.tool_settings.sculpt and bpy.context.tool_settings.sculpt.brush:
    s_brtype = bpy.context.tool_settings.sculpt.brush.name
    b_mirror_x = 1 if bpy.context.tool_settings.sculpt.use_symmetry_x else 0
    b_mirror_y = 1 if bpy.context.tool_settings.sculpt.use_symmetry_y else 0
    b_mirror_z = 1 if bpy.context.tool_settings.sculpt.use_symmetry_z else 0
    sculpt = '{"x":%d, "y":%d, "z":%d, "name":"%s"}' % (b_mirror_x, b_mirror_y, b_mirror_z, s_brtype)
else:
    sculpt = '{}'


# write to .tmp file, then rename to final filename
# allows us to detect if blender crashed before it finished writing ply file

fp = open('%s.tmp'%fn_ply, 'wt')
fp.write('ply\n')
fp.write('format ascii 1.0\n')
fp.write('element vertex %d\n' % len(lv))
fp.write('property float x\nproperty float y\nproperty float z\n')
fp.write('element face %d\n' % len(lg))
fp.write('property list uchar int vertex_index\n')
fp.write('comment camera: %s\n' % cam)
fp.write('comment sculpt: %s\n' % sculpt)
fp.write('comment active: %s\n' % aobj)
for comment in comments:
    fp.write('comment %s\n' % comment)
fp.write('end_header\n')
for v in lv:
    fp.write('%s\n' % ' '.join('%0.25f' % e for e in v))
for g in lg:
    fp.write('%d %s\n' % (len(g), ' '.join('%d' % e for e in g)))
fp.close()

shutil.move('%s.tmp'%fn_ply, fn_ply)

bpy.ops.wm.quit_blender()