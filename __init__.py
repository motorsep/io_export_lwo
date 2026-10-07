# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# LightWave Object (.lwo) exporter for idTech 4 / Fall of Phaeton
#
# Lineage:
#   2002  Anthony D'Agostino (Scorpius) - original LWO2 writer (Blender 2.2x)
#   Gert De Roost - Blender 2.6x/2.7x port, idTech mode, VMAD UVs and colors
#   4.0.0 (2026) motorsep/Claude - Blender 4.4+ / 5.2 rewrite for idTech 4:
#     * Non-destructive: works on evaluated depsgraph copies. The scene,
#       objects, selection and edit mode are never touched.
#     * LightWave-only output dropped: image-map BLOK/CLIP, endomorphs,
#       weight maps, edge weights, NORM vmaps, subpatches, extra SURF
#       channels. The engine (idRenderModelStatic::ConvertLWOToModelSurfaces)
#       reads none of it.
#     * One layer per file. The engine only reads the FIRST layer, so every
#       selected object is merged into it. "Batch" writes one file per object.
#     * Always triangulated (the engine skips non-triangles with a warning).
#     * Vertex colors from Color Attributes (POINT or CORNER domain, byte or
#       float) as RGBA VMAP + VMAD - the only color vmap type the engine reads.
#     * Sharp edges and flat faces become SMGP smoothing-group tags, honored
#       by the engine's lwGetVertNormals. Blender's shading survives the trip
#       without any engine change.
#     * Optional "MikkT" data for the Fall of Phaeton engine: explicit corner
#       normals (NORM, dim 3) and MikkTSpace tangents (TANG, dim 4: xyz +
#       bitangent sign) as VMAP/VMAD pairs. Stock idTech 4 parses unknown
#       vertex maps and ignores them, so the file stays loadable there.
#   4.4.0 (2026-10) motorsep/Claude - "Export animation frames": one LWO per
#       frame of the chosen Action(s), for the Fall of Phaeton mesh flipbook
#       compiler (buildMeshFlipbook). Keyframed transforms, shape keys and
#       armature deformation all come through the evaluated depsgraph, so the
#       same frame files work for any of them. Files are named
#       <object>_<00001>.lwo (optionally <object>_<action>_<00001>.lwo); the
#       frames are numbered contiguously so a .flipdef can name them as a
#       sourceRangeStart / sourceRangeEnd pair.

bl_info = {
    "name": "Export idTech 4 LWO (.lwo)",
    "author": "Anthony D'Agostino (Scorpius), Gert De Roost, motorsep/Claude",
    "version": (4, 4, 0),
    "blender": (4, 4, 0),
    "location": "File > Export > idTech 4 LWO (.lwo)",
    "description": "Export static meshes as LightWave LWO2 for idTech 4 engines",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export",
}

import math
import os
import struct
import time
from io import BytesIO

import bpy
import bmesh
import mathutils
from bpy_extras.io_utils import ExportHelper
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, StringProperty)


# =============================================================================
# LWO2 binary helpers
#
# LWO2 is an IFF container: big-endian, 4CC chunk ids, U4 chunk sizes, and
# every chunk padded to an even length (the pad byte is NOT counted in the
# size). SURF sub-chunks use U2 sizes. VX is a variable-length index: 2 bytes
# below 0xFF00, otherwise 4 bytes with the top byte set to 0xFF.
# =============================================================================

def lwo_string(s):
    """S0: NUL-terminated string padded to an even byte count."""
    b = s.encode('utf-8') + b'\0'
    if len(b) & 1:
        b += b'\0'
    return b


def vx(index):
    if index < 0xFF00:
        return struct.pack('>H', index)
    return struct.pack('>I', index | 0xFF000000)


def chunk(cid, payload):
    pad = b'\0' if len(payload) & 1 else b''
    return cid + struct.pack('>I', len(payload)) + payload + pad


def subchunk(cid, payload):
    pad = b'\0' if len(payload) & 1 else b''
    return cid + struct.pack('>H', len(payload)) + payload + pad


def vmap_chunk(cid, vtype, dim, name, records):
    """VMAP (records = (point, values...)) or VMAD (records = (point, poly, values...))."""
    out = BytesIO()
    out.write(vtype)
    out.write(struct.pack('>H', dim))
    out.write(lwo_string(name))
    fmt = '>%df' % dim
    if cid == b'VMAD':
        for rec in records:
            out.write(vx(rec[0]))
            out.write(vx(rec[1]))
            out.write(struct.pack(fmt, *rec[2:]))
    else:
        for rec in records:
            out.write(vx(rec[0]))
            out.write(struct.pack(fmt, *rec[1:]))
    return chunk(cid, out.getvalue())


# =============================================================================
# Smoothing groups (non-destructive, bmesh-based)
#
# The engine averages a vertex normal only across polygons that share the
# same smoothing group AND whose face normals are within the surface's SMAN
# angle. Flood-filling faces across non-sharp edges reproduces Blender's
# sharp-edge shading; a flat face gets a group of its own so nothing is
# averaged into it.
# =============================================================================

def compute_smoothing_groups(bm, first_group):
    """Return (dict face.index -> group id, next free group id)."""
    groups = {}
    gid = first_group
    for face in bm.faces:
        if face.index in groups:
            continue
        if not face.smooth:
            groups[face.index] = gid
            gid += 1
            continue
        stack = [face]
        while stack:
            f = stack.pop()
            if f.index in groups:
                continue
            groups[f.index] = gid
            for edge in f.edges:
                if not edge.smooth:
                    continue
                for linked in edge.link_faces:
                    if linked.smooth and linked.index not in groups:
                        stack.append(linked)
        gid += 1
    return groups, gid


def bmesh_needs_smoothing_groups(bm):
    has_flat = False
    has_smooth = False
    for f in bm.faces:
        if f.smooth:
            has_smooth = True
        else:
            has_flat = True
        if has_flat and has_smooth:
            return True
    if has_smooth:
        for e in bm.edges:
            if not e.smooth and len(e.link_faces) > 1:
                return True
    return False


# =============================================================================
# MultiUV: which UV map does a material sample?
#
# Blender's own answer is the UV Map node in the material's node tree, which
# is also what makes the viewport preview the right map. Prefer the node
# wired into an Image Texture's Vector input; otherwise any UV Map node.
# =============================================================================

def material_uv_map_name(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.type != 'TEX_IMAGE':
            continue
        vec = node.inputs.get('Vector')
        if vec is not None and vec.is_linked:
            src = vec.links[0].from_node
            if src.type == 'UVMAP' and src.uv_map:
                return src.uv_map
    for node in nodes:
        if node.type == 'UVMAP' and node.uv_map:
            return node.uv_map
    return None


# =============================================================================
# Builder
# =============================================================================

class LWOBuilder:
    """Accumulates one LWO2 layer from any number of mesh objects.

    Point and polygon indices are global across objects: the engine applies
    its per-chunk offsets to POLS but not to VMAP/VMAD/PTAG indices, so one
    PNTS and one POLS chunk with absolute indices is the only layout that
    stays consistent on the engine side.
    """

    UV_EPS = 1e-6
    COLOR_EPS = 0.5 / 255.0
    NORMAL_EPS = 1e-4

    # Vertex map names. The engine matches on vmap TYPE, not name.
    NORMAL_MAP_NAME = 'vert_normals'
    TANGENT_MAP_NAME = 'mikkt_tangents'
    ORIG_LOOP_LAYER = 'lwo_orig_loop'

    def __init__(self, context, options):
        self.context = context
        self.options = options
        self.warnings = []
        self.reset()

    def reset(self):
        self.points = []          # (x, y, z) in LW space, scaled
        self.polys = []           # (a, b, c) global point indices, LW winding
        self.poly_surf = []       # tag index per polygon
        self.poly_smgp = []       # smoothing group per polygon (0 = untagged)
        self.uv_maps = {}         # map name -> {'vmap': [(point, u, v)], 'vmad': [(point, poly, u, v)]}
        self.color_vmap = []      # (point, r, g, b, a)
        self.color_vmad = []      # (point, poly, r, g, b, a)
        self.normal_vmap = []     # (point, nx, ny, nz) LW space
        self.normal_vmad = []     # (point, poly, nx, ny, nz)
        self.tangent_vmap = []    # (point, tx, ty, tz, sign) LW space
        self.tangent_vmad = []    # (point, poly, tx, ty, tz, sign)
        self.tags = []            # material names, then smoothing group names
        self.tag_index = {}
        self.surface_has_smooth = {}   # material name -> any smooth-shaded face
        self.next_group = 1
        self.uv_name = None
        self.color_name = None

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def build(self, objects, layer_name):
        self.reset()
        for obj in objects:
            self._append_object(obj)
        if not self.polys:
            raise RuntimeError('Nothing to export: selected meshes have no faces')
        return self._serialize(layer_name)

    # -------------------------------------------------------------------------
    # Mesh preparation
    # -------------------------------------------------------------------------

    def _transform_matrix(self, obj):
        """World-space bake matrix from the enabled transform options, times
        the export scale. Fully non-destructive (nothing is applied)."""
        mat = mathutils.Matrix.Identity(4)
        loc, rot, scl = obj.matrix_world.decompose()
        if self.options['apply_scale']:
            mat = mathutils.Matrix.Diagonal((*scl, 1.0)) @ mat
        if self.options['apply_rotation']:
            mat = rot.to_matrix().to_4x4() @ mat
        if self.options['apply_location']:
            mat = mathutils.Matrix.Translation(loc) @ mat
        s = self.options['scale']
        return mathutils.Matrix.Diagonal((s, s, s, 1.0)) @ mat

    def _prepare_mesh(self, obj):
        """Return a temporary Mesh: modifiers evaluated (optional), transform
        baked, doubles removed (optional), triangulated. Caller removes it."""
        if obj.mode == 'EDIT':
            obj.update_from_editmode()

        if self.options['apply_modifiers']:
            depsgraph = self.context.evaluated_depsgraph_get()
            source = obj.evaluated_get(depsgraph)
            mesh = bpy.data.meshes.new_from_object(
                source, preserve_all_data_layers=True, depsgraph=depsgraph)
        else:
            mesh = bpy.data.meshes.new_from_object(obj)

        # MikkT: corner normals are captured from the untouched source mesh,
        # BEFORE any topology change. Blender stores custom split normals
        # relative to per-vertex fan spaces, and bmesh remove_doubles /
        # triangulate change those fans without re-encoding (measured: up to
        # 17 degrees of drift on a cube). An int corner layer carries each
        # final corner back to its source corner so the captured normals can
        # be re-applied to the triangulated mesh, where MikkTSpace is then
        # computed (Blender only computes tangents for tris/quads).
        capture = None
        if self.options['mikkt']:
            capture = {
                'vertex': [l.vertex_index for l in mesh.loops],
                'positions': [mathutils.Vector(v.co) for v in mesh.vertices],
                'normals': [mathutils.Vector(c.vector) for c in mesh.corner_normals],
            }

        bm = bmesh.new()
        bm.from_mesh(mesh)

        if capture is not None:
            orig_layer = bm.loops.layers.int.new(self.ORIG_LOOP_LAYER)
            for face in bm.faces:
                for loop in face.loops:
                    loop[orig_layer] = loop.index

        if self.options['remove_doubles']:
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)

        bmesh.ops.triangulate(bm, faces=bm.faces[:])

        # Drop loose vertices and edges: the engine never references them and
        # LW readers choke on 1- and 2-point polygons.
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')

        bm.verts.index_update()
        bm.faces.index_update()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        smoothing = None
        if self.options['smoothing_groups'] and bmesh_needs_smoothing_groups(bm):
            smoothing, self.next_group = compute_smoothing_groups(bm, self.next_group)

        bm.to_mesh(mesh)
        bm.free()

        # Which UV map each material slot samples (MultiUV) or the render
        # map for everything.
        uv_layers = self._resolve_uv_layers(obj, mesh)

        if capture is not None:
            self._finish_capture(obj, mesh, capture, uv_layers)

        # Bake the object transform last, on the final topology. A mirroring
        # transform flips the winding; the engine derives face normals from
        # winding, so the writer emits those polygons in reverse order.
        xform = self._transform_matrix(obj)
        mesh.transform(xform)
        flip = xform.determinant() < 0.0
        if capture is not None:
            m3 = xform.to_3x3()
            capture['normal_xform'] = m3.inverted_safe().transposed()
            capture['tangent_xform'] = m3
            # Mirroring flips the handedness of every tangent frame; the
            # bitangent sign records exactly that.
            capture['sign_flip'] = -1.0 if flip else 1.0

        return mesh, smoothing, capture, flip, uv_layers

    def _resolve_uv_layers(self, obj, mesh):
        """{material_index: uv_layer or None} for every slot used by a face.

        MultiUV off: the object's render UV map for every slot. MultiUV on:
        the map named by the material's UV Map node, falling back to the
        render map (with a warning) when the object has no map of that name.
        """
        default = self._pick_uv_layer(mesh)
        result = {}
        for mi in set(p.material_index for p in mesh.polygons):
            layer = default
            if self.options['multiuv']:
                slots = obj.material_slots
                mat = slots[mi].material if mi < len(slots) else None
                wanted = material_uv_map_name(mat)
                if wanted:
                    found = mesh.uv_layers.get(wanted)
                    if found is None:
                        self.warnings.append(
                            'Material "%s" samples UV map "%s" but object "%s" has no such map; '
                            'using "%s"' % (mat.name, wanted, obj.name,
                                            default.name if default else 'none'))
                    else:
                        layer = found
            result[mi] = layer
        return result

    def _finish_capture(self, obj, mesh, capture, uv_layers):
        """On the triangulated object-space mesh: re-apply the captured
        source normals as custom normals, compute MikkTSpace against them,
        and store per final corner: normal, tangent, bitangent sign."""
        attr = mesh.attributes.get(self.ORIG_LOOP_LAYER)
        if attr is None:
            raise RuntimeError('Internal error: source corner mapping missing')
        orig_of = attr.data
        loops = mesh.loops
        verts = mesh.vertices
        src_vertex = capture['vertex']
        src_pos = capture['positions']
        src_normals = capture['normals']

        normals = []
        for li in range(len(loops)):
            src = orig_of[li].value
            # Guard: the source corner must still sit on this vertex
            # (remove_doubles tolerance allowed).
            if (src_pos[src_vertex[src]] - verts[loops[li].vertex_index].co).length > 1e-3:
                raise RuntimeError(
                    'Internal error: corner mapping lost on object "%s"' % obj.name)
            normals.append(src_normals[src])
        mesh.normals_split_custom_set(normals)
        mesh.attributes.remove(attr)

        # Read the normals back from the mesh so they are exactly what
        # MikkTSpace sees (custom normals are quantized on storage).
        capture['normals'] = [mathutils.Vector(c.vector) for c in mesh.corner_normals]

        # MikkTSpace per UV map actually sampled: each polygon's corners take
        # the tangents computed against the map its material uses. Corners
        # without a map stay None and get no TANG entry.
        num_loops = len(mesh.loops)
        tangents = [None] * num_loops
        signs = [None] * num_loops
        layer_names = []
        for layer in uv_layers.values():
            if layer is not None and layer.name not in layer_names:
                layer_names.append(layer.name)
        if not layer_names:
            self.warnings.append(
                'Object "%s" has no UV map; MikkT tangents skipped, normals still exported'
                % obj.name)
        for lname in layer_names:
            # Precomputed frames win: a mesh that is one piece of a larger
            # continuous surface (terrain chunks) carries MikkT computed on
            # the WHOLE surface in corner attributes mikkt_tangent.<uv> /
            # mikkt_sign.<uv>. Recomputing here would average each border
            # vertex over this piece's faces only and the neighbouring piece
            # would disagree, showing a seam under normal mapping.
            pre_t = mesh.attributes.get('mikkt_tangent.' + lname)
            pre_s = mesh.attributes.get('mikkt_sign.' + lname)
            precomputed = (pre_t is not None and pre_s is not None
                           and pre_t.domain == 'CORNER' and pre_s.domain == 'CORNER'
                           and pre_t.data_type == 'FLOAT_VECTOR' and pre_s.data_type == 'FLOAT')
            if precomputed:
                print('LWO Export: "%s" uses precomputed MikkT frames for UV map "%s"' % (obj.name, lname))
            else:
                mesh.calc_tangents(uvmap=lname)
            try:
                loops = mesh.loops
                for poly in mesh.polygons:
                    layer = uv_layers.get(poly.material_index)
                    if layer is None or layer.name != lname:
                        continue
                    ls = poly.loop_start
                    for li in range(ls, ls + poly.loop_total):
                        if precomputed:
                            tangents[li] = mathutils.Vector(pre_t.data[li].vector)
                            signs[li] = pre_s.data[li].value
                        else:
                            tangents[li] = mathutils.Vector(loops[li].tangent)
                            signs[li] = loops[li].bitangent_sign
            finally:
                if not precomputed:
                    mesh.free_tangents()
        capture['tangents'] = tangents
        capture['signs'] = signs

    # -------------------------------------------------------------------------
    # Accumulation
    # -------------------------------------------------------------------------

    def _surface_tag(self, obj, material_index):
        slots = obj.material_slots
        mat = slots[material_index].material if material_index < len(slots) else None
        if mat is None:
            raise RuntimeError(
                'Object "%s" has faces without a material. Every face needs a '
                'material whose name is the engine material (e.g. models/props/crate)'
                % obj.name)
        name = mat.name
        idx = self.tag_index.get(name)
        if idx is None:
            idx = len(self.tags)
            self.tags.append(name)
            self.tag_index[name] = idx
            self.surface_has_smooth[name] = False
        return idx, name

    def _pick_uv_layer(self, mesh):
        layers = mesh.uv_layers
        if not layers:
            return None
        for layer in layers:
            if layer.active_render:
                return layer
        return layers.active or layers[0]

    def _pick_color_attribute(self, mesh):
        attrs = mesh.color_attributes
        if not attrs:
            return None
        idx = attrs.render_color_index
        if 0 <= idx < len(attrs):
            return attrs[idx]
        return attrs.active_color or attrs[0]

    def _append_object(self, obj):
        mesh, smoothing, capture, flip, uv_layers = self._prepare_mesh(obj)
        try:
            self._append_mesh(obj, mesh, smoothing, capture, flip, uv_layers)
        finally:
            bpy.data.meshes.remove(mesh)

    def _append_mesh(self, obj, mesh, smoothing, capture, flip, uv_layers):
        base_point = len(self.points)
        base_poly = len(self.polys)

        # Points: Blender is right-handed Z-up, LightWave left-handed Y-up.
        # Swapping Y and Z is the historic Blender->LWO convention the engine
        # undoes on load (ConvertLWOToModelSurfaces reads pos[0], pos[2], pos[1]).
        for v in mesh.vertices:
            x, y, z = v.co
            self.points.append((x, z, y))

        # Polygons: reversed winding to compensate the axis swap (a reflection),
        # reversed once more for mirrored object transforms. The surface tag
        # maps the face to its material; a per-material "all flat" check
        # decides SMAN later.
        loops = mesh.loops
        for poly in mesh.polygons:
            if poly.loop_total != 3:
                raise RuntimeError('Internal error: non-triangle after triangulation')
            ls = poly.loop_start
            a = loops[ls].vertex_index + base_point
            b = loops[ls + 1].vertex_index + base_point
            c = loops[ls + 2].vertex_index + base_point
            self.polys.append((a, b, c) if flip else (c, b, a))
            tag, name = self._surface_tag(obj, poly.material_index)
            self.poly_surf.append(tag)
            if poly.use_smooth:
                self.surface_has_smooth[name] = True
            group = smoothing.get(poly.index, 0) if smoothing else 0
            self.poly_smgp.append(group)

        # UVs, one map per material (MultiUV) or the render map for all.
        self._append_uvs(obj, mesh, uv_layers, base_point, base_poly)

        # Vertex colors: same VMAP + VMAD split, from the render color attribute.
        if self.options['vertex_colors']:
            attr = self._pick_color_attribute(mesh)
            if attr is not None:
                self._append_colors(mesh, attr, base_point, base_poly)

        # Explicit normals + MikkTSpace tangents (Fall of Phaeton engine only).
        if capture is not None:
            self._append_normals_and_tangents(obj, mesh, capture, base_point, base_poly)

    def _append_normals_and_tangents(self, obj, mesh, capture, base_point, base_poly):
        """Write the captured per-corner normals and tangents, transformed
        into export space.

        Normals go through the inverse-transpose, tangents through the
        matrix itself (keeps them perpendicular under non-uniform scale), the
        bitangent sign flips under mirroring. Axes are then swapped like
        positions; the sign is valid once the engine un-swaps them.
        """
        loops = mesh.loops
        eps = self.NORMAL_EPS

        n_xform = capture['normal_xform']
        t_xform = capture['tangent_xform']
        sign_flip = capture['sign_flip']
        src_normals = capture['normals']
        src_tangents = capture['tangents']
        src_signs = capture['signs']

        base_normal = {}
        base_tangent = {}
        for poly in mesh.polygons:
            ls = poly.loop_start
            for li in range(ls, ls + poly.loop_total):
                vi = loops[li].vertex_index
                src = li

                n = (n_xform @ src_normals[src]).normalized()
                rec = (n.x, n.z, n.y)
                known = base_normal.get(vi)
                if known is None:
                    base_normal[vi] = rec
                    self.normal_vmap.append((vi + base_point,) + rec)
                elif max(abs(known[k] - rec[k]) for k in range(3)) > eps:
                    self.normal_vmad.append((vi + base_point, poly.index + base_poly) + rec)

                t_src = src_tangents[src]
                if t_src is None:
                    continue
                t = t_xform @ t_src
                # Re-orthogonalize against the transformed normal so the
                # engine's cross(normal, tangent) bitangent stays exact.
                t = (t - n * n.dot(t)).normalized()
                # The engine's LWO loader inverts t on load (1 - v for every
                # TXUV value, Model.cpp), which mirrors texture space and with
                # it the bitangent. MikkT was computed in Blender's unflipped
                # UV space, so the handedness must be negated to describe the
                # same frame in the engine's space (the y/z axis swap is
                # undone symmetrically by the loader and does not change it).
                # Same fix as the ASE exporter 3.7.1, where it was verified
                # numerically against R_DeriveTangents.
                rec = (t.x, t.z, t.y, -src_signs[src] * sign_flip)
                known = base_tangent.get(vi)
                if known is None:
                    base_tangent[vi] = rec
                    self.tangent_vmap.append((vi + base_point,) + rec)
                elif max(abs(known[k] - rec[k]) for k in range(4)) > eps:
                    self.tangent_vmad.append((vi + base_point, poly.index + base_poly) + rec)

    def _append_uvs(self, obj, mesh, uv_layers, base_point, base_poly):
        """The vertex's first corner sets the VMAP value; corners that
        disagree (seams) get a per-polygon VMAD override, which the engine
        applies after the VMAP value.

        With MultiUV, polygons of different materials can sample different
        maps, written as separate sparse TXUV maps (the LightWave way). The
        engine concatenates all TXUV maps and keeps, per corner, the point
        entry then any polygon entry, so a vertex on the border between two
        maps must not carry a point entry in either: it gets polygon
        entries only, one per corner in the map that corner belongs to.
        """
        loops = mesh.loops
        eps = self.UV_EPS
        multi = self.options['multiuv']

        per_map = {}          # key -> {'base': {vi: (u, v)}, 'corners': [(vi, pi, u, v)]}
        missing = False
        for poly in mesh.polygons:
            layer = uv_layers.get(poly.material_index)
            if layer is None:
                missing = True
                continue
            if multi:
                key = layer.name
            else:
                if self.uv_name is None:
                    self.uv_name = layer.name
                key = self.uv_name
            entry = per_map.setdefault(key, {'base': {}, 'corners': []})
            data = layer.data
            ls = poly.loop_start
            for li in range(ls, ls + poly.loop_total):
                vi = loops[li].vertex_index
                u, v = data[li].uv
                entry['corners'].append((vi, poly.index, u, v))
                entry['base'].setdefault(vi, (u, v))

        if missing:
            self.warnings.append(
                'Object "%s" has faces without a UV map; the engine will warn about missing uv data'
                % obj.name)

        vertex_maps = {}
        for key, entry in per_map.items():
            for vi in entry['base']:
                vertex_maps.setdefault(vi, set()).add(key)

        for key, entry in per_map.items():
            store = self.uv_maps.setdefault(key, {'vmap': [], 'vmad': []})
            base = entry['base']
            for vi, (u, v) in base.items():
                if len(vertex_maps[vi]) == 1:
                    store['vmap'].append((vi + base_point, u, v))
            for vi, pi, u, v in entry['corners']:
                if len(vertex_maps[vi]) > 1:
                    store['vmad'].append((vi + base_point, pi + base_poly, u, v))
                    continue
                bu, bv = base[vi]
                if abs(bu - u) > eps or abs(bv - v) > eps:
                    store['vmad'].append((vi + base_point, pi + base_poly, u, v))

    def _append_colors(self, mesh, attr, base_point, base_poly):
        srgb = self.options['color_space'] == 'SRGB'
        if self.color_name is None:
            self.color_name = attr.name

        def read(item):
            c = item.color_srgb if srgb else item.color
            return (c[0], c[1], c[2], c[3])

        data = attr.data
        if attr.domain == 'POINT':
            for v in mesh.vertices:
                self.color_vmap.append((v.index + base_point,) + read(data[v.index]))
            return

        if attr.domain != 'CORNER':
            self.warnings.append(
                'Color attribute "%s" is on the %s domain; only Vertex and Face Corner '
                'colors are exported' % (attr.name, attr.domain))
            return

        loops = mesh.loops
        base_color = {}
        eps = self.COLOR_EPS
        for poly in mesh.polygons:
            ls = poly.loop_start
            for li in range(ls, ls + poly.loop_total):
                vi = loops[li].vertex_index
                col = read(data[li])
                known = base_color.get(vi)
                if known is None:
                    base_color[vi] = col
                    self.color_vmap.append((vi + base_point,) + col)
                elif max(abs(known[k] - col[k]) for k in range(4)) > eps:
                    self.color_vmad.append((vi + base_point, poly.index + base_poly) + col)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def _serialize(self, layer_name):
        opts = self.options
        num_materials = len(self.tags)

        # Smoothing-group tags share the TAGS list with materials. A polygon
        # whose whole surface is flat is handled by SMAN = 0 instead and gets
        # no SMGP entry.
        smgp_records = []
        group_tag = {}
        for pi, group in enumerate(self.poly_smgp):
            if group == 0:
                continue
            if not self.surface_has_smooth[self.tags[self.poly_surf[pi]]]:
                continue
            tag = group_tag.get(group)
            if tag is None:
                tag = len(self.tags)
                self.tags.append('smooth_%d' % group)
                group_tag[group] = tag
            smgp_records.append((pi, tag))

        body = BytesIO()

        # TAGS
        body.write(chunk(b'TAGS', b''.join(lwo_string(t) for t in self.tags)))

        # LAYR: number, flags, pivot, name. Pivot is unused by the engine.
        layr = struct.pack('>HH', 0, 0) + struct.pack('>fff', 0.0, 0.0, 0.0)
        layr += lwo_string(layer_name)
        body.write(chunk(b'LAYR', layr))

        # PNTS
        pnts = BytesIO()
        for p in self.points:
            pnts.write(struct.pack('>fff', *p))
        body.write(chunk(b'PNTS', pnts.getvalue()))

        # BBOX
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        zs = [p[2] for p in self.points]
        body.write(chunk(b'BBOX', struct.pack(
            '>6f', min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))))

        # Per-point maps
        for uv_name, uv_map in self.uv_maps.items():
            if uv_map['vmap']:
                body.write(vmap_chunk(b'VMAP', b'TXUV', 2, uv_name, uv_map['vmap']))
        if self.color_vmap:
            body.write(vmap_chunk(b'VMAP', b'RGBA', 4, self.color_name, self.color_vmap))
        if self.normal_vmap:
            body.write(vmap_chunk(b'VMAP', b'NORM', 3, self.NORMAL_MAP_NAME, self.normal_vmap))
        if self.tangent_vmap:
            body.write(vmap_chunk(b'VMAP', b'TANG', 4, self.TANGENT_MAP_NAME, self.tangent_vmap))

        # POLS
        pols = BytesIO()
        pols.write(b'FACE')
        for a, b, c in self.polys:
            pols.write(struct.pack('>H', 3))
            pols.write(vx(a))
            pols.write(vx(b))
            pols.write(vx(c))
        body.write(chunk(b'POLS', pols.getvalue()))

        # Per-polygon-vertex overrides (must follow POLS)
        for uv_name, uv_map in self.uv_maps.items():
            if uv_map['vmad']:
                body.write(vmap_chunk(b'VMAD', b'TXUV', 2, uv_name, uv_map['vmad']))
        if self.color_vmad:
            body.write(vmap_chunk(b'VMAD', b'RGBA', 4, self.color_name, self.color_vmad))
        if self.normal_vmad:
            body.write(vmap_chunk(b'VMAD', b'NORM', 3, self.NORMAL_MAP_NAME, self.normal_vmad))
        if self.tangent_vmad:
            body.write(vmap_chunk(b'VMAD', b'TANG', 4, self.TANGENT_MAP_NAME, self.tangent_vmad))

        # PTAG SURF: every polygon -> material tag
        ptag = BytesIO()
        ptag.write(b'SURF')
        for pi, tag in enumerate(self.poly_surf):
            ptag.write(vx(pi))
            ptag.write(vx(tag))
        body.write(chunk(b'PTAG', ptag.getvalue()))

        # PTAG SMGP: smoothing groups
        if smgp_records:
            ptag = BytesIO()
            ptag.write(b'SMGP')
            for pi, tag in smgp_records:
                ptag.write(vx(pi))
                ptag.write(vx(tag))
            body.write(chunk(b'PTAG', ptag.getvalue()))

        # SURF: one per material. COLR is the engine's fallback vertex color
        # when no RGBA map is present, so it must be white. SMAN is the
        # smoothing angle in radians; 0 means flat.
        smooth_angle = math.radians(opts['smoothing_angle'])
        for name in self.tags[:num_materials]:
            surf = BytesIO()
            surf.write(lwo_string(name))
            surf.write(lwo_string(''))
            surf.write(subchunk(b'COLR', struct.pack('>fff', 1.0, 1.0, 1.0) + vx(0)))
            sman = smooth_angle if self.surface_has_smooth[name] else 0.0
            surf.write(subchunk(b'SMAN', struct.pack('>f', sman)))
            body.write(chunk(b'SURF', surf.getvalue()))

        data = body.getvalue()
        return b'FORM' + struct.pack('>I', len(data) + 4) + b'LWO2' + data

    def summary(self):
        s = '%d points, %d triangles, %d surfaces, %d UV map(s)' % (
            len(self.points), len(self.polys), len(self.surface_has_smooth), len(self.uv_maps))
        if self.normal_vmap:
            s += ', MikkT: %d normals, %d tangents' % (
                len(self.normal_vmap) + len(self.normal_vmad),
                len(self.tangent_vmap) + len(self.tangent_vmad))
        return s


# =============================================================================
# Animation frames
#
# One LWO per frame of an Action. Nothing special is done for shape keys,
# armatures or keyed transforms: the scene is stepped with frame_set and the
# builder reads the evaluated depsgraph as it always does, so whatever
# Blender shows on that frame is what gets written. The only state touched is
# the active action/slot of the IDs the action drives, and the current frame;
# both are restored afterwards.
#
# Slotted actions (Blender 4.4+): Action.fcurves is gone in 5.2, curves live
# in layers -> strips -> channelbags, and an action evaluates to NOTHING on an
# ID until a slot is bound to it. Same handling as the MD5 exporter.
# =============================================================================

def iter_action_fcurves(action):
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        for fc in fcurves:
            yield fc
        return
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            channelbags = getattr(strip, "channelbags", None)
            if channelbags is None:
                continue
            for bag in channelbags:
                for fc in bag.fcurves:
                    yield fc


def action_target_kinds(action):
    """What the action drives: a set of 'POSE' (armature pose curves), 'KEY'
    (shape key values) and 'OBJECT' (object transforms / properties). A
    slotted action can hold several at once (Blender 5 files an object's
    shape key keys and its own keys in one action with two slots)."""
    kinds = set()
    slots = getattr(action, "slots", None)
    if slots:
        for s in slots:
            t = getattr(s, "target_id_type", 'OBJECT')
            if t == 'KEY':
                kinds.add('KEY')
            elif t == 'OBJECT':
                kinds.add('OBJECT')
    for fc in iter_action_fcurves(action):
        if fc.data_path.startswith("pose.bones"):
            kinds.discard('OBJECT')
            kinds.add('POSE')
        elif fc.data_path.startswith("key_blocks"):
            kinds.add('KEY')
        elif not slots:
            kinds.add('OBJECT')
    if not kinds:
        kinds.add('OBJECT')
    return kinds


def action_frame_range(action):
    start, end = action.frame_range
    return int(math.floor(start)), int(math.ceil(end))


def action_keyed_frames(action):
    """Sorted distinct frames that carry a key on any F-Curve of the action."""
    frames = set()
    for fc in iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            frames.add(int(round(kp.co[0])))
    return sorted(frames)


def assign_action(anim_data, action, want_type, slot=None):
    """Bind an action (and a slot of the wanted target type) to an AnimData."""
    anim_data.action = action
    if action is None or not hasattr(anim_data, "action_slot"):
        return
    if slot is not None:
        try:
            anim_data.action_slot = slot
            return
        except Exception as e:
            print("LWO Export: could not restore action slot: %s" % e)
    if anim_data.action_slot is not None:
        return
    slots = getattr(action, "slots", None)
    if not slots:
        return
    wanted = [s for s in slots if getattr(s, "target_id_type", 'OBJECT') == want_type]
    chosen = wanted[0] if wanted else slots[0]
    try:
        anim_data.action_slot = chosen
    except Exception as e:
        print("LWO Export: WARNING could not bind slot for '%s': %s" % (action.name, e))


def action_targets(action, mesh_objects):
    """[(id, slot_type)] for everything in the selection this action can
    drive; slot_type is what assign_action should bind ('OBJECT' or 'KEY')."""
    kinds = action_target_kinds(action)
    targets = []

    def add(ident, slot_type):
        if all(t[0] != ident for t in targets):
            targets.append((ident, slot_type))

    for obj in mesh_objects:
        if 'KEY' in kinds and obj.data.shape_keys is not None:
            add(obj.data.shape_keys, 'KEY')
        if 'POSE' in kinds:
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object is not None:
                    add(mod.object, 'OBJECT')
        if 'OBJECT' in kinds:
            add(obj, 'OBJECT')
    return kinds, targets


def safe_name(name):
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in '-_') else '_')
    return ''.join(out)


class LWOActionItem(bpy.types.PropertyGroup):
    export_action: BoolProperty(default=False, name="")


class LWO_UL_ActionsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "export_action", text=item.name)
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.prop(item, "export_action", text="")


class EXPORT_OT_idtech_lwo_select_actions(bpy.types.Operator):
    """(De-)Select all actions or invert the selection for export"""
    bl_idname = "export_scene.idtech_lwo_select_actions"
    bl_label = "Select actions"

    action: EnumProperty(
        items=(("SELECT", "Select all", ""),
               ("DESELECT", "Deselect all", ""),
               ("INVERT", "Invert selection", "")),
        default="SELECT")

    def execute(self, context):
        op = context.active_operator
        if op is None or not hasattr(op, "anim_actions"):
            return {'CANCELLED'}
        for a in op.anim_actions:
            if self.action == "DESELECT":
                a.export_action = False
            elif self.action == "INVERT":
                a.export_action = not a.export_action
            else:
                a.export_action = True
        return {'FINISHED'}


# =============================================================================
# Operator
# =============================================================================

class EXPORT_OT_idtech_lwo(bpy.types.Operator, ExportHelper):
    """Export selected meshes as a LightWave LWO2 object for idTech 4"""
    bl_idname = "export_scene.idtech_lwo"
    bl_label = "Export LWO"
    bl_options = {'PRESET'}
    filename_ext = ".lwo"
    filter_glob: StringProperty(default="*.lwo", options={'HIDDEN'})

    filepath: StringProperty(
        name="File Path",
        description="Output file path",
        maxlen=1024,
        default="",
        subtype='FILE_PATH',
    )

    # -- Essentials --

    option_apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Export the evaluated mesh (modifiers applied). "
                    "The mesh is always triangulated for the engine",
        default=True,
    )

    option_remove_doubles: BoolProperty(
        name="Remove Doubles",
        description="Merge coincident vertices (0.0001) before export. The engine "
                    "smooths normals across shared points only, so split "
                    "vertices along UV seams would otherwise shade as hard edges",
        default=True,
    )

    option_smoothing_groups: BoolProperty(
        name="Smoothing Groups From Sharp Edges",
        description="Write smoothing-group tags derived from sharp edges and "
                    "flat-shaded faces (including those set by a Smooth by Angle "
                    "modifier), so the engine reproduces Blender's shading",
        default=True,
    )

    option_smoothing_angle: FloatProperty(
        name="Smoothing Angle",
        description="Maximum angle in degrees between faces that still share a "
                    "smooth normal in the engine (SMAN). With smoothing groups on, "
                    "180 lets the sharp edges alone decide",
        min=0.0,
        max=180.0,
        default=180.0,
    )

    option_mikkt: BoolProperty(
        name="MikkT (Fall of Phaeton engine)",
        description="Also write explicit corner normals (NORM) and MikkTSpace "
                    "tangents (TANG) vertex maps. Only the Fall of Phaeton engine "
                    "reads them; stock idTech 4 ignores the extra maps and shades "
                    "from the smoothing data as usual, so the file stays classic-compatible",
        default=False,
    )

    option_multiuv: BoolProperty(
        name="MultiUV (per-material UV maps)",
        description="Each material samples the UV map named by the UV Map node in "
                    "its node tree; polygons are written with that map, as separate "
                    "sparse TXUV maps like LightWave does. Materials without a UV Map "
                    "node use the object's render UV map. See MULTIUV.md",
        default=False,
    )

    # -- Vertex colors --

    option_vertex_colors: BoolProperty(
        name="Vertex Colors",
        description="Export the render color attribute (camera icon in the Color "
                    "Attributes list) as an RGBA vertex map",
        default=True,
    )

    option_color_space: EnumProperty(
        name="Color Space",
        description="How to encode vertex colors in the file",
        items=(
            ('SRGB', "sRGB (as displayed)",
             "Write the colors as the viewport shows them. Matches the classic "
             "idTech 4 pipeline where vertex colors multiply display-space textures"),
            ('LINEAR', "Linear",
             "Write Blender's internal scene-linear values"),
        ),
        default='SRGB',
    )

    # -- Transformations --

    option_apply_scale: BoolProperty(
        name="Apply Scale",
        description="Bake object scale into vertex positions",
        default=True,
    )

    option_apply_rotation: BoolProperty(
        name="Apply Rotation",
        description="Bake object rotation into vertex positions",
        default=True,
    )

    option_apply_location: BoolProperty(
        name="Apply Location",
        description="Bake object location into vertex positions",
        default=True,
    )

    # -- Advanced --

    option_scale: FloatProperty(
        name="Scale",
        description="Multiply all vertex positions by this factor. "
                    "idTech 4 uses roughly 1 unit = 1 inch. "
                    "Default 1.0 exports at Blender's native scale",
        min=0.001,
        max=10000.0,
        soft_min=0.01,
        soft_max=1000.0,
        default=1.0,
    )

    option_batch: BoolProperty(
        name="Batch Export",
        description="Write one .lwo per selected object, named after the object, "
                    "next to the chosen file. Otherwise all selected objects are "
                    "merged into the single layer the engine reads",
        default=False,
    )

    # -- Animation frames --

    option_anim_frames: BoolProperty(
        name="Export Animation Frames",
        description="Instead of one file, write one .lwo per frame of the chosen "
                    "Action(s): <name>_00001.lwo, <name>_00002.lwo ... next to the "
                    "chosen file. Keyframed transforms, shape keys and armature "
                    "deformation are all taken from the evaluated frame. Feeds the "
                    "Fall of Phaeton mesh flipbook compiler (buildMeshFlipbook)",
        default=False,
    )

    option_anim_sel_only: BoolProperty(
        name="Only selected from list",
        description="Export only the ticked Actions. Off: every listed Action",
        default=False,
    )

    option_anim_keyed_only: BoolProperty(
        name="Only keyed frames",
        description="Write only the frames that carry a keyframe on the Action "
                    "(still numbered contiguously). Off: every frame of the "
                    "Action's range, interpolated frames included",
        default=False,
    )

    option_anim_action_prefix: BoolProperty(
        name="Action name in file names",
        description="<name>_<action>_00001.lwo instead of <name>_00001.lwo",
        default=False,
    )

    anim_actions: CollectionProperty(type=LWOActionItem)
    anim_actions_idx: IntProperty()

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text='Essentials:')
        box.prop(self, 'option_apply_modifiers')
        box.prop(self, 'option_remove_doubles')
        box.prop(self, 'option_smoothing_groups')
        box.prop(self, 'option_smoothing_angle')
        box.prop(self, 'option_mikkt')
        box.prop(self, 'option_multiuv')

        box = layout.box()
        box.label(text='Vertex Colors:')
        box.prop(self, 'option_vertex_colors')
        row = box.row()
        row.enabled = self.option_vertex_colors
        row.prop(self, 'option_color_space', text='')

        box = layout.box()
        box.label(text='Transformations:')
        box.prop(self, 'option_apply_scale')
        box.prop(self, 'option_apply_rotation')
        box.prop(self, 'option_apply_location')

        box = layout.box()
        box.label(text='Advanced:')
        box.prop(self, 'option_scale')
        box.prop(self, 'option_batch')

        box = layout.box()
        box.prop(self, 'option_anim_frames')
        if self.option_anim_frames:
            count = len(self.anim_actions)
            if count == 0:
                box.label(text='No Actions in this file', icon='ERROR')
            else:
                if self.option_anim_sel_only:
                    chosen = len([a for a in self.anim_actions if a.export_action])
                    box.label(text='Export actions: %d' % chosen)
                else:
                    box.label(text='Export actions: %d (all)' % count)
                box.prop(self, 'option_anim_sel_only')
                col = box.column()
                col.active = self.option_anim_sel_only
                col.template_list("LWO_UL_ActionsList", "",
                                  self, "anim_actions",
                                  self, "anim_actions_idx",
                                  rows=min(count, 8))
                sub = col.row(align=True)
                sub.operator(EXPORT_OT_idtech_lwo_select_actions.bl_idname,
                             text="Select").action = "SELECT"
                sub.operator(EXPORT_OT_idtech_lwo_select_actions.bl_idname,
                             text="Deselect").action = "DESELECT"
                sub.operator(EXPORT_OT_idtech_lwo_select_actions.bl_idname,
                             text="Invert").action = "INVERT"
                box.prop(self, 'option_anim_keyed_only')
                box.prop(self, 'option_anim_action_prefix')
                if not self.option_apply_modifiers:
                    box.label(text='Shape keys / armatures need Apply Modifiers', icon='ERROR')

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.selected_objects)

    def invoke(self, context, event):
        self.anim_actions.clear()
        for action in bpy.data.actions:
            item = self.anim_actions.add()
            item.name = action.name
        return super().invoke(context, event)

    def execute(self, context):
        start = time.perf_counter()

        options = {
            'apply_modifiers': self.option_apply_modifiers,
            'remove_doubles': self.option_remove_doubles,
            'smoothing_groups': self.option_smoothing_groups,
            'smoothing_angle': self.option_smoothing_angle,
            'mikkt': self.option_mikkt,
            'multiuv': self.option_multiuv,
            'vertex_colors': self.option_vertex_colors,
            'color_space': self.option_color_space,
            'apply_scale': self.option_apply_scale,
            'apply_rotation': self.option_apply_rotation,
            'apply_location': self.option_apply_location,
            'scale': self.option_scale,
        }

        mesh_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        mesh_objects.sort(key=lambda o: o.name)
        if not mesh_objects:
            self.report({'ERROR'}, 'No mesh objects selected')
            return {'CANCELLED'}

        builder = LWOBuilder(context, options)
        written = []
        try:
            if self.option_anim_frames:
                written = self._export_animation(context, builder, mesh_objects)
            elif self.option_batch:
                base_dir = os.path.dirname(self.filepath)
                for obj in mesh_objects:
                    stem = obj.name.replace('.', '_')
                    path = os.path.join(base_dir, stem + '.lwo')
                    data = builder.build([obj], stem)
                    self._write_file(path, data)
                    written.append((path, builder.summary()))
            else:
                stem = os.path.splitext(os.path.basename(self.filepath))[0]
                data = builder.build(mesh_objects, stem)
                self._write_file(self.filepath, data)
                written.append((self.filepath, builder.summary()))
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        for w in builder.warnings:
            self.report({'WARNING'}, w)
            print('LWO Export warning: ' + w)
        for path, summary in written:
            print('LWO Export: %s (%s)' % (path, summary))

        elapsed = time.perf_counter() - start
        self.report({'INFO'}, 'LWO export: %d file(s) in %.3fs' % (len(written), elapsed))
        return {'FINISHED'}

    def _write_file(self, path, data):
        try:
            with open(path, 'wb') as f:
                f.write(data)
        except IOError as e:
            raise RuntimeError('Could not write file: %s\n%s' % (path, e))

    # -------------------------------------------------------------------------
    # Animation frames
    # -------------------------------------------------------------------------

    def _chosen_actions(self):
        """Actions from the list (populated on invoke); from a script the list
        is empty and every action in the file is a candidate."""
        names = [a.name for a in self.anim_actions
                 if a.export_action or not self.option_anim_sel_only]
        if not self.anim_actions:
            names = [a.name for a in bpy.data.actions]
        actions = []
        for name in names:
            action = bpy.data.actions.get(name)
            if action is not None and action not in actions:
                actions.append(action)
        return actions

    def _export_animation(self, context, builder, mesh_objects):
        scene = context.scene
        base_dir = os.path.dirname(self.filepath)
        actions = self._chosen_actions()
        if not actions:
            raise RuntimeError('No Actions to export')
        if not self.option_apply_modifiers:
            builder.warnings.append(
                'Apply Modifiers is off: shape keys and armature deformation are not '
                'evaluated, only keyed object transforms will animate')

        # which files: one series per object (Batch) or one merged series
        if self.option_batch:
            series = [([obj], safe_name(obj.name)) for obj in mesh_objects]
        elif len(mesh_objects) == 1:
            series = [(mesh_objects, safe_name(mesh_objects[0].name))]
        else:
            series = [(mesh_objects, os.path.splitext(os.path.basename(self.filepath))[0])]

        saved_frame = scene.frame_current
        saved = {}       # id pointer -> (id, had_anim_data, action, slot)
        written = []

        # two actions into one series would overwrite each other's files
        use_tag = self.option_anim_action_prefix
        if len(actions) > 1 and not use_tag:
            use_tag = True
            builder.warnings.append(
                'Several Actions chosen: the action name is added to the file names '
                'so their frames do not overwrite each other')

        def remember(ident):
            if ident.as_pointer() in saved:
                return
            ad = ident.animation_data
            if ad is None:
                saved[ident.as_pointer()] = (ident, False, None, None)
            else:
                saved[ident.as_pointer()] = (ident, True, ad.action, getattr(ad, "action_slot", None))

        try:
            for action in actions:
                kinds, targets = action_targets(action, mesh_objects)
                if not targets:
                    builder.warnings.append(
                        'Action "%s" drives %s but nothing in the selection has it; skipped'
                        % (action.name, ', '.join(sorted(
                            {'KEY': 'shape keys', 'POSE': 'an armature', 'OBJECT': 'object transforms'}[k]
                            for k in kinds))))
                    continue

                # unbind every other target first so an action from a previous
                # pass does not keep animating alongside this one
                for ident, _t in targets:
                    remember(ident)
                target_ids = [t[0] for t in targets]
                for ptr, (ident, had, _a, _s) in saved.items():
                    ad = ident.animation_data
                    if ident in target_ids:
                        if ad is None:
                            ad = ident.animation_data_create()
                        want = next(t[1] for t in targets if t[0] == ident)
                        assign_action(ad, action, want)
                    elif ad is not None:
                        ad.action = None

                start, end = action_frame_range(action)
                if self.option_anim_keyed_only:
                    frames = [f for f in action_keyed_frames(action) if start <= f <= end]
                    if not frames:
                        builder.warnings.append(
                            'Action "%s" has no keyframes in its range; exporting every frame' % action.name)
                        frames = list(range(start, end + 1))
                else:
                    frames = list(range(start, end + 1))

                tag = ('_' + safe_name(action.name)) if use_tag else ''
                for number, frame in enumerate(frames, 1):
                    scene.frame_set(frame)
                    for objects, stem in series:
                        name = '%s%s_%05d' % (stem, tag, number)
                        path = os.path.join(base_dir, name + '.lwo')
                        data = builder.build(objects, name)
                        self._write_file(path, data)
                        written.append((path, 'frame %d, %s' % (frame, builder.summary())))
        finally:
            for ptr, (ident, had, action, slot) in saved.items():
                ad = ident.animation_data
                if not had:
                    if ad is not None:
                        ident.animation_data_clear()
                elif ad is not None:
                    want = 'KEY' if isinstance(ident, bpy.types.Key) else 'OBJECT'
                    assign_action(ad, action, want, slot)
            scene.frame_set(saved_frame)

        if not written:
            raise RuntimeError('No frames written: no chosen Action drives the selected objects')
        return written


# =============================================================================
# Registration
# =============================================================================

def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_idtech_lwo.bl_idname, text="idTech 4 LWO (.lwo)")


classes = (
    LWOActionItem,
    LWO_UL_ActionsList,
    EXPORT_OT_idtech_lwo_select_actions,
    EXPORT_OT_idtech_lwo,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
