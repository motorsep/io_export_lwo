# MultiUV: one UV map per material

## What the engine can do

idTech 4 stores exactly one texture coordinate per vertex. There are no UV
channels a material could switch between, so two UV maps can never apply to
the same polygon. What the engine does allow is different UV maps on
different polygons of one model, which is how id's LightWave models with
in-game GUI screens were built: the screen is its own surface with its own
map laid out in 0..1, the body is another surface with another map.

LightWave UV maps are sparse (a map only contains the points assigned to
it), so those files carry several `TXUV` maps that never overlap. The loader
concatenates every `TXUV` map and takes, per corner, the entry that exists.
Blender UV maps are dense (every corner has a value in every map), so the
exporter has to decide which map each polygon contributes. That decision is
what MultiUV automates.

## Setup in Blender

1. Create the UV maps you need on the mesh, for example `body` and `screen`,
   and unwrap each set of faces in its own map, each in 0..1. There is no
   need to move the other layout out of the way: maps of different
   materials never meet in the engine.
2. In each material's node tree add a **UV Map** node set to the map that
   material should sample and wire it into the **Vector** input of the
   material's Image Texture. This is the same node Blender needs to preview
   the right map in the viewport, so you are not adding anything extra.
3. Assign faces to materials as usual.
4. Export with **MultiUV (per-material UV maps)** ticked.

Materials are shared between objects and the node names the map, so a
material used on several objects works as long as each object has a UV map
of that name.

## What the exporter does

For every material on an object it reads the UV Map node (first the one
wired into an Image Texture, otherwise any UV Map node in the tree) and
writes the polygons of that material with the map it names. Each map
becomes its own sparse `TXUV` map in the file, named after the Blender map,
containing only those corners.

A vertex on the border between two maps gets per-polygon (`VMAD`) entries
only, one per corner, so the engine cannot pick the wrong map for it.

Fallbacks, all with a warning in the export report and console:

- Material without a UV Map node: the object's render UV map (camera icon),
  which is what the exporter uses for everything when MultiUV is off.
- Material naming a map the object does not have: the render UV map.
- Faces without any UV map: no coordinates, and the engine warns about
  missing uv data on load.

With MikkT enabled, tangents are computed per map too, so each corner's
tangent matches the map its material samples.

With MultiUV off the output is identical to earlier versions: one map per
object, the render UV map, for all of its faces.

## The terminal example

One mesh, faces split between `models/terminal/body` and a GUI material.
UV maps `body` and `screen`. Body material: UV Map node `body` into its
Image Texture. GUI material: UV Map node `screen` into its Image Texture
(the texture only serves the preview). Export with MultiUV on. The file
holds `TXUV "body"` covering the body corners and `TXUV "screen"` covering
the screen corners, the layout id's own terminals used.
