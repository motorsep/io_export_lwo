# io_export_lwo — Blender 5.2 LWO exporter for idTech 4 / Fall of Phaeton

Exports selected mesh objects as a LightWave LWO2 object (`.lwo`) shaped for
what `idRenderModelStatic::ConvertLWOToModelSurfaces` actually reads.
LightWave/Modo interchange is not a goal; the engine is.

## Install

Legacy add-on layout (same as `io_export_md5` / `io_export_idt4ase`):
copy the `io_export_lwo` folder into
`%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\` and enable
"Export idTech 4 LWO (.lwo)" in Preferences > Add-ons.
Requires Blender 4.4+ (uses corner normals / color attributes APIs); tested on 5.2.1 LTS.

## Usage

Select the mesh objects, `File > Export > idTech 4 LWO (.lwo)`.

* Material names must be the engine material names (`models/props/crate`).
  Every face needs a material; an empty slot aborts the export.
* All selected objects are merged into ONE layer. The engine only loads the
  first `LAYR` of a file, so separate layers would silently drop objects.
  "Batch Export" writes one file per object instead, named after the object.
* Always triangulated. The engine skips any polygon that is not a triangle.

## Options

| Option | Default | Notes |
| --- | --- | --- |
| Apply Modifiers | on | Exports the evaluated mesh (depsgraph). Non-destructive. |
| Remove Doubles | on | Merges coincident vertices (0.0001). The engine smooths normals only across shared points, so split vertices along UV seams would shade as hard edges. UVs and colors stay per-corner via `VMAD`. |
| Smoothing Groups From Sharp Edges | on | Sharp edges and flat faces (including those set by a Smooth by Angle modifier) become `PTAG SMGP` groups, honored by the engine's `lwGetVertNormals`. |
| Smoothing Angle | 180 | `SMAN` per surface, degrees. With groups on, 180 lets the sharp edges decide. A surface whose faces are all flat gets `SMAN 0`. |
| MultiUV (per-material UV maps) | off | Each material samples the UV map named by the UV Map node in its node tree; polygons are written with that map as separate sparse `TXUV` maps, the way LightWave files carry several maps. Materials without a UV Map node use the render UV map. Setup and rules in [MULTIUV.md](MULTIUV.md). |
| MikkT (Fall of Phaeton engine) | off | Adds explicit corner normals (`NORM`, dim 3) and MikkTSpace tangents (`TANG`, dim 4: xyz + bitangent sign) as `VMAP`/`VMAD` pairs. Stock idTech 4 parses and ignores unknown vertex maps, so the same file still loads there with classic smoothing. Lets you compare both engines side by side from one export. |
| Vertex Colors | on | Exports the render color attribute (camera icon) as `RGBA` `VMAP` + `VMAD`. POINT and CORNER domains, byte or float. Alpha is exported as authored. |
| Color Space | sRGB | sRGB = values as displayed (classic idTech 4 vertex-color pipeline). Linear = Blender's internal values. |
| Apply Scale / Rotation / Location | on | Bake the object's world transform. A mirrored (negative-scale) transform flips the winding back automatically. |
| Scale | 1.0 | Uniform multiplier on positions. |
| Batch Export | off | One `.lwo` per object next to the chosen file. |

## What is written

`TAGS` (materials, then `smooth_N` group tags), one `LAYR`, `PNTS`, `BBOX`,
`VMAP TXUV`, `VMAP RGBA`, `POLS FACE` (triangles), `VMAD TXUV` / `VMAD RGBA`
for corners that disagree with their vertex, `PTAG SURF`, `PTAG SMGP`, and one
`SURF` per material with `COLR` (white, the engine's fallback vertex color)
and `SMAN`.

Coordinates follow the historic Blender-to-LWO convention the engine undoes
on load: Y and Z swapped, polygon winding reversed.

## MikkT details

With the option on, normals are captured from the source mesh before any
topology change (Blender stores custom split normals relative to per-vertex
fan spaces, and triangulating in bmesh skews them by up to 17 degrees), then
re-applied as custom normals on the triangulated mesh, where Blender's
`calc_tangents` computes MikkTSpace against exactly those normals. Both are
transformed into export space (normals by the inverse-transpose, tangents by
the matrix, re-orthogonalized), axes swapped like positions. A mirrored
object transform negates the bitangent sign. Objects without a UV map get
normals only.

Engine contract (Fall of Phaeton): prefer `NORM` over the `SMAN`-derived
normal per corner, take `TANG` xyz as the tangent and w as the bitangent
sign after the usual Y/Z un-swap, mark the surface's tangents as calculated
so the UV-derived tangent pass is skipped. `VMAP` gives the per-vertex value,
`VMAD` overrides it per polygon corner, same as `TXUV`/`RGBA`.

## Not written (the engine ignores it)

Image maps (`BLOK`/`CLIP`), endomorphs, weight maps, edge weights, subpatches,
extra `SURF` channels. Without the MikkT option, vertex normals in the engine
come from `SMAN` + smoothing groups.

## Lineage

Anthony D'Agostino (Scorpius) 2002 LWO2 writer; Gert De Roost's Blender 2.7x
port with the idTech option; 4.0.0 rewrite for Blender 4.4+/5.2 by
motorsep/Claude. GPL v2 or later (see the header); distributed here under GPLv3.
