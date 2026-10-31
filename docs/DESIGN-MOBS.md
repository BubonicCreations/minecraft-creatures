# Mob models and builders

The contracts from the reskin project's `docs/DESIGN.md`, moved here with the code on 2026-10-31.

- Mobs (2026-10-26): builder `mob_skin` reads the vanilla model (`vanilla.geometries`, the 1.12+
  and the older inheriting formats, whose same-name bones add their cubes to the parent's: the
  woolly sheep is the shorn one plus wool; `texture_geometries` finds the model an entity draws a
  texture on, of the texture's shape, a model without a size counting as 64 x 64 when its UVs
  reach below row 32: taking the adult villager and sheep models for 64 x 32 picked the babies'
  for 17 textures of the first mobfaces run) and unfolds each cube's box UV the way Bedrock does (north = the
  face: head.north of a [8,8,8] cube at uv [0,0] is (8,8)-(16,16), a player skin's face). Each
  texel goes to the nearest `materials` colour; a material is a seamless photo's relief over
  vanilla's local colour, smoothed inside one UV face only (adjacent rectangles of the net are not
  adjacent on the model). `keep` colours inside `features` faces stay crisp (eyes).
- Mob faces (2026-10-26, the user: "these are amazing"; mob_skin's materials were "not great"):
  `mcreskin mobfaces <recipe>` (mobfaces.py) cuts every face of the model from vanilla, enlarges it
  to 1024 on its long side and lets Qwen's inpaint graph repaint all of it (image to image at
  `denoise`) with a prompt per face glob; `--pick TEXTURE SEED` copies a set to
  art/mobfaces/<texture with __>/<face>.png. Builder `mob_faces` (`paste_faces`) scales each picture
  into its face at size / 16 px per texel (at most 2048 a side) and `harmonize`s it: its colour
  over ~a texel is pulled to vanilla's, so faces painted one by one share one lighting and the
  cow's patches stay put. Face names: bone.face, bone#i.face for a bone's i-th cube.
  A recipe texture's `keep: {from: <seed>, faces: [globs], except: [globs]}` copies good faces of
  an earlier seed so a fix repaints only the rest; within a run a face whose vanilla texels match
  one already painted with the same prompt (`REUSE_TOL`) takes that picture, so tame and angry
  variants share most faces (6,400 pictures instead of 9,200 for the variants and babies).
  `keep: {from: art}` takes the picked set. `front: {bone, prompt, denoise}` paints the front of
  that bone's cubes (and its child bones', untilted, drawn) as the model shows them from straight
  ahead in one picture and cuts it back into their north faces: a snout or nose box then carries
  the face's own nose (the first run gave the wolf a whole face behind a second, separate nose).
  `mcreskin render --candidates SEED` draws a seed's pictures on the model before `--pick`.
- `mcreskin render` (render.py, 2026-10-26: "a way to render these animals outside of the game"):
  a numpy rasterizer of the vanilla model with the pack's colour, normal map and MER (simple light
  from the upper left, roughness highlight, MER glow), orthographic, six views per mob next to
  vanilla's. Bedrock model space is mirrored in x against the screen (Java's scale(-1, -1, 1),
  Blockbench's import): a point draws at (-x, y, z), a rotation [x, y, z] (degrees, x then y then
  z, about the pivot; a cube's defaults to its centre) turns by (-x, -y, z) there; box UV's left
  strip is the -x side. bind_pose_rotation turns the bone's own cubes only (the panda's and sheep's
  legs hang from the body bone but stand). Rest pose: the setup and always-on animations and each
  animation controller's resting state ("default", else "walking"), Molang read with queries and
  variables 0 (an adult, not moving); a rotation channel adds to the rest value (`this`), a
  position's `this` is the bone's pivot less its parent's (a root's parent at y 24: the Java model
  these were ported from), so the wolf's "-14 - this" is no move and guardian spikes move out.
  Render controllers' part_visibility hides bones no draw pass shows (saddle bags, llama chests).
  Bodies the game lays down in code (fox, polar bear, cat) and the villagers' hat brim: POSE_FIX.
  Not drawn as in game: the Ender Dragon (its neck, tail and second wing are placed in code),
  baby scaling, attachables, held items.
- Own mob models (2026-10-26; builder `mob_model`: `model` a Bedrock geometry shipped to
  models/entity/ under the vanilla id it replaces, `picture` its texture, `files` any extra pack
  files). mobmodel.py writes a model as boxes in code; glbmodel.py turns a GLB from the user's
  ComfyUI image-to-3D workflow (Pixal3D + TRELLIS.2, decimated to ~1,500 faces) into `poly_mesh`
  bones. Verified in game: Bedrock draws and animates poly meshes from a resource pack; its
  normalised UVs count v up (glTF's down: unflipped, the texture was a patchwork). Bones move
  rigidly (no skinning), so a mesh cut at a joint tears when it turns. `jointed` (the default)
  builds it like vanilla's own mobs for the game's own walk and look-at: legs cut below the belly
  and the head just behind the skull (ears with it), the body's holes sealed, each moving part a
  closed piece whose short tapered top reaches into the body. Tried and dropped: copies of body
  skin turned with a part (a fringe), slices spreading a bend (cracks along every slice), our own
  walk with knee bones (needs an entity file override; kept in glbmodel for reference).
  glbmodel.draw_mesh renders a model the game's way, with poses, before it ships. Other bodies
  (2026-10-30, not yet seen in game): `carve` and `pieces_past` cut parts along planes; vanilla's
  absolute sits are replaced for the mesh by `rest_pose`; saddles are the back's own surface raised
  (`add_saddle`); details in ../creatures/docs/LESSONS-MOBS.md. The pig
  (2026-10-27) shares geometry.pig.v3 between its climate variants, so its mesh is
  geometry.pig.mesh, drawn for the temperate adult by an entity file override and a copy of
  controller.render.pig.v4 (`mesh_for_variant`); a saddle still draws on vanilla's box body,
  ~1.5 px wider than the mesh each side.
