# Bubonic Creatures

A Minecraft **Bedrock** resource pack that replaces the game's blocky mobs with realistic 3D models
that still move with the game's own animations. It needs no behavior pack and no experiments. It was split off the
texture pack [Bubonic Realism](../minecraft-reskin-project) and works with or without it.

## Installing

For development on this PC (needs Minecraft fully restarted the first time):

    uv run tools/build.py --deploy

Then Settings → Global Resources → My Packs → **Bubonic Creatures DEV** → Activate. Put it **above** any Bubonic
Realism release that is still installed from before the split, because those carry the old mob textures.

For another computer:

    uv run tools/build.py --release

makes `dist/BubonicCreatures_<version>.mcpack`. Copy that one file over and open it. Each release raises the version,
because Minecraft won't import a version it already has.

## What's in it

Every mob in the game: 88 with their own model, 102 more coats, types and professions, and 88 babies. Steve,
Alex, NPCs and the gear animals wear (saddles, horse armour, harnesses) get the material reskin the mobs used to have. Blocks,
items, player armour and objects like chests, beds, signs and boats stay in Bubonic Realism.

## How a model is made

`docs/LESSONS-MOBS.md` has the whole story and the recipe. In short, run from this folder:

| Step | Command |
|---|---|
| Draft pictures on the user's ComfyUI (`tools/pictures.py` holds the prompts) | `uv run tools/pictures.py drafts <name>` |
| Compare them | `uv run tools/contact.py <out.jpg> 81,82,83 <name>...` |
| Cut the chosen one out onto black | `uv run tools/pictures.py <name>:<seed>` |
| Picture to 3D (Pixal3D + TRELLIS.2, about 2 minutes) | `uv run tools/run3d.py <name>:<seed>` |
| Joints, fit to vanilla, entity files (`tools/convert.py` holds the settings, `tools/plans.py` the cuts) | `uv run tools/convert.py <sheet_dir> <name>...` |
| Look at it | `uv run tools/check.py <sheet_dir> <name>...` or `uv run tools/check.py --rows <out.jpg> <name>...` |
| List everything in pack.yaml | `uv run tools/pack_entries.py` |
| The ender dragon has its own conversion | `uv run tools/dragon.py <sheet.jpg>` |

## Layout

`pack.yaml` (identity and texture list) · `art/models/mesh/` (each model's geometry, 2048 texture, entity file, render
controller, animations) · `art/blocks/` (the grains of the material reskins) · `tools/` · `comfy/` (picture graphs).
Git-ignored and only on the PC that made them: `work/` (pictures and GLBs, 3 GB), `art/models/mesh/*.glb`, `build/`, `dist/`.

The build and the mesh code (`glbmodel.py`, `render.py`, the `mob_model` builder) are the `mcreskin` tool in the reskin project,
which this repo uses from `../minecraft-reskin-project`: keep the two folders side by side. `vanilla` is a link to its copy of
Mojang's reference pack.
