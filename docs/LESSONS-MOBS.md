# Mobs: lessons learned (2026-10-26)

How the creatures went from vanilla's pixel boxes to a realistic cow that walks with the game's own
animations, what failed on the way, and the recipe for the next mob. Verified Bedrock facts are also
in DESIGN.md ("Own mob models", "Mob faces"); prompt wording in ART-NOTES.md ("Mob faces").

## The road, in short

| Try | What it was | Verdict |
|---|---|---|
| `mob_skin` | real material photos (fur, hide) as relief over vanilla's colours | "not great": still reads as vanilla's layout |
| `mob_faces` | Qwen repaints every face of vanilla's UV net as a photo | flat previews "amazing"; on the model: a wolf with two noses, human faces on blazes, villagers and sheep scrambled |
| own box model (`mobmodel.py`) | a 48-box cow in code, painted by projecting Qwen views onto it | shape liked, texture cartoonish |
| AI 3D model (`glbmodel.py`) | the user's ComfyUI image-to-3D (Pixal3D + TRELLIS.2) GLB as Bedrock polygon meshes | "the cow looks quite good now", once jointed for vanilla's animations |
| the pig (2026-10-27) | the recipe as is: one Qwen side view (head right, cut out onto pure black), 1,500 faces, 2048 texture, `height=15` (vanilla's pig is 16 px to the head top) | "still looks great" (the same small issues as the cow) |
| six more (2026-10-27) | sheep, polar bear, llama, goat, hoglin, mooshroom; see below | in-game verdict pending |
| fifty more (2026-10-30) | every remaining body plan but bosses: four-legged, small, fliers, swimmers, humanoids; see below | rendered and checked one by one; "This is looking really cool" |
| the rest (2026-10-31) | the last 30 mobs, every coat and profession, babies; see the end | rendered and checked; in-game verdict pending |

## Look at it on the model, not on a flat sheet

- Every big mistake was invisible in flat previews and obvious on the model: a whole wolf face behind a
  separate snout box (two noses), a 4-eyed cow face, a texture read upside down, legs tearing open.
  `mcreskin render` (vanilla models, rest pose from setup animations, part visibility) and
  `glbmodel.draw_mesh` (poly meshes, UVs read the game's way, poses) exist for this: render before
  deploying, and render the poses the game will use (widest step, head turned 60°).
- One in-game screenshot from the user (the wolf) was worth a whole night of flat previews.
- Pilot first, then batch. The 97-mob face run looked fine as nets and was wrong for most mobs;
  a 2-hour fix run was stopped halfway when the method itself was in doubt. That was the right call.

## Painting vanilla's boxes has a ceiling

- A box's faces are painted one by one, so features double or drift: a face picture on the head front
  plus a nose on the snout box makes two noses. Painting the front as one picture (`front:`) helps,
  but a realistic animal does not fit Minecraft's proportions anyway.
- "skin" in a prompt paints human faces and torsos; stylised mobs (ghast, blaze, strider) became
  people unless described as a mask with "no human features".
- Model facts that cost a day to find: an old model without a texture size is 64x64 when its UVs
  reach below row 32 (taking it for 64x32 picked the *baby* model for the adult villager and sheep);
  old-format child bones add their cubes to the parent's (the woolly sheep is the shorn one plus
  wool); bind_pose_rotation turns only the bone's own cubes; bodies are often laid down by setup
  animations or by code (fox, cat, polar bear).

## AI 3D models (ComfyUI)

- **Consistency comes from one image.** Four views painted separately are four different cows; one
  Qwen turnaround sheet (2x2 grid on black, `work/turnaround.py`) gives one cow.
- **Know the workflow's conventions before blaming the model.** The multiview workflow wants one
  6336x2688 black sheet cut at fixed spots, its CLIP vision set to DINOv3, and "left" means the
  animal's *own* left (head to the picture's left). Swapped sides grew ear flaps on the rump; a back
  view showing the ears over the rump did the same; leaving the back view out grew a second face.
  The user's single-image workflow avoided all of it and gave the best cow.
- Multiview texturing can paint features twice (eyes from the front view on the front of the face
  plus eyes from the side views): look at the head close up, paint the extras out (`retouch`).
- The texture carries the detail: 1,500 faces looked like 6,000 at play distance. Decimate in the
  workflow (it re-bakes the texture onto the low mesh).

## Bedrock polygon meshes

- Resource packs can replace a mob's model (`models/entity/*.geo.json` under the vanilla id), and
  Bedrock draws and animates `poly_mesh` bones. Normalised UVs count **v up** (glTF counts down).
- Entity textures are not in the block atlas; the budget is graphics memory (~50 MB per 2048 mob
  with its normal map and MER), so plan 2048 for big mobs, 1024 for everyday ones.
- **Bones are rigid; there is no skinning.** A realistic mesh cut at a joint opens when the joint
  turns. What did not fix it: copies of body skin turned with the part (a jagged fringe), spreading a
  bend over slices of skin (cracks along every slice), overlap rings (covered almost nothing). Our own
  walk with knee joints helped, but the game weighs a walk by `query.modified_move_speed` (~0.3-0.5
  for a strolling cow), so 20° became 6-10° ("the cow legs almost dont even move").
- **What works: build it the way vanilla builds mobs.** Cut the legs below the belly and the head just
  behind the skull (with its ears), seal the holes in the body, and make each moving part a closed
  piece whose short, tapered top reaches into the body. The game's own walk and head-turn then move
  parts that turn inside the body instead of tearing it. No animation or entity files needed.

## Tools that paid off

- Driving the user's ComfyUI through its API from the API-format workflow JSON: rerun with another
  input picture, face count or texture size and fetch the GLB without the user clicking anything.
- Blockbench through its MCP plugin, reached from WSL with Windows' own `curl.exe` (the plugin only
  listens on Windows' loopback). The plugin must be "MCP Server" by Jason J. Gardner and needs the
  network permission prompt accepted. Its Bedrock format holds boxes only, not meshes.

## Recipe for the next mob

1. Make the picture: one clean side view of the animal standing, legs straight and a little apart
   (or a 4-view sheet with `work/turnaround.py` for the multiview workflow).
2. Run the user's single-image workflow at ~1,500 faces and a 2048 texture; fetch the GLB.
3. Look at it from six angles (head, rump, under the body); fix doubled features.
4. `glbmodel.make(glb, "<vanilla geometry id>", "<texture key>", art/models/mesh, forward=...)`
   (jointed by default); check `draw_mesh` at vanilla's widest step and a 60° head turn.
5. Point the pack.yaml entry at it (`builder: mob_model`), build `--only` it with `--deploy`, and
   have the user look in game before doing the next one.
   - If the mob's variants share one vanilla geometry (the pig: temperate, warm and cold all draw
     geometry.pig.v3, only the texture differs), don't replace it: the other variants' textures would
     land on the mesh. Ship the mesh under a new id and `glbmodel.mesh_for_variant` (an entity file
     override plus a copy of its render controller that draws the mesh for one variant only), via
     the entry's `files:`. The cow didn't need this: its variants have their own geometries.
6. Other bodies: a `plan` of `glbmodel.carve` and `pieces_past` rules (work/models3d/mobs/plans.py); see below.

## Six more (2026-10-27): sheep, polar bear, llama, goat, hoglin, mooshroom

Scripts: `work/models3d/mobs/` (pictures.py drafts and cut-outs, run3d.py the 3D runs one at a
time, ~2 min each, convert.py conversion, review.py ours next to vanilla in the game's rest pose,
motion.py ours played by the mob's own animations: walk, run, look around, grazing, ram, rearing).
The first round looked fine on our check sheets and "all 6 have issues" in game. What we learned:

- **Check the model as the game poses it.** Our check sheets drew the bare mesh. `render.setup_pose`
  on our own bones (its `geometry=`) shows what the game does: a position channel's `this` is the
  bone's pivot less its parent's (a root's parent at y 24), so vanilla's rest moves ("-6 - this", the
  sheep's head; "-9 - this", the polar bear's body) cancel on vanilla's model and, on ours (hung from
  a root at 0), put the sheep's head 28 px and the bear's body 35 px underground. Fix:
  `glbmodel.cancel_rest_moves` hangs such a bone from a still anchor that makes `this` equal the
  constant. (An earlier "lift" read the moves as real and was wrong.)
- **Copy vanilla's hierarchy and body pivot** (`fit_to_vanilla`): the bear's head and legs hang from
  its body, and its body rears up about the hips (about the middle, its hindquarters sank).
- **Vanilla's moves are made for boxes.** Each mesh mob gets its own copies (`variant_animation`,
  drawn variant only; babies and other colours keep vanilla's): look-at clamped to ~35 degrees, a
  run's stride to 0.55 of vanilla's (44 degrees, not 80), the sheep's grazing and goat's ram without
  the 8-9 px head drop (it tore the head off), the bear's rearing without vanilla's 6 px hind-leg push.
- **Ball joints instead of sleeves and flat caps** (`split_clean`, `cut_plane`, `dome`): parts are cut
  on clean planes (legs under the belly, head at the skull) instead of by triangle centres (a zigzag
  whose teeth stuck out), closed by a shallow dome around the pivot, and the body by a cup behind it
  lit like the fur (facing in, it was a dark slash). Bits a cut leaves on the body that touch only a
  moving part go with it (the goat's horns sweep back past the cut). A full half ball (depth 1) and a
  cup a radius deep poked out of the goat's thin neck: depth 0.5 and 0.6.
- **Head cuts:** at the skull is best for all six; with the neck (sheep, llama from the shoulders) the
  seams were bigger, and the sheep's muzzle still could not reach the grass (a deep nod instead).
- **Sheep wool** takes the dye through the texture's alpha (opaque wool, alpha 3 skin), masked from a
  cut across the face (the head piece also holds the throat's fleece) and bright texels only (the
  brown ears stay undyed); the fleece is whitened, since the dye multiplies over it (cream: beige
  sheep, muddy colours).
- **Long hair breaks the 3D step** (twice: the shaggy goat, the hoglin with a fluffy mane): ask for
  short, smooth coats. **Colours of the game's mob** read better than "realistic" ones: the mooshroom
  is vivid red with big mushrooms, the hoglin pinkish tan with orange bristles.
- A Qwen batch once came back all black (NaNs); `/free` and running it again fixed it.

## Fifty more (2026-10-30)

Horse, donkey, mule, skeleton and zombie horse, wolf, cat, ocelot, fox, panda, camel, camel husk, armadillo,
sniffer, ravager, zoglin, trader llama, creeper; rabbit, frog, turtle, axolotl, chicken, strider, spider, cave
spider, silverfish, endermite; parrot, bee, allay, vex, phantom; cod, salmon, pufferfish, tropical fish, squid,
glow squid, dolphin, guardian, elder guardian, tadpole, nautilus; zombie, husk, skeleton, iron golem, enderman,
snow golem. The bat was tried three times and left vanilla (stiff triangles for wings; its flying pose collapses).

Same scripts, spec table in convert.py, cut rules in plans.py. `check.py` draws one sheet per mob: vanilla at the
same scale, ours all round and from below, the parts in flat colours, a step, the head turned and lowered, and the
mob's special moves, played from the animations it ships with. Every mob was looked at on that sheet before moving on.

What the 3D step can't take, and what it can:

- **Gaps break it.** Bare bones (ribs with black between them) came back as shreds, twice. "A statue carved from one
  solid block of pale ivory, the skeleton as shallow relief" gave the skeleton horse a whole body that still reads as bones.
- **Pure white on black bakes with black blotches** (hen, snowman, iron golem). Cream, pale blue-grey and mid grey don't.
- **Bristles, feelers and clear fins** make lumps, long sticks and dark blobs. Ask for "no hair", "no antennae", "solid
  opaque"; `clip` trims what is left (a cut leaves a hole: don't clip into the body).
- **Dark on black loses the shape** (panda's legs): ask for dark charcoal and bright light.
- **Pictures from above tilt the mesh**: `tip` turns it level (spider, phantom). `stand` turns a leaning upright body
  straight (humanoids, squid). Thin wings pictured head-on stay the weak spot.
- Scale floaters by length (`span`) and put them where vanilla's model is (`place_like_vanilla`): they don't stand on y 0.

Rigging:

- **Vanilla's sits place box parts by absolute numbers** ("-18 - this"): on a mesh, heads float and legs lie beside the
  body. `rest_pose` works a sit or lie-down out from the mob's own legs (the body tilts about the hips, front legs stay
  under the shoulders, hind legs fold) and `pose_animation` holds it for the mesh only.
- **Setups that lay a box body down** (wolf, fox, llama: "90 - this") must go for the mesh (`FLAT`), or it stands on its tail.
- Bone names match in any case: "Body" is the horse's `body`. Missing that left its legs behind when it reared.
- `links`: vanilla's empty bones between ours and their parents, which its animations also move (a rabbit's `backlegs`),
  turning about our children, not vanilla's pivot. `keep`: empty item bones (`rightItem`), so mobs still hold things.
- A horse's neck turns from the shoulders, so grazing at vanilla's 110 degrees put its head through its chest; 44 looks right.

Gear and variants:

- **Saddles fit the animal** (`add_saddle`): the back's own surface raised a little, as vanilla's saddle bone, with
  leather in an unused corner of the texture. Armoured horses, chested donkeys and mules, a saddled or armoured
  nautilus, a rolled-up armadillo and a sleeping fox fall back to vanilla's model.
- A horse's markings and armour layers are painted for vanilla's UVs: `layers` blanks them on the mesh. A tropical fish's
  material multiplies random colours over it: `material` gives the mesh a plain one.
- **Each model is one variant** (a chestnut horse, a tabby cat, a pale wolf, a temperate frog); the others stay vanilla.
- The sniffer's picture came out with four legs however it was asked, so its middle legs don't exist on ours.
- Two entities copying one render controller (zombie and husk, the spiders, the guardians, the llamas) need different ids: `rc_suffix`.

## The rest (2026-10-31): the last 30, every coat, babies

The pack moved to its own repo the same day; paths here are this repo's. Scripts that said `work/models3d/mobs` are in `tools/`.

What the 3D step can't take, more of it:

- **Skeletons are the hardest thing.** Every see-through rib cage and pelvis and every tattered cloak came back in shreds, however it was
  asked. What held: "a chunky toy figure sculpted from one solid piece of resin, arms and legs like sausages, ribs only shallowly engraved".
  It's less realistic: they read as stout big-headed figures.
- **Dark on black breaks more than the colour.** Dark robes shredded (witch, four tries), a black wolf lost its legs, a brown panda came back as
  confetti, twice. **Picture it pale, darken it after**: the witch in pale lavender, then blue-beats-green texels darkened to plum, held at once.
  Or recolour a coat that worked (`geo_of` with `darken`, `browned`, `dye`, `patina`, `recolour`): black wolf from ashen, brown panda from panda, zombie nautilus
  from nautilus, dyed shulkers, weathered copper golems.
- **Qwen won't paint some patterns** (a spotted or striped wolf came out plain): `coat_pattern` paints them, laid out by `position_map`, each texel's place on the animal.
- **One seed gives one animal.** Coats of a wolf at the same seed were almost the same wolf: use other seeds and strong words.
- **A cube pictured from above has a corner up**: `tip` turns it level, `forward` turns its face to the front. The 3D step also put the ghast's sad face on its back too: `plain_side` paints it out; the relief stays.
- Wings and rods aren't 3D step material: `card` (a flat two-sided mesh in a cut-out's outline), `rod` (a plain cylinder from a picture's strip), `attach` (add a part with its own texture, as many copies as needed).

Rigging:

- **Several models per entity** (`meshes_for_variants`, `family` in the specs): the first variant that holds draws. An edit of vanilla's animation is one copy for the whole family, because nesting one per model grows without end; a pose from a model's own legs adds one condition per model.
- **A mob's layers go too** (`hide_rc`): villager professions, warden glows, stray clothes are vanilla's boxes over the mesh. `show`: an illager's arms exist only when angry on vanilla's model, always on ours.
- **Babies** are the coat's own model at vanilla's baby height with its head 1.3 times bigger (`baby_of`), so no 3D run; vanilla's baby head-scaling is neutralised for them. They read as tiny adults.
- **The ender dragon** has no parents in its file and its neck and tail exist only in the game's code, as clones of `neck`. `tools/dragon.py` doesn't rely on how that's wired: same id, bones and pivots, each mesh part scaled into the box of vanilla's bone. Its check picture assembles a guess of the result; only the game can confirm it.


## When a mob stays vanilla in game (2026-10-31)
- **Read the game's content log first.** Settings > Creator turns on the log file; it lands in `%APPDATA%/Minecraft Bedrock/logs/ContentLog*.txt` and can stay empty until the game has loaded a world with the pack. Our renders can't show any of this.
- **Mojang's samples are every version of a mob merged, not what the game runs.** A copy can break its own format's rules and the game quietly keeps its own mob: format 1.10.0 on takes no `animation_controllers`, 1.8.0 takes no conditional render controllers. `glbmodel.entity_for_game` fixes both at write time. Twelve mobs were affected, the creeper and iron golem among them.
- **A higher `min_engine_version` in the game's file beats ours, with no error at all** (husk, rabbit). The game's files are in `data/resource_packs/vanilla*/__brarchive/`; `tools/gamecheck.py` compares against them, so run it after a game update.
- **Curves need constant keyframes:** `lerp_mode: catmullrom` errors once a value is wrapped in a Molang condition, so the edited copies drop it.
- Still logged and harmless: `Locator: Error ... armor_offset.default_neck` on 13 models. Our geometry has no locators; the game skips its own and draws the mob.
