"""Side views for the single-image 3D workflow, as for the pig (work/models3d/pig/picture.py): one
realistic animal in exact profile, head to the right, legs straight and apart, on black; three
seeds each. `on_black(name, seed)` cuts the chosen one out (RMBG-2.0) onto pure black."""
import sys
from pathlib import Path

from mcreskin.comfy import Comfy, load_graph, patch

OUT = Path("work/models3d/mobs")
POSE = ("standing on all four legs in exact side profile, its body parallel to the picture, head to the right "
        "looking straight ahead, all four legs straight, a little apart and clearly visible, the whole animal in the "
        "picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text")
ANIMALS = {
    "sheep": "one single realistic adult domestic sheep with a full thick white fleece, a pale cream face and ears, "
             "the fleece ending above its slim legs so the legs show below the belly",
    "polar_bear": "one single realistic adult polar bear with thick creamy white fur, black nose and eyes",
    "llama": "one single realistic adult llama with long creamy white and pale beige wool, long upright neck, "
             "banana shaped ears",
    "goat": "one single realistic adult white domestic goat with short smooth white hair (not shaggy), a small beard, "
            "and two thick ridged horns curving back from the top of its head",  # shaggy hair: a shapeless 3D blob
    "hoglin": "one single realistic huge wild hog with a smooth pinkish tan hide with short coarse orange-brown "
              "bristles (not long, not fluffy), a short bristly crest along its spine, a pink snout and large upcurved "
              "ivory tusks",  # v1 dark brown boar; v2 with a long fluffy mane: a shapeless 3D blob
    "mooshroom": "one single realistic adult cow with a vivid bright red hide and large white patches, "
                 "with three big red toadstool mushrooms with white spots growing upright from the middle of its back",  # v1: brown, small
    # the next round (2026-10-30): the colours of the game's mob, short smooth coats (long hair breaks the 3D step)
    "horse": "one single realistic adult chestnut brown horse with a short smooth glossy coat, a short trimmed dark mane "
             "close to its neck and a short dark tail",
    "donkey": "one single realistic adult grey-brown donkey with a short smooth coat, a pale muzzle and belly, long upright ears, "
              "a short stiff dark mane and a thin tail with a small tuft",
    "mule": "one single realistic adult dark brown mule with a short smooth coat, a pale muzzle, long upright ears, "
            "a short stiff mane and a short tail",
    "skeleton_horse": "one single horse statue carved from one solid block of pale ivory bone, the skeleton carved as shallow relief into its "
                      "solid body: ribs as shallow ridges, a skull-like face with dark eye sockets, no holes and no gaps anywhere, solid legs "
                      "with knobby joints, a short solid tail, no mane",  # bare bones or stretched skin: holes between the ribs tear the 3D step
    "zombie_horse": "one single rotting zombie horse with a smooth sickly dark green hide, patches of decayed grey-green flesh, "
                    "ribs showing in relief, pale dead eyes, a short ragged dark mane",
    "wolf": "one single realistic adult pale grey wolf with short smooth dense fur, a pale chest, upright ears and a straight tail "
            "hanging down",
    "cat": "one single realistic adult tabby house cat with short smooth brown striped fur, upright ears and a long tail held "
           "straight out behind it",
    "ocelot": "one single realistic adult ocelot with short smooth golden yellow fur with dark spots, upright ears and a long tail "
              "held straight out behind it",
    "fox": "one single realistic adult red fox with short smooth orange-red fur, a white chest and chin, dark brown lower legs and a "
           "thick tail held straight out level behind it, well above the ground and away from its legs",  # v1: hanging, it joined a hind leg
    "panda": "one single realistic adult giant panda standing on all fours with its legs clearly apart, short smooth dense white fur, "
             "dark charcoal grey ears, eye patches, legs and shoulders, lit brightly so that every leg stands out",  # v1: black legs lost on black
    "camel": "one single realistic adult dromedary camel with short smooth sandy tan hair, one large hump, a long curved neck "
             "and a small tail",
    "camel_husk": "one single undead mummified dromedary camel with dry cracked sand-brown hide, patches of torn dark skin, ribs "
                  "showing in relief, hollow dark eyes, one large withered hump, a long curved neck",
    "armadillo": "one single realistic adult armadillo with a pinkish tan banded armoured shell, a small narrow head, upright ears, "
                 "four short legs and a short tail",
    "sniffer": "one single large prehistoric animal with exactly six stout legs, three evenly spaced legs in a row along its side, "
               "a long low barrel shaped body with short smooth brick red hide and a broad mossy green back, a long wide flat "
               "olive green duck-like snout and small drooping ears",  # v1: four legs
    "ravager": "one single huge heavy beast with thick dark grey hide, a massive blocky head with a big brown muzzle, two large "
               "curved pale horns on its head, a thick neck and four very thick legs",
    "zoglin": "one single realistic huge wild hog with a smooth pale pink and grey hide with rotting green patches and scars, no long "
              "hair, a pink snout and large upcurved pale tusks",  # v1 with bristles: a lump with no legs
    "trader_llama": "one single realistic adult pale brown llama with short smooth wool, a long upright neck, banana shaped ears, "
                    "wearing a bright blue and gold patterned blanket on its back",
    "rabbit": "one single realistic adult brown wild rabbit with short smooth fur and a small white tail",
    "frog": "one single realistic large orange-brown frog with smooth skin, big eyes and a pale belly",
    "turtle": "one single realistic green sea turtle with a domed dark green shell and four big flippers",
    "axolotl": "one single realistic pink axolotl with smooth solid opaque pink skin all over, a wide flat head, short thick pink gills, four "
               "small legs and a short thick solid pink tail",  # v1: its clear tail fin came out a dark blob
    "chicken": "one single realistic pale cream hen with smooth tight cream feathers, a red comb and wattle, a yellow beak and yellow legs, "
               "brightly and evenly lit",  # v1 pure white on black: the texture baked with black blotches
    "strider": "one single strange creature with a large smooth oval crimson-red body without a neck, wide-set small eyes low on the body, a "
               "wide mouth, no hair, and two very long thick red legs",  # v1 with bristles: a lump with stubs for legs
    "spider": "one single huge realistic dark brown-grey spider with a fat round abdomen, a small head with red eyes and eight thick "
              "hairy legs",
    "cave_spider": "one single realistic dark blue-teal spider with a fat round abdomen, a small head with red eyes and eight thick legs",
    "silverfish": "one single big realistic silver-grey insect with a thick tapering segmented body and very short stout legs, like a woodlouse, "
                  "no antennae, no tail bristles",  # v1: broke in two; antennae became long sticks
    "endermite": "one single small strange dark purple mite-like creature with a thick short segmented body and very short stout legs, no "
                 "antennae",
    "bat": "one single realistic small dark brown bat with short smooth fur, large ears, a small body and short thin leathery brown wings, the "
           "wings well inside the picture with wide margins all round",  # v1: wings cut off by the picture's edge
    "parrot": "one single realistic scarlet macaw parrot with smooth bright red feathers, blue and yellow wings folded at its sides, a pale "
              "curved beak, a long red tail and two short grey legs",
    "bee": "one single realistic fat bumblebee with short smooth golden yellow and dark brown stripes, big dark eyes, a small stinger, six "
           "short legs and two pairs of clear wings held up and out",
    "allay": "one single tiny cute sky-blue fairy creature with a large round head, big dark eyes, a small tapering body with no legs, two "
             "small arms and two small pale blue insect wings on its back",
    "vex": "one single tiny pale grey flying imp creature with a large round head, dark eyes, a small slim tapering body with no legs, two thin "
           "arms and two small thin flat pale insect wings on its back",  # v1: its wings came out as a lump behind it
    "phantom": "one single flying creature like a manta ray with a long thin dark blue-grey body, a wide flat head with glowing green eyes, "
               "two long ragged leathery wings and a short thin tail",
    "cod": "one single realistic Atlantic cod fish, olive brown with dark speckles and a pale belly",
    "salmon": "one single realistic Atlantic salmon fish, silver with a rosy pink side and dark speckles on its back",
    "pufferfish": "one single realistic round pufferfish, yellow with brown patches on its back and a pale belly, short spines, big eyes",
    "tropicalfish": "one single realistic orange clownfish with three white bands edged with black",
    "squid": "one single realistic dark blue squid with a long pointed mantle at the top and eight thick arms hanging down from its head, "
             "the arms clearly apart from each other",
    "glow_squid": "one single squid with smooth glowing teal and cyan skin with bright pale blue spots, a long pointed mantle at the top "
                  "and eight thick arms hanging down from its head, the arms clearly apart from each other",
    "dolphin": "one single realistic grey bottlenose dolphin with a pale belly",
    "guardian": "one single strange large round armoured sea creature with a blue-green scaled body, a single huge round yellow eye in the middle "
                "of its face and no other eyes, no mouth, short thick orange spikes all over its body and a long thin tail with a small fin",  # v1 in profile: a two-eyed pufferfish
    "elder_guardian": "one single strange huge round armoured sea creature with a pale grey-white scaled body with purple patches, a single huge "
                      "round pale eye in the middle of its face and no other eyes, no mouth, short thick pale spikes all over its body and a long thin tail with a small fin",
    "tadpole": "one single realistic large dark brown tadpole with a round body and a long flat tail",
    "nautilus": "one single realistic chambered nautilus with a coiled shell striped white and orange-brown and many short thick tentacles",
    "zombie": "one single realistic rotting zombie man with grey-green decayed skin, a torn teal blue shirt and dark blue trousers",
    "skeleton": "one single skeleton statue carved from one solid block of pale bone, thick solid arm and leg bones, a big skull with dark eye "
                "sockets, the ribs as shallow ridges on a solid chest, no holes and no gaps anywhere",  # bare bones tear the 3D step (the skeleton horse)
    "iron_golem": "one single huge humanoid golem built of smooth mid grey iron plates with a few small green vines on it, a small head with a long "
                  "nose and red eyes, very wide shoulders, very long thick arms reaching its knees and short thick legs, evenly lit",  # v1 white-ish with many vines: the texture baked black and white
    "enderman": "one single very tall thin creature with dark charcoal grey skin, a square head with glowing magenta eyes, very long thin arms and "
                "very long thin legs, brightly lit so that it stands out",
    "snow_golem": "one single snowman made of a smaller ball of pale blue-grey snow stacked on a bigger ball of pale blue-grey snow, a carved orange "
                  "pumpkin as its head on top and two short thick wooden sticks as arms held out to its sides, evenly lit",  # pure white: the texture baked with black blotches
    "husk": "one single realistic mummified desert zombie man with dry cracked sand-brown skin, a tattered dusty brown shirt and trousers and sunken "
            "dark eyes",
    "creeper": "one single creature shaped like a tall smooth upright pillar, a long vertical cylinder body as wide as its head with "
               "no neck, mottled bright green and grey-green skin, no arms, a square face at the top of the pillar with two square "
               "black eyes and a black frowning mouth, and four very short stubby feet at the bottom of the pillar",  # v1: a goblin
}


SIDE = ("in exact side profile, its body parallel to the picture, head to the right, the whole animal in the picture with space "
        "around it, even soft lighting, plain pure black background, no floor, no shadow, no text")
POSES = {  # bodies that don't stand on four straight legs
    "rabbit": f"crouching on all four legs with its legs visible and its ears upright, {SIDE}",
    "frog": f"sitting on the ground with its four legs clearly visible and apart from its body, {SIDE}",
    "turtle": f"standing on its four flippers with the flippers pointing down and away from the shell, {SIDE}",
    "axolotl": f"standing on its four legs with its body raised and its tail straight out behind it, {SIDE}",
    "chicken": f"standing on its two legs, with its legs straight and clearly visible and apart, {SIDE}",
    "strider": f"standing on its two long legs, both legs straight and clearly apart, {SIDE}",
    "spider": f"standing on its eight legs with its body level and the legs clearly spread out and apart from each other, {SIDE}",  # v1 from above: its body came out tilted
    "silverfish": f"walking on the ground, its body long and straight, {SIDE}",
    "endermite": f"walking on the ground, its body long and straight, {SIDE}",
}
POSES["cave_spider"] = POSES["spider"]
WINGS = ("flying, facing the camera, seen from the front and a little above, both wings spread wide to its sides, the whole animal in the "
         "picture with space around it, brightly lit so that it stands out, plain pure black background, no floor, no shadow, no text")
SWIM = (f"swimming, its body straight and level, all its fins spread, {SIDE}")
UPRIGHT = ("standing upright facing the camera, full body, arms hanging straight down and clearly apart from its sides, legs straight and a little "
           "apart, the whole figure in the picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text")
POSES.update({"husk": UPRIGHT, "zombie": UPRIGHT, "skeleton": UPRIGHT, "iron_golem": UPRIGHT, "enderman": UPRIGHT,
              "snow_golem": "standing upright facing the camera, the whole figure in the picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text"})
POSES.update({"bat": WINGS, "phantom": WINGS,
              "parrot": f"standing on its two legs with its wings folded, {SIDE}",
              "bee": f"hovering, brightly lit so that it stands out, {SIDE}",
              "allay": f"hovering upright, facing the camera and a little to the right, wings spread to its sides, the whole creature in the picture, plain pure black background, no floor, no shadow, no text",
              "vex": f"hovering upright, facing the camera and a little to the right, wings spread to its sides, the whole creature in the picture, plain pure black background, no floor, no shadow, no text",
              "squid": f"hanging upright in the water, seen from the side, brightly lit so that it stands out, the whole animal in the picture, plain pure black background, no floor, no shadow, no text",
              "glow_squid": f"hanging upright in the water, seen from the side, the whole animal in the picture, plain pure black background, no floor, no shadow, no text",
              "cod": SWIM, "salmon": SWIM, "pufferfish": SWIM, "tropicalfish": SWIM, "dolphin": SWIM,
              "guardian": "swimming, facing the camera and turned a little to the right, its single eye clearly visible, its tail behind it to the left, the whole creature in the picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text",
              "elder_guardian": "swimming, facing the camera and turned a little to the right, its single eye clearly visible, its tail behind it to the left, the whole creature in the picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text",
              "tadpole": SWIM, "nautilus": SWIM})


# The last round (2026-10-31): bare bones stay statues; pale colours not pure white; dark bodies brightly lit.
BONES = ("thick solid arm and leg bones, a big skull with dark eye sockets, the ribs as shallow ridges on a solid chest, no holes and no gaps anywhere")
ANIMALS.update({
    "drowned": "one single realistic drowned zombie man with waterlogged teal-blue decayed skin, green algae patches on his head and shoulders, glowing pale cyan eyes, a torn rust-brown rag shirt and shredded dark teal trousers",
    "stray": f"one single skeleton statue carved from one solid block of pale frosty blue-grey bone, {BONES}, tattered grey-blue rags fused to its chest and hips",
    "bogged": f"one single skeleton statue carved from one solid block of mossy grey-green bone with patches of green moss, {BONES}, small red and brown mushrooms growing on top of its skull",
    "parched": f"one single skeleton statue carved from one solid block of dry sand-yellow bone, {BONES}, faded dusty cloth wrappings fused around its hips",
    "wither_skeleton": f"one single tall skeleton statue carved from one solid block of dark charcoal grey bone, {BONES}, brightly lit so that it stands out",
    "piglin": "one single realistic upright pig-man with pink-tan bare skin, a wide flat pig snout with two small upward tusks, long floppy ears, narrow eyes, a brown leather strap across the bare chest, a gold belt buckle and brown leather trousers",
    "piglin_brute": "one single massive muscular upright pig-man with scarred pink-tan skin, a wide flat pig snout with two big upward tusks, long floppy ears, a black sleeveless jerkin with gold trim, a wide gold belt and black trousers",
    "zombie_pigman": "one single rotting upright zombie pig-man with mottled pink and pale green decayed skin, one side of its face a bare bone skull, a wide flat pig snout with tusks, exposed ribs in relief and ragged brown trousers",
    "villager": "one single realistic elderly peasant man with a bald head, a very long large nose, one thick joined eyebrow, tan skin, wearing a plain brown wool robe to his knees, his hands tucked together inside his sleeves across his chest",
    "wandering_trader": "one single realistic elderly travelling merchant with a very long large nose, tan skin, wearing a deep blue hooded robe to his knees with red and gold trim, his hands tucked together inside his sleeves across his chest",
    "zombie_villager": "one single rotting zombie peasant man with green decayed skin, a bald head, a very long large nose, dark sunken eyes, wearing a tattered brown robe to his knees",
    "witch": "one single realistic old witch with grey-tan wrinkled skin, a very long large nose with a wart, a tall black pointed hat with a wide brim and a dark purple robe reaching her feet, her hands tucked together inside her sleeves across her chest, brightly lit so that she stands out",
    "pillager": "one single realistic hostile grey-skinned man with a very long large nose, a bald head, one thick dark eyebrow, wearing a dark red plaid shirt, a grey leather vest and dark brown trousers",
    "vindicator": "one single realistic hostile grey-skinned man with a very long large nose, a bald head, one thick dark eyebrow, wearing a charcoal grey leather coat and dark teal trousers",
    "evocation_illager": "one single realistic grey-skinned sorcerer with a very long large nose, a bald head, one thick dark eyebrow, wearing a long dark charcoal robe with thin gold trim reaching his shins, brightly lit so that he stands out",
    "warden": "one single huge humanoid cave monster with smooth dark teal-blue skin, no eyes, a wide head with a wide jaw, two large curved horn-like tendrils on its head, a ribcage in relief on its chest glowing bright cyan, very long thick arms and short thick legs, brightly lit so that it stands out",
    "creaking": "one single tall thin humanoid monster made of solid dark grey-brown tree bark and gnarled wood, glowing orange eyes, thick solid limbs, no holes and no gaps, brightly lit so that it stands out",
    "copper_golem": "one single small humanoid statue made of smooth shiny orange copper metal, a large wide head with a big nose and round eyes, a thin short lightning rod on top of its head, a small stubby body, short thick arms and short thick legs",
    "blaze": "one single strange fire creature statue carved from one solid block of glowing yellow and orange molten stone, a square head with dark eyes on top, a wide stack of many thick golden rods fused together below it, no holes and no gaps",
    "breeze": "one single strange wind creature statue, a big rounded pale blue and lilac stone head with white glowing eyes on top and three thick pale blue crystal rods fused to the underside of the head, pointing down, no holes and no gaps",
    "ghast": "one single huge floating pale grey ghost creature, a big rounded square balloon shaped body, a small sad face on its front with closed eyes and a small dark mouth, nine short thick pale tentacles hanging below it in three rows, evenly lit",
    "happy_ghast": "one single huge floating pale cream ghost creature, a big rounded square balloon shaped body, a small cute happy face on its front with round dark eyes and a small smile, nine short thick pale tentacles hanging below it in three rows, evenly lit",
    "magma_cube": "one single rounded cube shaped creature made of dark cracked volcanic rock, glowing orange lava in the cracks, two glowing orange eyes on its front",
    "slime": "one single rounded cube shaped creature made of solid opaque glossy bright green jelly, darker green soft square eyes and a small mouth on its front, evenly lit",
    "sulfur_cube": "one single rounded cube made of solid pale yellow and tan crumbly sulfur rock, with soft mottled colours, no face, evenly lit",
    "shulker": "one single rounded cube shaped closed shell made of purple-lilac ceramic plates with a ridged square pattern, a thin horizontal seam around its middle, evenly lit",
    "shulker_head": "one single small smooth pale yellow-green round head creature with two small dark eyes, no neck, no body, evenly lit",
    "wither": "one single three-headed monster statue carved from one solid block of dark charcoal grey stone, three big skulls side by side on a wide thick shoulder beam, the middle skull largest, a thick spine with shallow rib ridges below the beam tapering into a thick curled tail, no arms and no legs, no holes and no gaps, brightly lit so that it stands out",
    "zombie_nautilus": "one single dead chambered nautilus with a coiled sun-bleached pale cream shell overgrown with green moss and small coral, and many short thick dark teal tentacles",
    "ender_dragon": "one single huge wingless dragon with smooth dark charcoal grey scales and a row of low dark grey spikes along its back, a long neck stretched straight forward, a long thin tail stretched straight back, four legs, glowing magenta eyes, brightly lit so that it stands out",
    "dragon_wing": "one single huge bat-like dragon wing seen flat from directly above, spread wide, thin dark grey leathery membrane with pale grey bone struts, attached to nothing, evenly lit",
    "bat_body": "one single realistic small dark brown bat body with no wings at all, short smooth fur, large ears, a small furry head and short legs, brightly lit",
    "bat_wing": "one single bat wing seen flat from directly above, spread wide, thin brown leathery membrane with darker bone struts, attached to nothing, evenly lit",
})
FLAT_ON = "the whole thing in the picture with wide margins all round, plain pure black background, no floor, no shadow, no text"
FOLDED = ("standing upright facing the camera, full body, arms folded across the chest, legs straight and a little apart, the whole figure in the picture with space around it, even soft lighting, plain pure black background, no floor, no shadow, no text")
FRONT = "floating, facing the camera and turned a little to the right, the whole creature in the picture with space around it, plain pure black background, no floor, no shadow, no text"
for _n in ("drowned", "stray", "bogged", "parched", "wither_skeleton", "piglin", "piglin_brute", "zombie_pigman", "zombie_villager", "pillager", "vindicator", "evocation_illager", "warden", "creaking", "copper_golem", "wither"):
    POSES[_n] = UPRIGHT
for _n in ("villager", "wandering_trader", "witch"):
    POSES[_n] = FOLDED
for _n in ("blaze", "breeze", "ghast", "happy_ghast", "magma_cube", "slime", "sulfur_cube", "shulker", "shulker_head"):
    POSES[_n] = FRONT
POSES.update({"zombie_nautilus": SWIM, "dragon_wing": FLAT_ON, "bat_wing": FLAT_ON, "ender_dragon": f"standing on its four legs, {SIDE}", "bat_body": f"standing on its short legs, {SIDE}"})


# Second tries, from the first sheets: open ribs and pelvises came back with gaps (the 3D step shreds them), so the
# torso is covered; normal noses, so the nose is asked for harder; the wither grew legs; the blaze's and ghast's parts go separately.
ANIMALS.update({
    "stray": "one single skeleton with pale frosty blue-grey bones as thick as broom handles, a big skull with dark eye sockets, wearing a thick tattered grey-blue hooded cloak wrapped tightly around its whole torso and hips so that no ribs show, no gaps anywhere",
    "bogged": "one single skeleton with mossy grey-green bones as thick as broom handles, a big skull with dark eye sockets, small red and brown mushrooms growing on top of its skull, its whole torso and hips completely covered by a thick solid layer of green moss so that no ribs show, no gaps anywhere",
    "parched": "one single skeleton with dry sand-yellow bones as thick as broom handles, a big skull with dark eye sockets, its whole torso and hips tightly wrapped in thick dusty tan cloth bandages so that no ribs show, no gaps anywhere",
    "wither_skeleton": "one single tall skeleton with dark charcoal grey bones as thick as broom handles, a big dark skull with black eye sockets, wearing a thick tattered dark grey tunic wrapped tightly around its whole torso and hips so that no ribs show, no gaps anywhere, brightly lit so that it stands out",
    "villager": "one single elderly peasant man, a caricature with an enormous long bulbous nose hanging down past his mouth, a bald head, one thick joined eyebrow, tan skin, wearing a plain brown wool robe to his knees, his hands tucked together inside his sleeves across his chest",
    "wandering_trader": "one single elderly travelling merchant, a caricature with an enormous long bulbous nose hanging down past his mouth, tan skin, wearing a deep blue hooded robe to his knees with red and gold trim, his hands tucked together inside his sleeves across his chest",
    "witch": "one single old witch, a caricature with an enormous long crooked nose with a wart hanging down past her mouth, grey-tan wrinkled skin, a tall black pointed hat with a wide brim and a dark purple robe reaching her feet, her hands tucked together inside her sleeves across her chest, brightly lit so that she stands out",
    "warden": "one single huge humanoid cave monster with smooth dark teal-blue skin, a completely blank smooth face with no eyes at all above a wide dark mouth, two short curled tendrils on the sides of its head, a ribcage in relief on its chest glowing bright cyan, very long thick arms and short thick legs, brightly lit so that it stands out",
    "copper_golem": "one single small sturdy humanoid robot built of riveted orange copper plates, a wide box shaped head with a big blunt nose and two round dark eyes, a short copper lightning rod on top of its head, a blocky chest, short thick arms and short thick legs",
    "blaze_head": "one single square block shaped head of glowing molten yellow and orange fire rock, two dark eyes, no mouth, no neck, nothing below it",
    "blaze_rod": "one single thick straight glowing golden yellow metal rod standing upright, rounded ends, slightly uneven surface, evenly lit",
    "breeze": "one single strange wind creature statue, a big rounded block of pale blue and lilac wind-carved stone with swirling grooves, two glowing white eyes, no nose and no mouth, three thick pale blue crystal rods fused to its underside pointing down, no holes and no gaps",
    "wither": "one single monster bust statue carved from one solid block of dark charcoal grey stone, three big skulls side by side on a wide thick shoulder beam, the middle skull largest, below the beam a narrow chest with shallow rib ridges that tapers into one single thick curved tail, it has no arms and no legs at all, no holes and no gaps, brightly lit so that it stands out",
    "zombie_nautilus": "one single realistic dead chambered nautilus with a smooth coiled shell, pale bone-white with green moss patches, and many short thick dark teal tentacles",
})
POSES.update({"blaze_head": FRONT, "blaze_rod": "standing upright, the whole rod in the picture with space around it, plain pure black background, no floor, no shadow, no text"})


# Variants (2026-10-31): each coat its own picture and 3D run, in its parent's pose. Pale coats stay off pure white; dark ones are lit brightly.
def _horse(c): return f"one single realistic adult horse, {c}, with a short smooth glossy coat, a short trimmed mane close to its neck and a short tail"
def _wolf(c): return f"one single realistic adult wolf, {c}, with short smooth dense fur, upright ears and a straight tail hanging down"
def _cat(c): return f"one single realistic adult house cat, {c}, with short smooth fur, upright ears and a long tail held straight out behind it"
def _rabbit(c): return f"one single realistic adult rabbit, {c}, with short smooth fur and a small tail"
def _axolotl(c): return f"one single realistic axolotl with smooth solid opaque {c} skin all over, a wide flat head, short thick gills of the same colour, four small legs and a short thick solid tail"
def _llama(c, extra=""): return f"one single realistic adult llama with short smooth {c} wool, a long upright neck, banana shaped ears{extra}"
_BLANKET = ", wearing a bright blue and gold patterned blanket on its back"
def _fish(c, shape): return f"one single realistic {shape} tropical fish, {c}, solid opaque fins"
SLIM, TALL = "small slender", "tall flat disc shaped"
VARIANTS = {
    "cow_warm": ("cow", "one single realistic adult cow with a short smooth golden tan hide, darker brown lower legs and a pair of long upcurved pale horns"),
    "cow_cold": ("cow", "one single realistic adult cow with a short dense dark brown coat, a pale muzzle and short curved horns, brightly lit"),
    "pig_warm": ("pig", "one single realistic adult pig with a smooth ginger orange hide with darker brown patches"),
    "pig_cold": ("pig", "one single realistic adult pig with a smooth dark slate grey hide, a pale grey belly and a pink snout, brightly lit"),
    "chicken_warm": ("chicken", "one single realistic pale orange tan hen with smooth tight feathers, a red comb and wattle, a yellow beak and yellow legs, brightly and evenly lit"),
    "chicken_cold": ("chicken", "one single realistic slate blue grey hen with smooth tight feathers, a red comb and wattle, a yellow beak and yellow legs, brightly and evenly lit"),
    "mooshroom_brown": ("mooshroom", "one single realistic adult cow with a warm tan brown hide, with three big brown mushrooms with flat caps growing upright from the middle of its back"),
    "llama_white": ("llama", _llama("pale ivory")), "llama_brown": ("llama", _llama("rich reddish brown")), "llama_gray": ("llama", _llama("slate grey")),
    "trader_llama_white": ("llama", _llama("pale ivory", _BLANKET)), "trader_llama_brown": ("llama", _llama("rich reddish brown", _BLANKET)), "trader_llama_gray": ("llama", _llama("slate grey", _BLANKET)),
    "horse_white": ("horse", _horse("pale cream white with a pale grey mane")), "horse_creamy": ("horse", _horse("golden palomino cream with a pale mane")), "horse_brown": ("horse", _horse("bay brown with a black mane")),
    "horse_black": ("horse", _horse("dark charcoal black, brightly lit so that it stands out")), "horse_gray": ("horse", _horse("dapple grey with a dark grey mane")), "horse_darkbrown": ("horse", _horse("very dark chocolate brown with a black mane")),
    "wolf_ashen": ("wolf", _wolf("ash grey with a pale chest")), "wolf_black": ("wolf", _wolf("dark charcoal black, brightly lit so that it stands out")), "wolf_chestnut": ("wolf", _wolf("chestnut red brown with a pale chest")),
    "wolf_rusty": ("wolf", _wolf("rusty orange brown with a dark back")), "wolf_snowy": ("wolf", _wolf("pale cream white, evenly lit")), "wolf_spotted": ("wolf", _wolf("sandy tan with dark spots")),
    "wolf_striped": ("wolf", _wolf("sandy brown with dark stripes on its back")), "wolf_woods": ("wolf", _wolf("dark brown with a grey muzzle, brightly lit")),
    "cat_white": ("cat", _cat("pale cream white with blue eyes")), "cat_black": ("cat", _cat("a black and white tuxedo cat, brightly lit")), "cat_red": ("cat", _cat("ginger orange with darker stripes")),
    "cat_siamese": ("cat", _cat("a Siamese with a pale cream body and dark brown face, ears, legs and tail")), "cat_british": ("cat", _cat("a British shorthair, blue grey and stocky")), "cat_calico": ("cat", _cat("calico with patches of white, orange and black")),
    "cat_persian": ("cat", _cat("a pale golden cream Persian with a flat face")), "cat_ragdoll": ("cat", _cat("a ragdoll, pale cream with a dark grey face, ears and tail and white paws")), "cat_all_black": ("cat", _cat("dark charcoal black with yellow eyes, brightly lit so that it stands out")),
    "cat_jellie": ("cat", _cat("dark grey with white speckles and a white chest, brightly lit")),
    "fox_arctic": ("fox", "one single realistic adult arctic fox with short smooth pale cream fur, dark eyes and nose and a thick tail held straight out level behind it, well above the ground and away from its legs, evenly lit"),
    "panda_brown": ("panda", "one single realistic adult giant panda standing on all fours with its legs clearly apart, short smooth dense pale tan fur, dark brown ears, eye patches, legs and shoulders, lit brightly so that every leg stands out"),
    "rabbit_white": ("rabbit", _rabbit("pale cream white, evenly lit")), "rabbit_black": ("rabbit", _rabbit("dark charcoal black, brightly lit so that it stands out")), "rabbit_white_splotched": ("rabbit", _rabbit("pale cream with dark brown patches")),
    "rabbit_gold": ("rabbit", _rabbit("golden tan")), "rabbit_salt": ("rabbit", _rabbit("mottled salt and pepper grey")),
    "frog_cold": ("frog", "one single realistic large green frog with smooth skin, big eyes and a pale belly"), "frog_warm": ("frog", "one single realistic large pale cream frog with smooth skin, big eyes and faint brown marks, evenly lit"),
    "axolotl_cyan": ("axolotl", _axolotl("pale cyan")), "axolotl_gold": ("axolotl", _axolotl("golden yellow")), "axolotl_wild": ("axolotl", _axolotl("dark olive brown with speckles")), "axolotl_blue": ("axolotl", _axolotl("deep blue")),
    "parrot_blue": ("parrot", "one single realistic hyacinth macaw parrot with smooth bright blue feathers, a dark curved beak with a yellow ring, a long blue tail and two short grey legs"),
    "parrot_green": ("parrot", "one single realistic green macaw parrot with smooth bright green feathers, a pale curved beak, a long green tail and two short grey legs"),
    "parrot_yellow_blue": ("parrot", "one single realistic blue and gold macaw parrot with smooth blue back and wings and a golden yellow chest, a dark curved beak, a long blue tail and two short grey legs"),
    "parrot_grey": ("parrot", "one single realistic African grey parrot with smooth mid grey feathers, a dark curved beak, a short red tail and two short grey legs"),
    "sheep_sheared": ("sheep", "one single realistic adult sheep freshly shorn, with short smooth pale cream hide all over, a slim body and slim legs, a pale face and ears, evenly lit"),
    "pufferfish_mid": ("pufferfish", "one single realistic half inflated pufferfish, yellow with brown patches on its back and a pale belly, short spines, big eyes"),
    "pufferfish_large": ("pufferfish", "one single realistic fully inflated round pufferfish like a ball, yellow with brown patches on its back and a pale belly, short thick spines, big eyes"),
    "tropicalfish_a2": ("tropicalfish", _fish("bright blue with a yellow tail", SLIM)), "tropicalfish_a3": ("tropicalfish", _fish("red with a white stripe along its side", SLIM)), "tropicalfish_a4": ("tropicalfish", _fish("yellow with a black stripe along its side", SLIM)),
    "tropicalfish_a5": ("tropicalfish", _fish("green and pink like a wrasse", SLIM)), "tropicalfish_a6": ("tropicalfish", _fish("white with black spots", SLIM)),
    "tropicalfish_b1": ("tropicalfish", _fish("a yellow and black striped angelfish", TALL)), "tropicalfish_b2": ("tropicalfish", _fish("a blue tang with a dark pattern and yellow tail", TALL)), "tropicalfish_b3": ("tropicalfish", _fish("an orange and white butterflyfish", TALL)),
    "tropicalfish_b4": ("tropicalfish", _fish("a moorish idol with broad black, white and yellow bands", TALL)), "tropicalfish_b5": ("tropicalfish", _fish("a red and turquoise discus", TALL)), "tropicalfish_b6": ("tropicalfish", _fish("a pale silver and purple angelfish", TALL)),
}
_V = "one single elderly peasant man, a caricature with an enormous long bulbous nose hanging down past his mouth, a bald head, one thick joined eyebrow, tan skin, wearing "
_H = ", his hands tucked together inside his sleeves across his chest"
VARIANTS.update({
    "villager_farmer": ("villager", _V + "a wide straw hat, a plain brown robe to his knees with a yellow straw coloured apron" + _H),
    "villager_fisherman": ("villager", _V + "a floppy pale bucket hat, a blue grey robe to his knees and a tan vest" + _H),
    "villager_shepherd": ("villager", _V + "a soft brown wool hat, a thick brown robe to his knees with a pale wool shawl" + _H),
    "villager_fletcher": ("villager", _V + "a small pointed green hat with a red feather, a brown robe to his knees with a tan leather quiver strap" + _H),
    "villager_librarian": ("villager", _V + "a square red and gold scholar's hat, a deep red robe to his knees with a white collar" + _H),
    "villager_cartographer": ("villager", _V + "a round monocle, a brown robe to his knees with a white shirt and a gold trimmed tan vest" + _H),
    "villager_cleric": ("villager", _V + "a long magenta purple robe with gold trim to his knees" + _H),
    "villager_armorer": ("villager", _V + "a black iron welding mask pushed up on his head, a black leather apron over a brown robe to his knees" + _H),
    "villager_weaponsmith": ("villager", _V + "a black leather eye patch, a black leather apron over a dark brown robe to his knees" + _H),
    "villager_toolsmith": ("villager", _V + "a brown leather cap, a heavy black apron with tool loops over a brown robe to his knees" + _H),
    "villager_butcher": ("villager", _V + "a red headband, a white apron over a brown robe to his knees" + _H),
    "villager_leatherworker": ("villager", _V + "a thick tan leather apron and long leather gloves over a brown robe to his knees" + _H),
    "villager_mason": ("villager", _V + "a dark grey apron with a brown belt over a dark robe to his knees" + _H),
    "villager_nitwit": ("villager", _V + "a plain bright green robe to his knees" + _H),
})
for _n, (_parent, _text) in VARIANTS.items():
    ANIMALS[_n] = _text
    if _parent in POSES:
        POSES[_n] = POSES[_parent]

# third tries: a cube pictured from above came back with a corner up; a bat lying down faces the sky when stood upright
LEVEL = "seen straight on at eye level, one flat side square to the camera, its top not visible, the whole creature in the picture with space around it, plain pure black background, no floor, no shadow, no text"
ANIMALS["bat_upright"] = "one single realistic small furry bat with no wings at all, standing upright on its short hind legs like a little person, big upright ears, small dark eyes, a small snout, short smooth dark brown fur, brightly lit"
POSES["bat_upright"] = UPRIGHT
POSES["magma_cube"] = LEVEL

# fourth tries: every see-through skeleton and tattered cloak came back in shreds; a chunky solid figure is what the 3D step can hold
def _toy(c, extra=""): return (f"one single chunky toy figure of a skeleton sculpted from one solid piece of {c} resin, thick rounded arms and legs like sausages, a big round skull with shallow dark eye sockets, "
                               f"a solid barrel chest with ribs only shallowly engraved on its surface, solid hips, no see-through gaps anywhere, matte{extra}")
ANIMALS.update({
    "skeleton": _toy("pale ivory"), "stray": _toy("pale frosty blue-grey", ", a smooth solid grey-blue hooded cape moulded tight to its back and shoulders"),
    "bogged": _toy("mossy grey-green", ", patches of green moss and small red and brown mushrooms moulded on top of its skull"), "parched": _toy("dry sand-yellow", ", smooth tan cloth wrappings moulded around its hips and chest"),
    "wither_skeleton": _toy("mid charcoal grey", ", a tall figure, brightly lit so that it stands out"),
    "witch": "one single old witch, a caricature with an enormous long crooked nose with a wart hanging down past her mouth, grey-tan wrinkled skin, a tall plum purple pointed hat with a wide brim, and a smooth plum purple robe reaching her feet with no tatters, her hands tucked together inside her sleeves across her chest, brightly lit",
})

# second tries of coats that came out the same as their neighbours (one seed, one wolf) or not at all (a black panda)
ANIMALS.update({
    "wolf_chestnut": _wolf("deep red brown all over with a darker back"), "wolf_rusty": _wolf("bright rust orange all over with a black saddle on its back"),
    "wolf_spotted": _wolf("pale cream with large dark brown blotches all over its body"), "wolf_striped": _wolf("pale sand with bold dark brown stripes across its back and sides"),
    "wolf_woods": _wolf("very dark brown with a grey muzzle and a dark grey chest, brightly lit"),
    "panda_brown": "one single realistic adult Qinling panda, a rare brown panda standing on all fours with its legs clearly apart, short smooth dense pale tan fur, warm chocolate brown (not black) ears, eye patches, legs and shoulders, lit brightly so that every leg stands out",
})

ANIMALS["witch"] = ("one single old witch, a caricature with an enormous long crooked nose with a wart hanging down past her mouth, grey-tan wrinkled skin, a tall plum purple pointed hat with a wide brim, wearing a thick stiff solid bell shaped "
                    "plum purple robe like a heavy wool tent with a smooth straight hem down to her feet, her hands tucked together inside her wide sleeves across her chest, brightly lit")

ANIMALS["wolf_black"] = _wolf("dark grey, almost black, lit brightly with a strong rim light so that its legs, tail and head stand out clearly from the background")

# fifth try: dark robes on black shred; a pale lavender one holds, and convert.py darkens it back to the witch's plum
ANIMALS["witch"] = ("one single old witch, a caricature with an enormous long crooked nose with a wart hanging down past her mouth, tan wrinkled skin, a tall pale lavender pointed hat with a wide brim, wearing a thick stiff solid bell shaped "
                    "pale lavender robe like a heavy wool tent with a smooth straight hem down to her feet, her hands tucked together inside her wide sleeves across her chest, evenly lit")


def prompt(name: str) -> str:
    return f"Photorealistic studio photograph of {ANIMALS[name]}, {POSES.get(name, POSE)}"


def drafts(names=None, seeds=(81, 82, 83)) -> None:
    import os
    seeds = tuple(int(x) for x in os.environ.get("SEEDS", "").split(",") if x) or seeds
    c = Comfy()
    for name in names or ANIMALS:
        for seed in seeds:
            g = patch(load_graph(Path("comfy/qwen_txt2img.json")), "1", text=prompt(name))
            g = patch(g, "4", seed=seed)
            print(c.run(g, OUT, f"{name}_s{seed}"), flush=True)


def on_black(name: str, seed: int) -> Path:
    import numpy as np
    from PIL import Image
    g = patch(load_graph(Path("comfy/qwen_rmbg.json")), "1", text=prompt(name))
    g = patch(g, "4", seed=seed)
    cut, _ = Comfy().run(g, OUT, f"{name}_cut_s{seed}", node_ids=["7", "8"])
    rgba = np.asarray(Image.open(cut).convert("RGBA")).astype(np.float32) / 255
    out = OUT / f"{name}_black_s{seed}.png"
    Image.fromarray((rgba[..., :3] * rgba[..., 3:] * 255 + 0.5).astype(np.uint8), "RGB").save(out)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["drafts"]:
        drafts(sys.argv[2:])
    else:
        for arg in sys.argv[1:]:
            name, seed = arg.rsplit(":", 1)
            print(on_black(name, int(seed)), flush=True)
