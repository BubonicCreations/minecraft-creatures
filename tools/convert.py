"""The six GLBs (run3d.py) as jointed Bedrock meshes in art/models/mesh, each drawn for one variant
by an entity override + render controller copy (glbmodel.mesh_for_variant), and a check sheet per
animal: six angles at rest, a wide step, the head turned 60 degrees and lowered.
Usage: convert.py <sheet_dir> <name>..."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mcreskin import glbmodel as G
from mcreskin import imaging as I
from mcreskin import render as R
from mcreskin.build import PackBuilder

V = Path("vanilla/bedrock-samples-v1.26.50.4/resource_pack")
ART = Path("art/models/mesh")
WORK = Path("work/models3d/mobs")
B, CACHE = PackBuilder(root=Path.cwd()), {}
sys.path.insert(0, str(Path(__file__).parent))
import plans as P  # noqa: E402
CAP = 0.55  # a run's stride: vanilla's 80 degree swing at most this much (44 degrees)
WALK = lambda e, b, ch, i: f"({e}) * math.min(1.0, {CAP} / math.max(query.modified_move_speed, 0.01))" if ch == "rotation" and i == 0 else None
CLAMP = lambda deg: (lambda e, b, ch, i: f"math.clamp({e}, -{deg}, {deg})" if ch == "rotation" and i == 0 else None)
LOOK = lambda yaw, pitch=25: (lambda e, b, ch, i: (f"math.clamp({e}, -{pitch}, {pitch})" if i == 0 else
                                                   f"math.clamp({e}, -{yaw}, {yaw})" if i == 1 else None) if ch == "rotation" else None)
NO_DROP = lambda e, b, ch, i: "0.0" if ch == "position" else None
# rearing up: vanilla pushes the hind legs 6 px back to its hip pivot; a real bear's are there already
REAR = lambda e, b, ch, i: "0.0" if ch == "position" and b in ("leg0", "leg1") else None
QUAD = lambda yaw=35: {"walk": ("animation.quadruped.walk", WALK), "look_at_target": ("animation.common.look_at_target", LOOK(yaw))}
MOBS = {  # texture key, vanilla's texture (finds the entity) and geometry, what draws the mesh, height in px,
          # where the head is cut (head_from, ears: see glbmodel.split_quadruped), animation edits
    "sheep": dict(texture="entity/sheep/sheep_mesh", key="entity/sheep/sheep", gid="geometry.sheep.v1.8",
                  controller="controller.render.sheep.v2", variant="!query.is_baby && !query.is_sheared",
                  height=21.0, wool={"body"},
                  anims={**QUAD(), "grazing": ("animation.sheep.grazing.v2", "GRAZE")}),
    "polar_bear": dict(texture="entity/polar_bear/polar_bear_mesh", key="entity/polar_bear", gid="geometry.polarbear",
                       controller="controller.render.polar_bear.v2", variant="!query.is_baby", height=23.0, ears=False,
                       anims={**QUAD(30), "move": ("animation.polarbear.move", REAR)}),
    "llama": dict(texture="entity/llama/llama_mesh", key="entity/llama/llama_creamy", gid="geometry.llama.v1.8",
                  controller="controller.render.llama.v2", height=30.0,
                  variant="!query.is_baby && query.variant == 0 && variable.DecorTextureIndex == 0 && query.mark_variant != 1",
                  anims=QUAD()),
    "goat": dict(texture="entity/goat/goat_mesh", key="entity/goat/goat", gid="geometry.goat",
                 controller="controller.render.goat.v2", variant="!query.is_baby", height=22.0,
                 anims={"walk": ("animation.goat.walk", CLAMP(30)), "look_at_target": ("animation.goat.look_at_target", LOOK(35)),
                        "ram_attack": ("animation.goat.ram_attack", NO_DROP)}),
    "hoglin": dict(texture="entity/hoglin/hoglin_mesh", key="entity/hoglin/hoglin", gid="geometry.hoglin",
                   controller="controller.render.hoglin.v2", variant="!query.is_baby", height=29.0,
                   anims={"walk": ("animation.hoglin.walk", CLAMP(40)), "look_at_target": ("animation.hoglin.look_at_target", LOOK(25))}),
    "mooshroom": dict(texture="entity/cow/mooshroom_mesh", key="entity/cow/mooshroom_v2", gid="geometry.mooshroom.v2",
                      controller="controller.render.mooshroom.v2", variant="!query.is_baby && query.variant == 0", height=23.5,
                      anims=QUAD()),
}


SCALE = lambda k: (lambda e, b, ch, i: f"({e}) * {k}" if ch == "rotation" and i == 0 else None)
FLAT = lambda e, b, ch, i: "0.0" if ch == "rotation" else None  # a setup that lays a box body down: ours stands as it is
ZERO = lambda e, b, ch, i: "0.0" if ch in ("rotation", "position") else None  # a pose that places vanilla's model; ours is already there
QUARTERS = {"leg0": "right_hind_leg", "leg1": "left_hind_leg", "leg2": "right_front_leg", "leg3": "left_front_leg"}
HORSE = lambda eat=0.4: {"walk": ("animation.horse.v3.walk", CLAMP(35)), "look_at_player": ("animation.horse.v3.look_at_player", LOOK(20, 20)),
                         "eat": ("animation.horse.v3.eat", SCALE(eat))}
HORSE_SHOTS = [("grazing", "animation.horse.v3.eat", 0.0, {"variable.eat_anim": 1.0, "query.life_time": 0.0}),
               ("rearing", "animation.horse.v3.rear", 0.0, {"variable.stand_anim": 1.0, "variable.head_x_rot": 0.0, "query.life_time": 0.0})]
NEXT = {  # the next round (2026-10-30). `base`: vanilla's texture name in the entity file (finds its path)
    "horse": dict(base="base_chestnut", entity="horse_v3", gid="geometry.horse.v3", controller="controller.render.horse.v4",
                  variant="!query.is_baby && query.variant == 2 && !variable.armor_texture_slot",
                  layers={1: "Texture.markings_none"}, rename={"head": "Neck"}, head_from=0.1, anims=HORSE(), shots=HORSE_SHOTS, saddle="Saddle"),
    "donkey": dict(base="donkey", entity="donkey_v3", gid="geometry.horse.v3", controller="controller.render.donkey.v4",
                   variant="!query.is_baby && !query.is_chested", rename={"head": "Neck"}, head_from=0.1, anims=HORSE(), shots=HORSE_SHOTS, saddle="Saddle"),
    "mule": dict(base="mule", entity="mule_v3", gid="geometry.horse.v3", controller="controller.render.mule.v4",
                 variant="!query.is_baby && !query.is_chested", rename={"head": "Neck"}, head_from=0.1, anims=HORSE(), shots=HORSE_SHOTS, saddle="Saddle"),
    "skeleton_horse": dict(base="skeleton", entity="skeleton_horse_v3", gid="geometry.horse.v3", controller="controller.render.skeleton_horse.v4",
                           variant="!query.is_baby", rename={"head": "Neck"}, head_from=0.1, anims=HORSE(), shots=HORSE_SHOTS, saddle="Saddle"),
    "zombie_horse": dict(base="zombie", entity="zombie_horse_v3", gid="geometry.horse.v3", controller="controller.render.zombie_horse.v4",
                         variant="!query.is_baby", rename={"head": "Neck"}, head_from=0.1, anims=HORSE(), shots=HORSE_SHOTS, saddle="Saddle"),
    "wolf": dict(base="pale_default", gid="geometry.wolf", controller="controller.render.wolf.v2", variant="!query.is_baby && query.variant == 0",
                 anims={"wolf_setup": ("animation.wolf.setup", FLAT), "wolf_look_at": ("animation.common.look_at_target", LOOK(35)),
                        "wolf_leg_default": ("animation.wolf.leg_default", CLAMP(35)), "wolf_sitting": ("animation.wolf.sitting", "SIT")},
                 shots=[("sitting", "animation.wolf.sitting", 0.0, {})]),
    "cat": dict(base="tabby", gid="geometry.cat", controller="controller.render.cat.v2", variant="!query.is_baby && query.variant == 8", height=19.0,
                anims={"walk": ("animation.cat.walk", CLAMP(35)), "sprint": ("animation.cat.sprint", CLAMP(40)),
                       "look_at_target": ("animation.common.look_at_target", LOOK(35)),
                       "sit": ("animation.cat.sit", "SIT"), "lie_down": ("animation.cat.lie_down", "LIE")},
                shots=[("sit", "animation.cat.sit", 0.0, {}), ("lie down", "animation.cat.lie_down", 0.0, {}), ("sneak", "animation.cat.sneak", 0.0, {})]),
    "ocelot": dict(base="wild", gid="geometry.ocelot.v1.8", controller="controller.render.ocelot.v2", variant="!query.is_baby", height=13.0,
                   anims={"walk": ("animation.ocelot.walk", CLAMP(35)), "sprint": ("animation.ocelot.sprint", CLAMP(40)),
                          "look_at_target": ("animation.common.look_at_target", LOOK(35)), "sit": ("animation.ocelot.sit", "SIT")},
                   shots=[("sit", "animation.ocelot.sit", 0.0, {}), ("sneak", "animation.ocelot.sneak", 0.0, {})]),
    "fox": dict(base="red", gid="geometry.fox", controller="controller.render.fox.v2", variant="!query.is_baby && query.variant == 0 && !query.is_sleeping",
                anims={"setup": ("animation.fox.setup", FLAT), **QUAD(), "sit": ("animation.fox.sit", "SIT")},
                shots=[("sit", "animation.fox.sit", 0.0, {}), ("crouch", "animation.fox.crouch", 0.0, {"variable.crouch_amount": 1.0})]),
    "panda": dict(base="default", gid="geometry.panda", controller="controller.render.panda.v2", variant="!query.is_baby && query.variant == 0",
                  upright=45.0, anims={**QUAD(), "sitting": ("animation.panda.sitting", "SIT"), "lying": ("animation.panda.lying", "LIE")},
                  shots=[("sitting", "animation.panda.sitting", 0.0, {}), ("lying", "animation.panda.lying", 0.0, {}), ("rolling", "animation.panda.rolling", 0.0, {"variable.rolling": 1.0})]),
    "camel": dict(base="default", gid="geometry.camel", controller="controller.render.camel.v2", variant="!query.is_baby", head_from=0.1, max_tilt=50.0, saddle="saddle", saddle_at={"z": (0.42, 0.72)},
                  anims={"sit_down": ("animation.camel.sit_down", "SIT"), "sitting": ("animation.camel.sit", "SIT"), "stand_up": ("animation.camel.stand_up", "STAND")},
                  shots=[("walking", "animation.camel.walk", 0.3, {}), ("sitting", "animation.camel.sit", 0.0, {}), ("dash", "animation.camel.dash", 0.3, {})]),
    "camel_husk": dict(base="default", gid="geometry.camel_husk", controller="controller.render.camel_husk", variant="true", head_from=0.1, max_tilt=50.0, saddle="saddle", saddle_at={"z": (0.42, 0.72)}, tint=(0.62, 0.54, 0.46),
                       anims={"sit_down": ("animation.camel.sit_down", "SIT"), "sitting": ("animation.camel.sit", "SIT"), "stand_up": ("animation.camel.stand_up", "STAND")},
                  shots=[("walking", "animation.camel.walk", 0.3, {}), ("sitting", "animation.camel.sit", 0.0, {}), ("dash", "animation.camel.dash", 0.3, {})]),
    "armadillo": dict(base="default", gid="geometry.armadillo", controller="controller.render.armadillo.v2",
                      variant="!query.is_baby && !variable.use_rolled_up_model", anims={},
                      shots=[("walk", "animation.armadillo.walk", 0.3, {})]),
    "sniffer": dict(base="default", gid="geometry.sniffer", controller="controller.render.sniffer.v2", variant="!query.is_baby", rename=QUARTERS, anims={},
                    shots=[("walk", "animation.sniffer.walk", 0.3, {"variable.moving": 1.0}), ("dig", "animation.sniffer.dig", 2.0, {}), ("search", "animation.sniffer.search", 0.5, {})]),
    "ravager": dict(base="default", gid="geometry.ravager", controller="controller.render.ravager", variant="true",
                    rename={f"leg{i}": f"leg{i}" for i in range(4)}, anims={"walk": ("animation.ravager.walk", CLAMP(35))}),
    "zoglin": dict(base="default", gid="geometry.hoglin", controller="controller.render.zoglin.v2", variant="!query.is_baby",
                   anims={"walk": ("animation.hoglin.walk", CLAMP(40)), "look_at_target": ("animation.hoglin.look_at_target", LOOK(25))}),
    "trader_llama": dict(base="creamy", gid="geometry.llama.v1.8", controller="controller.render.llama.v2", rc_suffix="trader_mesh",
                         variant="!query.is_baby && query.variant == 0", layers={1: "Texture.decor_none", 2: "Texture.decor_none"}, anims=QUAD()),
    "creeper": dict(base="default", gid="geometry.creeper.v1.8", controller="controller.render.creeper", variant="true", align=False, plan=P.CREEPER,
                    rename={f"leg{i}": f"leg{i}" for i in range(4)},
                    anims={"creeper_head": ("animation.common.look_at_target", LOOK(35)), "creeper_legs": ("animation.creeper.legs", CLAMP(30))}),
}
TIP_SPIDER = -45.0
REAR_FRONT = lambda back_r, back_l, front_r, front_l: {"leg0": back_r, "leg1": back_l, "leg2": front_r, "leg3": front_l}
SMALL = {
    "rabbit": dict(base="brown", gid="geometry.rabbit.v2", controller="controller.render.rabbit.v2", variant="!query.is_baby && query.variant == 0",
                   rename=REAR_FRONT("rightBackLeg", "leftBackLeg", "rightFrontLeg", "leftFrontLeg"), links=True, anims={"move": ("animation.rabbit.move.v2", lambda e, b, ch, i: f"math.clamp({e}, -25, 25)" if ch == "rotation" and i == 0 and b == "body" else None)},
                   shots=[("hop 1", "animation.rabbit.move.v2", 0.1, {}), ("hop 2", "animation.rabbit.move.v2", 0.3, {})]),
    "frog": dict(base="temperate", gid="geometry.frog", controller="controller.render.frog", variant="query.variant == 0",
                 rename=REAR_FRONT("right_leg", "left_leg", "right_arm", "left_arm"), plan=P.parts(("head", "z", 0.25, "-", {"y": (0.35, 1.01)}), *P.legs_under(0.2)), anims={},
                 shots=[("walk", "animation.frog.walk", 0.3, {}), ("jump", "animation.frog.jump", 0.2, {}), ("swim", "animation.frog.swim", 0.3, {})]),
    "turtle": dict(base="default", gid="geometry.turtle", controller="controller.render.turtle.v2", variant="!query.is_baby",
                   height=12.0, rename={f"leg{i}": f"leg{i}" for i in range(4)}, plan=P.parts(("head", "z", 0.2, "-", {"y": (0.25, 1.01)}), *P.legs_under(0.35)), anims={},
                   shots=[("walk", "animation.turtle.ground_move", 0.3, {}), ("swim", "animation.turtle.move", 0.3, {})]),
    "axolotl": dict(base="lucy", gid="geometry.axolotl", controller="controller.render.axolotl.v2", variant="!query.is_baby && query.variant == 0",
                    rename=REAR_FRONT("right_leg", "left_leg", "right_arm", "left_arm"), plan=P.parts(("tail", "z", 0.5, "+"), ("head", "z", 0.17, "-"), ("leg0", "y", 0.28, "-", {"x": (-1.01, 0.0), "z": (0.3, 0.5)}), ("leg1", "y", 0.28, "-", {"x": (0.0, 1.01), "z": (0.3, 0.5)}),
                                 ("leg2", "y", 0.28, "-", {"x": (-1.01, 0.0), "z": (0.1, 0.3)}), ("leg3", "y", 0.28, "-", {"x": (0.0, 1.01), "z": (0.1, 0.3)})), anims={},
                    shots=[("walk", "animation.axolotl.walk_floor", 0.3, {}), ("swim", "animation.axolotl.swim", 0.3, {}), ("idle", "animation.axolotl.idle_floor", 0.5, {})]),
    "chicken": dict(base="default", gid="geometry.chicken.v1.12", controller="controller.render.chicken.v3", variant="!query.is_baby && v.index == 0",
                    plan=P.parts(("head", "z", 0.3, "-", {"y": (0.55, 1.01)}), ("leg0", "y", 0.26, "-", {"x": (-1.01, 0.0)}), ("leg1", "y", 0.26, "-", {"x": (0.0, 1.01)})),
                    rename={"leg0": "leg0", "leg1": "leg1"}, anims={"look_at_target": ("animation.common.look_at_target", LOOK(35))},
                    shots=[("walk", "animation.chicken.move", 0.3, {"query.modified_move_speed": 1.0})]),
    "strider": dict(base="default", gid="geometry.strider", controller="controller.render.strider.v2", variant="!query.is_baby", plan=P.parts(("right_leg", "y", 0.45, "-", {"x": (-1.01, 0.0)}), ("left_leg", "y", 0.45, "-", {"x": (0.0, 1.01)})),
                    anims={}, shots=[("walk", "animation.strider.walk", 0.3, {"query.modified_move_speed": 1.0})]),
    "spider": dict(base="default", gid="geometry.spider.v1.8", controller="controller.render.spider", variant="true", tip=TIP_SPIDER, align=False, span="vanilla", plan=P.spider, rename={"body": "body0"},
                   anims={"default_leg_pose": ("animation.spider.default_leg_pose", FLAT)}, shots=[("walk", "animation.spider.walk", 0.3, {"query.modified_move_speed": 1.0})]),
    "cave_spider": dict(base="default", gid="geometry.spider.v1.8", controller="controller.render.spider", rc_suffix="cave_mesh", variant="true", tip=TIP_SPIDER, align=False, span="vanilla", plan=P.spider, rename={"body": "body0"},
                        anims={"default_leg_pose": ("animation.spider.default_leg_pose", FLAT)}, shots=[("walk", "animation.spider.walk", 0.3, {"query.modified_move_speed": 1.0})]),
    "silverfish": dict(base="default", gid="geometry.silverfish", controller="controller.render.silverfish", variant="true", rename={"body": "bodyPart_2"}, wave=True, clip=(0.04, 0.5), span="vanilla", place_like_vanilla=True,
                       plan=P.chain([("bodyPart_1", 0.32), ("bodyPart_0", 0.18)], [("bodyPart_3", 0.5), ("bodyPart_4", 0.65), ("bodyPart_5", 0.78), ("bodyPart_6", 0.9)]),
                       anims={}, shots=[("move", "animation.silverfish.move", 0.3, {"query.modified_move_speed": 1.0})]),
    "endermite": dict(base="default", gid="geometry.endermite", controller="controller.render.endermite", variant="true", rename={"body": "section_2"}, wave=True, clip=(0.2, 1.0), span="vanilla", place_like_vanilla=True,
                      plan=P.chain([("section_1", 0.42), ("section_0", 0.22)], [("section_3", 0.72)]),
                      anims={}, shots=[("move", "animation.endermite.move", 0.3, {"query.modified_move_speed": 1.0})]),
}
FLOAT = dict(place_like_vanilla=True, align=False)
FLIERS = {
    "bat": dict(base="default", gid="geometry.bat_v2", controller="controller.render.bat", variant="true", forward=(0.0, 0.0, 1.0), height=12.0, rename={"head": "Head"}, **FLOAT,
                plan=P.parts(("head", "y", 0.68, "+", {"x": (-0.25, 0.25)}), ("rightWing", "x", -0.22, "-"), ("rightWingTip", "x", -0.62, "-", None, {"of": "rightWing"}),
                             ("leftWing", "x", 0.22, "+"), ("leftWingTip", "x", 0.62, "+", None, {"of": "leftWing"})),
                anims={}, shots=[("flying", "animation.bat.flying", 0.125, {}), ("resting", "animation.bat.resting", 0.0, {})]),
    "parrot": dict(base="red_blue", gid="geometry.parrot", controller="controller.render.parrot", variant="query.variant == 0", height=9.0,
                   plan=P.parrot, anims={"base": ("animation.parrot.base", FLAT)}, shots=[("flying", "animation.parrot.moving", 0.1, {"query.anim_time": 0.1}), ("sitting", "animation.parrot.sitting", 0.0, {})]),
    "bee": dict(base="default", gid="geometry.bee", controller="controller.render.bee.v2", variant="!query.is_baby", place_like_vanilla=True, height=15.0, plan=P.bee,
                anims={}, shots=[("flying", "animation.bee.flying", 0.1, {})]),
    "allay": dict(base="default", gid="geometry.allay", controller="controller.render.allay", variant="true", forward=(0.3, 0.0, 1.0), **FLOAT, plan=P.fairy("right_arm", "left_arm", "right_wing", "left_wing"),
                  rename={"head": "look_at"}, links=True, keep=["rightItem"], anims={"fly": ("animation.allay.fly", lambda e, b, ch, i: f"({e}) * 0.35" if ch == "rotation" and i == 0 and b == "body" else None)},
                  shots=[("fly", "animation.allay.fly", 0.2, {}), ("idle", "animation.allay.idle", 0.2, {})]),
    "vex": dict(base="default", gid="geometry.vex.v1.8", controller="controller.render.vex", variant="true", forward=(0.3, 0.0, 1.0), **FLOAT, plan=P.fairy("rightArm", "leftArm", "rightWing", "leftWing"),
                keep=["rightItem", "leftItem"], anims={}, shots=[("idle", "animation.vex.idle", 0.2, {}), ("charge", "animation.vex.charge", 0.2, {})]),
    "phantom": dict(base="default", gid="geometry.phantom", controller="controller.render.phantom", variant="true", forward=(0.0, 0.0, 1.0), tip=-20.0, span="vanilla", **FLOAT,
                    plan=P.parts(("head", "z", 0.25, "-", {"x": (-0.22, 0.22)}), ("wing1", "x", -0.2, "-"), ("wingtip1", "x", -0.6, "-", None, {"of": "wing1"}),
                                 ("wing0", "x", 0.2, "+"), ("wingtip0", "x", 0.6, "+", None, {"of": "wing0"}), ("tail", "z", 0.62, "+"), ("tailtip", "z", 0.82, "+", None, {"of": "tail"})),
                    wave=True, anims={"phantom_base_pose": ("animation.phantom.base_pose", NO_DROP)}, shots=[("flap", "animation.phantom.move", 0.3, {})]),
}
SWIM_ = dict(place_like_vanilla=True, span="vanilla")
SWAY = lambda k=0.5: (lambda e, b, ch, i: f"({e}) * {k}" if ch == "rotation" and i == 1 else None)  # vanilla's fish sweep 60 degrees
SWIMMERS = {
    "cod": dict(base="default", gid="geometry.cod", controller="controller.render.cod", variant="true", **SWIM_, plan=P.parts(("head", "z", 0.27, "-"), ("tailfin", "z", 0.74, "+")),
                anims={"swim": ("animation.cod.swim", SWAY())}, shots=[("swim", "animation.cod.swim", 0.3, {"query.anim_time": 0.3})]),
    "salmon": dict(base="default", gid="geometry.salmon", controller="controller.render.salmon", variant="true", **SWIM_, rename={"body": "body_front"},
                   plan=P.parts(("head", "z", 0.22, "-"), ("body_back", "z", 0.52, "+"), ("tailfin", "z", 0.84, "+", None, {"of": "body_back"})),
                   anims={"swim": ("animation.salmon.swim", SWAY())}, shots=[("swim", "animation.salmon.swim", 0.3, {"query.anim_time": 0.3})]),
    "pufferfish": dict(base="default", gid="geometry.pufferfish.small.v1.8", controller="controller.render.pufferfish.small", variant="true", **SWIM_, plan=P.parts(("tailfin", "z", 0.8, "+")),
                       anims={}, shots=[("swim", "animation.pufferfish.swim", 0.3, {"query.anim_time": 0.3})]),
    "tropicalfish": dict(base="typeA", gid="geometry.tropicalfish_a", controller="controller.render.tropicalfish", variant="variable.TropicalFish.Base == 0", **SWIM_, material="cod",
                         plan=P.parts(("tailfin", "z", 0.72, "+")), anims={}, shots=[("swim", "animation.tropicalfish.swim", 0.3, {"query.anim_time": 0.3})]),
    "squid": dict(base="default", gid="geometry.squid", controller="controller.render.squid.v2", variant="!query.is_baby", place_like_vanilla=True, align=False, wave=True, stand=True, undo_rest="body", height=25.0, plan=P.squid, anims={},
                  shots=[("swim", "animation.squid.move", 0.3, {"variable.tentacle_angle": 40.0})]),
    "glow_squid": dict(base="default", gid="geometry.squid", controller="controller.render.glow_squid.v2", variant="!query.is_baby", place_like_vanilla=True, align=False, wave=True, stand=True, undo_rest="body", height=25.0, plan=P.squid, anims={},
                       shots=[("swim", "animation.squid.move", 0.3, {"variable.tentacle_angle": 40.0})]),
    "dolphin": dict(base="default", gid="geometry.dolphin", controller="controller.render.dolphin.v2", variant="!query.is_baby", **SWIM_,
                    plan=P.parts(("head", "z", 0.28, "-"), ("tail", "z", 0.62, "+"), ("tail_fin", "z", 0.86, "+", None, {"of": "tail"})),
                    anims={}, shots=[("swim", "animation.dolphin.move", 0.3, {"query.anim_time": 0.3})]),
    "guardian": dict(base="default", gid="geometry.guardian.v1.8", controller="controller.render.guardian", rc_suffix="guardian_mesh", variant="!query.is_elder && !query.is_ghost", **SWIM_, forward=(0.7, 0.0, 0.7), align=False, rename={"body": "head"},
                     plan=P.parts(("tailpart0", "z", 0.58, "+"), ("tailpart1", "z", 0.72, "+", None, {"of": "tailpart0"}), ("tailpart2", "z", 0.86, "+", None, {"of": "tailpart1"})),
                     anims={}, shots=[("swim", "animation.guardian.swim", 0.3, {})]),
    "elder_guardian": dict(base="elder", gid="geometry.guardian.v1.8", controller="controller.render.guardian", rc_suffix="elder_mesh", variant="query.is_elder && !query.is_ghost", **SWIM_, forward=(0.7, 0.0, 0.7), align=False, rename={"body": "head"},
                           plan=P.parts(("tailpart0", "z", 0.58, "+"), ("tailpart1", "z", 0.72, "+", None, {"of": "tailpart0"}), ("tailpart2", "z", 0.86, "+", None, {"of": "tailpart1"})),
                           anims={}, shots=[("swim", "animation.guardian.swim", 0.3, {})]),
    "tadpole": dict(base="default", gid="geometry.tadpole", controller="controller.render.tadpole", variant="true", **SWIM_, plan=P.parts(("tail", "z", 0.42, "+")),
                    anims={}, shots=[("swim", "animation.tadpole.swim", 0.3, {"query.anim_time": 0.3})]),
    "nautilus": dict(base="default", gid="geometry.nautilus", controller="controller.render.nautilus", variant="!query.is_baby && !query.is_saddled && !variable.has_armor", **SWIM_, forward=(-1.0, 0.0, 0.0), plan=lambda m, bone, *rest: (m, bone), rename={"body": "head"}, anims={}),
}
UP = dict(forward=(0.0, 0.0, 1.0), align=False, stand=True)
ARMS = {"move": ("animation.humanoid.move", CLAMP(40))}
WALKER = [("walk", "animation.humanoid.move", 0.0, {"query.modified_distance_moved": 0.0, "query.modified_move_speed": 1.0}),
          ("arms raised", "animation.zombie.attack_bare_hand", 0.5, {"variable.attack_time": 0.5, "query.is_baby": 0.0})]
HUMANOIDS = {
    "zombie": dict(base="default", gid="geometry.zombie.v1.8", controller="controller.render.zombie.v2", variant="!query.is_baby", **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER),
    "husk": dict(base="default", gid="geometry.zombie.husk.v1.8", controller="controller.render.zombie.v2", rc_suffix="husk_mesh", variant="!query.is_baby", **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER),
    "skeleton": dict(base="default", gid="geometry.skeleton.v1.8", controller="controller.render.skeleton", variant="true", tint=(1.25, 1.25, 1.2), **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "enderman": dict(base="default", gid="geometry.enderman.v1.8", controller="controller.render.enderman", variant="true", **UP, plan=P.human(head=0.86, arm=0.5), keep=["rightItem"],
                     anims={"base_pose": ("animation.enderman.base_pose", FLAT)}, shots=[("arms and legs", "animation.enderman.arms_legs", 0.3, {"query.modified_move_speed": 1.0})]),
    "iron_golem": dict(base="default", gid="geometry.irongolem", controller="controller.render.iron_golem", variant="true", **UP,
                       rc_base={"geometry": "Geometry.default", "materials": [{"*": "Material.default"}], "textures": ["Texture.default"]}, drop_rc=["controller.render.iron_golem_cracks"],
                       tint=(1.75, 1.75, 1.75), plan=P.human(head=0.82, arm=0.55, legs=0.36, names=("arm0", "arm1", "leg0", "leg1")), anims={"walk": ("animation.iron_golem.walk", CLAMP(30))},
                       shots=[("walk", "animation.iron_golem.walk", 0.3, {"query.modified_move_speed": 1.0}), ("attack", "animation.iron_golem.attack", 0.1, {})]),
    "snow_golem": dict(base="default", gid="geometry.snowgolem.v1.8", controller="controller.render.snowgolem", variant="true", **UP, tint=(1.3, 1.3, 1.3), rename={"body": "piece2"},
                       plan=P.parts(("head", "y", 0.66, "+"), ("piece1", "y", 0.36, "+"), ("arm1", "x", -0.42, "-", None, {"of": "piece1"}), ("arm2", "x", 0.42, "+", None, {"of": "piece1"})), anims={}),
}
MOBS.update(NEXT)
MOBS.update(HUMANOIDS)
NEXT.update(HUMANOIDS)
MOBS.update(SMALL)
NEXT.update(SMALL)
MOBS.update(FLIERS)
NEXT.update(FLIERS)
MOBS.update(SWIMMERS)
NEXT.update(SWIMMERS)

LAST_HUMANS = {  # the last round (2026-10-31)
    "drowned": dict(base="default", gid="geometry.zombie.drowned.v1.16", controller="controller.render.zombie.v2", rc_suffix="drowned_mesh", variant="!query.is_baby", **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER),
    "stray": dict(base="default", gid="geometry.skeleton.stray.v1.8", controller="controller.render.stray", variant="true", hide_rc=["controller.render.stray_clothes"], **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "bogged": dict(base="default", gid="geometry.skeleton.bogged", controller="controller.render.bogged", variant="true", hide_rc=["controller.render.bogged_clothes"], **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "parched": dict(base="default", gid="geometry.parched", controller="controller.render.default", rc_suffix="parched_mesh", variant="true", **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "wither_skeleton": dict(base="default", gid="geometry.skeleton.wither.v1.8", controller="controller.render.wither_skeleton", variant="true", **UP, plan=P.human(arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "piglin": dict(base="default", gid="geometry.piglin", controller="controller.render.piglin.v2", variant="!query.is_baby", **UP, plan=P.human(arm=0.58, names=("rightarm", "leftarm", "rightleg", "leftleg")), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "piglin_brute": dict(base="default", gid="geometry.piglin", controller="controller.render.piglin_brute", variant="true", **UP, plan=P.human(arm=0.58, names=("rightarm", "leftarm", "rightleg", "leftleg")), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "zombie_pigman": dict(base="default", gid="geometry.piglin", controller="controller.render.zombie_pigman.v2", variant="!query.is_baby", **UP, plan=P.human(arm=0.58, names=("rightarm", "leftarm", "rightleg", "leftleg")), keep=["rightItem", "leftItem"], anims={}, shots=WALKER[:1]),
    "villager": dict(base="base", entity="villager_v2", gid="geometry.villager_v2", controller="controller.render.villager_v3_base", variant="!query.is_baby && variable.profession_index == 0",
                     hide_rc=["controller.render.villager_v3_masked", "controller.render.villager_v3_level"], **UP, plan=P.robed(), keep=["held_item"], anims={}),
    "wandering_trader": dict(base="default", gid="geometry.villager_v2", controller="controller.render.wandering_trader", variant="true", tint=(1.3, 1.3, 1.3), **UP, plan=P.robed(), keep=["held_item"], anims={}),
    "zombie_villager": dict(base="default", entity="zombie_villager_v2", gid="geometry.zombie.villager_v2", controller="controller.render.zombie_villager_v3_base", variant="!query.is_baby",
                            hide_rc=["controller.render.zombie_villager_v3_masked"], **UP, plan=P.human(head=0.76, arm=0.58), keep=["rightItem", "leftItem"], anims={}, shots=WALKER),
    "witch": dict(base="default", gid="geometry.villager.witch.v1.8", controller="controller.render.witch", variant="true", recolour="plum", **UP, plan=P.robed(head=0.68, legs=0.1), anims={}),
    "pillager": dict(base="default", gid="geometry.pillager", controller="controller.render.pillager", variant="true", **UP, plan=P.human(arm=0.58, names=("rightarm", "leftarm", "rightLeg", "leftLeg")), keep=["rightItem", "leftItem"], anims={}),
    "vindicator": dict(base="default", gid="geometry.vindicator.v1.8", controller="controller.render.vindicator", variant="true", show=["rightArm", "leftArm"], **UP, plan=P.human(arm=0.58, names=("rightArm", "leftArm", "leg0", "leg1")), keep=["rightItem", "leftItem"], anims={}),
    "evocation_illager": dict(base="default", gid="geometry.evoker.v1.8", controller="controller.render.evoker", variant="true", show=["rightArm", "leftArm"], **UP, plan=P.human(arm=0.58, legs=0.2), keep=["rightItem"], anims={}),
    "warden": dict(base="default", gid="geometry.warden", controller="controller.render.warden", variant="true",
                   hide_rc=["controller.render.warden_bioluminescent_layer", "controller.render.warden_spots_1", "controller.render.warden_spots_2", "controller.render.warden_tendrils", "controller.render.warden_heart"],
                   **UP, plan=P.human(head=0.78, arm=0.5, legs=0.3, names=("right_arm", "left_arm", "right_leg", "left_leg")), anims={},
                   shots=[("walk", "animation.warden.move", 0.3, {"query.modified_move_speed": 1.0, "query.modified_distance_moved": 3.0})]),
    "creaking": dict(base="default", gid="geometry.creaking", controller="controller.render.creaking", variant="true", hide_rc=["controller.render.creaking_eyes"], rename={"body": "upperBody"}, **UP, plan=P.human(head=0.84, arm=0.45, legs=0.42), anims={},
                     shots=[("walk", "animation.creaking.walk", 0.3, {"query.modified_move_speed": 1.0})]),
    "copper_golem": dict(base="default", gid="geometry.copper_golem", controller="controller.render.copper_golem", variant="v.index == 0", hide_rc=["controller.render.copper_golem_eyes"], **UP,
                         plan=P.parts(("head", "y", 0.6, "+"), ("right_arm", "x", -0.55, "-", {"y": (0.1, 0.6)}), ("left_arm", "x", 0.55, "+", {"y": (0.1, 0.6)}), ("right_leg", "y", 0.2, "-", {"x": (-1.01, 0.0)}), ("left_leg", "y", 0.2, "-", {"x": (0.0, 1.01)})), keep=["rightItem"], anims={},
                         shots=[("walk", "animation.copper_golem.walk", 0.3, {"query.modified_move_speed": 1.0})]),
}
MOBS.update(LAST_HUMANS)
NEXT.update(LAST_HUMANS)

WITHER_MOVE = lambda e, b, ch, i: "0.0" if ch == "position" else (f"({e}) * 0.3" if ch == "rotation" and i == 0 else None)
FACE_ON = dict(forward=(0.3, 0.0, 1.0), align=False)
SLICES = [(f"bodyCube_{i}", 1 - i / 8) for i in range(1, 8)]
ODD = {
    "blaze": dict(base="default", gid="geometry.blaze", controller="controller.render.blaze", variant="true", glb="blaze_head", height=8.0, **FACE_ON, plan=P.blaze(WORK / "blaze_rod_black_s91.png"), wave=True, anims={},
                  shots=[("flames", "animation.blaze.move", 0.3, {})]),
    "breeze": dict(base="default", gid="geometry.breeze", controller="controller.render.breeze", variant="true", hide_rc=["controller.render.breeze_eyes"], height=17.0, place_like_vanilla=True, rename={"body": "rods"}, links=True, **FACE_ON,
                   plan=P.parts(("head", "y", 0.36, "+")), wave=True, anims={}, shots=[("idle", "animation.breeze.idle", 0.5, {})]),
    "ghast": dict(base="default", gid="geometry.ghast", controller="controller.render.ghast", variant="true", place_like_vanilla=True, tint=(1.45, 1.4, 1.35), plain_back=("body", 0.2), glow={"when": "variable.ischarging", "below": 0.3, "color": (0.95, 0.2, 0.05)}, **FACE_ON, plan=P.ghast(at=0.3), wave=True, anims={}, shots=[("move", "animation.ghast.move", 0.25, {})]),
    "happy_ghast": dict(base="happy_ghast", gid="geometry.happy_ghast", controller="controller.render.happy_ghast", variant="!query.is_baby", place_like_vanilla=True, tint=(1.15, 1.15, 1.15), **FACE_ON, plan=P.ghast(at=0.3), wave=True, anims={}),
    "magma_cube": dict(base="default", gid="geometry.magma_cube_v2", controller="controller.render.magma_cube", variant="true", height=8.0, tip=-28.0, rename={"body": "bodyCube_0"}, links=True, forward=(-0.25, 0.0, 1.0), align=False, plan=P.chain(SLICES, [], axis="y"), wave=True, anims={},
                       shots=[("jump", "animation.magma_cube.move", 0.0, {"variable.slime_squish_factor": 0.8})]),
    "slime": dict(base="default", gid="geometry.slime", controller="controller.render.slime", variant="true", hide_rc=["controller.render.slime_armor"], height=8.0, tint=(1.15, 1.25, 1.1), rename={"body": "cube"}, **FACE_ON, plan=P.only("cube"), wave=True, anims={}),
    "sulfur_cube": dict(base="default", gid="geometry.sulfur_cube", controller="controller.render.sulfur_cube", variant="!query.is_baby && !variable.has_absorbed_tnt", hide_rc={"controller.render.sulfur_cube_outer": "variable.has_absorbed_block"},
                        height=18.0, rename={"body": "sulfur_cube"}, keep=["rightItem"], **FACE_ON, plan=P.only("sulfur_cube"), wave=True, anims={}),
    "shulker": dict(base="undyed", gid="geometry.shulker.v1.8", controller="controller.render.shulker", variant="query.variant == 16", height=16.0, rename={"body": "base"}, pivots={"base": [0.0, 0.0, 0.0]}, **FACE_ON, plan=P.shulker(WORK / "shulker_head_single_1500.glb"), wave=True, anims={},
                    shots=[("open", "animation.shulker.move", 0.0, {"variable.shulker.lidpositionfactor": 180.0, "variable.shulker.lidrotationfactor": 0.0, "query.life_time": 0.0})]),
    "wither": dict(base="default", gid="geometry.witherBoss", controller="controller.render.wither_boss", variant="true", hide_rc=["controller.render.wither_boss_armor_white", "controller.render.wither_boss_armor_blue"], rename={"body": "upperBodyPart1"}, place_like_vanilla=True,
                   forward=(0.0, 0.0, 1.0), align=False, plan=P.parts(("head2", "y", 0.62, "+", {"x": (-1.01, -0.36)}), ("head3", "y", 0.62, "+", {"x": (0.36, 1.01)}), ("head1", "y", 0.62, "+", {"x": (-0.36, 0.36)}), ("upperBodyPart2", "y", 0.5, "-"), ("upperBodyPart3", "y", 0.3, "-", None, {"of": "upperBodyPart2"})),
                   wave=True, anims={"move": ("animation.wither_boss.move", WITHER_MOVE)}, shots=[("move", "animation.wither_boss.move", 0.3, {"variable.upper_body_rotation": 20.0, "variable.body_base_rotation": 0.5})]),
    "zombie_nautilus": dict(base="default", gid="geometry.nautilus", controller="controller.render.zombie_nautilus", variant="!query.is_saddled && !variable.has_armor", glb="nautilus", recolour="mossy", **SWIM_, forward=(-1.0, 0.0, 0.0), plan=lambda m, bone, *rest: (m, bone), rename={"body": "head"}, anims={}),
    "bat": dict(base="default", gid="geometry.bat_v2", controller="controller.render.bat", variant="true", glb="bat_upright", forward=(0.0, 0.0, 1.0), stand=True, height=11.0, rename={"head": "Head"}, place_like_vanilla=True, align=False, plan=P.bat(str(WORK / "bat_wing_cut_s82_0.png")), wave=True, anims={},
                shots=[("flying", "animation.bat.flying", 0.125, {}), ("resting", "animation.bat.resting", 0.0, {})]),
}
MOBS.update(ODD)
NEXT.update(ODD)

# The other coats (2026-10-31): each its own model in its parent's pose and cuts, one entity file per family.
def coat(parent: str, name: str, base: str, variant: str, **over) -> None:
    spec = {k: v for k, v in MOBS[parent].items() if k not in ("key", "texture", "in_place", "done")}
    MOBS[name] = {**spec, "family": MOBS[parent].get("family", parent), "base": base, "variant": variant, **over}


def coats(parent: str, numbered: dict, pattern: str) -> None:
    """Each name's model drawn where `pattern` with its number holds; numbered: name -> (vanilla's texture name, number)."""
    for name, (base, k) in numbered.items():
        coat(parent, name, base, pattern.format(k=k))


MOBS["pig"] = dict(base="default", gid="geometry.pig.v3", controller="controller.render.pig.v4", variant="v.index == 0 && !query.is_baby", in_place=True, done=True, height=15.0, anims={})
MOBS["cow_warm"] = dict(base="warm", entity="cow", gid="geometry.cow.warm", controller="controller.render.cow.v3", variant="!query.is_baby && v.index == 1", anims=QUAD())
coat("cow_warm", "cow_cold", "cold", "!query.is_baby && v.index == 2", gid="geometry.cow.cold", material="cow")
MOBS["cow_warm"]["family"] = "cow_warm"
coat("pig", "pig_warm", "warm", "v.index == 1 && !query.is_baby")
coat("pig", "pig_cold", "cold", "v.index == 2 && !query.is_baby")
coat("chicken", "chicken_warm", "warm", "!query.is_baby && v.index == 1")
coat("chicken", "chicken_cold", "cold", "!query.is_baby && v.index == 2", gid="geometry.chicken.cold", material="chicken")
coat("mooshroom", "mooshroom_brown", "brown", "!query.is_baby && query.variant == 1", texture=None)
coats("llama", {"llama_white": ("white", 1), "llama_brown": ("brown", 2), "llama_gray": ("gray", 3)}, "!query.is_baby && query.variant == {k} && variable.DecorTextureIndex == 0 && query.mark_variant != 1")
coats("trader_llama", {"trader_llama_white": ("white", 1), "trader_llama_brown": ("brown", 2), "trader_llama_gray": ("gray", 3)}, "!query.is_baby && query.variant == {k}")
coats("horse", {"horse_white": ("base_white", 0), "horse_creamy": ("base_creamy", 1), "horse_brown": ("base_brown", 3), "horse_black": ("base_black", 4), "horse_gray": ("base_gray", 5), "horse_darkbrown": ("base_darkbrown", 6)},
      "!query.is_baby && query.variant == {k} && !variable.armor_texture_slot")
coats("wolf", {f"wolf_{c}": (f"{c}_default", i + 1) for i, c in enumerate(("ashen", "black", "chestnut", "rusty", "snowy", "spotted", "striped", "woods"))}, "!query.is_baby && query.variant == {k}")
MOBS["wolf_snowy"]["tint"] = (1.2, 1.22, 1.3)  # (they came out khaki)
MOBS["wolf_black"].update(geo_of="wolf_ashen", darken=(0.34, 0.34, 0.38))  # (two pictures lost their legs in the 3D step: dark on black)
MOBS["wolf_spotted"]["pattern"] = "spots"  # (Qwen made these two plain whatever it was told)
MOBS["wolf_striped"]["pattern"] = "stripes"
coats("cat", {"cat_white": ("white", 0), "cat_black": ("black", 1), "cat_red": ("red", 2), "cat_siamese": ("siamese", 3), "cat_british": ("british", 4), "cat_calico": ("calico", 5), "cat_persian": ("persian", 6), "cat_ragdoll": ("ragdoll", 7), "cat_all_black": ("all_black", 9), "cat_jellie": ("jellie", 10)},
      "!query.is_baby && query.variant == {k}")
coat("fox", "fox_arctic", "arctic", "!query.is_baby && query.variant == 1 && !query.is_sleeping", tint=(1.25, 1.27, 1.35))
MOBS["panda"]["variant"] = "!query.is_baby && query.variant != 4"  # the moods share a coat: only their faces differ on vanilla's boxes
coat("panda", "panda_brown", "brown", "!query.is_baby && query.variant == 4", geo_of="panda", browned=True)  # (its own pictures came back in shreds, twice)
coats("rabbit", {"rabbit_white": ("white", 1), "rabbit_black": ("black", 2), "rabbit_white_splotched": ("white_splotched", 3), "rabbit_gold": ("gold", 4), "rabbit_salt": ("salt", 5)}, "!query.is_baby && query.variant == {k}")
coats("frog", {"frog_cold": ("cold", 1), "frog_warm": ("warm", 2)}, "query.variant == {k}")
coats("axolotl", {"axolotl_cyan": ("cyan", 1), "axolotl_gold": ("gold", 2), "axolotl_wild": ("wild", 3), "axolotl_blue": ("blue", 4)}, "!query.is_baby && query.variant == {k}")
coats("parrot", {"parrot_blue": ("blue", 1), "parrot_green": ("green", 2), "parrot_yellow_blue": ("yellow_blue", 3), "parrot_grey": ("grey", 4)}, "query.variant == {k}")
coat("sheep", "sheep_sheared", "default", "!query.is_baby && query.is_sheared", gid="geometry.sheep.sheared.v1.8", wool=None, no_dye=True, texture=None)
# puffed up: its own rows in the first controller, which then draws every size
MOBS["pufferfish"].update(variant="query.variant == 0", rc_when="true", hide_rc=["controller.render.pufferfish.medium", "controller.render.pufferfish.large"])
coat("pufferfish", "pufferfish_mid", "default", "query.variant == 1", gid="geometry.pufferfish.mid.v1.8")
coat("pufferfish", "pufferfish_large", "default", "query.variant == 2", gid="geometry.pufferfish.large.v1.8")
_P = "math.mod(variable.TropicalFish.Pattern, 6) == {k}"
# the game's own file says more than the 1.26.50 sample, and the higher number wins (tools/gamecheck.py)
MOBS["husk"]["min_engine"] = "1.12.0"
MOBS["rabbit"]["min_engine"] = "1.26.1"
MOBS["tropicalfish"]["variant"] = "variable.TropicalFish.Base == 0 && " + _P.format(k=0)
for _k in range(1, 6):
    coat("tropicalfish", f"tropicalfish_a{_k + 1}", "typeA", "variable.TropicalFish.Base == 0 && " + _P.format(k=_k))
for _k in range(6):
    coat("tropicalfish", f"tropicalfish_b{_k + 1}", "typeB", "variable.TropicalFish.Base == 1 && " + _P.format(k=_k), gid="geometry.tropicalfish_b")
coats("villager", {"villager_farmer": ("farmer", 1), "villager_fisherman": ("fisherman", 2), "villager_shepherd": ("shepherd", 3), "villager_fletcher": ("fletcher", 4), "villager_librarian": ("librarian", 5), "villager_cartographer": ("cartographer", 6),
                   "villager_cleric": ("cleric", 7), "villager_armorer": ("armorer", 8), "villager_weaponsmith": ("weapon_smith", 9), "villager_toolsmith": ("tool_smith", 10), "villager_butcher": ("butcher", 11), "villager_leatherworker": ("leatherworker", 12),
                   "villager_mason": ("stonemason", 13), "villager_nitwit": ("nitwit", 14)}, "!query.is_baby && variable.profession_index == {k}")
MOBS["tropicalfish_a5"]["clip"] = (0.2, 1.01)  # (a stray speck in front of its mouth)
# the same model, recoloured: a dyed shulker's shell, a copper golem's patina
SHELLS = {"black": ((0.09, 0.09, 0.11), 0), "red": ((0.6, 0.13, 0.12), 1), "green": ((0.27, 0.38, 0.14), 2), "brown": ((0.38, 0.24, 0.13), 3), "blue": ((0.2, 0.25, 0.62), 4), "purple": ((0.44, 0.2, 0.62), 5), "cyan": ((0.13, 0.48, 0.55), 6),
          "silver": ((0.6, 0.6, 0.58), 7), "gray": ((0.28, 0.29, 0.31), 8), "pink": ((0.86, 0.47, 0.62), 9), "lime": ((0.45, 0.7, 0.14), 10), "yellow": ((0.9, 0.72, 0.15), 11), "light_blue": ((0.3, 0.6, 0.85), 12), "magenta": ((0.72, 0.28, 0.66), 13),
          "orange": ((0.9, 0.48, 0.12), 14), "white": ((0.85, 0.85, 0.83), 15)}
for _c, (_rgb, _k) in SHELLS.items():
    coat("shulker", f"shulker_{_c}", _c, f"query.variant == {_k}", geo_of="shulker", dye=_rgb)
for _i, (_c, _share) in enumerate((("exposed", 0.3), ("weathered", 0.62), ("oxidized", 0.92))):
    coat("copper_golem", f"copper_golem_{_c}", _c, f"v.index == {_i + 1}", geo_of="copper_golem", patina=_share)



def load(p: Path) -> dict:
    return json.loads(re.sub(r"(?m)//[^\n\"]*$", "", p.read_text(encoding="utf-8-sig")))


# Babies (2026-10-31): the coat's own model at the size of vanilla's baby, its head bigger, on vanilla's baby bones; no new 3D run.
NEUTRAL = lambda e, b, ch, i: "1.0" if ch == "scale" else "0.0"  # vanilla's baby head-scaling is for its boxes
GLB_OF = {"cow": Path("work/models3d/cow_single_1500_00001.glb"), "pig": Path("work/models3d/pig_single_1500_00001.glb")}
MOBS["cow"] = dict(base="default", entity="cow", gid="geometry.cow.v2", controller="controller.render.cow.v3", variant="!query.is_baby && v.index == 0", in_place=True, done=True, family="cow_warm", texture="entity/cow/cow_v2", height=23.0, anims=QUAD())
MOBS["pig"]["texture"] = "entity/pig/pig_v3"


def baby_of(parent: str) -> str | None:
    """Adds parent's baby to MOBS, if vanilla has a baby model for it: its name."""
    spec = MOBS[parent]
    if "is_baby" not in spec["variant"] or parent == "sheep_sheared" or parent.startswith("villager_"):  # (villagers learn a trade as adults)
        return None
    family = spec.get("family", parent)
    entity = load(V / "entity" / f"{spec.get('entity', family)}.entity.json")["minecraft:client_entity"]["description"]
    gid = entity["geometry"].get("baby")
    if not gid:
        return None
    texs = entity["textures"]
    base = next((b for b in (f"baby_{spec.get('base', '')}", f"baby_{spec.get('base', '').removeprefix('base_')}", "baby_default", "baby") if b in texs), next((k for k in texs if "baby" in k), None))
    if not base:
        return None
    anims = dict(spec.get("anims", {}))
    for short, source in entity.get("animations", {}).items():
        low = (short + " " + source).lower()
        if source.startswith("controller.") or short in anims:
            continue
        if "baby_transform" in low or "baby_scaling" in low or "big_head" in low:
            anims[short] = (source, NEUTRAL)
        elif "baby" in low and ("sit" in low):
            anims[short] = (source, "SIT")
        elif "baby" in low and "lie" in low:
            anims[short] = (source, "LIE")
    name = f"{parent}_baby"
    keep = {k: v for k, v in spec.items() if k not in ("key", "texture", "in_place", "done", "height", "saddle", "saddle_at", "shots", "wool", "no_dye", "tint", "plain_back", "glow", "recolour", "layers", "show", "rc_when")}
    MOBS[name] = {**keep, "family": family, "baby_from": parent, "gid": gid, "base": base, "variant": spec["variant"].replace("!query.is_baby", "query.is_baby"), "anims": anims, "shots": []}
    if spec.get("layers"):
        MOBS[name]["layers"] = spec["layers"]
    return name


BABIES = [n for n in [baby_of(p) for p in list(MOBS) if not MOBS[p].get("geo_of") and not MOBS[p].get("baby_from")] if n]
for _coat, _of, _base, _k in (("panda_brown", "panda", "baby_brown", 4), ("wolf_black", "wolf_ashen", "baby_black_default", 2)):  # a recoloured coat's baby: the same model, its own texture
    MOBS[f"{_coat}_baby"] = {**{k: v for k, v in MOBS[f"{_of}_baby"].items() if k not in ("key", "texture")}, "geo_of": f"{_of}_baby", "baby_from": _coat, "base": _base, "variant": f"query.is_baby && query.variant == {_k}"}
    BABIES.append(f"{_coat}_baby")


def vanilla_geometry(identifier: str) -> dict:
    for f in (V / "models/entity").glob("*.json"):
        for g in load(f).get("minecraft:geometry", []):
            if g["description"]["identifier"] == identifier:
                return g
    raise KeyError(identifier)


def fill(name: str) -> dict:
    """A spec with `key` (vanilla's texture path, from the entity file) and our `texture` beside it."""
    spec = MOBS[name]
    if spec.get("texture") is None:
        spec.pop("texture", None)
    if spec.get("baby_from"):
        spec.setdefault("texture", fill(spec["baby_from"])["texture"])
    if "base" in spec and "key" not in spec:
        tex = load(V / "entity" / f"{spec.get('entity', spec.get('family', name))}.entity.json")["minecraft:client_entity"]["description"]["textures"][spec["base"]]
        spec["key"] = tex.removeprefix("textures/")
        spec.setdefault("texture", f"{spec['key'].rsplit('/', 1)[0]}/{name}_mesh")
    return spec


def vanilla_extent(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """The corners of vanilla's model at rest, in model space."""
    qs = R.quads(B.van.geometry(spec["gid"]), R.setup_pose(B, spec["key"], spec["gid"], CACHE), R.hidden_bones(B, spec["key"], spec["gid"], CACHE))
    pts = np.concatenate([q[0][:3] for q in qs] + [(q[0][1] + q[0][2] - q[0][0])[None] for q in qs]) * [-1, 1, 1]
    return pts.min(0), pts.max(0)


def vanilla_height(spec: dict) -> float:
    """How tall vanilla's model stands at rest, what the game's hitbox and seats are made for."""
    low, high = vanilla_extent(spec)
    return float(high[1] - low[1])


def controllers_with(controller: str) -> dict:
    for f in (V / "render_controllers").glob("*.json"):
        d = load(f)
        if controller in d.get("render_controllers", {}):
            return d
    raise KeyError(controller)


def coat_pattern(geo: dict, rgb: np.ndarray, kind: str) -> np.ndarray:
    """Dark markings over a plain coat, laid out by each texel's place on the animal: blotches, or stripes across the back and sides."""
    from scipy import ndimage
    pos = G.position_map(geo, (rgb.shape[1], rgb.shape[0]))
    known = ~np.isnan(pos[..., 0])
    z = np.nan_to_num(pos[..., 2])
    y = np.nan_to_num(pos[..., 1])
    span = float(np.ptp(z[known]))
    wobble = ndimage.gaussian_filter(np.random.default_rng(3).random(z.shape, np.float32), 14)
    wobble = (wobble - wobble.mean()) / wobble.std()
    if kind == "stripes":
        high = np.clip((y - np.quantile(y[known], 0.45)) / (0.2 * float(np.ptp(y[known]))), 0, 1)  # none on the belly and legs
        mark = np.clip((np.sin(2 * np.pi * z / (0.11 * span) + 0.6 * wobble) - 0.15) / 0.25, 0, 1) * high
    else:
        mark = np.clip((ndimage.gaussian_filter(np.random.default_rng(8).random(z.shape, np.float32), 18) - 0.5) * 900, 0, 1)
    mark = mark * known
    return rgb * (1 - 0.72 * mark[..., None]) + 0.72 * mark[..., None] * rgb * np.array([0.32, 0.25, 0.2])


def moss(rgb: np.ndarray, seed: int = 5) -> np.ndarray:
    """Pale bone with soft patches of moss green; the shading kept."""
    from scipy import ndimage
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    bone = np.clip(0.25 + 0.95 * lum, 0, 1)[..., None] * np.array([0.93, 0.89, 0.76])
    noise = ndimage.gaussian_filter(np.random.default_rng(seed).random(lum.shape, np.float32), 24)
    noise = (noise - noise.mean()) / noise.std()
    patch = np.clip((noise - 0.5) / 0.5, 0, 1)[..., None]
    green = np.clip(0.2 + 0.8 * lum, 0, 1)[..., None] * np.array([0.42, 0.55, 0.24])
    return bone * (1 - patch) + green * patch


def whiten(tex: np.ndarray, wool: np.ndarray) -> np.ndarray:
    """Fleece near white, its shading kept: the sheep material multiplies the dye over it, so a
    cream fleece made a beige white sheep and muddy colours (vanilla's wool is ~0.92 grey)."""
    lum = tex[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    grey = np.clip(lum * (0.86 / max(float(np.median(lum[wool])), 1e-3)), 0, 1)[..., None] * np.array([0.98, 0.975, 0.95])
    out = tex.copy()
    out[..., :3] = np.where(wool[..., None], 0.9 * grey + 0.1 * tex[..., :3], tex[..., :3])
    return out


def graze_edit(geo: dict):
    """The sheep's grazing for our head (cut with its neck): no drop, and a pitch that brings the
    muzzle near the grass instead of vanilla's ~36 degrees (its box head also drops 9 px)."""
    bones = {b["name"]: b for b in geo["minecraft:geometry"][0]["bones"]}
    head = bones["head"]
    pts = np.array(head["poly_mesh"]["positions"])
    piv = np.array(head["pivot"])
    tip = pts[pts[:, 2].argmin()]  # the muzzle: furthest forward (-z)
    d = tip - piv
    want = 2.0 - piv[1]  # the muzzle's height above the pivot when grazing: 2 px off the ground
    r = float(np.hypot(d[1], d[2]))
    a0 = np.degrees(np.arctan2(d[1], -d[2]))  # its angle above forward now
    a1 = np.degrees(np.arcsin(np.clip(want / r, -1, 1)))
    pitch = float(np.clip(a0 - a1, 20, 65))  # a head cut at the skull can't reach: a deep nod
    print(f"sheep grazing: head pitch {pitch:.0f} degrees (muzzle {r:.1f} px from the neck pivot)")
    return lambda e, b, ch, i: "0.0" if ch == "position" else (f"({e}) * {pitch / 36:.3f}" if ch == "rotation" and i == 0 else None)


def edited(anim: dict, variant: str, edit, geo: dict, spec: dict) -> dict:
    """Vanilla's animation as this mob's mesh plays it: an edit of its channels, or a pose of its own."""
    if edit == "STAND":
        return G.pose_animation(anim, variant, {})
    if edit in ("SIT", "LIE"):
        head = spec.get("rename", {}).get("head", "head")
        return G.pose_animation(anim, variant, G.rest_pose(geo, head, edit.lower(), spec.get("max_tilt", 40.0), spec.get("upright")))
    return G.variant_animation(anim, variant, edit)


def family_of(family: str) -> list[str]:
    """The models that share one entity file, those made so far: the lead first."""
    names = [n for n, s in MOBS.items() if s.get("family", n) == family and (ART / f"{s.get('geo_of', n)}.mesh.geo.json").exists() and ((ART / f"{n}_mesh.png").exists() or s.get("in_place") or s.get("baby_from"))]
    return sorted(names, key=lambda n: n != family)


def any_of(names: list[str]) -> str:
    return fill(names[0])["variant"] if len(names) == 1 else " || ".join(f"({fill(n)['variant']})" for n in names)


def family_anims(family: str, names: list[str]) -> tuple[dict, dict]:
    """The family's animations: an edit of vanilla's is one copy for all of them (nesting one per model grows
    without end), a pose worked out from a model's own legs one more condition per model."""
    vanilla_anims = R._entity_data(B, CACHE)["anims"]
    todo: dict = {}
    for n in names:
        for short, (source, edit) in fill(n)["anims"].items():
            todo.setdefault(short, (source, []))[1].append((n, edit))
    anims, ids = {}, {}
    for short, (source, members) in todo.items():
        if source not in vanilla_anims:  # (vanilla's fox names a baby animation its files don't have)
            continue
        cur = vanilla_anims[source]
        shared: dict = {}
        for n, edit in members:
            if callable(edit):
                shared.setdefault(id(edit), (edit, []))[1].append(n)
        for edit, group in shared.values():
            cur = G.variant_animation(cur, any_of(group), edit)
        for n, edit in members:
            if not callable(edit):
                geo = json.loads((ART / f"{fill(n).get('geo_of', n)}.mesh.geo.json").read_text())
                edit = graze_edit(geo) if edit == "GRAZE" else edit
                cur = edited(cur, fill(n)["variant"], edit, geo, fill(n))
        ids[short] = f"animation.{family}.mesh.{short}"
        anims[ids[short]] = cur
    return anims, ids


def write_family(family: str) -> None:
    """The entity file, render controller and animations of a mob and its other coats (`family`), one entity
    file drawing each model for its own variant."""
    names = family_of(family)
    lead = fill(family)
    entity_name = lead.get("entity", family)
    anims, ids = family_anims(family, names)
    if anims:
        (ART / f"{family}.mesh.animation.json").write_text(json.dumps(G.plain_molang({"format_version": "1.8.0", "animations": anims}), indent=2) + "\n")
    rc_file = {"format_version": "1.8.0", "render_controllers": {lead["controller"]: lead["rc_base"]}} if "rc_base" in lead else controllers_with(lead["controller"])
    members = []
    for n in names:
        spec = fill(n)
        members.append({"key": "mesh" if n == family else f"mesh_{n}", "geometry": f"geometry.{spec.get('geo_of', n)}.mesh", "variant": spec["variant"],
                        "texture": None if spec.get("in_place") else "textures/" + spec["texture"], "layers": spec.get("layers"), "material": spec.get("material"),
                        "alt": (spec["glow"]["when"], "textures/" + spec["texture"].replace("_mesh", "_alt_mesh")) if spec.get("glow") else None})
    entity, rc = G.meshes_for_variants(load(V / "entity" / f"{entity_name}.entity.json"), rc_file, lead["controller"], members, ids, lead.get("rc_suffix", "mesh"))
    every = any_of(names)
    if lead.get("rc_when"):  # (the small pufferfish's controller draws every size)
        ours = f"{lead['controller']}.{lead.get('rc_suffix', 'mesh')}"
        entity["minecraft:client_entity"]["description"]["render_controllers"] = [{ours: lead["rc_when"]} if isinstance(c, dict) and ours in c else c for c in entity["minecraft:client_entity"]["description"]["render_controllers"]]
    desc = entity["minecraft:client_entity"]["description"]
    desc["render_controllers"] = [c for c in desc["render_controllers"] if (c if isinstance(c, str) else next(iter(c))) not in lead.get("drop_rc", ())]
    # layers painted for vanilla's boxes: not over the mesh. `hide_rc`: a list, or {controller: when it still draws over the mesh}
    hidden = lead.get("hide_rc", ())
    extra = hidden if isinstance(hidden, dict) else {h: None for h in hidden}

    def hide(c):
        name, cond = (c, None) if isinstance(c, str) else next(iter(c.items()))
        if name not in extra:
            return c
        keep = f"!({every})" if not extra[name] else f"!({every}) || ({extra[name]})"
        return {name: f"({cond}) && ({keep})" if cond else keep}

    desc["render_controllers"] = [hide(c) for c in desc["render_controllers"]]
    ctrl = next(iter(rc["render_controllers"].values()))  # parts vanilla shows only sometimes, always there on the mesh (its arms: there's no folded pair)
    show = {x.lower() for x in lead.get("show", ())}
    for entry in ctrl.get("part_visibility", []):
        for part in entry:
            if part.lower() in show:
                entry[part] = f"({every}) || ({entry[part]})"
    (ART / f"{entity_name}.entity.json").write_text(json.dumps(G.entity_for_game(G.plain_molang(entity), lead.get("min_engine")), indent=2) + "\n")
    (ART / f"{family}.mesh.render_controllers.json").write_text(json.dumps(G.plain_molang(rc), indent=2) + "\n")


def recoloured(name: str) -> None:
    """A coat that is its parent's model with another texture (`dye`: a shulker's shell; `patina`: copper going green)."""
    from scipy import ndimage
    spec = fill(name)
    tex = I.load_rgba(ART / f"{spec['geo_of']}_mesh.png")
    rgb = tex[..., :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    if "darken" in spec:
        rgb = rgb * np.array(spec["darken"])
    elif spec.get("browned"):
        dark = np.clip((0.5 - lum) / 0.25, 0, 1)[..., None]
        rgb = rgb * np.array([1.0, 0.93, 0.82]) * (1 - dark) + np.clip(0.14 + 1.25 * lum, 0, 1)[..., None] * np.array([0.72, 0.46, 0.28]) * dark
    elif "dye" in spec:
        shell = np.arange(tex.shape[1]) < 0.75 * tex.shape[1]  # the head has the atlas's last quarter (plans.shulker)
        rel = lum / max(float(np.median(lum[:, shell])), 1e-3)
        dyed = np.clip(rel[..., None] * np.array(spec["dye"]), 0, 1)
        rgb = np.where(shell[None, :, None], dyed, rgb)
    else:
        noise = ndimage.gaussian_filter(np.random.default_rng(11).random(lum.shape, np.float32), 12)
        noise = (noise - noise.min()) / np.ptp(noise)
        cut = float(np.quantile(noise, 1 - spec["patina"]))
        patch = np.clip((noise - cut) / 0.04, 0, 1)[..., None]
        green = np.clip(0.28 + 1.1 * lum, 0, 1)[..., None] * np.array([0.42, 0.72, 0.6])
        rgb = rgb * (1 - patch) + green * patch
    tex[..., :3] = rgb
    I.save_png(tex, ART / f"{name}_mesh.png")
    write_family(spec.get("family", name))


def convert(name: str) -> tuple[dict, np.ndarray]:
    if MOBS[name].get("geo_of") and MOBS[name].get("baby_from"):  # a recoloured coat's baby: its model and its parent's texture are both there
        write_family(MOBS[name].get("family", name))
        return None, None
    if MOBS[name].get("geo_of"):
        recoloured(name)
        return None, None
    spec = fill(name)
    parent = spec.get("baby_from")
    if parent:  # its parent's GLB; its parent's texture, already finished, stays as it is
        glb = GLB_OF.get(parent) or WORK / f"{MOBS[parent].get('glb', parent)}_single_1500.glb"
    else:
        glb = ART / f"{name}_single_1500.glb"
        glb.write_bytes((WORK / f"{spec.get('glb', name)}_single_1500.glb").read_bytes())
    height = spec.get("height") or vanilla_height(spec)
    span = spec.get("span")
    if span == "vanilla":
        low, high = vanilla_extent(spec)
        span = float(max(high[0] - low[0], high[2] - low[2]))
    print(f"{name}: height {height:.1f} px, span {span}", flush=True)
    make = lambda out, head_from: G.make(glb, f"geometry.{name}.mesh", spec["texture"], out, height=height, legs=spec.get("legs", 4),
                                         forward=spec.get("forward", (1.0, 0.0, 0.0)), ears=spec.get("ears", True), head_from=head_from,
                                         align=spec.get("align", True), plan=spec.get("plan"), tip=spec.get("tip", 0.0), span=span, stand=spec.get("stand", False), clip=spec.get("clip"))
    if parent:
        gpath, tpath = make(WORK / "baby_tmp", spec.get("head_from"))
        gpath = ART / gpath.name
        geo = json.loads((WORK / "baby_tmp" / gpath.name).read_text())
        tex = I.load_rgba(tpath)
        for bn in geo["minecraft:geometry"][0]["bones"]:  # a baby's big head
            if bn["name"] == "head" and bn.get("poly_mesh"):
                piv = np.array(bn["pivot"], float)
                bn["poly_mesh"]["positions"] = np.round(piv + (np.array(bn["poly_mesh"]["positions"], float) - piv) * 1.3, 4).tolist()
    else:
        gpath, tpath = make(ART, spec.get("head_from"))
        geo = json.loads(gpath.read_text())
        tex = I.load_rgba(tpath)
    if spec.get("plain_back"):  # the 3D step doubled a feature on the far side: bone (or None) and how squarely it must face back
        tex = G.plain_side(geo, tex, lambda c, n: n[2] > spec["plain_back"][1], spec["plain_back"][0])
        I.save_png(tex, tpath)
    if spec.get("tint"):  # the mob's darker or stranger colour than the picture came out
        tex[..., :3] = np.clip(tex[..., :3] * np.array(spec["tint"]), 0, 1)
        I.save_png(tex, tpath)
    if spec.get("pattern"):
        tex[..., :3] = coat_pattern(geo, tex[..., :3], spec["pattern"])
        I.save_png(tex, tpath)
    if spec.get("recolour") == "plum":  # robe and hat, pictured pale because dark ones shred, darkened: where blue beats green
        purple = np.clip((tex[..., 2] - tex[..., 1] - 0.005) / 0.03, 0, 1)[..., None]
        tex[..., :3] = tex[..., :3] * (1 - purple) + tex[..., :3] * np.array([0.5, 0.3, 0.56]) * purple
        I.save_png(tex, tpath)
    if spec.get("recolour") == "mossy":  # the same model long dead: bleached, with patches of moss (its own picture came back in shreds)
        tex[..., :3] = moss(tex[..., :3])
        I.save_png(tex, tpath)
    if spec.get("glow"):  # a second texture: the face's dark features lit (the ghast shooting)
        mask = G.texel_mask(geo, (tex.shape[1], tex.shape[0]), lambda c, n: n[2] < -0.3 and c[1] > 0.5 * (np.array(next(b for b in geo["minecraft:geometry"][0]["bones"] if b["name"] == "body")["poly_mesh"]["positions"])[:, 1].min() + np.array(next(b for b in geo["minecraft:geometry"][0]["bones"] if b["name"] == "body")["poly_mesh"]["positions"])[:, 1].max()), {"body"})
        lum = tex[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        lit = mask & (lum < spec["glow"]["below"])
        alt = tex.copy()
        alt[..., :3] = np.where(lit[..., None], np.array(spec["glow"]["color"]), alt[..., :3] * np.array([1.0, 0.92, 0.9]))
        I.save_png(alt, ART / f"{name}_alt_mesh.png")
    if spec.get("no_dye"):  # a shorn sheep: no wool for the dye to go on
        tex[..., 3] = 3 / 255
        I.save_png(tex, tpath)
    if spec.get("wool"):  # alpha: where the dye goes. From a cut across the face (the moving head also
        # holds the fleece of the throat), and only bright texels (the brown ears stay brown, undyed)
        face, _ = make(WORK / "face_cut", 0.7)
        wool = G.bone_mask(json.loads(face.read_text()), (tex.shape[1], tex.shape[0]), spec["wool"]) > 0.5
        lum = tex[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        wool &= lum > 0.55 * float(np.median(lum[wool]))
        tex = whiten(tex, wool)
        tex[..., 3] = np.where(wool, 1.0, 3 / 255)
        I.save_png(tex, tpath)
    if spec.get("place_like_vanilla"):  # floaters: ours goes where vanilla's model is, not on y 0
        low, high = vanilla_extent(spec)
        pts = np.concatenate([np.array(bn["poly_mesh"]["positions"]) for bn in geo["minecraft:geometry"][0]["bones"] if bn.get("poly_mesh")])
        centre = (pts.min(0) + pts.max(0)) / 2
        target = (low + high) / 2
        if spec.get("undo_rest"):  # the game moves the bone again at rest (the squid's body): ours must start that much lower
            target = target - np.array(R.setup_pose(B, spec["key"], spec["gid"], CACHE).get(spec["undo_rest"], {}).get("position") or [0, 0, 0], float)
        geo = G.shifted(geo, [target[0] - centre[0], target[1] - centre[1], target[2] - centre[2]])
    if spec.get("saddle"):
        geo, tex = G.add_saddle(geo, tex, spec["saddle"], **spec.get("saddle_at", {}))
        I.save_png(tex, tpath)
    fitted = G.fit_to_vanilla(geo, B.van.geometry(spec["gid"]), rename=spec.get("rename"), links=spec.get("links", False))
    if spec.get("keep"):
        fitted = G.keep_bones(fitted, B.van.geometry(spec["gid"]), spec["keep"])
    for bn in fitted["minecraft:geometry"][0]["bones"]:  # where vanilla's animations turn a whole body about (the shulker on a wall)
        bn["pivot"] = spec.get("pivots", {}).get(bn["name"], bn["pivot"])
    rest = R.setup_pose(B, spec["key"], spec["gid"], CACHE, geometry=fitted["minecraft:geometry"][0])
    if not spec.get("wave"):  # (a wave along the body isn't a rest move: it isn't "c - this")
        fitted = G.cancel_rest_moves(fitted, rest)
        after = R.setup_pose(B, spec["key"], spec["gid"], CACHE, geometry=fitted["minecraft:geometry"][0])
        moved = {k: v["position"] for k, v in after.items() if max(abs(x) for x in v.get("position") or [0]) > 1e-3}
        print(f"{name}: rest moves before {[(k, v.get('position')) for k, v in rest.items() if v.get('position')]}, after {moved or 'none'}")
        assert all(max(abs(x) for x in v) < 2.0 for v in moved.values()), moved  # vanilla's small fixed nudges stay (the bee's bob)
    gpath.write_text(json.dumps(fitted, separators=(",", ":")))
    write_family(spec.get("family", name))
    print(name, [b["name"] for b in fitted["minecraft:geometry"][0]["bones"]], flush=True)
    return geo, tex  # as made: standard bone names, for the check renders


def sheet(name: str, geo: dict, tex: np.ndarray, out: Path) -> None:
    t = tex.copy()
    if MOBS[name].get("wool"):  # how a blue-dyed sheep would look: wool tinted, skin not
        t[..., :3] = t[..., :3] * (1 - t[..., 3:]) + t[..., :3] * np.array([0.25, 0.4, 0.95]) * t[..., 3:]
    t = t * 255
    step = {"leg0": [35, 0, 0], "leg1": [-35, 0, 0], "leg2": [-35, 0, 0], "leg3": [35, 0, 0]}
    views = [(0, 8, None), (45, 10, None), (90, 5, None), (180, 10, None), (270, 5, None),
             (30, -55, None), (35, 8, step), (25, 12, {"head": [0, 60, 0]}), (300, 12, {"head": [0, -60, 0]}),
             (40, 10, {"head": [30, 0, 0]})]
    tiles = [np.asarray(G.draw_mesh(geo, t, yaw, pitch, px=300, pose=pose))[..., :3] for yaw, pitch, pose in views]
    img = np.concatenate([np.concatenate(tiles[:5], 1), np.concatenate(tiles[5:], 1)], 0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out / f"{name}_mesh.jpg", quality=86)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for name in sys.argv[2:]:
        try:
            geo, tex = convert(name)
        except Exception as e:  # one mob's failure doesn't stop the rest
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        if name not in ("sheep", "polar_bear", "llama", "goat", "hoglin", "mooshroom"):  # check.py draws the others
            continue
        sheet(name, geo, tex, out)
