# creatures

Minecraft **Bedrock** resource pack "Bubonic Creatures": realistic AI-made 3D mobs jointed for the game's own animations. Split off
`../minecraft-reskin-project` on 2026-10-31. Player guide and commands: `README.md`. What failed, what works, the recipe for
the next model: `docs/LESSONS-MOBS.md`.

## Commands
- `~/.local/bin/uv run tools/build.py` (DEV) / `--deploy` / `--release`; everything runs from this folder (the tools use relative paths).
- Tool code is `mcreskin` from `../minecraft-reskin-project` (`glbmodel.py`, `render.py`, build): its tests live there (`uv run pytest -q` there).

## Rules
- Render each model and look at it before moving on (`tools/check.py`); the user asked for this, and earlier rounds that skipped it had to be redone. The user then judges in game.
- Each mob's model must be drawn only for its own variant: `variant` in `tools/convert.py`. Everything else falls back to vanilla's model.
- Never write into the user's Minecraft folders except `development_resource_packs/BubonicCreaturesDEV` (`tools/build.py --deploy`).
- A mob stays vanilla in game: read the game's content log, then run `tools/gamecheck.py`; our renders can't show it (`docs/LESSONS-MOBS.md`).
- ComfyUI: one job at a time, on `http://172.26.96.1:8188`. Conserve tokens: no agents or workflows; explain commands in plain words.
- Commit or push only when asked. A release build bumps the pack's version.
