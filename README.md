# sonic_maker

A level editor for Sonic the Hedgehog (Genesis) that edits terrain **while the
original game engine is running it**.

The editor replaces Green Hill Zone's live, decompressed terrain mappings with a
small private level in Genesis RAM. Sonic's real 68k collision routines keep
reading those mappings, so slopes, walls, ceilings, inertia, rolling and jumping
are all handled by the ROM rather than reimplemented in Python. Paint a tile and
Sonic is standing on it the next frame.

## Requirements

- Python 3.12
- `numpy`, `pygame`, `stable-retro` (see `requirements.txt`)
- A Sonic the Hedgehog (Genesis) ROM imported into stable-retro

The ROM is not distributed here. Import your own copy once:

```sh
python3 -m retro.import /path/to/folder-containing-the-rom
```

`stable_retro` then resolves it by itself; nothing in this repo points at a ROM
file directly.

## Running

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 sonic_maker.py
```

Useful flags: `--scale {2,3,4}`, `--level-file`, `--levels-dir`,
`--sprites`, `--textures`, `--visualizations`, and `--export-textures DIR`.

## The four tabs

**Tiles** — paint raw Sonic 1 collision masks, pick solidity (top / all / sides),
flip them, or stamp ready-made ramps that include their own backing.

**Terrain** — draw a freehand stroke or a cubic spline and it is fitted to the
closest real Sonic 1 floor masks, column by column, with a seam cost that keeps
the surface continuous. Sonic can run across the result immediately.

**Markers** — place the green start and red finish flags.

**Visual** — artwork, which is kept strictly parallel to collision, exactly as
the ROM does it. Nothing painted here changes what Sonic can stand on.

- *Materials* — semantic materials (grass, soil, rock, sand, platform, water,
  decor) painted per block. Left alone, every block picks its own material from
  how deep it sits under the surface, so any level is textured on first launch.
- *Sprites* — drop images from `sonic_maker_sprites/` anywhere in the world,
  drag to move, drag the grip to scale, and choose whether each sits in front of
  or behind Sonic.
- *Capture* — set the four edges of a frame one at a time and export what is
  inside it as a PNG template for another editor.

`A` toggles the collision overlay, `V` the artwork, `G` the grid.

## Folders

| Folder | Holds |
| --- | --- |
| `sonic_maker_levels/` | named level saves (JSON) |
| `sonic_maker_sprites/` | decoration images for the Visual tab |
| `sonic_maker_textures/` | optional `<material>.png` artwork replacing the procedural placeholders |
| `sonic_maker_visualizations/` | PNG templates written by the Capture tool |

Levels are a sparse JSON format at version 3. Version 1 and 2 saves still load;
they simply have no artwork block and fall back to the automatic style.

## Semantic textures

`--export-textures sonic_maker_textures` writes, for every material, an exact
lossless mask PNG in that material's flat colour, a procedural placeholder, and
a `materials.json` describing them. Texture the mask however you like, save the
result beside it as `<material>.png`, and pass `--textures` that folder.

Terrain artwork is always drawn *through* the ROM's own collision masks, so a
replacement texture can never invent geometry, widen a ledge or paint ground
over a pit.

## Tests

```sh
pip install pytest
python3 -m pytest tests/ -q
```

The suite covers the tile format, terrain fitting, artwork, sprite loading and
capture, and boots the real emulator to assert that edited collision actually
changes what Sonic does.
