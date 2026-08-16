# sonic_maker

A level editor for Sonic the Hedgehog (Genesis) that edits terrain **while the
original game engine is running it**.

<img src="media/secret_forest.gif" width="640" alt="A level built in the editor, played start to finish">

*Secret Forest — built in the editor, then played through to the finish flag.*

The editor replaces Green Hill Zone's live, decompressed terrain mappings with a
small private level in Genesis RAM. Sonic's real 68k collision routines keep
reading those mappings, so slopes, walls, ceilings, inertia, rolling and jumping
are all handled by the ROM rather than reimplemented in Python. Paint a tile and
Sonic is standing on it the next frame.

## Requirements

- Python 3.12
- `numpy`, `pygame`, `stable-retro` (see `requirements.txt`)
- A Sonic the Hedgehog (Genesis) ROM you own — see below

## Getting the ROM

**No ROM is included here, and nothing in this repo points at one.** You need
your own legally obtained copy, imported into `stable-retro` once.

### 1. Buy a copy

The usual route is Steam:

- **[SEGA Mega Drive and Genesis Classics](https://store.steampowered.com/app/34270/)**
  — the collection Sonic the Hedgehog ships in. It installs the original Mega
  Drive ROMs as plain files on disk, which is exactly what this editor needs.
- The standalone **Sonic The Hedgehog** listing (app 71113) still works if you
  already own it, but it has been delisted and can no longer be bought.

> **Not Sonic Origins.** That is a remake built on the Retro Engine and ships
> its own asset formats rather than a raw Genesis ROM, so it will not work here.

Any other legally obtained copy of the same ROM works too. The import step
matches on file content, not on where the file came from, so what matters is
the hash in step 3 — not the shop.

### 2. Find the ROM file

In Steam: right-click the game → **Manage** → **Browse local files**. Look for a
folder named `uncompressed ROMs`. Typical locations:

| OS | Path |
| --- | --- |
| Windows | `C:\Program Files (x86)\Steam\steamapps\common\Sega Classics\uncompressed ROMs\` |
| Linux | `~/.steam/steam/steamapps/common/Sega Classics/uncompressed ROMs/` |
| macOS | `~/Library/Application Support/Steam/steamapps/common/Sega Classics/uncompressed ROMs/` |

The file is around 512 KB with a `.bin` or `.md` extension. Exact filenames vary
between releases, so verify by hash rather than by name.

### 3. Check you have the right revision

This editor targets **REV01**, the revision the Sonic Retro disassembly
documents. It refuses to start on anything else rather than misread it.

| | |
| --- | --- |
| Size | 524,288 bytes (512 KB) |
| SHA-1 | `69e102855d4389c3fd1a8f3dc7d193f8eee5fe5b` |
| Header serial | `GM 00004049-01` |

```sh
sha1sum "Sonic The Hedgehog.bin"          # Linux
shasum -a 1 "Sonic The Hedgehog.bin"      # macOS
certutil -hashfile "Sonic The Hedgehog.bin" SHA1   # Windows
```

### 4. Import it

Point the importer at the **folder** containing the ROM:

```sh
python3 -m retro.import "/path/to/uncompressed ROMs"
```

It identifies ROMs by SHA-1 and copies the matching one into its own data
directory, so the filename does not matter. You should see
`Imported 1 games` (or more, if the folder holds the whole bundle).

Confirm it took:

```sh
python3 -c "import stable_retro as retro; print(retro.data.get_original_romfile_path('SonicTheHedgehog-Genesis-v0'))"
```

That path is the only place the editor ever reads the ROM from.

## Running

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 sonic_maker.py
```

Useful flags: `--scale {2,3,4}`, `--level-file`, `--levels-dir`,
`--sprites`, `--textures`, `--visualizations`, and `--export-textures DIR`.

## Playing the original game

`play_sonic.py` plays the stock ROM with the keyboard — no editing, no custom
level. It uses the same ROM and the same RAM map as the editor, so it doubles as
a quick check that your import worked and as a reference for what unmodified
Sonic 1 terrain looks like.

```sh
python3 play_sonic.py                 # Green Hill Act 1
python3 play_sonic.py --level 3-2     # zone-act in game order (Spring Yard Act 2)
python3 play_sonic.py --level LabyrinthZone.Act1
```

Arrows move, <kbd>Up</kbd> or <kbd>Space</kbd> jumps, <kbd>Down</kbd> ducks and
rolls, <kbd>C</kbd> and <kbd>V</kbd> save and load a checkpoint, <kbd>Esc</kbd>
quits.

<kbd>A</kbd> toggles the collision overlay — green tops, blue solid, orange
sides — decoded from the live chunk mappings and the ROM's height masks. It is
the read-only counterpart to what the Tiles tab writes.

Only levels that have a savestate are offered, so `6-3` is absent: Sonic 1's
Scrap Brain Act 3 leads into Final Zone and has no start state.

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
| `sonic_maker_textures/` | optional `<material>.png` artwork replacing the procedural placeholders, either loose or in a `<theme>/` subfolder |
| `sonic_maker_visualizations/` | PNG templates written by the Capture tool |
| `media/` | the demo clip above |

Levels are a sparse JSON format at version 3. Version 1 and 2 saves still load;
they simply have no artwork block and fall back to the automatic style.

## Semantic textures

`--export-textures sonic_maker_textures` writes, for every material, an exact
lossless mask PNG in that material's flat colour, a procedural placeholder, and
a `materials.json` describing them. Texture the mask however you like, save the
result beside it as `<material>.png`, and pass `--textures` that folder.

A level's artwork is looked up under its own theme name first and falls back to
the shared folder, so several looks can live side by side:

```
sonic_maker_textures/
  grass.png                     # used by any theme without its own
  forbidden_forrest/grass.png   # used by levels whose theme is forbidden_forrest
```

Any material a theme does not supply falls back to its procedural placeholder,
so a partial set is fine.

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

## License

[MIT](LICENSE) — for the code, the tree sprites in `sonic_maker_sprites/`, and
the demo clip in `media/`, all of which are original to this project.

It cannot and does not grant any rights to Sonic the Hedgehog itself. The ROM,
its collision data, its graphics and its characters remain the property of SEGA.
This editor ships none of them: it reads the collision tables and Sonic's sprite
out of the copy *you* supply, at runtime, and writes neither to disk.

## Disclaimer

An unofficial fan project. Not affiliated with, endorsed by, or sponsored by
SEGA. "Sonic the Hedgehog" is a trademark of SEGA.
