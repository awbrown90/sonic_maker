#!/usr/bin/env python3
"""Edit Sonic 1 collision tiles while playing with the original physics.

The editor replaces Green Hill's live, decompressed terrain mappings with a
small private level.  Sonic's real Genesis collision routines continue to
read those mappings, so slopes, walls, ceilings, inertia, rolling, and jumping
are all handled by the game rather than reimplemented in Python.

Controls
--------
    LEFT / RIGHT          run
    DOWN                  duck / roll
    UP or SPACE           jump

    Left mouse            paint (drag to draw)
    Right mouse           erase
    Middle mouse          pick a tile from the level
    Mouse wheel           scroll the collision inventory

    T                     cycle Tiles / Terrain / Markers / Visual tabs
    Terrain / Freehand    drag a surface; release to generate it
    Terrain / Spline      left-click/add/drag anchors, right-click removes
                          an anchor; click Generate after adjustments
    Markers               choose Green Start or Red Finish, then click
                          the world to place it; drag either flag to adjust
    Visual / Materials    pick a semantic material and paint it over blocks;
                          right-click restores the automatic material
    Visual / Sprites      pick a decoration, click the world to drop it,
                          drag it to move, drag its grip (or use the wheel)
                          to scale it, and choose which side of Sonic it
                          sits on; right-click or DEL removes one
    Visual / Capture      set the four edges of a frame one at a time, then
                          export what is inside it as a PNG template;
                          right-click abandons the edge being set

    Ctrl+Z / Ctrl+Y       undo / redo
    Ctrl+S / Ctrl+O       open named Save / Load browser
    R                     reset Sonic to the start
    P                     pause/unpause physics (editing still works)
    G                     toggle grid
    A                     toggle the collision overlay
    V                     toggle the visual artwork layer
    E                     toggle eraser
    ESC                   quit

The Terrain tab fits a freehand stroke or cubic spline to the closest native
Sonic 1 floor masks.  It creates a green top-only surface with solid $FF
backing, so Sonic can run across the result immediately.  The ready-made ramp
buttons in the Tiles tab remain available for exact hand construction.

Artwork and collision
---------------------
Sonic 1 never derives graphics from collision.  A placed 16x16 block resolves
into four 8x8 art tiles for the drawing routine and, quite separately, into a
height mask plus surface angle for the collision routine; that is why animated
waterfalls can rewrite VRAM without altering a single solid pixel.  The editor
mirrors that split.  Physics still reads the mapping words in `tiles`, while a
parallel `visual` description dresses the same world with semantic materials.

Materials are flat, uniform colours with alpha - deliberately so.  Each one
exports an exact lossless mask PNG that an image generator can texture, and the
result is always drawn *through* the ROM's own collision masks.  Generated
pixels can therefore never invent geometry, move a ledge, or hide a pit::

    sonic_maker.py --export-textures sonic_maker_textures
    # texture each <material>.mask.png, save it as <material>.png beside it
    sonic_maker.py --textures sonic_maker_textures

Without any of that the editor draws procedural placeholders built from the
same flat colours, so every existing level is textured on first launch.

Sprites are the free-standing half of the same idea.  Any image dropped into
`sonic_maker_sprites/` shows up in the Visual tab's library and can be placed
anywhere, at any size, on either side of Sonic.  Like every material, a sprite
is decoration alone and cannot be stood on.

A sprite's own alpha channel is authoritative and is never second-guessed.
Two things are repaired around it, both driven by what the file itself
records rather than by assumption, and neither ever widens a silhouette:

  * A subject cut out of a light page usually keeps a pixel or two of that
    page opaque just inside the mask.  The page colour is still sitting under
    the transparent pixels, so that residue can be recognised and dropped.
  * Art exported without premultiplication keeps the colour it was composed
    against in its soft edges.  Opaque colour is bled outwards underneath
    them, which changes no alpha value at all.

Only artwork carrying no transparency whatsoever - a JPEG, a flat BMP - falls
back to keying a background out, and even then only where that background is
reachable from the image border, so an enclosed highlight survives.

Capturing a template
--------------------
The Capture tool sends a stretch of level out to whatever editor you would
rather draw in.  Its frame is always a rectangle, so it is set one edge at a
time: arm an edge, click the world (or the map beside the controls, for an
edge that is off screen), and the next edge arms itself.  While an edge is
armed, everything outside the frame is dimmed and a line tracks the cursor,
so both where the frame sits in the level and what it will hold are visible
the whole time.

Export writes `sonic_maker_visualizations/<level>_<x>x<y>_<w>x<h>.png` at one
image pixel per world pixel, holding the artwork and the block grid but never
Sonic.  Anything painted over that template therefore lands back on exactly
the blocks it covers.

The renderer composes, in order: parallax background, terrain artwork behind
Sonic, the optional translucent collision overlay, Sonic, foreground artwork,
then the editor guides.

This first version edits static terrain only.  Monitors, bridges, moving
platforms, springs, and loop plane-switchers are objects rather than terrain
tiles and are intentionally not included yet.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import stable_retro as retro


GAME = "SonicTheHedgehog-Genesis-v0"
STATE = "GreenHillZone.Act1"
FPS = 60
NATIVE_WIDTH = 320
NATIVE_HEIGHT = 224
PANEL_WIDTH = 320
PANEL_HEIGHT = 672

# The custom world occupies 64 unique 256x256 chunks.  Their mappings fill
# $FF0000-$FF7FFF; $FF8000 holds a private block->collision identity table.
WORLD_TILES_X = 256
WORLD_TILES_Y = 64
CHUNK_TILES = 16
CHUNKS_X = WORLD_TILES_X // CHUNK_TILES
CHUNKS_Y = WORLD_TILES_Y // CHUNK_TILES
WORLD_WIDTH = WORLD_TILES_X * 16
WORLD_HEIGHT = WORLD_TILES_Y * 16
GROUND_TILE_Y = 56
GROUND_Y = GROUND_TILE_Y * 16
SPAWN_X = 160
SPAWN_Y = GROUND_Y - 20
MARKER_FOOT_OFFSET = 20
FINISH_X = 1024
FINISH_Y = GROUND_Y
FINISH_HALF_WIDTH = 18
FINISH_HEIGHT = 64

RAM_BASE = 0xFF0000
RAM_CHUNKS = 0x0000
RAM_COLLISION_TABLE = 0x8000
RAM_LAYOUT = 0xA400
RAM_SONIC_GFX = 0xC800
RAM_SONIC = 0xD000
RAM_OBJECTS_AFTER_SONIC = 0xD040
RAM_OBJECTS_END = 0xF000
RAM_SPRITE_COUNT = 0xF62C
RAM_CAMERA_X = 0xF700
RAM_CAMERA_Y = 0xF704
RAM_COLLISION_INDEX = 0xF796
RAM_SPRITES = 0xF800
RAM_WATER_PALETTE = 0xFA80
RAM_DRY_PALETTE = 0xFB00

ROM_COLLISION_ANGLES = 0x62900
ROM_COLLISION_HEIGHTS = 0x62A00
WRAM_STATE_OFFSET = 16

SONIC_TILE = 0x780
SONIC_TILE_COUNT = 0x17
MAX_SPRITES = 0x50

SOLID_NONE = 0x0000
SOLID_TOP = 0x2000
SOLID_SIDES = 0x4000
SOLID_ALL = 0x6000
SOLIDITIES = (SOLID_TOP, SOLID_ALL, SOLID_SIDES)
FLIP_X = 0x0800
FLIP_Y = 0x1000

COLLISION_BACKGROUND = (4, 10, 20)
COLLISION_GRID = (15, 30, 44)
COLLISION_CHUNK_GRID = (39, 68, 91)
COLLISION_TOP = (34, 197, 94)
COLLISION_ALL = (28, 126, 214)
COLLISION_SIDES = (238, 154, 44)
COLLISION_ALPHAS = (0.25, 0.40, 0.60, 0.85)
PANEL_BG = (18, 24, 33)
PANEL_EDGE = (55, 68, 82)
TEXT = (225, 232, 240)
MUTED = (145, 158, 173)
ACCENT = (255, 220, 72)
START_MARKER_COLOR = (49, 220, 105)
FINISH_MARKER_COLOR = (235, 67, 79)

# Useful shapes lead the inventory; every other non-empty ROM mask follows.
FAVORITE_COLLISION_IDS = (
    0x0F,  # flat 15-pixel surface, angle $00
    0x08,  # half-height platform
    0x01,  # thin platform
    0x50,  # one-tile 45-degree ramp
    0x1C, 0x1D,  # two-tile gentler ramp
    0x18, 0x19, 0x1A, 0x1B,  # four-tile gentle ramp
    0x1E, 0x1F,  # two-tile steep ramp
    0xFB, 0xFC, 0xFD, 0xFE, 0xFF,  # full masks with different angle metadata
)

STAMP_45_RIGHT = "45_right"
STAMP_45_LEFT = "45_left"
STAMP_GENTLE_RIGHT = "gentle_right"
STAMP_GENTLE_LEFT = "gentle_left"
SAFE_STAMPS = (
    STAMP_45_RIGHT, STAMP_45_LEFT,
    STAMP_GENTLE_RIGHT, STAMP_GENTLE_LEFT,
)

RAW_MASK_NOTES = {
    0x01: "1px raw mask; can tunnel",
    0x18: "gentle ramp segment 1/4",
    0x19: "gentle ramp segment 2/4",
    0x1A: "gentle ramp segment 3/4",
    0x1B: "gentle ramp segment 4/4",
    0x1C: "two-part ramp segment 1/2",
    0x1D: "two-part ramp segment 2/2",
    0x1E: "steep ramp segment 1/2",
    0x1F: "steep ramp segment 2/2",
    0x50: "raw 45-degree ramp piece",
}

FORMAT_NAME = "sonic-maker-collision-level"
FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, FORMAT_VERSION)
VISUAL_FORMAT_VERSION = 3
DEFAULT_LEVEL_FILE = Path(__file__).with_name("sonic_maker_level.json")
DEFAULT_LEVEL_DIRECTORY = Path(__file__).with_name("sonic_maker_levels")
DEFAULT_TEXTURE_DIRECTORY = Path(__file__).with_name("sonic_maker_textures")
DEFAULT_SPRITE_DIRECTORY = Path(__file__).with_name("sonic_maker_sprites")
DEFAULT_VISUALIZATION_DIRECTORY = Path(__file__).with_name(
    "sonic_maker_visualizations")
INVALID_FILENAME_CHARACTERS = frozenset('/\\<>:"|?*')


# ---------------------------------------------------------------------------
# Semantic visual layer
#
# The ROM keeps artwork and collision in parallel: one placed block resolves
# into four 8x8 art tiles for the drawing routine and, independently, into a
# height mask for the collision routine.  These definitions are the artwork
# half.  Nothing here can change what Sonic can stand on.
# ---------------------------------------------------------------------------

MATERIAL_AUTO = "auto"
VISUAL_FLAG_FOREGROUND = 0x1
VISUAL_FLAGS = VISUAL_FLAG_FOREGROUND
DEFAULT_THEME = "green_hill_semantic"


@dataclass(frozen=True)
class Material:
    """One semantic material: a flat mask colour plus placeholder artwork.

    ``mask_color`` is the exact, lossless colour an image generator keys on;
    ``base`` only tints the procedural stand-in until real art replaces it.
    ``kind`` decides whether the art is clipped to the collision mask
    (``terrain``) or painted across a whole block regardless of it
    (``overlay``, the animated-waterfall case).
    """

    key: str
    label: str
    mask_color: tuple[int, int, int]
    base: tuple[int, int, int]
    kind: str = "terrain"
    alpha: int = 255
    note: str = ""


MATERIALS: dict[str, Material] = {
    material.key: material for material in (
        Material("grass", "GRASS", (0, 255, 0), (60, 172, 64),
                 note="lit ribbon along every exposed surface"),
        Material("soil", "SOIL", (128, 64, 0), (122, 76, 42),
                 note="earth directly under the surface"),
        Material("rock", "ROCK", (128, 128, 128), (104, 110, 124),
                 note="cliff faces, walls, and deep fill"),
        Material("sand", "SAND", (255, 200, 128), (214, 188, 132),
                 note="beach and desert floor"),
        Material("platform", "PLATFORM", (255, 255, 0), (198, 166, 74),
                 note="built ledges and brickwork"),
        Material("water", "WATER", (0, 255, 255), (54, 132, 214),
                 kind="overlay", alpha=150,
                 note="translucent; ignores collision entirely"),
        Material("decor", "DECOR", (255, 0, 255), (188, 92, 176),
                 kind="overlay",
                 note="foliage and props; ignores collision"),
    )
}
MATERIAL_ORDER = (MATERIAL_AUTO,) + tuple(MATERIALS)
MATERIAL_INDEX = {key: index + 1 for index, key in enumerate(MATERIALS)}
EDITOR_TABS = ("tiles", "terrain", "markers", "visual")

# Depth in pixels below the exposed top of a solid run.  These thresholds are
# the whole of the automatic style: grass ribbon, soil body, rock at depth.
GRASS_DEPTH = 5
ROCK_DEPTH = 28

CHUNK_PIXELS = CHUNK_TILES * 16
VISUAL_MARGIN = 16
VISUAL_TOP_MARGIN = 32
TEXTURE_SHEET = 64
PARALLAX_WIDTH = 512
HORIZON_Y = int(NATIVE_HEIGHT * 0.62)

# Free-standing decoration images.  Unlike materials these are not tied to the
# block grid at all: they sit at any world pixel, at any size, on either side
# of Sonic, and they never touch collision.
VISUAL_MODES = ("materials", "sprites", "capture")
CAPTURE_EDGES = ("left", "right", "top", "bottom")
CAPTURE_MIN_SIZE = 16
CAPTURE_SNAP = 16
SPRITE_EXTENSIONS = (".png", ".webp", ".gif", ".bmp", ".jpg", ".jpeg", ".tga")
SPRITE_SOURCE_LIMIT = 352
SPRITE_MIN_SIZE = 8
SPRITE_MAX_SIZE = 1024
SPRITE_DEFAULT_HEIGHT = 96
SPRITE_STEP = 1.25


def finish_marker_reached(player_x: int, player_y: int,
                          finish_x: int, finish_y: int) -> bool:
    """Whether Sonic's center/feet overlap the finish flag's trigger area."""
    feet_y = player_y + MARKER_FOOT_OFFSET
    return (abs(player_x - finish_x) <= FINISH_HALF_WIDTH and
            finish_y - FINISH_HEIGHT <= feet_y <= finish_y + 12)


def normalize_level_filename(name: str) -> str:
    """Return a safe JSON filename for a user-entered level name."""
    if not isinstance(name, str):
        raise ValueError("level name must be text")
    name = name.strip()
    if name.lower().endswith(".json"):
        name = name[:-5].rstrip()
    if not name:
        raise ValueError("enter a level name")
    if len(name) > 80:
        raise ValueError("level name is too long")
    if name in (".", "..") or name.startswith("."):
        raise ValueError("level name cannot start with a dot")
    if name.endswith((".", " ")):
        raise ValueError("level name cannot end with a dot or space")
    if (any(character in INVALID_FILENAME_CHARACTERS for character in name) or
            any(ord(character) < 32 for character in name)):
        raise ValueError("level name contains an invalid character")
    return name + ".json"


def list_level_files(directory: os.PathLike | str,
                     extra_paths: Iterable[os.PathLike | str] = ()) \
        -> list[Path]:
    """List named saves, with same-named directory files taking priority."""
    directory = Path(directory).expanduser().resolve()
    found: dict[str, Path] = {}
    try:
        children = list(directory.iterdir())
    except FileNotFoundError:
        children = ()
    for path in children:
        if (path.is_file() and not path.is_symlink() and
                path.suffix.lower() == ".json"):
            found[path.name.casefold()] = path.resolve()
    for raw_path in extra_paths:
        path = Path(raw_path).expanduser().resolve()
        key = path.name.casefold()
        if path.is_file() and path.suffix.lower() == ".json" and key not in found:
            found[key] = path
    return sorted(found.values(), key=lambda path: path.name.casefold())


def find_level_file(directory: os.PathLike | str, name: str,
                    extra_paths: Iterable[os.PathLike | str] = ()) -> Path:
    """Resolve a typed level name to a listed save without path traversal."""
    filename = normalize_level_filename(name)
    for path in list_level_files(directory, extra_paths):
        if path.name.casefold() == filename.casefold():
            return path
    raise FileNotFoundError(filename)


def encode_tile(collision_id: int, solidity: int = SOLID_ALL,
                flip_x: bool = False, flip_y: bool = False) -> int:
    """Encode one editor tile as Sonic 1's native 16x16 mapping word."""
    if not isinstance(collision_id, int) or not 0 <= collision_id <= 0xFF:
        raise ValueError("collision_id must be between 0 and 255")
    if solidity not in (SOLID_NONE, SOLID_TOP, SOLID_SIDES, SOLID_ALL):
        raise ValueError("invalid solidity")
    if collision_id == 0 or solidity == SOLID_NONE:
        return 0
    return (collision_id | solidity |
            (FLIP_X if flip_x else 0) |
            (FLIP_Y if flip_y else 0))


def validate_tile_word(word: int) -> int:
    if not isinstance(word, int) or not 0 <= word <= 0x7FFF:
        raise ValueError("tile word must be a 15-bit integer")
    if word == 0:
        return word
    if word & 0x0700:
        raise ValueError("editor collision IDs must fit in one byte")
    if not (word & 0xFF):
        raise ValueError("non-empty tile has collision ID zero")
    if (word & 0x6000) not in SOLIDITIES:
        raise ValueError("non-empty tile has no solidity flags")
    return word


def decode_tile(word: int) -> tuple[int, int, bool, bool]:
    validate_tile_word(word)
    if word == 0:
        return 0, SOLID_NONE, False, False
    return (word & 0xFF, word & 0x6000,
            bool(word & FLIP_X), bool(word & FLIP_Y))


def safe_stamp_pattern(name: str) -> tuple[tuple[int, int, int], ...]:
    """Return a native-safe ramp pattern relative to its low-end cell.

    Sonic's floor sensor examines the tile containing his feet.  A raw ramp
    placed above the starter surface is therefore hidden by the flat tile
    directly below it.  These patterns replace that tile with full backing
    and add a short upper ledge, matching how Sonic 1 composes its slopes.
    """
    if name == STAMP_45_RIGHT:
        surfaces = ((0, 0x50, False),
                    (1, 0x0F, False), (2, 0x0F, False))
    elif name == STAMP_45_LEFT:
        surfaces = ((0, 0x50, True),
                    (-1, 0x0F, False), (-2, 0x0F, False))
    elif name == STAMP_GENTLE_RIGHT:
        surfaces = tuple((offset, collision_id, False)
                         for offset, collision_id in enumerate(
                             (0x18, 0x19, 0x1A, 0x1B, 0x0F, 0x0F)))
    elif name == STAMP_GENTLE_LEFT:
        surfaces = tuple((-offset, collision_id, offset < 4)
                         for offset, collision_id in enumerate(
                             (0x18, 0x19, 0x1A, 0x1B, 0x0F, 0x0F)))
    else:
        raise ValueError(f"unknown safe stamp {name!r}")

    result = []
    for dx, collision_id, flip_x in surfaces:
        result.append((dx, 0, encode_tile(collision_id, SOLID_TOP,
                                          flip_x=flip_x)))
        result.append((dx, 1, encode_tile(0xFF, SOLID_ALL)))
    return tuple(result)


def chunk_word_offset(tile_x: int, tile_y: int) -> int:
    """Return a tile's word offset inside the custom Genesis chunk buffer."""
    if not (0 <= tile_x < WORLD_TILES_X and
            0 <= tile_y < WORLD_TILES_Y):
        raise IndexError("tile is outside the editable world")
    chunk_x, local_x = divmod(tile_x, CHUNK_TILES)
    chunk_y, local_y = divmod(tile_y, CHUNK_TILES)
    chunk_number = chunk_y * CHUNKS_X + chunk_x + 1
    return ((chunk_number - 1) * 0x200 +
            local_y * 0x20 + local_x * 2)


def _limit_surface_slope(values: np.ndarray,
                         maximum: float = 1.0) -> np.ndarray:
    """Limit a heightfield to a 45-degree grade in either direction."""
    result = np.asarray(values, dtype=float).copy()
    for index in range(1, len(result)):
        result[index] = np.clip(result[index],
                                result[index - 1] - maximum,
                                result[index - 1] + maximum)
    for index in range(len(result) - 2, -1, -1):
        result[index] = np.clip(result[index],
                                result[index + 1] - maximum,
                                result[index + 1] + maximum)
    return result


def freehand_surface(points: Iterable[tuple[float, float]],
                     smooth_radius: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Turn a possibly backtracking mouse stroke into a smooth heightfield."""
    points = list(points)
    if len(points) < 2:
        raise ValueError("draw a longer terrain path")
    samples: dict[int, float] = {}
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))))
        for step in range(steps + 1):
            amount = step / steps
            x = int(round(x0 + (x1 - x0) * amount))
            samples[x] = y0 + (y1 - y0) * amount
    if len(samples) < 2:
        raise ValueError("terrain path needs horizontal distance")
    known_x = np.array(sorted(samples), dtype=float)
    known_y = np.array([samples[int(x)] for x in known_x], dtype=float)
    start = max(0, int(math.ceil(known_x[0])))
    end = min(WORLD_WIDTH - 1, int(math.floor(known_x[-1])))
    if end <= start:
        raise ValueError("terrain path needs horizontal distance")
    x_values = np.arange(start, end + 1, dtype=float)
    y_values = np.interp(x_values, known_x, known_y)
    if smooth_radius and len(y_values) > 2:
        radius = min(int(smooth_radius), max(1, (len(y_values) - 1) // 2))
        ramp = np.arange(1, radius + 2, dtype=float)
        kernel = np.concatenate((ramp, ramp[-2::-1]))
        kernel /= kernel.sum()
        padded = np.pad(y_values, radius, mode="edge")
        y_values = np.convolve(padded, kernel, mode="valid")
    y_values = _limit_surface_slope(y_values)
    y_values = np.clip(y_values, 1, WORLD_HEIGHT - 2)
    return x_values, y_values


def spline_surface(points: Iterable[tuple[float, float]]) \
        -> tuple[np.ndarray, np.ndarray]:
    """Sample a draggable-anchor cubic Hermite spline as a heightfield."""
    by_x: dict[int, float] = {}
    for x, y in points:
        by_x[int(round(x))] = float(y)
    if len(by_x) < 2:
        raise ValueError("a spline needs at least two anchor points")
    anchor_x = np.array(sorted(by_x), dtype=float)
    anchor_y = np.array([by_x[int(x)] for x in anchor_x], dtype=float)
    secants = np.diff(anchor_y) / np.diff(anchor_x)
    tangents = np.empty_like(anchor_y)
    tangents[0], tangents[-1] = secants[0], secants[-1]
    if len(anchor_y) > 2:
        tangents[1:-1] = ((anchor_y[2:] - anchor_y[:-2]) /
                          (anchor_x[2:] - anchor_x[:-2]))

    start = max(0, int(math.ceil(anchor_x[0])))
    end = min(WORLD_WIDTH - 1, int(math.floor(anchor_x[-1])))
    if end <= start:
        raise ValueError("spline needs horizontal distance")
    x_values = np.arange(start, end + 1, dtype=float)
    y_values = np.empty_like(x_values)
    segment = 0
    for index, x in enumerate(x_values):
        while (segment + 1 < len(anchor_x) - 1 and
               x > anchor_x[segment + 1]):
            segment += 1
        x0, x1 = anchor_x[segment:segment + 2]
        width = x1 - x0
        t = np.clip((x - x0) / width, 0.0, 1.0)
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        y_values[index] = (h00 * anchor_y[segment] +
                           h10 * width * tangents[segment] +
                           h01 * anchor_y[segment + 1] +
                           h11 * width * tangents[segment + 1])
    y_values = _limit_surface_slope(y_values)
    y_values = np.clip(y_values, 1, WORLD_HEIGHT - 2)
    return x_values, y_values


@dataclass(frozen=True)
class Edit:
    x: int
    y: int
    before: int
    after: int


@dataclass(frozen=True)
class VisualEdit:
    """An appearance-only change; it never reaches the emulator's RAM."""

    x: int
    y: int
    before: tuple[str, int] | None
    after: tuple[str, int] | None


@dataclass(frozen=True)
class PlacedSprite:
    """One decoration image anchored by its base, in world pixels.

    ``x``/``y`` are the bottom centre, so a tree stands where it was clicked
    and stays planted while it is scaled.
    """

    art: str
    x: int
    y: int
    width: int
    height: int
    flags: int = 0

    @property
    def front(self) -> bool:
        return bool(self.flags & VISUAL_FLAG_FOREGROUND)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.x - self.width // 2, self.y - self.height,
                self.width, self.height)

    def contains(self, x: float, y: float) -> bool:
        left, top, width, height = self.bounds
        return left <= x < left + width and top <= y < top + height


@dataclass(frozen=True)
class CaptureRegion:
    """The rectangle of world exported as a drawing template."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def edge(self, name: str) -> int:
        if name not in CAPTURE_EDGES:
            raise ValueError(f"unknown capture edge {name!r}")
        return getattr(self, name)

    def with_edge(self, name: str, value: float,
                  snap: bool = True) -> "CaptureRegion":
        """Move one edge, keeping the rectangle valid rather than flipping it.

        An edge dragged past its opposite number stops a block short of it, so
        the region never inverts midway through setting the four sides.
        """
        value = int(round(value))
        if snap:
            value = int(round(value / CAPTURE_SNAP)) * CAPTURE_SNAP
        if name == "left":
            value = max(0, min(value, self.right - CAPTURE_MIN_SIZE))
        elif name == "right":
            value = min(WORLD_WIDTH, max(value, self.left + CAPTURE_MIN_SIZE))
        elif name == "top":
            value = max(0, min(value, self.bottom - CAPTURE_MIN_SIZE))
        else:
            value = min(WORLD_HEIGHT, max(value, self.top + CAPTURE_MIN_SIZE))
        return replace(self, **{name: value})

    @classmethod
    def around(cls, left: float, top: float, width: float, height: float,
               snap: bool = True) -> "CaptureRegion":
        step = CAPTURE_SNAP if snap else 1
        first_x = max(0, int(left) // step * step)
        first_y = max(0, int(top) // step * step)
        last_x = min(WORLD_WIDTH, -(-int(left + width) // step) * step)
        last_y = min(WORLD_HEIGHT, -(-int(top + height) // step) * step)
        return cls(first_x, first_y,
                   max(first_x + CAPTURE_MIN_SIZE, last_x),
                   max(first_y + CAPTURE_MIN_SIZE, last_y))

    def validated(self) -> "CaptureRegion":
        if not (0 <= self.left < self.right <= WORLD_WIDTH and
                0 <= self.top < self.bottom <= WORLD_HEIGHT):
            raise ValueError("capture region is outside the editable world")
        if (self.width < CAPTURE_MIN_SIZE or
                self.height < CAPTURE_MIN_SIZE):
            raise ValueError("capture region is too small")
        return self


@dataclass(frozen=True)
class SpriteEdit:
    """A decoration appearing, moving, resizing, or being removed."""

    sprite_id: int
    before: PlacedSprite | None
    after: PlacedSprite | None


AnyEdit = Edit | VisualEdit | SpriteEdit


class EditableLevel:
    """Serializable logical copy of the collision world plus its artwork.

    ``cells`` holds the native mapping words Sonic's collision routine reads.
    ``visual`` is the parallel appearance description: a sparse override per
    16x16 block that never participates in physics.
    """

    def __init__(self, cells: Iterable[int] | None = None,
                 spawn_x: int = SPAWN_X, spawn_y: int = SPAWN_Y,
                 finish_x: int = FINISH_X, finish_y: int = FINISH_Y,
                 theme: str = DEFAULT_THEME,
                 visual: dict[tuple[int, int], tuple[str, int]] | None = None,
                 sprites: Iterable[PlacedSprite] = (),
                 capture: CaptureRegion | None = None):
        size = WORLD_TILES_X * WORLD_TILES_Y
        self.cells = list(cells) if cells is not None else [0] * size
        if len(self.cells) != size:
            raise ValueError(f"level must contain exactly {size} tiles")
        self.cells = [validate_tile_word(word) for word in self.cells]
        self.spawn_x = int(spawn_x)
        self.spawn_y = int(spawn_y)
        self.finish_x = int(finish_x)
        self.finish_y = int(finish_y)
        self.theme = str(theme) or DEFAULT_THEME
        self.visual: dict[tuple[int, int], tuple[str, int]] = {}
        for (x, y), (material, flags) in dict(visual or {}).items():
            self.set_visual(x, y, material, flags)
        self.sprites: dict[int, PlacedSprite] = {}
        self._next_sprite_id = 0
        for sprite in sprites:
            self.add_sprite(sprite)
        self.capture = capture.validated() if capture is not None else None
        self._validate_markers()

    @classmethod
    def with_ground(cls) -> "EditableLevel":
        level = cls()
        filler = encode_tile(0xFF, SOLID_ALL)
        for x in range(WORLD_TILES_X):
            for y in range(GROUND_TILE_Y, WORLD_TILES_Y):
                level.set_word(x, y, filler)
        return level

    def _validate_markers(self) -> None:
        if not (0 <= self.spawn_x < WORLD_WIDTH and
                0 <= self.spawn_y < WORLD_HEIGHT):
            raise ValueError("spawn point is outside the editable world")
        if not (0 <= self.finish_x < WORLD_WIDTH and
                0 <= self.finish_y < WORLD_HEIGHT):
            raise ValueError("finish point is outside the editable world")

    @property
    def start_marker(self) -> tuple[int, int]:
        """Marker base at Sonic's feet; spawn_y stores his native center."""
        return (self.spawn_x,
                min(WORLD_HEIGHT - 1,
                    self.spawn_y + MARKER_FOOT_OFFSET))

    @property
    def finish_marker(self) -> tuple[int, int]:
        return self.finish_x, self.finish_y

    def set_start_marker(self, x: float, y: float) -> None:
        self.spawn_x = max(0, min(WORLD_WIDTH - 1, int(round(x))))
        marker_y = max(MARKER_FOOT_OFFSET,
                       min(WORLD_HEIGHT - 1, int(round(y))))
        self.spawn_y = marker_y - MARKER_FOOT_OFFSET

    def set_finish_marker(self, x: float, y: float) -> None:
        self.finish_x = max(0, min(WORLD_WIDTH - 1, int(round(x))))
        self.finish_y = max(0, min(WORLD_HEIGHT - 1, int(round(y))))

    @staticmethod
    def _index(x: int, y: int) -> int:
        if not (0 <= x < WORLD_TILES_X and 0 <= y < WORLD_TILES_Y):
            raise IndexError("tile is outside the editable world")
        return y * WORLD_TILES_X + x

    def word_at(self, x: int, y: int) -> int:
        if not (0 <= x < WORLD_TILES_X and 0 <= y < WORLD_TILES_Y):
            return 0
        return self.cells[y * WORLD_TILES_X + x]

    def set_word(self, x: int, y: int, word: int) -> int:
        index = self._index(x, y)
        word = validate_tile_word(word)
        previous = self.cells[index]
        self.cells[index] = word
        return previous

    def visual_at(self, x: int, y: int) -> tuple[str, int] | None:
        """Return a block's ``(material, flags)`` override, or None for auto."""
        return self.visual.get((x, y))

    def set_visual(self, x: int, y: int, material: str | None,
                   flags: int = 0) -> tuple[str, int] | None:
        """Override or clear one block's appearance; returns the old value.

        Passing None or ``MATERIAL_AUTO`` restores the automatic material,
        which is derived from the collision mask rather than stored.
        """
        self._index(x, y)
        previous = self.visual.get((x, y))
        if material is None or material == MATERIAL_AUTO:
            self.visual.pop((x, y), None)
            return previous
        if material not in MATERIALS:
            raise ValueError(f"unknown visual material {material!r}")
        if not isinstance(flags, int) or flags & ~VISUAL_FLAGS:
            raise ValueError("unsupported visual flags")
        self.visual[(x, y)] = (material, int(flags))
        return previous

    def apply_visual(self, x: int, y: int,
                     value: tuple[str, int] | None) -> tuple[str, int] | None:
        """Restore a whole override value; the undo/redo counterpart."""
        if value is None:
            return self.set_visual(x, y, None)
        return self.set_visual(x, y, value[0], value[1])

    @staticmethod
    def validate_sprite(sprite: PlacedSprite) -> PlacedSprite:
        if not isinstance(sprite.art, str) or not sprite.art:
            raise ValueError("a sprite needs an artwork name")
        if not (0 <= sprite.x < WORLD_WIDTH and 0 <= sprite.y < WORLD_HEIGHT):
            raise ValueError("sprite is outside the editable world")
        if not (SPRITE_MIN_SIZE <= sprite.width <= SPRITE_MAX_SIZE and
                SPRITE_MIN_SIZE <= sprite.height <= SPRITE_MAX_SIZE):
            raise ValueError("sprite size is out of range")
        if not isinstance(sprite.flags, int) or sprite.flags & ~VISUAL_FLAGS:
            raise ValueError("unsupported sprite flags")
        return sprite

    def add_sprite(self, sprite: PlacedSprite) -> int:
        """Place a decoration and return the id that identifies it for undo."""
        self.validate_sprite(sprite)
        sprite_id = self._next_sprite_id
        self._next_sprite_id += 1
        self.sprites[sprite_id] = sprite
        return sprite_id

    def sprite_at(self, sprite_id: int | None) -> PlacedSprite | None:
        return self.sprites.get(sprite_id) if sprite_id is not None else None

    def apply_sprite(self, sprite_id: int,
                     sprite: PlacedSprite | None) -> PlacedSprite | None:
        """Set or remove one decoration; the undo/redo counterpart."""
        previous = self.sprites.get(sprite_id)
        if sprite is None:
            self.sprites.pop(sprite_id, None)
        else:
            self.sprites[sprite_id] = self.validate_sprite(sprite)
            self._next_sprite_id = max(self._next_sprite_id, sprite_id + 1)
        return previous

    def to_dict(self) -> dict:
        tiles = []
        for index, word in enumerate(self.cells):
            if word:
                y, x = divmod(index, WORLD_TILES_X)
                tiles.append([x, y, word])
        cells = [[x, y, material, flags]
                 for (x, y), (material, flags)
                 in sorted(self.visual.items(), key=lambda item: item[0][::-1])]
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "size": [WORLD_TILES_X, WORLD_TILES_Y],
            "spawn": [self.spawn_x, self.spawn_y],
            "finish": [self.finish_x, self.finish_y],
            "tiles": tiles,
            "visual": {
                "theme": self.theme,
                "cells": cells,
                "sprites": [
                    {"art": sprite.art, "x": sprite.x, "y": sprite.y,
                     "width": sprite.width, "height": sprite.height,
                     "flags": sprite.flags}
                    for sprite in self.sprites.values()
                ],
                "capture": (None if self.capture is None else {
                    "left": self.capture.left, "top": self.capture.top,
                    "right": self.capture.right,
                    "bottom": self.capture.bottom}),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EditableLevel":
        if not isinstance(data, dict):
            raise ValueError("level JSON must contain an object")
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a Sonic Maker collision level")
        version = data.get("version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError("unsupported Sonic Maker level version")
        if data.get("size") != [WORLD_TILES_X, WORLD_TILES_Y]:
            raise ValueError("level has an unsupported size")
        spawn = data.get("spawn")
        if (not isinstance(spawn, list) or len(spawn) != 2 or
                not all(isinstance(value, int) for value in spawn)):
            raise ValueError("level has an invalid spawn point")
        finish = (data.get("finish") if version >= 2
                  else [FINISH_X, FINISH_Y])
        if (not isinstance(finish, list) or len(finish) != 2 or
                not all(isinstance(value, int) for value in finish)):
            raise ValueError("level has an invalid finish point")
        tiles = data.get("tiles")
        if not isinstance(tiles, list):
            raise ValueError("level tiles must be a list")
        theme, visual, sprites, capture = cls._visual_from_dict(
            data.get("visual") if version >= VISUAL_FORMAT_VERSION else None)
        level = cls(spawn_x=spawn[0], spawn_y=spawn[1],
                    finish_x=finish[0], finish_y=finish[1],
                    theme=theme, visual=visual, sprites=sprites,
                    capture=capture)
        seen = set()
        for item in tiles:
            if (not isinstance(item, list) or len(item) != 3 or
                    not all(isinstance(value, int) for value in item)):
                raise ValueError("each level tile must be [x, y, word]")
            x, y, word = item
            if (x, y) in seen:
                raise ValueError(f"duplicate tile at {x}, {y}")
            seen.add((x, y))
            level.set_word(x, y, word)
        return level

    @classmethod
    def _visual_from_dict(cls, data: dict | None) \
            -> tuple[str, dict[tuple[int, int], tuple[str, int]],
                     list[PlacedSprite], CaptureRegion | None]:
        """Read the version-3 artwork block; older saves get the auto style."""
        if data is None:
            return DEFAULT_THEME, {}, [], None
        if not isinstance(data, dict):
            raise ValueError("level visual data must be an object")
        theme = data.get("theme", DEFAULT_THEME)
        if not isinstance(theme, str) or not theme:
            raise ValueError("level visual theme must be a non-empty name")
        cells = data.get("cells", [])
        if not isinstance(cells, list):
            raise ValueError("level visual cells must be a list")
        visual: dict[tuple[int, int], tuple[str, int]] = {}
        for item in cells:
            if (not isinstance(item, list) or len(item) != 4 or
                    not isinstance(item[0], int) or
                    not isinstance(item[1], int) or
                    not isinstance(item[2], str) or
                    not isinstance(item[3], int)):
                raise ValueError(
                    "each visual cell must be [x, y, material, flags]")
            x, y, material, flags = item
            if (x, y) in visual:
                raise ValueError(f"duplicate visual cell at {x}, {y}")
            visual[(x, y)] = (material, flags)
        return (theme, visual, cls._sprites_from_list(data.get("sprites", [])),
                cls._capture_from_dict(data.get("capture")))

    @staticmethod
    def _capture_from_dict(data: object) -> CaptureRegion | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("level capture region must be an object")
        try:
            return CaptureRegion(
                int(data["left"]), int(data["top"]),
                int(data["right"]), int(data["bottom"])).validated()
        except (KeyError, TypeError) as error:
            raise ValueError("capture region needs four edges") from error

    @staticmethod
    def _sprites_from_list(items: object) -> list[PlacedSprite]:
        if not isinstance(items, list):
            raise ValueError("level visual sprites must be a list")
        sprites = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each visual sprite must be an object")
            art = item.get("art")
            flags = item.get("flags", 0)
            if not isinstance(art, str) or not isinstance(flags, int):
                raise ValueError("a visual sprite needs an art name and flags")
            try:
                sprites.append(PlacedSprite(
                    art, int(item["x"]), int(item["y"]),
                    int(item["width"]), int(item["height"]), flags))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"visual sprite {art!r} has invalid geometry") from error
        return sprites

    def save(self, path: os.PathLike | str) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: os.PathLike | str) -> "EditableLevel":
        with Path(path).expanduser().open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def save_named_level(level: EditableLevel, directory: os.PathLike | str,
                     name: str) -> tuple[Path, bool]:
    """Save into the level directory and report whether a file was replaced."""
    directory = Path(directory).expanduser().resolve()
    filename = normalize_level_filename(name)
    existing = next(
        (path for path in list_level_files(directory)
         if path.name.casefold() == filename.casefold()), None)
    target = existing or (directory / filename)
    replaced = existing is not None
    level.save(target)
    return target, replaced


class EditHistory:
    """Group collision and appearance edits into one undoable stroke.

    ``undo``/``redo`` return only the collision changes, because those are the
    ones that have to be mirrored into live WRAM.  Use ``peek_undo`` /
    ``peek_redo`` to see the whole group, e.g. to invalidate cached artwork.
    """

    def __init__(self, limit: int = 256):
        self.limit = limit
        self.undo_stack: list[list[AnyEdit]] = []
        self.redo_stack: list[list[AnyEdit]] = []

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()

    def record(self, edits: Iterable[AnyEdit]) -> None:
        change = [edit for edit in edits if edit.before != edit.after]
        if not change:
            return
        self.undo_stack.append(change)
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def peek_undo(self) -> tuple[AnyEdit, ...]:
        return tuple(self.undo_stack[-1]) if self.undo_stack else ()

    def peek_redo(self) -> tuple[AnyEdit, ...]:
        return tuple(self.redo_stack[-1]) if self.redo_stack else ()

    @staticmethod
    def _apply(level: EditableLevel, edit: AnyEdit, value: object) -> None:
        if isinstance(edit, SpriteEdit):
            level.apply_sprite(edit.sprite_id, value)
        elif isinstance(edit, VisualEdit):
            level.apply_visual(edit.x, edit.y, value)
        else:
            level.set_word(edit.x, edit.y, value)

    def undo(self, level: EditableLevel) -> list[tuple[int, int, int]]:
        if not self.undo_stack:
            return []
        change = self.undo_stack.pop()
        for edit in reversed(change):
            self._apply(level, edit, edit.before)
        self.redo_stack.append(change)
        return [(edit.x, edit.y, edit.before) for edit in change
                if isinstance(edit, Edit)]

    def redo(self, level: EditableLevel) -> list[tuple[int, int, int]]:
        if not self.redo_stack:
            return []
        change = self.redo_stack.pop()
        for edit in change:
            self._apply(level, edit, edit.after)
        self.undo_stack.append(change)
        return [(edit.x, edit.y, edit.after) for edit in change
                if isinstance(edit, Edit)]


class CollisionAssets:
    """Decode Sonic 1's collision masks and live Sonic sprite tiles."""

    def __init__(self, rom: bytes):
        collision_end = ROM_COLLISION_HEIGHTS + 256 * 16
        if (len(rom) < collision_end or
                rom[ROM_COLLISION_HEIGHTS:ROM_COLLISION_HEIGHTS + 16]
                != bytes(16)):
            raise ValueError("unsupported Sonic 1 ROM revision")
        self.rom = rom
        self.angles = rom[ROM_COLLISION_ANGLES:ROM_COLLISION_ANGLES + 256]
        self.heights = np.frombuffer(
            rom, dtype=np.int8, count=256 * 16,
            offset=ROM_COLLISION_HEIGHTS).reshape(256, 16).copy()
        self.masks = np.zeros((256, 4, 16, 16), dtype=bool)
        for collision_id in range(1, 256):
            heights = self.heights[collision_id]
            mask = self.masks[collision_id, 0]
            for x, raw_height in enumerate(heights):
                height = int(raw_height)
                if height > 0:
                    mask[16 - height:, x] = True
                elif height < 0:
                    mask[:-height, x] = True
            self.masks[collision_id, 1] = mask[:, ::-1]
            self.masks[collision_id, 2] = mask[::-1, :]
            self.masks[collision_id, 3] = mask[::-1, ::-1]

    @staticmethod
    def _u8(ram: np.ndarray, address: int) -> int:
        return int(ram[address ^ 1])

    @staticmethod
    def _u16(ram: np.ndarray, address: int) -> int:
        return int(ram[address]) | (int(ram[address + 1]) << 8)

    @staticmethod
    def camera(ram: np.ndarray) -> tuple[int, int]:
        return (CollisionAssets._u16(ram, RAM_CAMERA_X),
                CollisionAssets._u16(ram, RAM_CAMERA_Y))

    @staticmethod
    def color_for(solidity: int) -> tuple[int, int, int]:
        if solidity == SOLID_ALL:
            return COLLISION_ALL
        if solidity == SOLID_TOP:
            return COLLISION_TOP
        return COLLISION_SIDES

    @staticmethod
    def _genesis_rgb(value: int) -> tuple[int, int, int]:
        red = (value >> 1) & 7
        green = (value >> 5) & 7
        blue = (value >> 9) & 7
        red = ((red << 2) | (red >> 2)) << 3
        green = ((green << 3) | (green >> 1)) << 2
        blue = ((blue << 2) | (blue >> 2)) << 3
        return red, green, blue

    def _palette(self, ram: np.ndarray, address: int, line: int):
        start = address + line * 32
        return [self._genesis_rgb(self._u16(ram, start + index * 2))
                for index in range(16)]

    def _sonic_tiles(self, ram: np.ndarray) -> np.ndarray:
        tiles = np.zeros((SONIC_TILE_COUNT, 8, 8), dtype=np.uint8)
        for tile in range(SONIC_TILE_COUNT):
            tile_address = RAM_SONIC_GFX + tile * 32
            for y in range(8):
                for pair in range(4):
                    packed = self._u8(ram, tile_address + y * 4 + pair)
                    tiles[tile, y, pair * 2] = packed >> 4
                    tiles[tile, y, pair * 2 + 1] = packed & 0xF
        return tiles

    def draw_sonic(self, image: np.ndarray, ram: np.ndarray) -> None:
        height, width = image.shape[:2]
        tiles = self._sonic_tiles(ram)
        pieces = []
        sprite_count = min(self._u8(ram, RAM_SPRITE_COUNT), MAX_SPRITES)
        for sprite in range(sprite_count):
            address = RAM_SPRITES + sprite * 8
            tile_word = self._u16(ram, address + 4)
            tile_id = tile_word & 0x7FF
            if not SONIC_TILE <= tile_id < SONIC_TILE + SONIC_TILE_COUNT:
                continue
            size = self._u8(ram, address + 2)
            tiles_wide = ((size >> 2) & 3) + 1
            tiles_high = (size & 3) + 1
            x = (self._u16(ram, address + 6) & 0x1FF) - 128
            y = (self._u16(ram, address) & 0x1FF) - 128
            pieces.append((x, y, tiles_wide, tiles_high,
                           tile_id - SONIC_TILE, tile_word))

        for x, y, tiles_wide, tiles_high, first_tile, tile_word in reversed(pieces):
            indices = np.zeros((tiles_high * 8, tiles_wide * 8),
                               dtype=np.uint8)
            for tile_x in range(tiles_wide):
                for tile_y in range(tiles_high):
                    source = first_tile + tile_x * tiles_high + tile_y
                    if source < SONIC_TILE_COUNT:
                        indices[tile_y * 8:(tile_y + 1) * 8,
                                tile_x * 8:(tile_x + 1) * 8] = tiles[source]
            if tile_word & FLIP_X:
                indices = indices[:, ::-1]
            if tile_word & FLIP_Y:
                indices = indices[::-1, :]
            x0, x1 = max(0, x), min(width, x + indices.shape[1])
            y0, y1 = max(0, y), min(height, y + indices.shape[0])
            if x0 >= x1 or y0 >= y1:
                continue
            visible = indices[y0 - y:y1 - y, x0 - x:x1 - x]
            opaque = visible != 0
            if not np.any(opaque):
                continue
            palette_line = (tile_word >> 13) & 3
            palette_address = (RAM_WATER_PALETTE
                               if self._u8(ram, RAM_SONIC + 0x22) & 0x40
                               else RAM_DRY_PALETTE)
            palette = self._palette(ram, palette_address, palette_line)
            region = image[y0:y1, x0:x1]
            for color_index in range(1, 16):
                region[opaque & (visible == color_index)] = palette[color_index]

    def collision_layer(self, level: EditableLevel, left: int, top: int,
                        width: int = NATIVE_WIDTH,
                        height: int = NATIVE_HEIGHT) \
            -> tuple[np.ndarray, np.ndarray]:
        """Rasterise a region's collision masks into colours plus coverage."""
        colors = np.zeros((height, width, 3), dtype=np.uint8)
        covered = np.zeros((height, width), dtype=bool)
        for tile_y in range(top // 16, (top + height + 15) // 16 + 1):
            offset_y = tile_y * 16 - top
            for tile_x in range(left // 16, (left + width + 15) // 16 + 1):
                word = level.word_at(tile_x, tile_y)
                if not word:
                    continue
                collision_id, solidity, flip_x, flip_y = decode_tile(word)
                flip = int(flip_x) | (int(flip_y) << 1)
                mask = self.masks[collision_id, flip]
                offset_x = tile_x * 16 - left
                x0, x1 = max(0, offset_x), min(width, offset_x + 16)
                y0, y1 = max(0, offset_y), min(height, offset_y + 16)
                if x0 >= x1 or y0 >= y1:
                    continue
                visible = mask[y0 - offset_y:y1 - offset_y,
                               x0 - offset_x:x1 - offset_x]
                colors[y0:y1, x0:x1][visible] = self.color_for(solidity)
                covered[y0:y1, x0:x1] |= visible
        return colors, covered


def _value_noise(rng: np.random.Generator, size: int,
                 cells: int) -> np.ndarray:
    """Smoothed random field that wraps exactly at ``size`` pixels."""
    grid = rng.random((cells, cells))
    samples = np.arange(size, dtype=float) * cells / size
    low = np.floor(samples).astype(int) % cells
    high = (low + 1) % cells
    fade = samples - np.floor(samples)
    fade = fade * fade * (3.0 - 2.0 * fade)
    rows = grid[low] * (1.0 - fade)[:, None] + grid[high] * fade[:, None]
    return rows[:, low] * (1.0 - fade)[None, :] + rows[:, high] * fade[None, :]


def _value_noise_1d(rng: np.random.Generator, size: int,
                    cells: int) -> np.ndarray:
    grid = rng.random(cells)
    samples = np.arange(size, dtype=float) * cells / size
    low = np.floor(samples).astype(int) % cells
    high = (low + 1) % cells
    fade = samples - np.floor(samples)
    fade = fade * fade * (3.0 - 2.0 * fade)
    return grid[low] * (1.0 - fade) + grid[high] * fade


def _tileable_noise(rng: np.random.Generator, size: int,
                    octaves: Iterable[int]) -> np.ndarray:
    field = np.zeros((size, size))
    total = 0.0
    for index, cells in enumerate(octaves):
        weight = 0.5 ** index
        field += weight * _value_noise(rng, size, cells)
        total += weight
    return field / total


def blend_rgba(destination: np.ndarray, source: np.ndarray) -> None:
    """Alpha-composite an RGBA block onto an opaque RGB region, in place."""
    alpha = source[..., 3:4].astype(np.uint16)
    if not alpha.any():
        return
    blended = (destination.astype(np.uint16) * (255 - alpha) +
               source[..., :3].astype(np.uint16) * alpha + 127) // 255
    destination[:] = blended.astype(np.uint8)


def composite_rgba(target: np.ndarray, source: np.ndarray) -> None:
    """Source-over one RGBA block onto another RGBA block, in place."""
    source_alpha = source[..., 3].astype(np.float32) / 255.0
    kept = (target[..., 3].astype(np.float32) / 255.0) * (1.0 - source_alpha)
    combined = source_alpha + kept
    safe = np.maximum(combined, 1e-6)[..., None]
    rgb = (source[..., :3].astype(np.float32) * source_alpha[..., None] +
           target[..., :3].astype(np.float32) * kept[..., None]) / safe
    target[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    target[..., 3] = np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def blend_lines(view: np.ndarray, color: tuple[int, int, int],
                amount: float) -> None:
    """Tint a strided row/column view, so grid lines can sit over artwork."""
    if amount >= 1.0:
        view[:] = color
        return
    tinted = (view.astype(np.float32) * (1.0 - amount) +
              np.asarray(color, dtype=np.float32) * amount)
    view[:] = np.clip(tinted, 0, 255).astype(np.uint8)


def chunks_touching(tile_x: int, tile_y: int) -> tuple[tuple[int, int], ...]:
    """Chunks whose cached artwork depends on one block, margins included."""
    left, top = tile_x * 16, tile_y * 16
    chunk_x, chunk_y = tile_x // CHUNK_TILES, tile_y // CHUNK_TILES
    touched = []
    for candidate_y in range(chunk_y - 1, chunk_y + 2):
        for candidate_x in range(chunk_x - 1, chunk_x + 2):
            if not (0 <= candidate_x < CHUNKS_X and
                    0 <= candidate_y < CHUNKS_Y):
                continue
            if (left < (candidate_x + 1) * CHUNK_PIXELS + VISUAL_MARGIN and
                    left + 16 > candidate_x * CHUNK_PIXELS - VISUAL_MARGIN and
                    top < (candidate_y + 1) * CHUNK_PIXELS + VISUAL_MARGIN and
                    top + 16 > candidate_y * CHUNK_PIXELS - VISUAL_TOP_MARGIN):
                touched.append((candidate_x, candidate_y))
    return tuple(touched)


@dataclass(frozen=True)
class ParallaxBand:
    image: np.ndarray
    scroll_x: float
    scroll_y: float
    offset_y: int = 0


class SemanticTheme:
    """Seamless world textures for every semantic material, plus a sky.

    Each material starts as a procedural stand-in built from one flat colour,
    so an image generator can texture the matching mask PNG and the result can
    be dropped back in as ``<material>.png``.  Terrain art is only ever drawn
    through the ROM's collision masks, so replacing a sheet cannot move a
    ledge, widen a platform, or paint ground over a pit.
    """

    SHEET = TEXTURE_SHEET

    def __init__(self, name: str = DEFAULT_THEME, seed: int = 0x50173,
                 texture_dir: os.PathLike | str | None = None):
        self.name = name
        self.seed = int(seed)
        self.texture_dir = (Path(texture_dir).expanduser()
                            if texture_dir is not None else None)
        self.sheets: dict[str, np.ndarray] = {}
        self.replaced: tuple[str, ...] = ()
        self.reload()
        self.sky = self._sky_gradient()
        self.bands = self._parallax_bands(
            np.random.default_rng(self.seed + 977))

    def _texture_search_dirs(self) -> tuple[Path, ...]:
        """Theme folder first, then the shared texture directory."""
        if self.texture_dir is None:
            return ()
        found = []
        theme_dir = self.texture_dir / self.name
        if theme_dir.is_dir():
            found.append(theme_dir)
        if self.texture_dir.is_dir():
            found.append(self.texture_dir)
        return tuple(found)

    def reload(self) -> tuple[str, ...]:
        """Rebuild the placeholder sheets, then overlay any art on disk."""
        rng = np.random.default_rng(self.seed)
        self.sheets = {key: self._procedural_sheet(key, rng)
                       for key in MATERIALS}
        replaced = []
        search_dirs = self._texture_search_dirs()
        if search_dirs:
            for key, material in MATERIALS.items():
                path = next((folder / f"{key}.png" for folder in search_dirs
                             if (folder / f"{key}.png").is_file()), None)
                if path is None:
                    continue
                try:
                    self.sheets[key] = self._load_sheet(path, material)
                except (pygame.error, ValueError, OSError):
                    continue
                replaced.append(key)
        self.replaced = tuple(replaced)
        return self.replaced

    @classmethod
    def _load_sheet(cls, path: Path, material: Material) -> np.ndarray:
        surface = pygame.image.load(str(path))
        squared = pygame.Surface((cls.SHEET, cls.SHEET), pygame.SRCALPHA)
        if surface.get_size() != (cls.SHEET, cls.SHEET):
            padded = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            padded.blit(surface, (0, 0))
            surface = pygame.transform.smoothscale(
                padded, (cls.SHEET, cls.SHEET))
        squared.blit(surface, (0, 0))
        rgb = pygame.surfarray.array3d(squared).transpose(1, 0, 2)
        alpha = pygame.surfarray.array_alpha(squared).transpose(1, 0)
        return np.dstack([rgb, np.minimum(alpha, material.alpha)])

    def _procedural_sheet(self, key: str,
                          rng: np.random.Generator) -> np.ndarray:
        size = self.SHEET
        material = MATERIALS[key]
        base = np.asarray(material.base, dtype=np.float32)
        grain = _tileable_noise(rng, size, (4, 8, 16))
        fine = _tileable_noise(rng, size, (16, 32))
        alpha = np.full((size, size), float(material.alpha), dtype=np.float32)
        tint = np.ones(3, dtype=np.float32)
        rows = np.arange(size, dtype=np.float32)[:, None]

        if key == "grass":
            blades = _value_noise_1d(rng, size, 32)[None, :]
            shade = 0.80 + 0.30 * grain + 0.22 * (blades - 0.5)
        elif key == "soil":
            shade = 0.82 + 0.30 * grain
            shade = np.where(fine > 0.82, shade * 0.76, shade)
        elif key == "rock":
            facets = np.floor(_value_noise(rng, size, 9) * 5.0) / 4.0
            shade = 0.76 + 0.22 * facets + 0.24 * fine
        elif key == "sand":
            ripples = 0.5 + 0.5 * np.sin(rows * np.pi / 8.0 + grain * 2.0)
            shade = 0.90 + 0.10 * ripples + 0.14 * (fine - 0.5)
        elif key == "platform":
            shade, tint = self._brick_shade(size, grain)
        elif key == "water":
            waves = 0.5 + 0.5 * np.sin(rows * np.pi / 6.0 + grain * 3.0)
            shade = 0.84 + 0.26 * waves
        else:
            clumps = _tileable_noise(rng, size, (3, 6, 12))
            shade = 0.76 + 0.38 * clumps
            alpha = np.where(clumps > 0.46, float(material.alpha), 0.0)

        colors = base[None, None, :] * shade[..., None] * tint[None, None, :]
        return np.dstack([np.clip(colors, 0, 255).astype(np.uint8),
                          alpha.astype(np.uint8)])

    @staticmethod
    def _brick_shade(size: int,
                     grain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.arange(size)[:, None]
        columns = np.arange(size)[None, :]
        offset = ((rows // 8) % 2) * 16
        mortar = ((rows % 8) == 0) | (((columns + offset) % 32) == 0)
        shade = 0.86 + 0.22 * grain
        shade = np.where((rows % 8) == 1, shade * 1.12, shade)
        shade = np.where(mortar, 0.56, shade)
        return shade, np.array([1.0, 0.97, 0.86], dtype=np.float32)

    def _dark_forest_theme(self) -> bool:
        name = self.name.casefold()
        return any(token in name for token in (
            "forbidden", "forrest", "forest", "darkwood"))

    def _sky_gradient(self) -> np.ndarray:
        ramp = np.linspace(0.0, 1.0, WORLD_HEIGHT, dtype=np.float32)[:, None]
        if self._dark_forest_theme():
            sky = (np.array((8, 10, 18), dtype=np.float32) * (1.0 - ramp) +
                   np.array((26, 34, 42), dtype=np.float32) * ramp)
            below = np.clip(
                (np.arange(WORLD_HEIGHT, dtype=np.float32) - GROUND_Y) / 96.0,
                0.0, 1.0)[:, None]
            sky = sky * (1.0 - below) + np.array(
                (6, 10, 12), dtype=np.float32) * below
            return np.clip(sky, 0, 255).astype(np.uint8)
        sky = (np.array((74, 150, 236), dtype=np.float32) * (1.0 - ramp) +
               np.array((178, 216, 246), dtype=np.float32) * ramp)
        below = np.clip(
            (np.arange(WORLD_HEIGHT, dtype=np.float32) - GROUND_Y) / 96.0,
            0.0, 1.0)[:, None]
        sky = sky * (1.0 - below) + np.array(
            (26, 42, 74), dtype=np.float32) * below
        return np.clip(sky, 0, 255).astype(np.uint8)

    def _parallax_bands(self,
                        rng: np.random.Generator) -> tuple[ParallaxBand, ...]:
        # Distant bands are hazed toward the sky so the playfield, which is the
        # only thing Sonic can touch, always reads as the nearest layer.
        if self._dark_forest_theme():
            return (
                self._mist_band(rng),
                self._hill_band(rng, 86, (16, 22, 30), (24, 32, 40),
                                0.22, 0.28, -40, cells=7),
                self._hill_band(rng, 118, (10, 22, 16), (18, 36, 24),
                                0.50, 0.56, -8, cells=11),
            )
        return (
            self._cloud_band(rng),
            self._hill_band(rng, 74, (124, 172, 202), (162, 202, 224),
                            0.30, 0.34, -32, cells=5),
            self._hill_band(rng, 98, (86, 148, 150), (124, 184, 164),
                            0.55, 0.60, -6, cells=8),
        )

    @staticmethod
    def _mist_band(rng: np.random.Generator, height: int = 52,
                   count: int = 10) -> ParallaxBand:
        rows = np.arange(height, dtype=np.float32)[:, None]
        columns = np.arange(PARALLAX_WIDTH, dtype=np.float32)[None, :]
        field = np.zeros((height, PARALLAX_WIDTH), dtype=np.float32)
        for _ in range(count):
            center_x = rng.uniform(0, PARALLAX_WIDTH)
            center_y = rng.uniform(height * 0.25, height * 0.85)
            spread_x = rng.uniform(28, 70)
            spread_y = rng.uniform(5, 12)
            distance = np.abs(columns - center_x)
            distance = np.minimum(distance, PARALLAX_WIDTH - distance)
            field += np.exp(-((distance / spread_x) ** 2 +
                              ((rows - center_y) / spread_y) ** 2))
        alpha = np.clip((field - 0.38) * 1.8, 0.0, 1.0)
        band = np.zeros((height, PARALLAX_WIDTH, 4), dtype=np.uint8)
        band[..., :3] = (np.array((48, 62, 68), dtype=np.float32) *
                         (0.70 + 0.30 * alpha)[..., None]).astype(np.uint8)
        band[..., 3] = (alpha * 120).astype(np.uint8)
        return ParallaxBand(band, 0.10, 0.14, -88)

    @staticmethod
    def _hill_band(rng: np.random.Generator, height: int,
                   color: tuple[int, int, int],
                   ridge_color: tuple[int, int, int],
                   scroll_x: float, scroll_y: float, offset_y: int,
                   cells: int) -> ParallaxBand:
        """A ridge line over a solid skirt, so the band never cuts off flat."""
        skirt = NATIVE_HEIGHT
        profile = 0.30 + 0.55 * _value_noise_1d(rng, PARALLAX_WIDTH, cells)
        crest = (height * (1.0 - profile)).astype(int)[None, :]
        rows = np.arange(height + skirt)[:, None]
        filled = rows >= crest
        shade = np.clip(1.0 - (rows - crest) / float(height) * 0.45, 0.5, 1.0)
        band = np.zeros((height + skirt, PARALLAX_WIDTH, 4), dtype=np.uint8)
        band[..., :3] = np.clip(
            np.asarray(color, dtype=np.float32)[None, None, :] *
            shade[..., None], 0, 255).astype(np.uint8)
        band[..., 3] = 255
        band[filled & (rows < crest + 3), :3] = ridge_color
        band[~filled] = 0
        return ParallaxBand(band, scroll_x, scroll_y, offset_y + skirt)

    @staticmethod
    def _cloud_band(rng: np.random.Generator, height: int = 46,
                    count: int = 14) -> ParallaxBand:
        rows = np.arange(height, dtype=np.float32)[:, None]
        columns = np.arange(PARALLAX_WIDTH, dtype=np.float32)[None, :]
        field = np.zeros((height, PARALLAX_WIDTH), dtype=np.float32)
        for _ in range(count):
            center_x = rng.uniform(0, PARALLAX_WIDTH)
            center_y = rng.uniform(height * 0.30, height * 0.80)
            spread_x = rng.uniform(16, 44)
            spread_y = rng.uniform(4, 9)
            distance = np.abs(columns - center_x)
            distance = np.minimum(distance, PARALLAX_WIDTH - distance)
            field += np.exp(-((distance / spread_x) ** 2 +
                              ((rows - center_y) / spread_y) ** 2))
        alpha = np.clip((field - 0.45) * 2.6, 0.0, 1.0)
        band = np.zeros((height, PARALLAX_WIDTH, 4), dtype=np.uint8)
        band[..., :3] = (np.array((246, 250, 255), dtype=np.float32) *
                         (0.88 + 0.12 * alpha)[..., None]).astype(np.uint8)
        band[..., 3] = (alpha * 235).astype(np.uint8)
        return ParallaxBand(band, 0.12, 0.16, -94)

    def draw_background(self, image: np.ndarray, left: int,
                        top_row: int) -> None:
        height, width = image.shape[:2]
        rows = np.clip(top_row + np.arange(height), 0, len(self.sky) - 1)
        image[:] = self.sky[rows][:, None, :]
        # The horizon is a fraction of the view, so an export of an unusual
        # shape places its bands the same way the live screen would.
        horizon = int(height * 0.62)
        ground_on_screen = GROUND_Y - top_row
        for band in self.bands:
            bottom = band.offset_y + int(round(
                horizon + (ground_on_screen - horizon) * band.scroll_y))
            top = bottom - band.image.shape[0]
            y0, y1 = max(0, top), min(height, bottom)
            if y0 >= y1:
                continue
            columns = (np.arange(width) +
                       int(left * band.scroll_x)) % PARALLAX_WIDTH
            blend_rgba(image[y0:y1],
                       band.image[y0 - top:y1 - top][:, columns])


class VisualTerrain:
    """Cache 256x256 chunks of artwork derived from the collision map.

    Sonic redraws terrain a chunk at a time and leaves the rest alone; the
    same unit works here, so painting one block rebuilds one chunk rather
    than the world.  Chunks are rendered with a margin because a pixel's
    material depends on how deep it sits under the surface above it.
    """

    MAX_CACHED = 48

    def __init__(self, assets: CollisionAssets, theme: SemanticTheme):
        self.assets = assets
        self.theme = theme
        self.cache: dict[tuple[int, int],
                         tuple[np.ndarray | None, np.ndarray | None]] = {}
        self.order: list[tuple[int, int]] = []
        self.builds = 0

    def clear(self) -> None:
        self.cache.clear()
        self.order.clear()

    def invalidate(self, tile_x: int, tile_y: int) -> None:
        for key in chunks_touching(tile_x, tile_y):
            if key in self.cache:
                del self.cache[key]
                self.order.remove(key)

    def layers(self, level: EditableLevel, chunk_x: int, chunk_y: int) \
            -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (chunk_x, chunk_y)
        if key not in self.cache:
            self.cache[key] = self._build(level, chunk_x, chunk_y)
            self.order.append(key)
            self.builds += 1
            while len(self.order) > self.MAX_CACHED:
                del self.cache[self.order.pop(0)]
        return self.cache[key]

    def blit(self, image: np.ndarray, level: EditableLevel, left: int,
             top: int, foreground: bool = False) -> None:
        height, width = image.shape[:2]
        first_x = max(0, left // CHUNK_PIXELS)
        last_x = min(CHUNKS_X - 1, (left + width - 1) // CHUNK_PIXELS)
        first_y = max(0, top // CHUNK_PIXELS)
        last_y = min(CHUNKS_Y - 1, (top + height - 1) // CHUNK_PIXELS)
        for chunk_y in range(first_y, last_y + 1):
            for chunk_x in range(first_x, last_x + 1):
                layer = self.layers(level, chunk_x, chunk_y)[int(foreground)]
                if layer is None:
                    continue
                origin_x = chunk_x * CHUNK_PIXELS - left
                origin_y = chunk_y * CHUNK_PIXELS - top
                x0 = max(0, origin_x)
                x1 = min(width, origin_x + CHUNK_PIXELS)
                y0 = max(0, origin_y)
                y1 = min(height, origin_y + CHUNK_PIXELS)
                if x0 >= x1 or y0 >= y1:
                    continue
                blend_rgba(image[y0:y1, x0:x1],
                           layer[y0 - origin_y:y1 - origin_y,
                                 x0 - origin_x:x1 - origin_x])

    def _build(self, level: EditableLevel, chunk_x: int, chunk_y: int) \
            -> tuple[np.ndarray | None, np.ndarray | None]:
        left = chunk_x * CHUNK_PIXELS - VISUAL_MARGIN
        top = chunk_y * CHUNK_PIXELS - VISUAL_TOP_MARGIN
        width = CHUNK_PIXELS + 2 * VISUAL_MARGIN
        height = CHUNK_PIXELS + VISUAL_TOP_MARGIN + VISUAL_MARGIN
        overrides = self._overrides(level, chunk_x, chunk_y)
        solid, walls = self._solid_masks(level, left, top, width, height)
        if not solid.any() and not overrides:
            return None, None

        back = np.zeros((height, width, 4), dtype=np.uint8)
        front = np.zeros((height, width, 4), dtype=np.uint8)
        rows = (np.arange(height) + top) % TEXTURE_SHEET
        columns = (np.arange(width) + left) % TEXTURE_SHEET

        if solid.any():
            depth = self._depth(solid)
            index = self._auto_materials(solid, walls, depth)
            for (tile_x, tile_y), (key, _flags) in overrides.items():
                box = self._block_slice(tile_x, tile_y, left, top,
                                        width, height)
                if box is None or MATERIALS[key].kind != "terrain":
                    continue
                rows_slice, columns_slice = box
                block = index[rows_slice, columns_slice]
                block[solid[rows_slice, columns_slice]] = MATERIAL_INDEX[key]
            self._paint_materials(back, index, rows, columns)
            self._shade_terrain(back, solid, depth)

        # Terrain art flagged foreground moves to the plane in front of Sonic,
        # mirroring the Genesis priority bit that the drawing code reads from
        # the art tile rather than from anything the collision routine sees.
        for (tile_x, tile_y), (key, flags) in overrides.items():
            box = self._block_slice(tile_x, tile_y, left, top, width, height)
            if (box is None or MATERIALS[key].kind != "terrain" or
                    not flags & VISUAL_FLAG_FOREGROUND):
                continue
            rows_slice, columns_slice = box
            front[rows_slice, columns_slice] = back[rows_slice, columns_slice]
            back[rows_slice, columns_slice] = 0

        for (tile_x, tile_y), (key, flags) in overrides.items():
            box = self._block_slice(tile_x, tile_y, left, top, width, height)
            if box is None or MATERIALS[key].kind != "overlay":
                continue
            rows_slice, columns_slice = box
            target = front if flags & VISUAL_FLAG_FOREGROUND else back
            patch = self.theme.sheets[key][
                np.ix_(rows[rows_slice], columns[columns_slice])]
            composite_rgba(target[rows_slice, columns_slice], patch)

        keep_y = slice(VISUAL_TOP_MARGIN, VISUAL_TOP_MARGIN + CHUNK_PIXELS)
        keep_x = slice(VISUAL_MARGIN, VISUAL_MARGIN + CHUNK_PIXELS)
        back = back[keep_y, keep_x].copy()
        front = front[keep_y, keep_x].copy()
        return (back if back[..., 3].any() else None,
                front if front[..., 3].any() else None)

    @staticmethod
    def _overrides(level: EditableLevel, chunk_x: int, chunk_y: int) \
            -> dict[tuple[int, int], tuple[str, int]]:
        if not level.visual:
            return {}
        first_x = (chunk_x * CHUNK_PIXELS - VISUAL_MARGIN) // 16
        last_x = ((chunk_x + 1) * CHUNK_PIXELS + VISUAL_MARGIN - 1) // 16
        first_y = (chunk_y * CHUNK_PIXELS - VISUAL_TOP_MARGIN) // 16
        last_y = ((chunk_y + 1) * CHUNK_PIXELS + VISUAL_MARGIN - 1) // 16
        return {(x, y): value for (x, y), value in level.visual.items()
                if first_x <= x <= last_x and first_y <= y <= last_y}

    def _solid_masks(self, level: EditableLevel, left: int, top: int,
                     width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        solid = np.zeros((height, width), dtype=bool)
        walls = np.zeros((height, width), dtype=bool)
        for tile_y in range(top // 16, -(-(top + height) // 16)):
            offset_y = tile_y * 16 - top
            for tile_x in range(left // 16, -(-(left + width) // 16)):
                word = level.word_at(tile_x, tile_y)
                if not word:
                    continue
                collision_id, solidity, flip_x, flip_y = decode_tile(word)
                mask = self.assets.masks[collision_id,
                                         int(flip_x) | (int(flip_y) << 1)]
                offset_x = tile_x * 16 - left
                x0, x1 = max(0, offset_x), min(width, offset_x + 16)
                y0, y1 = max(0, offset_y), min(height, offset_y + 16)
                if x0 >= x1 or y0 >= y1:
                    continue
                piece = mask[y0 - offset_y:y1 - offset_y,
                             x0 - offset_x:x1 - offset_x]
                solid[y0:y1, x0:x1] |= piece
                if solidity == SOLID_SIDES:
                    walls[y0:y1, x0:x1] |= piece
        return solid, walls

    @staticmethod
    def _depth(solid: np.ndarray) -> np.ndarray:
        """Pixels below the top of each solid run; 1 is the exposed surface."""
        depth = np.zeros(solid.shape, dtype=np.int16)
        running = np.zeros(solid.shape[1], dtype=np.int16)
        for row in range(solid.shape[0]):
            running = np.where(solid[row], running + 1, 0)
            depth[row] = running
        return depth

    @staticmethod
    def _auto_materials(solid: np.ndarray, walls: np.ndarray,
                        depth: np.ndarray) -> np.ndarray:
        """The default style: grass ribbon, soil body, rock deep and at edges."""
        index = np.zeros(solid.shape, dtype=np.uint8)
        index[solid] = MATERIAL_INDEX["soil"]
        index[solid & (depth >= ROCK_DEPTH)] = MATERIAL_INDEX["rock"]
        exposed = solid & ~(np.roll(solid, 1, axis=1) &
                            np.roll(solid, -1, axis=1))
        rim = (exposed | np.roll(exposed, 1, axis=1) |
               np.roll(exposed, -1, axis=1))
        index[solid & rim & (depth > GRASS_DEPTH)] = MATERIAL_INDEX["rock"]
        index[solid & (depth <= GRASS_DEPTH)] = MATERIAL_INDEX["grass"]
        index[walls] = MATERIAL_INDEX["rock"]
        return index

    def _paint_materials(self, art: np.ndarray, index: np.ndarray,
                         rows: np.ndarray, columns: np.ndarray) -> None:
        for key, value in MATERIAL_INDEX.items():
            mask = index == value
            if not mask.any():
                continue
            art[mask] = self.theme.sheets[key][np.ix_(rows, columns)][mask]

    @staticmethod
    def _shade_terrain(art: np.ndarray, solid: np.ndarray,
                       depth: np.ndarray) -> None:
        limited = np.clip(depth, 0, ROCK_DEPTH).astype(np.float32)
        light = 1.0 - 0.32 * (limited / float(ROCK_DEPTH))
        light[depth == 1] = 1.28
        boundary = solid & ~(np.roll(solid, 1, axis=0) &
                             np.roll(solid, -1, axis=0) &
                             np.roll(solid, 1, axis=1) &
                             np.roll(solid, -1, axis=1))
        light[boundary & (depth > 1)] *= 0.62
        shaded = art[..., :3].astype(np.float32) * light[..., None]
        art[..., :3] = np.clip(shaded, 0, 255).astype(np.uint8)

    @staticmethod
    def _block_slice(tile_x: int, tile_y: int, left: int, top: int,
                     width: int, height: int) -> tuple[slice, slice] | None:
        offset_x, offset_y = tile_x * 16 - left, tile_y * 16 - top
        x0, x1 = max(0, offset_x), min(width, offset_x + 16)
        y0, y1 = max(0, offset_y), min(height, offset_y + 16)
        if x0 >= x1 or y0 >= y1:
            return None
        return slice(y0, y1), slice(x0, x1)


class SpriteLibrary:
    """Load decoration images from a folder and hand out scaled RGBA copies.

    The file's own alpha channel is authoritative: whatever a PNG declares is
    what gets drawn, untouched.  Only artwork that carries no transparency at
    all - a JPEG, a flat BMP - falls back to keying a background out, and even
    then only where it is reachable from the image border, so a bright
    highlight enclosed by the subject survives.

    Colour under translucent pixels is repaired, because art exported without
    premultiplication keeps whatever it was composed against - usually white -
    in its soft edges, which reads as a pale halo over a dark background.
    That correction never changes a single alpha value.
    """

    MAX_SCALED = 96

    def __init__(self, directory: os.PathLike | str | None = None):
        self.directory = (Path(directory).expanduser()
                          if directory is not None else None)
        self.names: tuple[str, ...] = ()
        self.sources: dict[str, np.ndarray] = {}
        self.failed: tuple[str, ...] = ()
        self._scaled: dict[tuple[str, int, int], np.ndarray] = {}
        self._scaled_order: list[tuple[str, int, int]] = []
        self._thumbnails: dict[tuple[str, int, int], pygame.Surface] = {}
        self.reload()

    def reload(self) -> tuple[str, ...]:
        self.sources.clear()
        self._scaled.clear()
        self._scaled_order.clear()
        self._thumbnails.clear()
        names, failed = [], []
        if self.directory is not None and self.directory.is_dir():
            for path in sorted(self.directory.iterdir(),
                               key=lambda item: item.name.casefold()):
                if (not path.is_file() or
                        path.suffix.lower() not in SPRITE_EXTENSIONS or
                        path.stem in self.sources):
                    continue
                try:
                    self.sources[path.stem] = self._load(path)
                except (pygame.error, ValueError, OSError):
                    failed.append(path.name)
                    continue
                names.append(path.stem)
        self.names = tuple(names)
        self.failed = tuple(failed)
        return self.names

    def source(self, name: str) -> np.ndarray | None:
        return self.sources.get(name)

    def aspect(self, name: str, fallback: float = 1.0) -> float:
        source = self.sources.get(name)
        if source is None or not source.shape[0]:
            return fallback
        return source.shape[1] / source.shape[0]

    def scaled(self, name: str, width: int, height: int) -> np.ndarray | None:
        source = self.sources.get(name)
        if source is None or width < 1 or height < 1:
            return None
        key = (name, width, height)
        if key not in self._scaled:
            # Nearest sampling: these are pixel-art props, and it also keeps
            # keyed edges from bleeding back into a halo.
            surface = pygame.transform.scale(
                self._surface(source), (width, height))
            self._scaled[key] = self._array(surface)
            self._scaled_order.append(key)
            while len(self._scaled_order) > self.MAX_SCALED:
                del self._scaled[self._scaled_order.pop(0)]
        return self._scaled[key]

    def thumbnail(self, name: str, width: int,
                  height: int) -> pygame.Surface | None:
        source = self.sources.get(name)
        if source is None:
            return None
        key = (name, width, height)
        if key not in self._thumbnails:
            ratio = min(width / source.shape[1], height / source.shape[0])
            size = (max(1, round(source.shape[1] * ratio)),
                    max(1, round(source.shape[0] * ratio)))
            self._thumbnails[key] = pygame.transform.scale(
                self._surface(source), size)
        return self._thumbnails[key]

    @staticmethod
    def _surface(rgba: np.ndarray) -> pygame.Surface:
        return pygame.image.frombuffer(
            np.ascontiguousarray(rgba).tobytes(), rgba.shape[1::-1], "RGBA")

    @staticmethod
    def _array(surface: pygame.Surface) -> np.ndarray:
        return np.dstack([
            pygame.surfarray.array3d(surface).transpose(1, 0, 2),
            pygame.surfarray.array_alpha(surface).transpose(1, 0)])

    @classmethod
    def _load(cls, path: Path) -> np.ndarray:
        surface = pygame.image.load(str(path))
        width, height = surface.get_size()
        if not width or not height:
            raise ValueError("image has no pixels")
        # Does the file itself describe transparency?  A 32-bit PNG does even
        # when every pixel happens to be opaque, and so does a palette image
        # with a transparent index.  Trust either; guess only when there is
        # nothing to trust.
        supplied = (surface.get_masks()[3] != 0 or
                    surface.get_colorkey() is not None)
        if not supplied or surface.get_bitsize() != 32:
            canvas = pygame.Surface((width, height), pygame.SRCALPHA)
            canvas.fill((0, 0, 0, 0))
            canvas.blit(surface, (0, 0))
            surface = canvas
        longest = max(width, height)
        if longest > SPRITE_SOURCE_LIMIT:
            ratio = SPRITE_SOURCE_LIMIT / longest
            surface = pygame.transform.scale(
                surface, (max(1, round(width * ratio)),
                          max(1, round(height * ratio))))
        rgba = cls._array(surface)
        if not supplied:
            rgba[..., 3] = cls._key_background(rgba[..., :3])
        return cls._crop(cls._unmatte(cls._trim_matte(rgba)))

    @classmethod
    def _trim_matte(cls, rgba: np.ndarray) -> np.ndarray:
        """Drop the ring of flattening background left inside the silhouette.

        Cutting a subject out of a light page usually leaves a pixel or two of
        that page opaque just inside the alpha edge, which reads as a bright
        rim once the sprite is drawn over anything darker.  The page colour is
        still recorded under the transparent pixels, so the residue can be
        identified from the file rather than guessed at: an edge pixel that
        matches the page, where the artwork plainly does not, is background.

        Transparency whose colour was simply zeroed is not treated as a page,
        so a deliberate dark outline is never mistaken for residue.
        """
        alpha = rgba[..., 3]
        clear = alpha < 128
        if not clear.any() or clear.all():
            return rgba
        colour = rgba[..., :3].astype(np.int16)
        page = np.median(colour[clear], axis=0)
        if page.min() < 150:
            return rgba
        interior = colour[~clear & ~cls._grow(clear)]
        if (not len(interior) or
                np.abs(np.median(interior, axis=0) - page).max() <= 60):
            return rgba

        alpha = alpha.copy()
        for _ in range(3):
            clear = alpha < 128
            residue = (~clear & cls._grow(clear) &
                       (np.abs(colour - page).max(axis=2) <= 90))
            if not residue.any():
                break
            alpha[residue] = 0
        trimmed = rgba.copy()
        trimmed[..., 3] = alpha
        return trimmed

    @staticmethod
    def _unmatte(rgba: np.ndarray) -> np.ndarray:
        """Bleed opaque colour outwards under the translucent pixels.

        Only the colour beneath the alpha changes; the alpha channel the file
        declared is passed through byte for byte.  This is what removes the
        pale rim around art that was flattened against a white page.
        """
        alpha = rgba[..., 3]
        if not ((alpha > 0) & (alpha < 255)).any():
            return rgba
        colour = rgba[..., :3].astype(np.float32)
        settled = (alpha == 255).astype(np.float32)
        for _ in range(4):
            totals = np.zeros_like(colour)
            counts = np.zeros_like(settled)
            for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
                totals += np.roll(colour * settled[..., None], step, axis=axis)
                counts += np.roll(settled, step, axis=axis)
            spread = (counts > 0) & (settled == 0)
            if not spread.any():
                break
            colour[spread] = totals[spread] / counts[spread][:, None]
            settled[spread] = 1.0
        repaired = rgba.copy()
        repaired[..., :3] = np.clip(colour, 0, 255).astype(np.uint8)
        return repaired

    @staticmethod
    def _grow(mask: np.ndarray) -> np.ndarray:
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        return grown

    @classmethod
    def _key_background(cls, rgb: np.ndarray) -> np.ndarray:
        """Alpha for flattened art: bright, colourless, border-connected."""
        values = rgb.astype(np.int16)
        spread = values.max(axis=2) - values.min(axis=2)
        darkest = values.min(axis=2)
        strict = (spread <= 30) & (darkest >= 175)
        loose = (spread <= 52) & (darkest >= 132)

        background = np.zeros(strict.shape, dtype=bool)
        background[0, :] = background[-1, :] = True
        background[:, 0] = background[:, -1] = True
        background &= strict
        while True:
            grown = cls._grow(background) & strict
            if np.array_equal(grown, background):
                break
            background = grown
        # Two loose passes eat the compression halo left around the subject.
        for _ in range(2):
            background |= cls._grow(background) & loose
        return np.where(background, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop(rgba: np.ndarray) -> np.ndarray:
        """Trim transparent margins so the base anchor sits on the artwork."""
        opaque = rgba[..., 3] > 8
        if not opaque.any():
            raise ValueError("image is fully transparent")
        rows = np.flatnonzero(opaque.any(axis=1))
        columns = np.flatnonzero(opaque.any(axis=0))
        return rgba[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1].copy()


@dataclass(frozen=True)
class ViewOptions:
    show_visual: bool = True
    show_collision: bool = True
    collision_alpha: float = 0.40
    collision_outline: bool = False
    show_grid: bool = True


class LevelRenderer:
    """Compose the layers in the order Sonic's own display pipeline implies."""

    def __init__(self, assets: CollisionAssets, theme: SemanticTheme,
                 sprites: SpriteLibrary | None = None):
        self.assets = assets
        self.theme = theme
        self.sprites = sprites if sprites is not None else SpriteLibrary(None)
        self.terrain = VisualTerrain(assets, theme)

    def invalidate(self, tile_x: int, tile_y: int) -> None:
        self.terrain.invalidate(tile_x, tile_y)

    def invalidate_all(self) -> None:
        self.terrain.clear()

    def render(self, level: EditableLevel, ram: np.ndarray,
               view: ViewOptions = ViewOptions()) -> np.ndarray:
        image = np.empty((NATIVE_HEIGHT, NATIVE_WIDTH, 3), dtype=np.uint8)
        camera_x, camera_y = self.assets.camera(ram)
        self.compose(image, level, camera_x, camera_y, view,
                     between_planes=lambda target:
                     self.assets.draw_sonic(target, ram))
        return image

    def render_region(self, level: EditableLevel, left: int, top: int,
                      width: int, height: int,
                      view: ViewOptions = ViewOptions()) -> np.ndarray:
        """Render any rectangle of the world at one image pixel per world
        pixel, with no Sonic in it - the level as artwork rather than as a
        moment of play."""
        if width < 1 or height < 1:
            raise ValueError("capture region has no area")
        image = np.empty((height, width, 3), dtype=np.uint8)
        self.compose(image, level, left, top, view)
        return image

    def compose(self, image: np.ndarray, level: EditableLevel, left: int,
                top: int, view: ViewOptions,
                between_planes=None) -> None:
        """Paint the layered scene for one region of the world.

        ``between_planes`` runs where Sonic belongs: after the terrain behind
        him, before the foreground plane in front of him.
        """
        if view.show_visual:
            self.theme.draw_background(image, left, top)
            self.terrain.blit(image, level, left, top)
            self.draw_sprites(image, level, left, top)
        else:
            image[:] = COLLISION_BACKGROUND
        if view.show_collision:
            self._draw_collision(image, level, left, top, view)
        if view.show_grid:
            self._draw_grid(image, left, top, view.show_visual)
        if between_planes is not None:
            between_planes(image)
        if view.show_visual:
            self.terrain.blit(image, level, left, top, foreground=True)
            self.draw_sprites(image, level, left, top, front=True)

    def draw_sprites(self, image: np.ndarray, level: EditableLevel,
                     view_left: int, view_top: int,
                     front: bool = False) -> None:
        height, width = image.shape[:2]
        for sprite in level.sprites.values():
            if sprite.front != front:
                continue
            patch = self.sprites.scaled(sprite.art, sprite.width,
                                        sprite.height)
            if patch is None:
                continue
            left, top, sprite_width, sprite_height = sprite.bounds
            left -= view_left
            top -= view_top
            x0, x1 = max(0, left), min(width, left + sprite_width)
            y0, y1 = max(0, top), min(height, top + sprite_height)
            if x0 >= x1 or y0 >= y1:
                continue
            blend_rgba(image[y0:y1, x0:x1],
                       patch[y0 - top:y1 - top, x0 - left:x1 - left])

    def _draw_collision(self, image: np.ndarray, level: EditableLevel,
                        left: int, top: int, view: ViewOptions) -> None:
        height, width = image.shape[:2]
        colors, covered = self.assets.collision_layer(
            level, left, top, width, height)
        amount = view.collision_alpha
        if view.collision_outline:
            covered &= ~(np.roll(covered, 1, axis=0) &
                         np.roll(covered, -1, axis=0) &
                         np.roll(covered, 1, axis=1) &
                         np.roll(covered, -1, axis=1))
            amount = min(1.0, amount + 0.35)
        if not covered.any():
            return
        if not view.show_visual:
            image[covered] = colors[covered]
            return
        blended = (image[covered].astype(np.float32) * (1.0 - amount) +
                   colors[covered].astype(np.float32) * amount)
        image[covered] = np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_grid(image: np.ndarray, camera_x: int, camera_y: int,
                   over_artwork: bool) -> None:
        amount = 0.35 if over_artwork else 1.0
        for step, color in ((16, COLLISION_GRID),
                            (CHUNK_PIXELS, COLLISION_CHUNK_GRID)):
            blend_lines(image[:, (-camera_x) % step::step], color, amount)
            blend_lines(image[(-camera_y) % step::step, :], color, amount)


def visualization_filename(stem: str, region: CaptureRegion) -> str:
    """Name a capture after the level and the rectangle it came from."""
    cleaned = "".join(
        "_" if character in INVALID_FILENAME_CHARACTERS or ord(character) < 32
        else character
        for character in (stem or "untitled").strip()) or "untitled"
    return (f"{cleaned[:60]}_{region.left}x{region.top}"
            f"_{region.width}x{region.height}.png")


def export_visualization(renderer: LevelRenderer, level: EditableLevel,
                         region: CaptureRegion,
                         directory: os.PathLike | str, stem: str,
                         view: ViewOptions = ViewOptions()) -> Path:
    """Write one region of the level out as a PNG drawing template.

    The image is one pixel per world pixel, so anything drawn over it in
    another editor maps straight back onto the blocks it covers.
    """
    region = region.validated()
    image = renderer.render_region(level, region.left, region.top,
                                   region.width, region.height, view)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / visualization_filename(stem, region)
    surface = pygame.image.frombuffer(
        np.ascontiguousarray(image).tobytes(),
        (region.width, region.height), "RGB")
    pygame.image.save(surface, str(target))
    return target


def export_material_assets(theme: SemanticTheme,
                           directory: os.PathLike | str) -> list[Path]:
    """Write each material's lossless mask, placeholder art, and metadata.

    The mask PNG is the contract with an image generator: texture it, keep the
    silhouette, and drop the result back beside it as ``<material>.png``.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    described = []
    for key, material in MATERIALS.items():
        sheet = theme.sheets[key]
        mask = np.zeros_like(sheet)
        mask[..., :3] = material.mask_color
        mask[..., 3] = sheet[..., 3]
        for image, path in ((mask, directory / f"{key}.mask.png"),
                            (sheet, directory / f"{key}.png")):
            surface = pygame.image.frombuffer(
                np.ascontiguousarray(image).tobytes(),
                image.shape[1::-1], "RGBA")
            pygame.image.save(surface, str(path))
            written.append(path)
        described.append({
            "key": key,
            "label": material.label,
            "mask_color": list(material.mask_color),
            "mask_hex": "#%02x%02x%02x" % material.mask_color,
            "kind": material.kind,
            "alpha": material.alpha,
            "clipped_to_collision": material.kind == "terrain",
            "note": material.note,
        })
    metadata = directory / "materials.json"
    with metadata.open("w", encoding="utf-8") as handle:
        json.dump({
            "theme": theme.name,
            "tile": TEXTURE_SHEET,
            "seamless": True,
            "usage": "replace <material>.png with art that tiles seamlessly "
                     "at this size; terrain art is always clipped to the "
                     "ROM collision mask, so it cannot change geometry",
            "materials": described,
        }, handle, indent=2)
        handle.write("\n")
    written.append(metadata)
    return written


@dataclass(frozen=True)
class TerrainColumn:
    tile_x: int
    tile_y: int
    collision_id: int
    flip_x: bool = False

    @property
    def surface_word(self) -> int:
        return encode_tile(self.collision_id, SOLID_TOP,
                           flip_x=self.flip_x)


@dataclass(frozen=True)
class _SurfaceCandidate:
    collision_id: int
    flip_x: bool
    offsets: tuple[int, ...]
    angle: int


@dataclass(frozen=True)
class _FitOption:
    local_cost: float
    tile_y: int
    candidate: _SurfaceCandidate
    absolute_profile: tuple[float, ...]


class TerrainFitter:
    """Fit a pixel heightfield to Sonic 1's real floor collision masks."""

    LOCAL_CHOICES = 24

    def __init__(self, assets: CollisionAssets):
        self.assets = assets
        self.candidates = self._surface_candidates()

    def _surface_candidates(self) -> tuple[_SurfaceCandidate, ...]:
        candidates = []
        seen = set()
        for collision_id in range(1, 256):
            heights = self.assets.heights[collision_id]
            # Negative and zero columns are useful for ceilings, loop seams,
            # and partial objects, but not for a continuous grassy heightfield.
            if not np.all(heights > 0):
                continue
            for flip_x in (False, True):
                profile = heights[::-1] if flip_x else heights
                offsets = tuple(int(16 - height) for height in profile)
                raw_angle = self.assets.angles[collision_id]
                angle = (0 if raw_angle == 0xFF else
                         ((-raw_angle) & 0xFF if flip_x else raw_angle))
                key = offsets, angle
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_SurfaceCandidate(
                    collision_id, flip_x, offsets, angle))
        if not candidates:
            raise RuntimeError("ROM contains no usable terrain collision masks")
        return tuple(candidates)

    @staticmethod
    def _angle_distance(first: int, second: int) -> int:
        return abs(((first - second + 128) & 0xFF) - 128)

    def _local_options(self, tile_x: int, x_values: np.ndarray,
                       y_values: np.ndarray) -> list[_FitOption]:
        pixel_x = tile_x * 16 + np.arange(16, dtype=float)
        target = np.interp(pixel_x, x_values, y_values)
        tangent = int(round(
            math.atan2(target[-1] - target[0], 15) * 128 / math.pi)) & 0xFF
        center_row = int(np.median(target) // 16)
        options = []
        for tile_y in range(max(0, center_row - 1),
                            min(WORLD_TILES_Y, center_row + 2)):
            for candidate in self.candidates:
                absolute = np.asarray(candidate.offsets, dtype=float) + tile_y * 16
                height_error = float(np.mean((absolute - target) ** 2))
                angle_error = self._angle_distance(candidate.angle, tangent)
                options.append(_FitOption(
                    height_error + 0.08 * angle_error ** 2,
                    tile_y, candidate, tuple(float(value) for value in absolute)))
        options.sort(key=lambda option: option.local_cost)
        return options[:self.LOCAL_CHOICES]

    def fit(self, x_values: np.ndarray,
            y_values: np.ndarray) -> list[TerrainColumn]:
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        if (x_values.ndim != 1 or y_values.ndim != 1 or
                len(x_values) != len(y_values) or len(x_values) < 2):
            raise ValueError("terrain curve must contain matching x/y samples")
        order = np.argsort(x_values)
        x_values, y_values = x_values[order], y_values[order]
        first_tile = max(0, int(x_values[0]) // 16)
        last_tile = min(WORLD_TILES_X - 1, int(x_values[-1]) // 16)
        if last_tile < first_tile:
            raise ValueError("terrain curve is outside the level")
        columns = [
            (tile_x, self._local_options(tile_x, x_values, y_values))
            for tile_x in range(first_tile, last_tile + 1)
        ]

        cumulative: list[list[float]] = []
        backtrack: list[list[int]] = []
        for index, (_, options) in enumerate(columns):
            if index == 0:
                cumulative.append([option.local_cost for option in options])
                backtrack.append([-1] * len(options))
                continue
            previous_options = columns[index - 1][1]
            previous_costs = cumulative[-1]
            costs, choices = [], []
            for option in options:
                possible = []
                for previous_index, previous in enumerate(previous_options):
                    seam = (previous.absolute_profile[-1] -
                            option.absolute_profile[0])
                    seam_cost = 3 * seam ** 2
                    if abs(seam) > 4:
                        seam_cost += 200 * (abs(seam) - 4) ** 2
                    possible.append(previous_costs[previous_index] +
                                    option.local_cost + seam_cost)
                best = min(range(len(possible)), key=possible.__getitem__)
                costs.append(possible[best])
                choices.append(best)
            cumulative.append(costs)
            backtrack.append(choices)

        selected = min(range(len(cumulative[-1])),
                       key=cumulative[-1].__getitem__)
        result = []
        for index in range(len(columns) - 1, -1, -1):
            tile_x, options = columns[index]
            option = options[selected]
            result.append(TerrainColumn(
                tile_x, option.tile_y,
                option.candidate.collision_id,
                option.candidate.flip_x))
            selected = backtrack[index][selected]
        result.reverse()
        return result


def apply_terrain_columns(level: EditableLevel,
                          columns: Iterable[TerrainColumn]) -> list[Edit]:
    """Rebuild selected map columns as a surface with solid backing."""
    edits = []
    backing = encode_tile(0xFF, SOLID_ALL)
    for column in columns:
        if not (0 <= column.tile_x < WORLD_TILES_X and
                0 <= column.tile_y < WORLD_TILES_Y):
            continue
        for tile_y in range(WORLD_TILES_Y):
            if tile_y < column.tile_y:
                word = 0
            elif tile_y == column.tile_y:
                word = column.surface_word
            else:
                word = backing
            previous = level.word_at(column.tile_x, tile_y)
            if previous != word:
                level.set_word(column.tile_x, tile_y, word)
                edits.append(Edit(column.tile_x, tile_y, previous, word))
    return edits


class SonicRuntime:
    """Own the emulator and mirror EditableLevel changes into live WRAM."""

    def __init__(self):
        self.env = retro.make(GAME, STATE, render_mode=None)
        self.env.reset()
        self.buttons = self.env.buttons
        self.button_indices = {name: index
                               for index, name in enumerate(self.buttons)}
        self.memory = self.env.data.memory
        rom_path = retro.data.get_original_romfile_path(GAME)
        self.rom = Path(rom_path).read_bytes()
        noop = [0] * len(self.buttons)
        # The state starts at the act card; wait until native control unlocks.
        for _ in range(230):
            self.env.step(noop)
        self.level: EditableLevel | None = None
        self.respawned = False

    @staticmethod
    def _state_u8(state: bytearray, offset: int, value: int) -> None:
        state[WRAM_STATE_OFFSET + (offset ^ 1)] = value & 0xFF

    @staticmethod
    def _state_u16(state: bytearray, offset: int, value: int) -> None:
        state[WRAM_STATE_OFFSET + offset] = value & 0xFF
        state[WRAM_STATE_OFFSET + offset + 1] = (value >> 8) & 0xFF

    def install(self, level: EditableLevel) -> None:
        state = bytearray(self.env.em.get_state())
        if len(state) < WRAM_STATE_OFFSET + 0x10000:
            raise RuntimeError("unexpected Genesis save-state layout")

        # Zero all custom chunk mappings and the private collision index.
        state[WRAM_STATE_OFFSET + RAM_CHUNKS:
              WRAM_STATE_OFFSET + RAM_COLLISION_TABLE + 0x800] = \
            bytes(RAM_COLLISION_TABLE + 0x800)
        # The foreground layout has eight rows with a $80-byte stride.
        state[WRAM_STATE_OFFSET + RAM_LAYOUT:
              WRAM_STATE_OFFSET + RAM_LAYOUT + 0x400] = bytes(0x400)

        for chunk_y in range(CHUNKS_Y):
            for chunk_x in range(CHUNKS_X):
                chunk_number = chunk_y * CHUNKS_X + chunk_x + 1
                layout_offset = RAM_LAYOUT + chunk_y * 0x80 + chunk_x
                self._state_u8(state, layout_offset, chunk_number)

        # Mapping block N directly to collision shape N makes the entire ROM
        # collision-mask library available to the inventory without terrain
        # artwork dependencies.
        for collision_id in range(256):
            self._state_u8(state, RAM_COLLISION_TABLE + collision_id,
                           collision_id)

        for index, word in enumerate(level.cells):
            if not word:
                continue
            tile_y, tile_x = divmod(index, WORLD_TILES_X)
            self._state_u16(state, chunk_word_offset(tile_x, tile_y), word)

        # A 68k long pointer is two independently word-swapped words in the
        # serialized RAM image: $00FF:$8000 -> $FF8000.
        self._state_u16(state, RAM_COLLISION_INDEX, 0x00FF)
        self._state_u16(state, RAM_COLLISION_INDEX + 2, 0x8000)

        # Remove every object except Sonic.  Terrain remains native, while
        # monitors, enemies, and GHZ set pieces cannot invisibly interfere.
        state[WRAM_STATE_OFFSET + RAM_OBJECTS_AFTER_SONIC:
              WRAM_STATE_OFFSET + RAM_OBJECTS_END] = \
            bytes(RAM_OBJECTS_END - RAM_OBJECTS_AFTER_SONIC)
        self._put_player_in_state(state, level)
        self.env.em.set_state(bytes(state))
        self.env.data.update_ram()
        self.level = level

    def _put_player_in_state(self, state: bytearray,
                             level: EditableLevel) -> None:
        for offset, value in (
                (RAM_SONIC + 0x08, level.spawn_x),
                (RAM_SONIC + 0x0A, 0),
                (RAM_SONIC + 0x0C, level.spawn_y),
                (RAM_SONIC + 0x0E, 0),
                (RAM_SONIC + 0x10, 0),
                (RAM_SONIC + 0x12, 0),
                (RAM_SONIC + 0x14, 0),
                (RAM_SONIC + 0x30, 60),
                (RAM_CAMERA_X, max(0, level.spawn_x - NATIVE_WIDTH // 2)),
                (RAM_CAMERA_Y, min(0x300,
                                   max(0, level.spawn_y - NATIVE_HEIGHT // 2)))):
            self._state_u16(state, offset, value)
        self._state_u8(state, RAM_SONIC + 0x16, 19)
        self._state_u8(state, RAM_SONIC + 0x17, 9)
        self._state_u8(state, RAM_SONIC + 0x1C, 5)
        self._state_u8(state, RAM_SONIC + 0x22, 0)
        self._state_u8(state, RAM_SONIC + 0x24, 2)
        self._state_u8(state, RAM_SONIC + 0x26, 0)

    def write_tile(self, x: int, y: int, word: int) -> None:
        validate_tile_word(word)
        address = RAM_BASE + chunk_word_offset(x, y)
        self.memory.assign(address, ">u2", word)

    def action(self, held: set[int]) -> list[int]:
        action = [0] * len(self.buttons)
        if pygame.K_LEFT in held:
            action[self.button_indices["LEFT"]] = 1
        if pygame.K_RIGHT in held:
            action[self.button_indices["RIGHT"]] = 1
        if pygame.K_DOWN in held:
            action[self.button_indices["DOWN"]] = 1
        if pygame.K_UP in held or pygame.K_SPACE in held:
            action[self.button_indices["B"]] = 1
        return action

    def _strip_non_sonic_objects(self) -> None:
        state = bytearray(self.env.em.get_state())
        start = WRAM_STATE_OFFSET + RAM_OBJECTS_AFTER_SONIC
        end = WRAM_STATE_OFFSET + RAM_OBJECTS_END
        if any(state[start:end]):
            state[start:end] = bytes(end - start)
            self.env.em.set_state(bytes(state))
            self.env.data.update_ram()

    def step(self, held: set[int]) -> None:
        self.respawned = False
        # Keep the custom collision table authoritative even if a native game
        # event attempts to restore the zone's original pointer.
        self.memory.assign(RAM_BASE + RAM_COLLISION_INDEX,
                           ">u4", RAM_BASE + RAM_COLLISION_TABLE)
        self.memory.assign(RAM_BASE + RAM_SONIC + 0x30, ">u2", 60)
        out = self.env.step(self.action(held))
        done = ((len(out) == 5 and (out[2] or out[3])) or
                (len(out) == 4 and out[2]))
        self._strip_non_sonic_objects()
        if done:
            self.env.reset()
            noop = [0] * len(self.buttons)
            for _ in range(230):
                self.env.step(noop)
            assert self.level is not None
            self.install(self.level)
            self.respawned = True
            return

        info = self.player_info()
        if (info["y"] > WORLD_HEIGHT + 128 or
                info["x"] > WORLD_WIDTH + 128 or
                info["routine"] >= 6):
            self.reset_player()
            self.respawned = True

    def reset_player(self) -> None:
        assert self.level is not None
        level = self.level
        assignments = (
            (RAM_SONIC + 0x08, ">u2", level.spawn_x),
            (RAM_SONIC + 0x0A, ">u2", 0),
            (RAM_SONIC + 0x0C, ">u2", level.spawn_y),
            (RAM_SONIC + 0x0E, ">u2", 0),
            (RAM_SONIC + 0x10, ">u2", 0),
            (RAM_SONIC + 0x12, ">u2", 0),
            (RAM_SONIC + 0x14, ">u2", 0),
            (RAM_SONIC + 0x16, "|u1", 19),
            (RAM_SONIC + 0x17, "|u1", 9),
            (RAM_SONIC + 0x1C, "|u1", 5),
            (RAM_SONIC + 0x22, "|u1", 0),
            (RAM_SONIC + 0x24, "|u1", 2),
            (RAM_SONIC + 0x26, "|u1", 0),
            (RAM_SONIC + 0x30, ">u2", 60),
            (RAM_CAMERA_X, ">u2", max(0, level.spawn_x - 160)),
            (RAM_CAMERA_Y, ">u2", min(0x300,
                                       max(0, level.spawn_y - 112))),
        )
        for offset, kind, value in assignments:
            self.memory.assign(RAM_BASE + offset, kind, value)
        self._strip_non_sonic_objects()

    def player_info(self) -> dict[str, int | bool]:
        get = self.memory.extract
        status = get(RAM_BASE + RAM_SONIC + 0x22, "|u1")
        return {
            "x": get(RAM_BASE + RAM_SONIC + 0x08, ">u2"),
            "y": get(RAM_BASE + RAM_SONIC + 0x0C, ">u2"),
            "xvel": get(RAM_BASE + RAM_SONIC + 0x10, ">i2"),
            "yvel": get(RAM_BASE + RAM_SONIC + 0x12, ">i2"),
            "inertia": get(RAM_BASE + RAM_SONIC + 0x14, ">i2"),
            "angle": get(RAM_BASE + RAM_SONIC + 0x26, "|u1"),
            "routine": get(RAM_BASE + RAM_SONIC + 0x24, "|u1"),
            "airborne": bool(status & 0x02),
        }

    def camera(self) -> tuple[int, int]:
        return (self.memory.extract(RAM_BASE + RAM_CAMERA_X, ">u2"),
                self.memory.extract(RAM_BASE + RAM_CAMERA_Y, ">u2"))

    def close(self) -> None:
        self.env.close()


def line_cells(start: tuple[int, int], end: tuple[int, int]):
    """Integer Bresenham line used so fast mouse drags do not leave gaps."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


class MakerApp:
    PALETTE_COLUMNS = 6
    PALETTE_ROWS = 4
    PALETTE_CELL = 44
    SPRITE_COLUMNS = 4
    SPRITE_SLOTS = 8

    def __init__(self, runtime: SonicRuntime, assets: CollisionAssets,
                 level: EditableLevel, level_path: Path | None, scale: int = 3,
                 headless: bool = False, max_frames: int = 0,
                 levels_dir: Path | None = None,
                 texture_dir: Path | None = None,
                 sprite_dir: Path | None = None,
                 visualization_dir: Path | None = None):
        self.runtime = runtime
        self.assets = assets
        self.level = level
        self.level_path = (Path(level_path).expanduser().resolve()
                           if level_path is not None else None)
        self.levels_dir = Path(
            levels_dir or DEFAULT_LEVEL_DIRECTORY).expanduser().resolve()
        self.storage_error = ""
        try:
            self.levels_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.storage_error = str(error)
        extras = [DEFAULT_LEVEL_FILE.expanduser().resolve()]
        if self.level_path is not None:
            extras.append(self.level_path)
        self.extra_level_paths = tuple(dict.fromkeys(extras))
        self.scale = scale
        self.headless = headless
        self.max_frames = max_frames
        self.canvas_width = NATIVE_WIDTH * scale
        self.canvas_height = NATIVE_HEIGHT * scale
        self.window_height = max(self.canvas_height, PANEL_HEIGHT)
        self.panel_x = self.canvas_width

        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.canvas_width + PANEL_WIDTH, self.window_height))
        pygame.display.set_caption("Sonic Maker - live native collision")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 30)
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.banner_font = pygame.font.Font(None, 56)

        self.texture_dir = (Path(texture_dir).expanduser()
                            if texture_dir is not None else None)
        self.sprite_dir = (Path(sprite_dir).expanduser()
                           if sprite_dir is not None else None)
        self.visualization_dir = Path(
            visualization_dir or DEFAULT_VISUALIZATION_DIRECTORY
        ).expanduser().resolve()
        self.theme = SemanticTheme(level.theme, texture_dir=self.texture_dir)
        self.sprites = SpriteLibrary(self.sprite_dir)
        self.renderer = LevelRenderer(assets, self.theme, self.sprites)

        self.held: set[int] = set()
        self.running = True
        self.paused = False
        self.show_grid = True
        self.show_visual = True
        self.show_collision = True
        self.collision_alpha = ViewOptions().collision_alpha
        self.collision_outline = False
        self.visual_material = "grass"
        self.visual_foreground = False
        self.visual_mode = "materials"
        self.visual_stroke_before: \
            dict[tuple[int, int], tuple[str, int] | None] | None = None
        self.stroke_material: tuple[str | None, int] = (MATERIAL_AUTO, 0)
        self.sprite_choice: str | None = (self.sprites.names[0]
                                          if self.sprites.names else None)
        self.sprite_scroll = 0
        self.sprite_front = False
        self.selected_sprite: int | None = None
        self.sprite_drag: str | None = None
        self.sprite_drag_offset = (0.0, 0.0)
        self.sprite_before: PlacedSprite | None = None
        self.capture_edge: str | None = None
        self.capture_queue: tuple[str, ...] = ()
        self.capture_snap = True
        self.capture_grid = True
        self.capture_collision = False
        self.last_capture: Path | None = None
        self._minimap: pygame.Surface | None = None
        self.editor_tab = "tiles"
        self.terrain_tool = "freehand"
        self.terrain_fitter = TerrainFitter(assets)
        self.freehand_points: list[tuple[float, float]] = []
        self.spline_points: list[tuple[float, float]] = []
        self.terrain_drawing = False
        self.drag_anchor: int | None = None
        self.last_terrain_columns = 0
        self.marker_tool = "start"
        self.marker_dragging = False
        self.marker_drag_offset = (0.0, 0.0)
        self.finish_armed = True
        self.finish_cooldown = 0
        self.completion_until = 0
        self.selected_id = 0x0F
        self.solidity = SOLID_ALL
        self.flip_x = False
        self.flip_y = False
        self.erase_mode = False
        self.stamp_mode: str | None = None
        useful = [cid for cid in FAVORITE_COLLISION_IDS
                  if np.any(assets.masks[cid, 0])]
        self.palette_ids = useful + [
            cid for cid in range(1, 256)
            if cid not in useful and np.any(assets.masks[cid, 0])
        ]
        self.palette_scroll = 0
        self.history = EditHistory()
        self.stroke_before: dict[tuple[int, int], int] | None = None
        self.stroke_word = 0
        self.stroke_stamp: str | None = None
        self.stroke_last: tuple[int, int] | None = None
        self.frame = 0
        self.message = "LIVE GENESIS COLLISION"
        self.message_until = 180
        self.thumbnail_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.file_dialog: str | None = None
        self.dialog_name = (self.level_path.stem
                            if self.level_path is not None else "untitled")
        self.dialog_files: list[Path] = []
        self.dialog_selected: Path | None = None
        self.dialog_scroll = 0
        self.dialog_error = ""
        self.dialog_select_all = False
        self._make_rects()
        self.runtime.install(level)
        info = self.runtime.player_info()
        self.finish_armed = not finish_marker_reached(
            int(info["x"]), int(info["y"]),
            self.level.finish_x, self.level.finish_y)

    def _make_rects(self) -> None:
        x = self.panel_x + 12
        self.buttons = {
            "view_visual": pygame.Rect(x + 152, 7, 68, 22),
            "view_collision": pygame.Rect(x + 224, 7, 68, 22),
            "tab_tiles": pygame.Rect(x, 36, 70, 24),
            "tab_terrain": pygame.Rect(x + 73, 36, 70, 24),
            "tab_markers": pygame.Rect(x + 146, 36, 70, 24),
            "tab_visual": pygame.Rect(x + 219, 36, 70, 24),
            "reset": pygame.Rect(x, 64, 92, 28),
            "save": pygame.Rect(x + 100, 64, 92, 28),
            "load": pygame.Rect(x + 200, 64, 92, 28),
            "new": pygame.Rect(x, 98, 92, 28),
            "pause": pygame.Rect(x + 100, 98, 92, 28),
            "grid": pygame.Rect(x + 200, 98, 92, 28),
            "top": pygame.Rect(x, 232, 92, 28),
            "all": pygame.Rect(x + 100, 232, 92, 28),
            "sides": pygame.Rect(x + 200, 232, 92, 28),
            "flip_x": pygame.Rect(x, 283, 92, 28),
            "flip_y": pygame.Rect(x + 100, 283, 92, 28),
            "erase": pygame.Rect(x + 200, 283, 92, 28),
            "stamp_45_right": pygame.Rect(x, 331, 142, 28),
            "stamp_45_left": pygame.Rect(x + 150, 331, 142, 28),
            "stamp_gentle_right": pygame.Rect(x, 365, 142, 28),
            "stamp_gentle_left": pygame.Rect(x + 150, 365, 142, 28),
            "terrain_freehand": pygame.Rect(x, 142, 142, 32),
            "terrain_spline": pygame.Rect(x + 150, 142, 142, 32),
            "terrain_generate": pygame.Rect(x, 184, 92, 30),
            "terrain_clear": pygame.Rect(x + 100, 184, 92, 30),
            "terrain_undo": pygame.Rect(x + 200, 184, 92, 30),
            "marker_start": pygame.Rect(x, 142, 142, 32),
            "marker_finish": pygame.Rect(x + 150, 142, 142, 32),
            "marker_reset": pygame.Rect(x, 184, 292, 30),
            "visual_mode_materials": pygame.Rect(x, 128, 94, 26),
            "visual_mode_sprites": pygame.Rect(x + 99, 128, 94, 26),
            "visual_mode_capture": pygame.Rect(x + 198, 128, 94, 26),
            "capture_left": pygame.Rect(x, 254, 142, 28),
            "capture_right": pygame.Rect(x + 150, 254, 142, 28),
            "capture_top": pygame.Rect(x, 288, 142, 28),
            "capture_bottom": pygame.Rect(x + 150, 288, 142, 28),
            "capture_sequence": pygame.Rect(x, 322, 142, 28),
            "capture_view": pygame.Rect(x + 150, 322, 142, 28),
            "capture_snap": pygame.Rect(x, 356, 142, 28),
            "capture_grid": pygame.Rect(x + 150, 356, 142, 28),
            "capture_collision": pygame.Rect(x, 390, 142, 28),
            "capture_export": pygame.Rect(x + 150, 390, 142, 28),
            "visual_foreground": pygame.Rect(x, 330, 142, 28),
            "visual_outline": pygame.Rect(x + 150, 330, 142, 28),
            "visual_opacity": pygame.Rect(x, 364, 142, 28),
            "visual_reload": pygame.Rect(x + 150, 364, 142, 28),
            "sprite_plane": pygame.Rect(x, 342, 142, 28),
            "sprite_delete": pygame.Rect(x + 150, 342, 142, 28),
            "sprite_smaller": pygame.Rect(x, 376, 142, 28),
            "sprite_bigger": pygame.Rect(x + 150, 376, 142, 28),
        }
        for index, key in enumerate(MATERIAL_ORDER):
            row, column = divmod(index, 2)
            self.buttons[f"material_{key}"] = pygame.Rect(
                x + column * 150, 174 + row * 34, 142, 30)
        self.sprite_slots = tuple(
            pygame.Rect(x + column * 73, 174 + row * 73, 70, 70)
            for row, column in (divmod(index, self.SPRITE_COLUMNS)
                                for index in range(self.SPRITE_SLOTS)))
        self.minimap_rect = pygame.Rect(x, 174, 288, 72)
        self.preview_rect = pygame.Rect(x, 142, 72, 72)
        self.palette_rect = pygame.Rect(x, 415,
                                        self.PALETTE_COLUMNS * self.PALETTE_CELL,
                                        self.PALETTE_ROWS * self.PALETTE_CELL)
        window = self.screen.get_rect()
        self.dialog_rect = pygame.Rect(0, 0, 560, 540)
        self.dialog_rect.center = window.center
        dx, dy = self.dialog_rect.x, self.dialog_rect.y
        self.dialog_input_rect = pygame.Rect(dx + 24, dy + 104, 512, 38)
        self.dialog_list_rect = pygame.Rect(dx + 24, dy + 181, 512, 242)
        self.dialog_cancel_rect = pygame.Rect(dx + 242, dy + 476, 136, 38)
        self.dialog_confirm_rect = pygame.Rect(dx + 400, dy + 476, 136, 38)

    @property
    def brush_word(self) -> int:
        if self.erase_mode:
            return 0
        return encode_tile(self.selected_id, self.solidity,
                           self.flip_x, self.flip_y)

    def notify(self, text: str, frames: int = 150) -> None:
        self.message = text
        self.message_until = self.frame + frames

    def tile_from_mouse(self, position: tuple[int, int]) -> tuple[int, int] | None:
        mx, my = position
        if not (0 <= mx < self.canvas_width and 0 <= my < self.canvas_height):
            return None
        camera_x, camera_y = self.runtime.camera()
        tile_x = (mx // self.scale + camera_x) // 16
        tile_y = (my // self.scale + camera_y) // 16
        if 0 <= tile_x < WORLD_TILES_X and 0 <= tile_y < WORLD_TILES_Y:
            return tile_x, tile_y
        return None

    def world_from_mouse(self, position: tuple[int, int]) \
            -> tuple[float, float] | None:
        mx, my = position
        if not (0 <= mx < self.canvas_width and 0 <= my < self.canvas_height):
            return None
        camera_x, camera_y = self.runtime.camera()
        return (min(WORLD_WIDTH - 1,
                    max(0.0, mx / self.scale + camera_x)),
                min(WORLD_HEIGHT - 1,
                    max(0.0, my / self.scale + camera_y)))

    def _visible_button_names(self) -> tuple[str, ...]:
        common = ("view_visual", "view_collision", "tab_tiles", "tab_terrain",
                  "tab_markers", "tab_visual", "reset", "save", "load",
                  "new", "pause", "grid")
        if self.editor_tab == "terrain":
            return common + ("terrain_freehand", "terrain_spline",
                             "terrain_generate", "terrain_clear",
                             "terrain_undo")
        if self.editor_tab == "markers":
            return common + ("marker_start", "marker_finish",
                             "marker_reset")
        if self.editor_tab == "visual":
            modes = ("visual_mode_materials", "visual_mode_sprites",
                     "visual_mode_capture")
            if self.visual_mode == "capture":
                return common + modes + tuple(
                    f"capture_{edge}" for edge in CAPTURE_EDGES) + (
                    "capture_sequence", "capture_view", "capture_snap",
                    "capture_grid", "capture_collision", "capture_export")
            if self.visual_mode == "sprites":
                return common + modes + ("sprite_plane", "sprite_delete",
                                         "sprite_smaller", "sprite_bigger")
            return common + modes \
                + tuple(f"material_{key}" for key in MATERIAL_ORDER) \
                + ("visual_foreground", "visual_outline", "visual_opacity",
                   "visual_reload")
        return common + ("top", "all", "sides", "flip_x", "flip_y",
                         "erase", "stamp_45_right", "stamp_45_left",
                         "stamp_gentle_right", "stamp_gentle_left")

    def view_options(self) -> ViewOptions:
        return ViewOptions(self.show_visual, self.show_collision,
                           self.collision_alpha, self.collision_outline,
                           self.show_grid)

    def _nearest_anchor(self, point: tuple[float, float],
                        maximum: float = 10.0) -> int | None:
        if not self.spline_points:
            return None
        distances = [math.hypot(x - point[0], y - point[1])
                     for x, y in self.spline_points]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        return nearest if distances[nearest] <= maximum else None

    def _terrain_mouse_down(self, button: int,
                            position: tuple[int, int]) -> None:
        point = self.world_from_mouse(position)
        if point is None:
            return
        if self.terrain_tool == "freehand":
            if button == 1:
                self.freehand_points = [point]
                self.terrain_drawing = True
            return
        nearest = self._nearest_anchor(point)
        if button == 1:
            if nearest is None:
                self.spline_points.append(point)
                nearest = len(self.spline_points) - 1
            self.drag_anchor = nearest
        elif button == 3 and nearest is not None:
            del self.spline_points[nearest]
            self.notify("SPLINE ANCHOR REMOVED", 90)

    def _terrain_mouse_motion(self, position: tuple[int, int]) -> None:
        point = self.world_from_mouse(position)
        if point is None:
            return
        if self.terrain_tool == "freehand" and self.terrain_drawing:
            if (not self.freehand_points or
                    math.dist(self.freehand_points[-1], point) >= 1.5):
                self.freehand_points.append(point)
        elif self.terrain_tool == "spline" and self.drag_anchor is not None:
            self.spline_points[self.drag_anchor] = point

    def _terrain_mouse_up(self, button: int) -> None:
        if button != 1:
            return
        if self.terrain_tool == "freehand" and self.terrain_drawing:
            self.terrain_drawing = False
            if len(self.freehand_points) >= 2:
                self.generate_terrain()
        self.drag_anchor = None

    def _terrain_curve(self) -> tuple[np.ndarray, np.ndarray]:
        if self.terrain_tool == "freehand":
            return freehand_surface(self.freehand_points)
        return spline_surface(self.spline_points)

    def generate_terrain(self) -> None:
        try:
            x_values, y_values = self._terrain_curve()
            columns = self.terrain_fitter.fit(x_values, y_values)
        except ValueError as error:
            self.notify(str(error).upper(), 210)
            return
        edits = apply_terrain_columns(self.level, columns)
        for edit in edits:
            self.runtime.write_tile(edit.x, edit.y, edit.after)
            self._touch_cell(edit.x, edit.y)
        self.history.record(edits)
        self.last_terrain_columns = len(columns)
        self.notify(f"GENERATED {len(columns)} TERRAIN COLUMNS", 180)

    def clear_terrain_curve(self) -> None:
        self.freehand_points.clear()
        self.spline_points.clear()
        self.terrain_drawing = False
        self.drag_anchor = None
        self.notify("TERRAIN PATH CLEARED", 90)

    def _marker_at_point(self, point: tuple[float, float]) -> str | None:
        choices = (
            ("start", self.level.start_marker),
            ("finish", self.level.finish_marker),
        )
        hits = []
        for name, (marker_x, marker_y) in choices:
            dx = abs(point[0] - marker_x)
            dy = marker_y - point[1]
            if dx <= 16 and -12 <= dy <= 54:
                hits.append((dx + abs(dy) * 0.2, name))
        return min(hits)[1] if hits else None

    def _set_marker_at(self, marker: str,
                       point: tuple[float, float]) -> None:
        if marker == "start":
            self.level.set_start_marker(*point)
        elif marker == "finish":
            self.level.set_finish_marker(*point)
            self.finish_armed = False
            self.finish_cooldown = 0
        else:
            raise ValueError("unknown marker")

    def _marker_mouse_down(self, button: int,
                           position: tuple[int, int]) -> None:
        if button != 1:
            return
        point = self.world_from_mouse(position)
        if point is None:
            return
        clicked_marker = self._marker_at_point(point)
        if clicked_marker is not None:
            self.marker_tool = clicked_marker
            marker_point = (self.level.start_marker
                            if clicked_marker == "start"
                            else self.level.finish_marker)
            self.marker_drag_offset = (
                marker_point[0] - point[0], marker_point[1] - point[1])
        else:
            self.marker_drag_offset = (0.0, 0.0)
        adjusted = (point[0] + self.marker_drag_offset[0],
                    point[1] + self.marker_drag_offset[1])
        self._set_marker_at(self.marker_tool, adjusted)
        self.marker_dragging = True

    def _marker_mouse_motion(self, position: tuple[int, int]) -> None:
        if not self.marker_dragging:
            return
        point = self.world_from_mouse(position)
        if point is not None:
            adjusted = (point[0] + self.marker_drag_offset[0],
                        point[1] + self.marker_drag_offset[1])
            self._set_marker_at(self.marker_tool, adjusted)

    def _marker_mouse_up(self, button: int) -> None:
        if button != 1 or not self.marker_dragging:
            return
        self.marker_dragging = False
        self.marker_drag_offset = (0.0, 0.0)
        marker = (self.level.start_marker if self.marker_tool == "start"
                  else self.level.finish_marker)
        self.notify(
            f"{self.marker_tool.upper()} MARKER x={marker[0]} y={marker[1]}",
            150)

    def reset_sonic_to_start(self) -> None:
        self.runtime.reset_player()
        self.finish_armed = False
        self.finish_cooldown = 20
        self.notify("SONIC RESET TO START")

    def _update_level_completion(self) -> None:
        if self.finish_cooldown > 0:
            self.finish_cooldown -= 1
        info = self.runtime.player_info()
        inside = finish_marker_reached(
            int(info["x"]), int(info["y"]),
            self.level.finish_x, self.level.finish_y)
        if not inside:
            self.finish_armed = True
            return
        if not self.finish_armed or self.finish_cooldown:
            return
        self.finish_armed = False
        self.finish_cooldown = 45
        self.runtime.reset_player()
        self.completion_until = self.frame + 180
        self.notify("LEVEL COMPLETE!", 180)

    def _set_editor_tab(self, tab: str) -> None:
        if tab not in EDITOR_TABS:
            raise ValueError("unknown editor tab")
        if tab == self.editor_tab:
            return
        self._finish_stroke()
        self.terrain_drawing = False
        self.drag_anchor = None
        self.marker_dragging = False
        self.marker_drag_offset = (0.0, 0.0)
        self.sprite_drag = None
        self.sprite_before = None
        self.editor_tab = tab
        self.notify(f"{tab.upper()} EDITOR", 90)

    def _set_visual_mode(self, mode: str) -> None:
        if mode not in VISUAL_MODES:
            raise ValueError("unknown visual mode")
        if mode == self.visual_mode:
            return
        self._finish_stroke()
        self.sprite_drag = None
        self.sprite_before = None
        self.visual_mode = mode
        self.notify("SPRITE PLACEMENT" if mode == "sprites"
                    else "MATERIAL PAINTING", 120)

    def toggle_visual_layer(self) -> None:
        self.show_visual = not self.show_visual
        self.notify("ARTWORK ON" if self.show_visual else "ARTWORK OFF", 120)

    def toggle_collision_layer(self) -> None:
        self.show_collision = not self.show_collision
        self.notify("COLLISION OVERLAY ON" if self.show_collision
                    else "COLLISION OVERLAY OFF", 120)

    def reload_artwork(self) -> None:
        replaced = self.theme.reload()
        names = self.sprites.reload()
        self.renderer.invalidate_all()
        if self.sprite_choice not in names:
            self.sprite_choice = names[0] if names else None
        if replaced or names:
            self.notify(f"RELOADED {len(replaced)} MATERIALS, "
                        f"{len(names)} SPRITES", 210)
        elif self.texture_dir is None and self.sprite_dir is None:
            self.notify("NO ART FOLDERS; USING PLACEHOLDERS", 210)
        else:
            self.notify("NO ARTWORK FOUND IN THOSE FOLDERS", 210)

    # -- sprite placement ---------------------------------------------------

    @staticmethod
    def _sprite_handle(sprite: PlacedSprite) -> tuple[float, float]:
        """World position of the resize grip.

        It sits beside the base rather than at the top corner: a tall tree's
        top is often above the camera, but wherever you dropped a sprite is by
        definition on screen.  Dragging away from the base grows it.
        """
        return (sprite.x + sprite.width / 2, sprite.y)

    def _sprite_at_point(self, point: tuple[float, float]) -> int | None:
        """Topmost decoration under the cursor, ignoring transparent pixels."""
        for sprite_id, sprite in reversed(list(self.level.sprites.items())):
            if not sprite.contains(*point):
                continue
            patch = self.sprites.scaled(sprite.art, sprite.width,
                                        sprite.height)
            # A sprite whose image has gone missing still draws an outline, so
            # it stays selectable and can be removed rather than being stuck.
            if patch is None:
                return sprite_id
            left, top, _width, _height = sprite.bounds
            if patch[int(point[1] - top), int(point[0] - left), 3] <= 8:
                continue
            return sprite_id
        return None

    def _resized_sprite(self, sprite: PlacedSprite,
                        height: float) -> PlacedSprite:
        aspect = self.sprites.aspect(
            sprite.art, sprite.width / max(1, sprite.height))
        new_height = int(round(min(SPRITE_MAX_SIZE,
                                   max(SPRITE_MIN_SIZE, height))))
        new_width = int(round(min(SPRITE_MAX_SIZE,
                                  max(SPRITE_MIN_SIZE, new_height * aspect))))
        return replace(sprite, width=new_width, height=new_height)

    def _select_sprite(self, sprite_id: int | None) -> None:
        self.selected_sprite = sprite_id
        sprite = self.level.sprite_at(sprite_id)
        if sprite is not None:
            self.sprite_front = sprite.front

    def _change_sprite(self, sprite_id: int,
                       sprite: PlacedSprite | None) -> None:
        """Apply one finished decoration change and make it undoable."""
        before = self.level.apply_sprite(sprite_id, sprite)
        if before != sprite:
            self.history.record([SpriteEdit(sprite_id, before, sprite)])

    def _place_sprite(self, point: tuple[float, float]) -> None:
        if self.sprite_choice is None:
            self.notify("ADD IMAGES TO THE SPRITE FOLDER", 210)
            return
        source = self.sprites.source(self.sprite_choice)
        if source is None:
            self.notify("THAT SPRITE IMAGE IS MISSING", 210)
            return
        sprite = self._resized_sprite(
            PlacedSprite(self.sprite_choice, int(point[0]), int(point[1]),
                         SPRITE_DEFAULT_HEIGHT, SPRITE_DEFAULT_HEIGHT,
                         VISUAL_FLAG_FOREGROUND if self.sprite_front else 0),
            SPRITE_DEFAULT_HEIGHT)
        self.selected_sprite = self.level.add_sprite(sprite)
        # The gesture is still open: dragging now moves the sprite that was
        # just dropped, and mouse-up records placement and move together.
        self.sprite_before = None
        self.sprite_drag = "move"
        self.sprite_drag_offset = (0.0, 0.0)
        self.notify(f"PLACED {sprite.art.upper()}", 120)

    def _sprite_mouse_down(self, button: int,
                           position: tuple[int, int]) -> None:
        point = self.world_from_mouse(position)
        if point is None:
            return
        if button == 3:
            hit = self._sprite_at_point(point)
            if hit is not None:
                self._change_sprite(hit, None)
                if self.selected_sprite == hit:
                    self.selected_sprite = None
                self.notify("SPRITE REMOVED", 120)
            return
        if button != 1:
            return

        selected = self.level.sprite_at(self.selected_sprite)
        grip = max(5.0, 8.0 / self.scale)
        if (selected is not None and
                math.dist(self._sprite_handle(selected), point) <= grip):
            self.sprite_before = selected
            self.sprite_drag = "scale"
            return

        hit = self._sprite_at_point(point)
        if hit is None:
            self._place_sprite(point)
            return
        sprite = self.level.sprites[hit]
        self._select_sprite(hit)
        self.sprite_before = sprite
        self.sprite_drag = "move"
        self.sprite_drag_offset = (sprite.x - point[0], sprite.y - point[1])

    def _sprite_mouse_motion(self, position: tuple[int, int]) -> None:
        sprite = self.level.sprite_at(self.selected_sprite)
        if self.sprite_drag is None or sprite is None:
            return
        point = self.world_from_mouse(position)
        if point is None:
            return
        if self.sprite_drag == "move":
            self.level.apply_sprite(self.selected_sprite, replace(
                sprite,
                x=int(round(min(WORLD_WIDTH - 1, max(
                    0, point[0] + self.sprite_drag_offset[0])))),
                y=int(round(min(WORLD_HEIGHT - 1, max(
                    0, point[1] + self.sprite_drag_offset[1]))))))
            return
        # Scaling keeps the base planted and the aspect intact; the pointer's
        # distance out from the base sets the half-width.
        aspect = self.sprites.aspect(
            sprite.art, sprite.width / max(1, sprite.height))
        half_width = max(SPRITE_MIN_SIZE / 2.0, point[0] - sprite.x)
        self.level.apply_sprite(
            self.selected_sprite,
            self._resized_sprite(sprite, half_width * 2.0 / max(aspect, 1e-3)))

    def _sprite_mouse_up(self, button: int) -> None:
        if button != 1 or self.sprite_drag is None:
            return
        sprite = self.level.sprite_at(self.selected_sprite)
        if sprite is not None and sprite != self.sprite_before:
            self.history.record([SpriteEdit(
                self.selected_sprite, self.sprite_before, sprite)])
        self.sprite_drag = None
        self.sprite_before = None

    def _scale_selected_sprite(self, factor: float) -> None:
        sprite = self.level.sprite_at(self.selected_sprite)
        if sprite is None:
            self.notify("SELECT A SPRITE FIRST", 120)
            return
        resized = self._resized_sprite(sprite, sprite.height * factor)
        self._change_sprite(self.selected_sprite, resized)
        self.notify(f"SPRITE {resized.width}x{resized.height}", 90)

    def _set_selected_sprite_plane(self, front: bool) -> None:
        self.sprite_front = front
        sprite = self.level.sprite_at(self.selected_sprite)
        if sprite is not None:
            self._change_sprite(self.selected_sprite, replace(
                sprite,
                flags=VISUAL_FLAG_FOREGROUND if front else 0))
        self.notify("SPRITE IN FRONT OF SONIC" if front
                    else "SPRITE BEHIND SONIC", 120)

    def _delete_selected_sprite(self) -> None:
        if self.level.sprite_at(self.selected_sprite) is None:
            self.notify("SELECT A SPRITE FIRST", 120)
            return
        self._change_sprite(self.selected_sprite, None)
        self.selected_sprite = None
        self.notify("SPRITE REMOVED", 120)

    # -- capture region -----------------------------------------------------

    def view_region(self) -> CaptureRegion:
        """The rectangle the camera is looking at right now."""
        camera_x, camera_y = self.runtime.camera()
        return CaptureRegion.around(camera_x, camera_y, NATIVE_WIDTH,
                                    NATIVE_HEIGHT, self.capture_snap)

    def _ensure_capture_region(self) -> CaptureRegion:
        if self.level.capture is None:
            self.level.capture = self.view_region()
        return self.level.capture

    def _announce_capture_edge(self) -> None:
        remaining = len(self.capture_queue)
        step = "" if not remaining else f"  ({remaining} more)"
        self.notify(f"CLICK THE {self.capture_edge.upper()} EDGE{step}", 240)

    def start_capture_sequence(self) -> None:
        """Walk the four edges in turn, which is how a frame gets set."""
        self._ensure_capture_region()
        self.capture_edge, self.capture_queue = CAPTURE_EDGES[0], \
            CAPTURE_EDGES[1:]
        self._announce_capture_edge()

    def arm_capture_edge(self, edge: str) -> None:
        if edge not in CAPTURE_EDGES:
            raise ValueError(f"unknown capture edge {edge!r}")
        self._ensure_capture_region()
        self.capture_queue = ()
        self.capture_edge = edge
        self._announce_capture_edge()

    def _set_capture_edge(self, point: tuple[float, float]) -> None:
        if self.capture_edge is None:
            return
        region = self._ensure_capture_region()
        value = point[0] if self.capture_edge in ("left", "right") else point[1]
        self.level.capture = region.with_edge(
            self.capture_edge, value, self.capture_snap)
        if self.capture_queue:
            self.capture_edge = self.capture_queue[0]
            self.capture_queue = self.capture_queue[1:]
            self._announce_capture_edge()
            return
        self.capture_edge = None
        current = self.level.capture
        self.notify(f"REGION {current.width} x {current.height}", 180)

    def _capture_mouse_down(self, button: int,
                            position: tuple[int, int]) -> None:
        if button == 3:
            if self.capture_edge is not None:
                self.capture_edge = None
                self.capture_queue = ()
                self.notify("EDGE SETTING CANCELLED", 120)
            return
        if button != 1:
            return
        point = self.world_from_mouse(position)
        if point is None:
            return
        if self.capture_edge is None:
            self.notify("PICK AN EDGE TO SET FIRST", 150)
            return
        self._set_capture_edge(point)

    def _minimap_point(self, position: tuple[int, int]) \
            -> tuple[float, float] | None:
        if not self.minimap_rect.collidepoint(position):
            return None
        return ((position[0] - self.minimap_rect.x) *
                WORLD_WIDTH / self.minimap_rect.width,
                (position[1] - self.minimap_rect.y) *
                WORLD_HEIGHT / self.minimap_rect.height)

    def _minimap_click(self, position: tuple[int, int]) -> bool:
        """The map sets edges too, so a wide frame needs no scrolling."""
        point = self._minimap_point(position)
        if point is None:
            return False
        if self.capture_edge is None:
            self.notify("PICK AN EDGE TO SET FIRST", 150)
        else:
            self._set_capture_edge(point)
        return True

    def use_view_as_capture(self) -> None:
        self.level.capture = self.view_region()
        self.capture_edge = None
        self.capture_queue = ()
        region = self.level.capture
        self.notify(f"REGION {region.width} x {region.height}", 180)

    def capture_view_options(self) -> ViewOptions:
        return ViewOptions(True, self.capture_collision,
                           self.collision_alpha, self.collision_outline,
                           self.capture_grid)

    def export_capture(self) -> None:
        region = self.level.capture
        if region is None:
            self.notify("SET A CAPTURE REGION FIRST", 180)
            return
        stem = (self.level_path.stem if self.level_path is not None
                else self.dialog_name)
        try:
            path = export_visualization(
                self.renderer, self.level, region, self.visualization_dir,
                stem, self.capture_view_options())
        except (OSError, ValueError, pygame.error) as error:
            self.notify(f"EXPORT FAILED: {error}".upper(), 240)
            return
        self.last_capture = path
        self.notify(f"SAVED {path.name}", 240)

    def _sprite_library_click(self, position: tuple[int, int]) -> bool:
        for slot, rect in enumerate(self.sprite_slots):
            if not rect.collidepoint(position):
                continue
            index = self.sprite_scroll + slot
            if index < len(self.sprites.names):
                self.sprite_choice = self.sprites.names[index]
                self.notify(f"SPRITE {self.sprite_choice.upper()}", 120)
            return True
        return False

    def _scroll_sprite_library(self, direction: int) -> None:
        hidden = max(0, len(self.sprites.names) - self.SPRITE_SLOTS)
        maximum = ((hidden + self.SPRITE_COLUMNS - 1) //
                   self.SPRITE_COLUMNS) * self.SPRITE_COLUMNS
        self.sprite_scroll = max(
            0, min(maximum,
                   self.sprite_scroll - direction * self.SPRITE_COLUMNS))

    def _touch_cell(self, x: int, y: int) -> None:
        """Drop the cached artwork a block edit invalidates."""
        self.renderer.invalidate(x, y)
        self._minimap = None

    def _paint_cell(self, x: int, y: int, word: int) -> None:
        assert self.stroke_before is not None
        key = (x, y)
        previous = self.level.word_at(x, y)
        self.stroke_before.setdefault(key, previous)
        if previous != word:
            self.level.set_word(x, y, word)
            self.runtime.write_tile(x, y, word)
            self._touch_cell(x, y)

    def _paint_brush_at(self, x: int, y: int) -> None:
        if self.stroke_stamp is None or self.stroke_word == 0:
            self._paint_cell(x, y, self.stroke_word)
            return
        for dx, dy, word in safe_stamp_pattern(self.stroke_stamp):
            stamp_x, stamp_y = x + dx, y + dy
            if (0 <= stamp_x < WORLD_TILES_X and
                    0 <= stamp_y < WORLD_TILES_Y):
                self._paint_cell(stamp_x, stamp_y, word)

    def _paint_visual_cell(self, x: int, y: int) -> None:
        assert self.visual_stroke_before is not None
        material, flags = self.stroke_material
        wanted = (None if material in (None, MATERIAL_AUTO)
                  else (material, flags))
        previous = self.level.visual_at(x, y)
        self.visual_stroke_before.setdefault((x, y), previous)
        if previous != wanted:
            self.level.apply_visual(x, y, wanted)
            self.renderer.invalidate(x, y)

    def _continue_stroke(self, position: tuple[int, int]) -> None:
        tile = self.tile_from_mouse(position)
        if tile is None:
            self.stroke_last = None
            return
        start = self.stroke_last or tile
        visual = self.visual_stroke_before is not None
        for x, y in line_cells(start, tile):
            if visual:
                self._paint_visual_cell(x, y)
            else:
                self._paint_brush_at(x, y)
        self.stroke_last = tile

    def _start_stroke(self, button: int, position: tuple[int, int]) -> None:
        if self.tile_from_mouse(position) is None:
            return
        if self.editor_tab == "visual":
            self.visual_stroke_before = {}
            self.stroke_material = (
                (None, 0) if button == 3 else
                (self.visual_material,
                 VISUAL_FLAG_FOREGROUND if self.visual_foreground else 0))
        else:
            self.stroke_before = {}
            self.stroke_word = 0 if button == 3 else self.brush_word
            self.stroke_stamp = None if button == 3 else self.stamp_mode
        self.stroke_last = None
        self._continue_stroke(position)

    def _finish_stroke(self) -> None:
        edits: list[Edit | VisualEdit] = []
        if self.stroke_before is not None:
            edits += [Edit(x, y, before, self.level.word_at(x, y))
                      for (x, y), before in self.stroke_before.items()]
        if self.visual_stroke_before is not None:
            edits += [VisualEdit(x, y, before, self.level.visual_at(x, y))
                      for (x, y), before in self.visual_stroke_before.items()]
        if self.stroke_before is None and self.visual_stroke_before is None:
            return
        self.history.record(edits)
        self.stroke_before = None
        self.visual_stroke_before = None
        self.stroke_last = None
        self.stroke_stamp = None

    def _pick(self, position: tuple[int, int]) -> None:
        tile = self.tile_from_mouse(position)
        if tile is None:
            return
        if self._placing_sprites():
            point = self.world_from_mouse(position)
            hit = self._sprite_at_point(point) if point is not None else None
            if hit is None:
                self.notify("NO SPRITE UNDER THE CURSOR", 120)
                return
            self._select_sprite(hit)
            self.sprite_choice = self.level.sprites[hit].art
            self.notify(f"PICKED {self.sprite_choice.upper()}", 120)
            return
        if self.editor_tab == "visual":
            override = self.level.visual_at(*tile)
            self.visual_material = (MATERIAL_AUTO if override is None
                                    else override[0])
            self.visual_foreground = bool(
                override is not None and override[1] & VISUAL_FLAG_FOREGROUND)
            self.notify(f"PICKED MATERIAL {self.visual_material.upper()}")
            return
        word = self.level.word_at(*tile)
        if not word:
            self.erase_mode = True
            self.stamp_mode = None
            self.notify("ERASER")
            return
        (self.selected_id, self.solidity,
        self.flip_x, self.flip_y) = decode_tile(word)
        self.erase_mode = False
        self.stamp_mode = None
        self.notify(f"PICKED TILE {self.selected_id:02X}")

    def _apply_history(self, group: Iterable[AnyEdit],
                       changes: list[tuple[int, int, int]],
                       label: str) -> None:
        touched = tuple(group)
        if not touched:
            self.notify(f"NOTHING TO {label}")
            return
        for x, y, word in changes:
            self.runtime.write_tile(x, y, word)
        for edit in touched:
            # Sprites are drawn straight from the level, so only block edits
            # have to drop cached chunk artwork.
            if not isinstance(edit, SpriteEdit):
                self._touch_cell(edit.x, edit.y)
            elif self.level.sprite_at(edit.sprite_id) is None:
                if self.selected_sprite == edit.sprite_id:
                    self.selected_sprite = None
        self.notify(label)

    def undo(self) -> None:
        self._finish_stroke()
        group = self.history.peek_undo()
        self._apply_history(group, self.history.undo(self.level), "UNDO")

    def redo(self) -> None:
        self._finish_stroke()
        group = self.history.peek_redo()
        self._apply_history(group, self.history.redo(self.level), "REDO")

    @property
    def dialog_rows(self) -> int:
        return self.dialog_list_rect.height // 30

    def _refresh_dialog_files(self) -> None:
        try:
            self.dialog_files = list_level_files(
                self.levels_dir, self.extra_level_paths)
        except OSError as error:
            self.dialog_files = []
            self.dialog_error = f"CANNOT READ SAVE DIRECTORY: {error}"
        maximum = max(0, len(self.dialog_files) - self.dialog_rows)
        self.dialog_scroll = min(self.dialog_scroll, maximum)

    def _select_dialog_file(self, index: int) -> None:
        if not 0 <= index < len(self.dialog_files):
            return
        self.dialog_selected = self.dialog_files[index]
        self.dialog_name = self.dialog_selected.stem
        self.dialog_select_all = False
        self.dialog_error = ""
        if index < self.dialog_scroll:
            self.dialog_scroll = index
        elif index >= self.dialog_scroll + self.dialog_rows:
            self.dialog_scroll = index - self.dialog_rows + 1

    def open_file_dialog(self, mode: str) -> None:
        if mode not in ("save", "load"):
            raise ValueError("file dialog mode must be save or load")
        self._finish_stroke()
        self.held.clear()
        self.terrain_drawing = False
        self.drag_anchor = None
        self.marker_dragging = False
        self.marker_drag_offset = (0.0, 0.0)
        self.file_dialog = mode
        self.dialog_error = self.storage_error
        self.dialog_scroll = 0
        self.dialog_selected = None
        self.dialog_name = (self.level_path.stem
                            if self.level_path is not None else "untitled")
        self._refresh_dialog_files()

        if self.level_path is not None:
            current = self.level_path.resolve()
            for index, path in enumerate(self.dialog_files):
                if path == current:
                    self._select_dialog_file(index)
                    break
        if mode == "load" and self.dialog_selected is None and self.dialog_files:
            self._select_dialog_file(0)
        self.dialog_select_all = mode == "save"
        pygame.key.start_text_input()
        pygame.key.set_text_input_rect(self.dialog_input_rect)

    def _close_file_dialog(self) -> None:
        self.file_dialog = None
        self.dialog_error = ""
        self.dialog_select_all = False
        pygame.key.stop_text_input()

    def _confirm_file_dialog(self) -> None:
        if self.file_dialog is None:
            return
        mode = self.file_dialog
        try:
            if mode == "save":
                path, replaced = save_named_level(
                    self.level, self.levels_dir, self.dialog_name)
                self.level_path = path
            else:
                path = find_level_file(
                    self.levels_dir, self.dialog_name,
                    self.extra_level_paths)
                level = EditableLevel.load(path)
                self._finish_stroke()
                self.level = level
                self.history.clear()
                self.freehand_points.clear()
                self.spline_points.clear()
                self.last_terrain_columns = 0
                self.selected_sprite = None
                self.sprite_drag = None
                self.sprite_before = None
                if level.theme != self.theme.name:
                    self.theme = SemanticTheme(
                        level.theme, texture_dir=self.texture_dir)
                    self.renderer = LevelRenderer(
                        self.assets, self.theme, self.sprites)
                self.renderer.invalidate_all()
                self._minimap = None
                self.runtime.install(level)
                self.level_path = path
                info = self.runtime.player_info()
                self.finish_armed = not finish_marker_reached(
                    int(info["x"]), int(info["y"]),
                    level.finish_x, level.finish_y)
                self.finish_cooldown = 20
                self.completion_until = 0
                replaced = False
        except FileNotFoundError:
            self.dialog_error = "THAT SAVED LEVEL DOES NOT EXIST"
            return
        except (OSError, ValueError, IndexError,
                json.JSONDecodeError) as error:
            self.dialog_error = f"{mode.upper()} FAILED: {error}"
            return

        self.dialog_name = path.stem
        self._close_file_dialog()
        if mode == "save":
            verb = "OVERWROTE" if replaced else "SAVED"
            self.notify(f"{verb} {path.name}", 210)
        else:
            self.notify(f"LOADED {path.name}", 210)

    def save(self) -> None:
        self.open_file_dialog("save")

    def load(self) -> None:
        self.open_file_dialog("load")

    def _handle_file_dialog_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.WINDOWFOCUSLOST:
            self.held.clear()
            return
        if event.type == pygame.KEYDOWN:
            mods = getattr(event, "mod", pygame.key.get_mods())
            if event.key == pygame.K_ESCAPE:
                self._close_file_dialog()
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm_file_dialog()
            elif event.key == pygame.K_BACKSPACE:
                if self.dialog_select_all:
                    self.dialog_name = ""
                else:
                    self.dialog_name = self.dialog_name[:-1]
                self.dialog_select_all = False
                self.dialog_selected = None
                self.dialog_error = ""
            elif event.key == pygame.K_a and mods & pygame.KMOD_CTRL:
                self.dialog_select_all = True
            elif event.key in (pygame.K_UP, pygame.K_DOWN):
                if self.dialog_selected in self.dialog_files:
                    index = self.dialog_files.index(self.dialog_selected)
                else:
                    index = -1 if event.key == pygame.K_DOWN else 0
                change = 1 if event.key == pygame.K_DOWN else -1
                index = max(0, min(len(self.dialog_files) - 1,
                                   index + change))
                self._select_dialog_file(index)
            return
        if event.type == pygame.TEXTINPUT:
            entered = event.text.replace("\n", "").replace("\r", "")
            if not entered:
                return
            if self.dialog_select_all:
                self.dialog_name = ""
            self.dialog_name = (self.dialog_name + entered)[:85]
            self.dialog_select_all = False
            self.dialog_selected = None
            self.dialog_error = ""
            return
        if event.type == pygame.MOUSEWHEEL:
            maximum = max(0, len(self.dialog_files) - self.dialog_rows)
            self.dialog_scroll = max(
                0, min(maximum, self.dialog_scroll - event.y * 2))
            return
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        if self.dialog_confirm_rect.collidepoint(event.pos):
            self._confirm_file_dialog()
        elif self.dialog_cancel_rect.collidepoint(event.pos):
            self._close_file_dialog()
        elif self.dialog_input_rect.collidepoint(event.pos):
            self.dialog_select_all = True
        elif self.dialog_list_rect.collidepoint(event.pos):
            row = (event.pos[1] - self.dialog_list_rect.y) // 30
            index = self.dialog_scroll + row
            self._select_dialog_file(index)
            if (self.file_dialog == "load" and
                    getattr(event, "clicks", 1) >= 2):
                self._confirm_file_dialog()

    def new_ground(self) -> None:
        self._finish_stroke()
        self.level = EditableLevel.with_ground()
        self.level_path = None
        self.dialog_name = "untitled"
        self.history.clear()
        self.freehand_points.clear()
        self.spline_points.clear()
        self.last_terrain_columns = 0
        self.selected_sprite = None
        self.sprite_drag = None
        self.sprite_before = None
        self.renderer.invalidate_all()
        self._minimap = None
        self.runtime.install(self.level)
        self.finish_armed = True
        self.finish_cooldown = 20
        self.completion_until = 0
        self.notify("NEW GROUND PLANE")

    def _palette_click(self, position: tuple[int, int]) -> bool:
        if not self.palette_rect.collidepoint(position):
            return False
        local_x = position[0] - self.palette_rect.x
        local_y = position[1] - self.palette_rect.y
        column = local_x // self.PALETTE_CELL
        row = local_y // self.PALETTE_CELL
        index = self.palette_scroll + row * self.PALETTE_COLUMNS + column
        if index < len(self.palette_ids):
            self.selected_id = self.palette_ids[index]
            self.erase_mode = False
            self.stamp_mode = None
            self.notify(f"COLLISION TILE {self.selected_id:02X}", 90)
        return True

    def _button_click(self, position: tuple[int, int]) -> bool:
        for name in self._visible_button_names():
            rect = self.buttons[name]
            if not rect.collidepoint(position):
                continue
            if name.startswith("tab_"):
                self._set_editor_tab(name[4:])
            elif name.startswith("material_"):
                self.visual_material = name[9:]
                self.notify(f"MATERIAL {self.visual_material.upper()}", 120)
            elif name == "view_visual":
                self.toggle_visual_layer()
            elif name == "view_collision":
                self.toggle_collision_layer()
            elif name == "visual_foreground":
                self.visual_foreground = not self.visual_foreground
                self.notify("FOREGROUND PLANE" if self.visual_foreground
                            else "BACKGROUND PLANE", 120)
            elif name == "visual_outline":
                self.collision_outline = not self.collision_outline
                self.notify("COLLISION OUTLINE" if self.collision_outline
                            else "COLLISION FILL", 120)
            elif name == "visual_opacity":
                step = (COLLISION_ALPHAS.index(self.collision_alpha) + 1
                        if self.collision_alpha in COLLISION_ALPHAS else 0)
                self.collision_alpha = COLLISION_ALPHAS[
                    step % len(COLLISION_ALPHAS)]
                self.notify(
                    f"COLLISION {round(self.collision_alpha * 100)}%", 120)
            elif name == "visual_reload":
                self.reload_artwork()
            elif name.startswith("visual_mode_"):
                self._set_visual_mode(name[len("visual_mode_"):])
            elif name == "sprite_plane":
                self._set_selected_sprite_plane(not self.sprite_front)
            elif name == "sprite_delete":
                self._delete_selected_sprite()
            elif name == "sprite_smaller":
                self._scale_selected_sprite(1.0 / SPRITE_STEP)
            elif name == "sprite_bigger":
                self._scale_selected_sprite(SPRITE_STEP)
            elif name.startswith("capture_") and name[8:] in CAPTURE_EDGES:
                self.arm_capture_edge(name[8:])
            elif name == "capture_sequence":
                self.start_capture_sequence()
            elif name == "capture_view":
                self.use_view_as_capture()
            elif name == "capture_snap":
                self.capture_snap = not self.capture_snap
                self.notify("SNAPPING TO BLOCKS" if self.capture_snap
                            else "FREE PIXEL EDGES", 120)
            elif name == "capture_grid":
                self.capture_grid = not self.capture_grid
                self.notify("GRID IN EXPORT" if self.capture_grid
                            else "NO GRID IN EXPORT", 120)
            elif name == "capture_collision":
                self.capture_collision = not self.capture_collision
                self.notify("COLLISION IN EXPORT" if self.capture_collision
                            else "ART ONLY IN EXPORT", 120)
            elif name == "capture_export":
                self.export_capture()
            elif name == "reset":
                self.reset_sonic_to_start()
            elif name == "save":
                self.save()
            elif name == "load":
                self.load()
            elif name == "new":
                self.new_ground()
            elif name == "pause":
                self.paused = not self.paused
                self.notify("PAUSED" if self.paused else "LIVE")
            elif name == "grid":
                self.show_grid = not self.show_grid
            elif name == "terrain_freehand":
                self.terrain_tool = "freehand"
                self.terrain_drawing = False
                self.drag_anchor = None
                self.notify("FREEHAND TERRAIN BRUSH", 90)
            elif name == "terrain_spline":
                self.terrain_tool = "spline"
                self.terrain_drawing = False
                self.drag_anchor = None
                self.notify("SPLINE TERRAIN TOOL", 90)
            elif name == "terrain_generate":
                self.generate_terrain()
            elif name == "terrain_clear":
                self.clear_terrain_curve()
            elif name == "terrain_undo":
                self.undo()
            elif name == "marker_start":
                self.marker_tool = "start"
                self.marker_dragging = False
                self.notify("PLACE OR DRAG START MARKER", 120)
            elif name == "marker_finish":
                self.marker_tool = "finish"
                self.marker_dragging = False
                self.notify("PLACE OR DRAG FINISH MARKER", 120)
            elif name == "marker_reset":
                self.reset_sonic_to_start()
            elif name == "top":
                self.solidity, self.erase_mode, self.stamp_mode = \
                    SOLID_TOP, False, None
            elif name == "all":
                self.solidity, self.erase_mode, self.stamp_mode = \
                    SOLID_ALL, False, None
            elif name == "sides":
                self.solidity, self.erase_mode, self.stamp_mode = \
                    SOLID_SIDES, False, None
            elif name == "flip_x":
                self.flip_x, self.erase_mode = not self.flip_x, False
                self.stamp_mode = None
            elif name == "flip_y":
                self.flip_y, self.erase_mode = not self.flip_y, False
                self.stamp_mode = None
            elif name == "erase":
                self.erase_mode = not self.erase_mode
                self.stamp_mode = None
            elif name == "stamp_45_right":
                self.stamp_mode = STAMP_45_RIGHT
                self.solidity = SOLID_TOP
                self.erase_mode = False
            elif name == "stamp_45_left":
                self.stamp_mode = STAMP_45_LEFT
                self.solidity = SOLID_TOP
                self.erase_mode = False
            elif name == "stamp_gentle_right":
                self.stamp_mode = STAMP_GENTLE_RIGHT
                self.solidity = SOLID_TOP
                self.erase_mode = False
            elif name == "stamp_gentle_left":
                self.stamp_mode = STAMP_GENTLE_LEFT
                self.solidity = SOLID_TOP
                self.erase_mode = False
            return True
        return False

    def _scroll_palette(self, direction: int) -> None:
        visible = self.PALETTE_COLUMNS * self.PALETTE_ROWS
        hidden = max(0, len(self.palette_ids) - visible)
        maximum = ((hidden + self.PALETTE_COLUMNS - 1) //
                   self.PALETTE_COLUMNS) * self.PALETTE_COLUMNS
        self.palette_scroll = max(
            0, min(maximum,
                   self.palette_scroll - direction * self.PALETTE_COLUMNS * 2))

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.file_dialog is not None:
                self._handle_file_dialog_event(event)
            elif event.type == pygame.KEYDOWN:
                mods = getattr(event, "mod", pygame.key.get_mods())
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_z and mods & pygame.KMOD_CTRL:
                    self.undo()
                elif event.key == pygame.K_y and mods & pygame.KMOD_CTRL:
                    self.redo()
                elif event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                    self.save()
                elif event.key == pygame.K_o and mods & pygame.KMOD_CTRL:
                    self.load()
                elif event.key == pygame.K_r:
                    self.reset_sonic_to_start()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    self.notify("PAUSED" if self.paused else "LIVE")
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_a:
                    self.toggle_collision_layer()
                elif event.key == pygame.K_v:
                    self.toggle_visual_layer()
                elif event.key == pygame.K_t:
                    self._set_editor_tab(EDITOR_TABS[
                        (EDITOR_TABS.index(self.editor_tab) + 1) %
                        len(EDITOR_TABS)])
                elif (event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and
                        self._placing_sprites()):
                    self._delete_selected_sprite()
                elif event.key == pygame.K_e and self.editor_tab == "tiles":
                    self.erase_mode = not self.erase_mode
                    self.stamp_mode = None
                    self.notify("ERASER" if self.erase_mode else "BRUSH")
                if self.file_dialog is None:
                    self.held.add(event.key)
            elif event.type == pygame.KEYUP:
                self.held.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.held.clear()
                self._finish_stroke()
                self.terrain_drawing = False
                self.drag_anchor = None
                self.marker_dragging = False
                self.marker_drag_offset = (0.0, 0.0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (4, 5):
                    self._wheel(1 if event.button == 4 else -1)
                elif event.button == 2 and self.editor_tab in ("tiles",
                                                              "visual"):
                    self._pick(event.pos)
                elif event.button in (1, 3):
                    if event.pos[0] >= self.panel_x:
                        if event.button == 1:
                            self._panel_click(event.pos)
                    elif self.editor_tab == "terrain":
                        self._terrain_mouse_down(event.button, event.pos)
                    elif self.editor_tab == "markers":
                        self._marker_mouse_down(event.button, event.pos)
                    elif self._placing_sprites():
                        self._sprite_mouse_down(event.button, event.pos)
                    elif self._capturing():
                        self._capture_mouse_down(event.button, event.pos)
                    else:
                        self._start_stroke(event.button, event.pos)
            elif event.type == pygame.MOUSEMOTION:
                if self.editor_tab == "terrain":
                    self._terrain_mouse_motion(event.pos)
                elif self.editor_tab == "markers":
                    self._marker_mouse_motion(event.pos)
                elif self._capturing():
                    pass          # the armed edge previews under the cursor
                elif self._placing_sprites():
                    self._sprite_mouse_motion(event.pos)
                elif (self.stroke_before is not None or
                        self.visual_stroke_before is not None):
                    self._continue_stroke(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    if self.editor_tab == "terrain":
                        self._terrain_mouse_up(event.button)
                    elif self.editor_tab == "markers":
                        self._marker_mouse_up(event.button)
                    elif self._placing_sprites():
                        self._sprite_mouse_up(event.button)
                    elif self._capturing():
                        pass
                    else:
                        self._finish_stroke()
            elif event.type == pygame.MOUSEWHEEL:
                self._wheel(event.y)

    def _placing_sprites(self) -> bool:
        return self.editor_tab == "visual" and self.visual_mode == "sprites"

    def _capturing(self) -> bool:
        return self.editor_tab == "visual" and self.visual_mode == "capture"

    def _panel_click(self, position: tuple[int, int]) -> None:
        if self._button_click(position):
            return
        if self.editor_tab == "tiles":
            self._palette_click(position)
        elif self._placing_sprites():
            self._sprite_library_click(position)
        elif self._capturing():
            self._minimap_click(position)

    def _wheel(self, direction: int) -> None:
        if self.editor_tab == "tiles":
            self._scroll_palette(direction)
        elif self._placing_sprites():
            if pygame.mouse.get_pos()[0] >= self.panel_x:
                self._scroll_sprite_library(direction)
            else:
                self._scale_selected_sprite(SPRITE_STEP ** direction)

    def _mask_surface(self, collision_id: int, solidity: int,
                      flip_x: bool, flip_y: bool, size: int) -> pygame.Surface:
        flip = int(flip_x) | (int(flip_y) << 1)
        key = collision_id, solidity, flip, size
        cached = self.thumbnail_cache.get(key)
        if cached is not None:
            return cached
        pixels = np.empty((16, 16, 3), dtype=np.uint8)
        pixels[:] = (6, 12, 19)
        pixels[self.assets.masks[collision_id, flip]] = \
            self.assets.color_for(solidity)
        surface = pygame.image.frombuffer(
            pixels.tobytes(), (16, 16), "RGB").copy()
        surface = pygame.transform.scale(surface, (size, size))
        self.thumbnail_cache[key] = surface
        return surface

    def _draw_button(self, name: str, label: str, active: bool = False,
                     color: tuple[int, int, int] | None = None) -> None:
        rect = self.buttons[name]
        fill = color if color is not None else ((62, 76, 92) if active
                                                else (36, 46, 58))
        pygame.draw.rect(self.screen, fill, rect, border_radius=4)
        pygame.draw.rect(self.screen, ACCENT if active else PANEL_EDGE,
                         rect, 2, border_radius=4)
        text = self.small_font.render(label, True, TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_terrain_panel(self, x: int) -> None:
        self.screen.blit(self.small_font.render(
            "TERRAIN SURFACE", True, MUTED), (x, 130))
        self._draw_button("terrain_freehand", "FREEHAND BRUSH",
                          self.terrain_tool == "freehand")
        self._draw_button("terrain_spline", "SPLINE ANCHORS",
                          self.terrain_tool == "spline")
        self._draw_button("terrain_generate", "GENERATE")
        self._draw_button("terrain_clear", "CLEAR PATH")
        self._draw_button("terrain_undo", "UNDO")

        if self.terrain_tool == "freehand":
            count = len(self.freehand_points)
            tool_lines = (
                "FREEHAND BRUSH",
                "Drag a yellow surface across the world.",
                "Release to smooth, fit, and generate it.",
                f"Current path: {count} sampled point{'s' if count != 1 else ''}",
            )
        else:
            count = len(self.spline_points)
            tool_lines = (
                "SPLINE ANCHORS",
                "Left-click to add; drag to reposition.",
                "Right-click an anchor to remove it.",
                f"Click Generate  |  {count} anchor{'s' if count != 1 else ''}",
            )
        for row, label in enumerate(tool_lines):
            color = TEXT if row == 0 else MUTED
            self.screen.blit(self.small_font.render(label, True, color),
                             (x, 231 + row * 19))

        self.screen.blit(self.small_font.render(
            "GENERATED COLLISION", True, MUTED), (x, 321))
        legend = (
            (ACCENT, "yellow line: requested surface"),
            (COLLISION_TOP, "green: fitted native slope mask"),
            (COLLISION_ALL, "blue: solid $FF terrain backing"),
        )
        for row, (color, label) in enumerate(legend):
            top = 347 + row * 25
            pygame.draw.rect(self.screen, color, (x, top, 15, 15))
            self.screen.blit(self.small_font.render(label, True, TEXT),
                             (x + 23, top + 1))

        notes = (
            "The closest real Sonic 1 masks are selected",
            "for every 16-pixel column along the curve.",
            "The curve is limited to a continuous 45° grade.",
            "Generation clears above and fills below it.",
            "Ctrl+Z undoes the complete generated surface.",
        )
        for row, label in enumerate(notes):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 435 + row * 19))

        generated = ("No terrain generated yet" if not self.last_terrain_columns
                     else f"Last result: {self.last_terrain_columns} tile columns")
        self.screen.blit(self.small_font.render(generated, True, ACCENT),
                         (x, 545))
        self.screen.blit(self.small_font.render(
            "Tip: Pause physics while drawing precise curves.", True, MUTED),
            (x, 568))

    def _draw_visual_panel(self, x: int) -> None:
        self._draw_button("visual_mode_materials", "MATERIALS",
                          self.visual_mode == "materials")
        self._draw_button("visual_mode_sprites", "SPRITES",
                          self.visual_mode == "sprites")
        self._draw_button("visual_mode_capture", "CAPTURE",
                          self.visual_mode == "capture")
        if self.visual_mode == "sprites":
            self._draw_sprite_panel(x)
        elif self.visual_mode == "capture":
            self._draw_capture_panel(x)
        else:
            self._draw_material_panel(x)

    def _draw_capture_panel(self, x: int) -> None:
        heading = ("CAPTURE REGION  (click the world or the map)"
                   if self.capture_edge is not None
                   else "CAPTURE REGION  (pick an edge to move)")
        self.screen.blit(self.small_font.render(heading, True, MUTED),
                         (x, 160))
        self._draw_minimap()

        region = self.level.capture
        for edge in CAPTURE_EDGES:
            value = "--" if region is None else region.edge(edge)
            self._draw_button(f"capture_{edge}", f"{edge.upper()}  {value}",
                              edge == self.capture_edge)
        self._draw_button("capture_sequence", "SET ALL 4 EDGES",
                          bool(self.capture_queue))
        self._draw_button("capture_view", "USE THIS VIEW")
        self._draw_button("capture_snap",
                          "SNAP TO BLOCKS" if self.capture_snap
                          else "FREE PIXELS", self.capture_snap)
        self._draw_button("capture_grid",
                          "WITH GRID" if self.capture_grid else "NO GRID",
                          self.capture_grid)
        self._draw_button("capture_collision",
                          "WITH COLLISION" if self.capture_collision
                          else "ART ONLY", self.capture_collision)
        self._draw_button("capture_export", "EXPORT PNG", False,
                          (25, 100, 57) if region is not None else None)

        if region is None:
            lines = (
                "NO REGION YET",
                "SET ALL 4 EDGES walks left, right, top,",
                "bottom in turn.  USE THIS VIEW takes the",
                "rectangle on screen as a starting point.",
            )
        else:
            blocks = f"{region.width // 16} x {region.height // 16} blocks"
            lines = (
                f"{region.width} x {region.height} px   ({blocks})",
                f"from x={region.left} y={region.top} "
                f"to x={region.right} y={region.bottom}",
                "Exported one image pixel per world pixel, so",
                "drawing over it maps back onto these blocks.",
            )
        for row, label in enumerate(lines):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True,
                                       TEXT if row == 0 else MUTED),
                (x, 424 + row * 19))

        notes = (
            "Everything outside the frame is dimmed, so what",
            "stays bright is what the PNG will hold.  Sonic",
            "is never included.",
        )
        for row, label in enumerate(notes):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 500 + row * 19))

        latest = (f"saved {self.last_capture.name}"
                  if self.last_capture is not None else "nothing exported yet")
        for row, label in enumerate((latest, str(self.visualization_dir))):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True,
                                       ACCENT if row == 0 else MUTED),
                (x, 561 + row * 18))

    def _draw_sprite_panel(self, x: int) -> None:
        folder = self.sprite_dir
        heading = ("SPRITE LIBRARY  (wheel to scroll)" if self.sprites.names
                   else "SPRITE LIBRARY  (folder is empty)")
        self.screen.blit(self.small_font.render(heading, True, MUTED),
                         (x, 160))
        for slot, rect in enumerate(self.sprite_slots):
            index = self.sprite_scroll + slot
            pygame.draw.rect(self.screen, (27, 36, 47), rect)
            if index >= len(self.sprites.names):
                pygame.draw.rect(self.screen, (20, 27, 36), rect, 1)
                continue
            name = self.sprites.names[index]
            chosen = name == self.sprite_choice
            thumbnail = self.sprites.thumbnail(name, rect.width - 8,
                                               rect.height - 22)
            if thumbnail is not None:
                self.screen.blit(thumbnail, thumbnail.get_rect(
                    center=(rect.centerx, rect.y + (rect.height - 18) // 2)))
            label = self._ellipsize(name, self.small_font, rect.width - 6)
            text = self.small_font.render(label, True,
                                          ACCENT if chosen else TEXT)
            self.screen.blit(text, text.get_rect(
                center=(rect.centerx, rect.bottom - 9)))
            pygame.draw.rect(self.screen, ACCENT if chosen else PANEL_EDGE,
                             rect, 2 if chosen else 1)

        self.screen.blit(self.small_font.render("PLACED SPRITE", True, MUTED),
                         (x, 326))
        self._draw_button("sprite_plane",
                          "FRONT OF SONIC" if self.sprite_front
                          else "BEHIND SONIC", self.sprite_front)
        self._draw_button("sprite_delete", "DELETE")
        self._draw_button("sprite_smaller", "SMALLER  -")
        self._draw_button("sprite_bigger", "BIGGER  +")

        sprite = self.level.sprite_at(self.selected_sprite)
        if sprite is None:
            lines = (
                "NOTHING SELECTED",
                "Click empty space to drop the chosen sprite.",
                "Click a placed one to select and drag it.",
                "Middle-click picks the sprite under the cursor.",
            )
        else:
            plane = "in front of Sonic" if sprite.front else "behind Sonic"
            missing = self.sprites.source(sprite.art) is None
            lines = (
                f"{sprite.art.upper()}"
                f"{'  (image missing)' if missing else ''}",
                f"base x={sprite.x}  y={sprite.y}",
                f"size {sprite.width} x {sprite.height} px, {plane}",
                "Drag the grip out from the base to scale.",
            )
        for row, label in enumerate(lines):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True,
                                       TEXT if row == 0 else MUTED),
                (x, 410 + row * 19))

        notes = (
            "The wheel over the canvas scales too; DEL or a",
            "right-click removes.  Sprites are decoration:",
            "they never change what Sonic can stand on.",
        )
        for row, label in enumerate(notes):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 486 + row * 19))

        if folder is None:
            location = "no sprite folder"
        elif not folder.is_dir():
            location = f"{folder} (not created yet)"
        else:
            location = str(folder)
        summary = (f"{len(self.level.sprites)} placed  |  "
                   f"{len(self.sprites.names)} in library")
        if self.sprites.failed:
            summary += f"  |  {len(self.sprites.failed)} unreadable"
        for row, label in enumerate((summary, f"sprites: {location}")):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True,
                                       ACCENT if row == 0 else MUTED),
                (x, 545 + row * 18))

    def _draw_material_panel(self, x: int) -> None:
        self.screen.blit(self.small_font.render(
            "SEMANTIC MATERIALS  (paint over blocks)", True, MUTED), (x, 160))
        for key in MATERIAL_ORDER:
            name = f"material_{key}"
            active = key == self.visual_material
            if key == MATERIAL_AUTO:
                self._draw_button(name, "AUTO (CLEAR)", active)
            else:
                material = MATERIALS[key]
                rect = self.buttons[name]
                self._draw_button(name, material.label, active)
                swatch = pygame.Rect(rect.x + 6, rect.y + 7, 16, 16)
                pygame.draw.rect(self.screen, material.base, swatch)
                pygame.draw.rect(self.screen, material.mask_color, swatch, 1)

        self.screen.blit(self.small_font.render("BLOCK OPTIONS", True, MUTED),
                         (x, 314))
        self._draw_button("visual_foreground",
                          "FRONT OF SONIC" if self.visual_foreground
                          else "BEHIND SONIC", self.visual_foreground)
        self._draw_button("visual_outline",
                          "COLL OUTLINE" if self.collision_outline
                          else "COLL FILL", self.collision_outline)
        self._draw_button("visual_opacity",
                          f"COLL {round(self.collision_alpha * 100)}%")
        self._draw_button("visual_reload", "RELOAD ART")

        if self.visual_material == MATERIAL_AUTO:
            lines = (
                "AUTO",
                "Derived from the collision mask itself:",
                "grass ribbon, soil body, rock deep and",
                "along cliff faces.  Right-click also clears.",
            )
        else:
            material = MATERIALS[self.visual_material]
            clipping = ("clipped to the collision mask"
                        if material.kind == "terrain"
                        else "fills the block, ignores collision")
            lines = (
                f"{material.label}  mask #{'%02x%02x%02x' % material.mask_color}",
                material.note,
                clipping,
                f"art: {self.visual_material}.png"
                f"{'  (loaded)' if self.visual_material in self.theme.replaced else '  (placeholder)'}",
            )
        for row, label in enumerate(lines):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True, TEXT if row == 0 else MUTED),
                (x, 402 + row * 19))

        notes = (
            "Artwork and collision stay separate, exactly as",
            "in the ROM: painting here never changes what",
            "Sonic can stand on.  A toggles the overlay.",
        )
        for row, label in enumerate(notes):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 480 + row * 19))

        if self.texture_dir is None:
            folder = "none; procedural placeholders"
        elif not self.texture_dir.is_dir():
            folder = f"{self.texture_dir} (not created yet)"
        else:
            folder = f"{self.texture_dir} ({len(self.theme.replaced)} loaded)"
        summary = (f"theme {self.level.theme}  |  "
                   f"{len(self.level.visual)} painted block"
                   f"{'s' if len(self.level.visual) != 1 else ''}")
        for row, label in enumerate((summary, f"art folder: {folder}")):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(
                self.small_font.render(label, True,
                                       ACCENT if row == 0 else MUTED),
                (x, 545 + row * 18))

    def _draw_markers_panel(self, x: int) -> None:
        self.screen.blit(self.small_font.render(
            "LEVEL MARKERS", True, MUTED), (x, 130))
        self._draw_button("marker_start", "GREEN START",
                          self.marker_tool == "start",
                          (25, 100, 57) if self.marker_tool == "start" else None)
        self._draw_button("marker_finish", "RED FINISH",
                          self.marker_tool == "finish",
                          (126, 42, 48) if self.marker_tool == "finish" else None)
        self._draw_button("marker_reset", "RESET SONIC TO GREEN START")

        start_x, start_y = self.level.start_marker
        finish_x, finish_y = self.level.finish_marker
        selected_color = (START_MARKER_COLOR if self.marker_tool == "start"
                          else FINISH_MARKER_COLOR)
        selected_label = ("START" if self.marker_tool == "start"
                          else "FINISH")
        pygame.draw.rect(self.screen, selected_color, (x, 235, 15, 15))
        self.screen.blit(self.font.render(
            f"{selected_label} MARKER SELECTED", True, TEXT), (x + 24, 233))

        coordinates = (
            f"Green start base:  x={start_x:4}  y={start_y:4}",
            f"Sonic spawn center: x={self.level.spawn_x:4}  y={self.level.spawn_y:4}",
            f"Red finish base:    x={finish_x:4}  y={finish_y:4}",
        )
        for row, label in enumerate(coordinates):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 270 + row * 20))

        self.screen.blit(self.small_font.render(
            "POSITIONING", True, MUTED), (x, 345))
        instructions = (
            "Choose Green Start or Red Finish above.",
            "Click the world to place the selected flag.",
            "Drag either visible flag to reposition it.",
            "The flag base is the surface/feet position.",
            "Pause physics for precise marker placement.",
        )
        for row, label in enumerate(instructions):
            self.screen.blit(self.small_font.render(label, True, TEXT),
                             (x, 370 + row * 20))

        self.screen.blit(self.small_font.render(
            "FINISH BEHAVIOR", True, MUTED), (x, 485))
        finish_notes = (
            "Touching the red pole displays LEVEL COMPLETE,",
            "then immediately returns Sonic to the green start.",
            "Both marker positions are stored in the level JSON.",
        )
        for row, label in enumerate(finish_notes):
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 510 + row * 20))

    def _draw_panel(self) -> None:
        panel = pygame.Rect(self.panel_x, 0, PANEL_WIDTH, self.window_height)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.line(self.screen, PANEL_EDGE,
                         (self.panel_x, 0), (self.panel_x, self.window_height), 2)
        x = self.panel_x + 12
        self.screen.blit(self.title_font.render("SONIC MAKER", True, TEXT),
                         (x, 9))
        for tab, label in zip(EDITOR_TABS,
                              ("TILES", "TERRAIN", "MARKERS", "VISUAL")):
            self._draw_button(f"tab_{tab}", label, self.editor_tab == tab)
        self._draw_button("view_visual", "ART", self.show_visual,
                          (32, 84, 60) if self.show_visual else None)
        self._draw_button("view_collision", "COLL", self.show_collision,
                          (22, 72, 125) if self.show_collision else None)

        self._draw_button("reset", "RESET SONIC")
        self._draw_button("save", "SAVE")
        self._draw_button("load", "LOAD")
        self._draw_button("new", "NEW GROUND")
        self._draw_button("pause", "PAUSED" if self.paused else "LIVE",
                          self.paused)
        self._draw_button("grid", "GRID", self.show_grid)

        pygame.draw.rect(self.screen, (8, 14, 22), self.preview_rect)
        if self.stamp_mode in (STAMP_45_RIGHT, STAMP_45_LEFT):
            preview_id = 0x50
            preview_flip = self.stamp_mode == STAMP_45_LEFT
            preview_flip_y = False
            preview_solidity = SOLID_TOP
        elif self.stamp_mode in (STAMP_GENTLE_RIGHT, STAMP_GENTLE_LEFT):
            preview_id = 0x18
            preview_flip = self.stamp_mode == STAMP_GENTLE_LEFT
            preview_flip_y = False
            preview_solidity = SOLID_TOP
        else:
            preview_id = self.selected_id
            preview_flip = self.flip_x
            preview_flip_y = self.flip_y
            preview_solidity = self.solidity
        preview = self._mask_surface(preview_id, preview_solidity,
                                     preview_flip, preview_flip_y, 64)
        self.screen.blit(preview, preview.get_rect(center=self.preview_rect.center))
        pygame.draw.rect(self.screen, ACCENT if not self.erase_mode else PANEL_EDGE,
                         self.preview_rect, 2)
        if self.stamp_mode:
            gentle = self.stamp_mode in (STAMP_GENTLE_RIGHT,
                                         STAMP_GENTLE_LEFT)
            direction = "LEFT" if self.stamp_mode.endswith("left") else "RIGHT"
            labels = (
                f"{'GENTLE' if gentle else '45-DEGREE'} RAMP",
                f"climbs {direction}",
                "includes backing + ledge",
                "click ramp's low-end cell",
            )
        else:
            angle = self.assets.angles[self.selected_id]
            labels = (
                f"Raw collision ${self.selected_id:02X}",
                f"native angle ${angle:02X}",
                RAW_MASK_NOTES.get(self.selected_id, "single raw mask"),
                "L paint / R erase",
            )
        for row, label in enumerate(labels):
            self.screen.blit(self.small_font.render(label, True,
                                                    TEXT if row < 2 else MUTED),
                             (x + 84, 145 + row * 17))

        self.screen.blit(self.small_font.render("SOLIDITY", True, MUTED),
                         (x, 216))
        self._draw_button("top", "TOP (GREEN)", self.solidity == SOLID_TOP,
                          (25, 100, 57) if self.solidity == SOLID_TOP else None)
        self._draw_button("all", "ALL (BLUE)", self.solidity == SOLID_ALL,
                          (22, 72, 125) if self.solidity == SOLID_ALL else None)
        self._draw_button("sides", "SIDES (ORANGE)",
                          self.solidity == SOLID_SIDES,
                          (126, 81, 24) if self.solidity == SOLID_SIDES else None)

        self.screen.blit(self.small_font.render("TOOLS", True, MUTED), (x, 267))
        self._draw_button("flip_x", "FLIP X", self.flip_x)
        self._draw_button("flip_y", "FLIP Y", self.flip_y)
        self._draw_button("erase", "ERASER", self.erase_mode)

        self.screen.blit(self.small_font.render(
            "READY-MADE RAMPS  (click their low end)", True, MUTED), (x, 315))
        self._draw_button("stamp_45_right", "45 UP  >",
                          self.stamp_mode == STAMP_45_RIGHT)
        self._draw_button("stamp_45_left", "<  45 UP",
                          self.stamp_mode == STAMP_45_LEFT)
        self._draw_button("stamp_gentle_right", "GENTLE UP  >",
                          self.stamp_mode == STAMP_GENTLE_RIGHT)
        self._draw_button("stamp_gentle_left", "<  GENTLE UP",
                          self.stamp_mode == STAMP_GENTLE_LEFT)

        self.screen.blit(self.small_font.render(
            "RAW COLLISION MASKS  (wheel to scroll)", True, MUTED), (x, 399))
        pygame.draw.rect(self.screen, (10, 16, 24), self.palette_rect)
        visible = self.PALETTE_COLUMNS * self.PALETTE_ROWS
        for visible_index, collision_id in enumerate(
                self.palette_ids[self.palette_scroll:self.palette_scroll + visible]):
            row, column = divmod(visible_index, self.PALETTE_COLUMNS)
            rect = pygame.Rect(
                self.palette_rect.x + column * self.PALETTE_CELL,
                self.palette_rect.y + row * self.PALETTE_CELL,
                self.PALETTE_CELL, self.PALETTE_CELL)
            selected = collision_id == self.selected_id and not self.erase_mode
            pygame.draw.rect(self.screen, (27, 36, 47), rect.inflate(-2, -2))
            icon = self._mask_surface(collision_id, self.solidity,
                                      False, False, 32)
            self.screen.blit(icon, (rect.x + 5, rect.y + 5))
            label = self.small_font.render(f"{collision_id:02X}", True,
                                           ACCENT if selected else TEXT)
            self.screen.blit(label, (rect.right - 17, rect.bottom - 14))
            if selected:
                pygame.draw.rect(self.screen, ACCENT, rect.inflate(-1, -1), 2)

        if self.editor_tab != "tiles":
            pygame.draw.rect(
                self.screen, PANEL_BG,
                (self.panel_x + 2, 128, PANEL_WIDTH - 2, 470))
            if self.editor_tab == "terrain":
                self._draw_terrain_panel(x)
            elif self.editor_tab == "markers":
                self._draw_markers_panel(x)
            else:
                self._draw_visual_panel(x)

        info = self.runtime.player_info()
        state = "AIR" if info["airborne"] else "GROUND"
        current_file = (self.level_path.name
                        if self.level_path is not None else "unsaved")
        bottom_lines = (
            f"Sonic x={info['x']:4} y={info['y']:4}  {state}",
            f"speed={info['inertia']:5}  angle=${info['angle']:02X}",
            f"level: {current_file}",
        )
        for row, label in enumerate(bottom_lines):
            label = self._ellipsize(label, self.small_font, PANEL_WIDTH - 24)
            self.screen.blit(self.small_font.render(label, True, MUTED),
                             (x, 603 + row * 17))
        if self.message and self.frame < self.message_until:
            label = self.small_font.render(self.message[:43], True, ACCENT)
            self.screen.blit(label, (x, 654))

    def _draw_terrain_overlay(self) -> None:
        camera_x, camera_y = self.runtime.camera()

        def screen_point(point: tuple[float, float]) -> tuple[int, int]:
            return (int(round((point[0] - camera_x) * self.scale)),
                    int(round((point[1] - camera_y) * self.scale)))

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(
            0, 0, self.canvas_width, self.canvas_height))
        if self.terrain_tool == "freehand":
            raw_points = [screen_point(point)
                          for point in self.freehand_points]
            if len(raw_points) >= 2:
                pygame.draw.lines(self.screen, MUTED, False, raw_points,
                                  max(1, self.scale))
        else:
            ordered = sorted(self.spline_points, key=lambda point: point[0])
            control_points = [screen_point(point) for point in ordered]
            if len(control_points) >= 2:
                pygame.draw.lines(self.screen, MUTED, False,
                                  control_points, max(1, self.scale))

        try:
            x_values, y_values = self._terrain_curve()
        except ValueError:
            x_values = y_values = np.empty(0)
        if len(x_values) >= 2:
            visible = ((x_values >= camera_x - 2) &
                       (x_values <= camera_x + NATIVE_WIDTH + 2))
            curve_points = [screen_point((float(x), float(y)))
                            for x, y in zip(x_values[visible], y_values[visible])]
            if len(curve_points) >= 2:
                pygame.draw.lines(self.screen, ACCENT, False, curve_points,
                                  max(3, self.scale))

        if self.terrain_tool == "spline":
            radius = max(5, self.scale * 2)
            for index, point in enumerate(self.spline_points):
                center = screen_point(point)
                color = ACCENT if index == self.drag_anchor else TEXT
                pygame.draw.circle(self.screen, PANEL_BG, center, radius + 2)
                pygame.draw.circle(self.screen, color, center, radius, 2)
                pygame.draw.circle(self.screen, color, center, 2)
        self.screen.set_clip(previous_clip)

    def _draw_marker_overlay(self) -> None:
        camera_x, camera_y = self.runtime.camera()
        previous_clip = self.screen.get_clip()
        canvas = pygame.Rect(0, 0, self.canvas_width, self.canvas_height)
        self.screen.set_clip(canvas)

        if self.editor_tab == "markers":
            trigger = pygame.Surface(
                (self.canvas_width, self.canvas_height), pygame.SRCALPHA)
            trigger_rect = pygame.Rect(
                int((self.level.finish_x - FINISH_HALF_WIDTH - camera_x) *
                    self.scale),
                int((self.level.finish_y - FINISH_HEIGHT - camera_y) *
                    self.scale),
                FINISH_HALF_WIDTH * 2 * self.scale,
                (FINISH_HEIGHT + 12) * self.scale)
            pygame.draw.rect(trigger, (*FINISH_MARKER_COLOR, 35), trigger_rect)
            pygame.draw.rect(trigger, (*FINISH_MARKER_COLOR, 135),
                             trigger_rect, max(1, self.scale))
            self.screen.blit(trigger, (0, 0))

        markers = (
            ("start", self.level.start_marker, START_MARKER_COLOR, "START"),
            ("finish", self.level.finish_marker, FINISH_MARKER_COLOR, "FINISH"),
        )
        for name, (world_x, world_y), color, label in markers:
            base_x = int(round((world_x - camera_x) * self.scale))
            base_y = int(round((world_y - camera_y) * self.scale))
            pole_top = base_y - 48 * self.scale
            width = 16 * self.scale
            selected = (self.editor_tab == "markers" and
                        self.marker_tool == name)
            pygame.draw.line(self.screen, (226, 232, 238),
                             (base_x, base_y), (base_x, pole_top),
                             max(2, self.scale))
            flag = [(base_x, pole_top),
                    (base_x + width, pole_top + 8 * self.scale),
                    (base_x, pole_top + 16 * self.scale)]
            pygame.draw.polygon(self.screen, color, flag)
            pygame.draw.lines(self.screen, ACCENT if selected else TEXT,
                              True, flag, max(1, self.scale))
            radius = 5 * self.scale
            pygame.draw.circle(self.screen, PANEL_BG,
                               (base_x, base_y), radius + 2)
            pygame.draw.circle(self.screen, ACCENT if selected else color,
                               (base_x, base_y), radius, max(2, self.scale))
            label_surface = self.small_font.render(
                label, True, ACCENT if selected else color)
            label_x = base_x + 4 * self.scale
            label_y = pole_top - label_surface.get_height() - 2
            self.screen.blit(label_surface, (label_x, label_y))
        self.screen.set_clip(previous_clip)

    def _draw_sprite_overlay(self) -> None:
        camera_x, camera_y = self.runtime.camera()
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(
            0, 0, self.canvas_width, self.canvas_height))

        for sprite_id, sprite in self.level.sprites.items():
            left, top, width, height = sprite.bounds
            rect = pygame.Rect((left - camera_x) * self.scale,
                               (top - camera_y) * self.scale,
                               width * self.scale, height * self.scale)
            selected = sprite_id == self.selected_sprite
            plane_color = (FINISH_MARKER_COLOR if sprite.front
                           else START_MARKER_COLOR)
            pygame.draw.rect(self.screen, ACCENT if selected else plane_color,
                             rect, 2 if selected else 1)
            # The base marker is what the sprite is anchored by.
            base = ((sprite.x - camera_x) * self.scale,
                    (sprite.y - camera_y) * self.scale)
            pygame.draw.line(self.screen, plane_color,
                             (base[0] - 5, base[1]), (base[0] + 5, base[1]),
                             max(1, self.scale // 2))
            if not selected:
                continue
            grip = self._sprite_handle(sprite)
            handle = pygame.Rect(0, 0, 5 * self.scale, 5 * self.scale)
            handle.center = (int((grip[0] - camera_x) * self.scale),
                             int((grip[1] - camera_y) * self.scale))
            pygame.draw.rect(self.screen, PANEL_BG, handle)
            pygame.draw.rect(self.screen, ACCENT, handle, max(1, self.scale // 2))
        self.screen.set_clip(previous_clip)

    def _minimap_surface(self) -> pygame.Surface:
        """One pixel per block, so the whole world fits beside the controls."""
        if self._minimap is None:
            words = np.array(self.level.cells, dtype=np.int32).reshape(
                WORLD_TILES_Y, WORLD_TILES_X)
            pixels = np.empty((WORLD_TILES_Y, WORLD_TILES_X, 3),
                              dtype=np.uint8)
            pixels[:] = (14, 20, 29)
            for solidity, color in ((SOLID_TOP, COLLISION_TOP),
                                    (SOLID_ALL, COLLISION_ALL),
                                    (SOLID_SIDES, COLLISION_SIDES)):
                filled = (words != 0) & ((words & 0x6000) == solidity)
                pixels[filled] = tuple(value * 2 // 3 + 46 for value in color)
            surface = pygame.image.frombuffer(
                pixels.tobytes(), (WORLD_TILES_X, WORLD_TILES_Y), "RGB")
            self._minimap = pygame.transform.scale(
                surface, self.minimap_rect.size)
        return self._minimap

    def _draw_minimap(self) -> None:
        rect = self.minimap_rect
        self.screen.blit(self._minimap_surface(), rect)
        scale_x = rect.width / WORLD_WIDTH
        scale_y = rect.height / WORLD_HEIGHT

        def to_map(world_x: float, world_y: float) -> tuple[int, int]:
            return (int(rect.x + world_x * scale_x),
                    int(rect.y + world_y * scale_y))

        for sprite in self.level.sprites.values():
            pygame.draw.circle(self.screen, (188, 92, 176),
                               to_map(sprite.x, sprite.y), 1)
        camera_x, camera_y = self.runtime.camera()
        view = pygame.Rect(to_map(camera_x, camera_y),
                           (max(2, int(NATIVE_WIDTH * scale_x)),
                            max(2, int(NATIVE_HEIGHT * scale_y))))
        pygame.draw.rect(self.screen, (150, 165, 180), view, 1)
        info = self.runtime.player_info()
        pygame.draw.circle(self.screen, START_MARKER_COLOR,
                           to_map(int(info["x"]), int(info["y"])), 2)
        if self.level.capture is not None:
            region = self.level.capture
            frame = pygame.Rect(
                to_map(region.left, region.top),
                (max(2, int(region.width * scale_x)),
                 max(2, int(region.height * scale_y))))
            pygame.draw.rect(self.screen, ACCENT, frame, 2)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1)

    def _draw_capture_overlay(self) -> None:
        camera_x, camera_y = self.runtime.camera()
        canvas = pygame.Rect(0, 0, self.canvas_width, self.canvas_height)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(canvas)

        region = self.level.capture
        if region is not None:
            frame = pygame.Rect(
                int((region.left - camera_x) * self.scale),
                int((region.top - camera_y) * self.scale),
                region.width * self.scale, region.height * self.scale)
            # Everything outside the frame is dimmed, so what stays bright is
            # exactly what the exported PNG will contain.
            shade = pygame.Surface(canvas.size, pygame.SRCALPHA)
            shade.fill((5, 9, 16, 165))
            inside = frame.clip(canvas)
            if inside.width and inside.height:
                shade.fill((0, 0, 0, 0), inside)
            self.screen.blit(shade, (0, 0))
            pygame.draw.rect(self.screen, ACCENT, frame, max(2, self.scale))
            self._draw_capture_edge_labels(region, frame)

        if self.capture_edge is not None:
            self._draw_capture_edge_preview(camera_x, camera_y)
        self.screen.set_clip(previous_clip)

    def _draw_capture_edge_labels(self, region: CaptureRegion,
                                  frame: pygame.Rect) -> None:
        armed = self.capture_edge
        edges = (
            ("left", (frame.left, frame.centery), region.left, True),
            ("right", (frame.right, frame.centery), region.right, True),
            ("top", (frame.centerx, frame.top), region.top, False),
            ("bottom", (frame.centerx, frame.bottom), region.bottom, False),
        )
        for name, (point_x, point_y), value, vertical in edges:
            color = FINISH_MARKER_COLOR if name == armed else ACCENT
            if name == armed:
                if vertical:
                    pygame.draw.line(self.screen, color, (point_x, 0),
                                     (point_x, self.canvas_height),
                                     max(2, self.scale))
                else:
                    pygame.draw.line(self.screen, color, (0, point_y),
                                     (self.canvas_width, point_y),
                                     max(2, self.scale))
            label = self.small_font.render(
                f"{name.upper()} {value}", True, color)
            box = label.get_rect()
            box.center = (min(max(point_x, box.width // 2 + 2),
                              self.canvas_width - box.width // 2 - 2),
                          min(max(point_y, box.height // 2 + 2),
                              self.canvas_height - box.height // 2 - 2))
            backing = pygame.Surface(box.size, pygame.SRCALPHA)
            backing.fill((8, 12, 20, 200))
            self.screen.blit(backing, box)
            self.screen.blit(label, box)

    def _draw_capture_edge_preview(self, camera_x: int,
                                   camera_y: int) -> None:
        point = self.world_from_mouse(pygame.mouse.get_pos())
        if point is None:
            return
        vertical = self.capture_edge in ("left", "right")
        value = point[0] if vertical else point[1]
        if self.capture_snap:
            value = round(value / CAPTURE_SNAP) * CAPTURE_SNAP
        if vertical:
            screen_x = int((value - camera_x) * self.scale)
            pygame.draw.line(self.screen, START_MARKER_COLOR, (screen_x, 0),
                             (screen_x, self.canvas_height), max(1, self.scale))
        else:
            screen_y = int((value - camera_y) * self.scale)
            pygame.draw.line(self.screen, START_MARKER_COLOR, (0, screen_y),
                             (self.canvas_width, screen_y), max(1, self.scale))
        label = self.small_font.render(
            f"SET {self.capture_edge.upper()} = {int(value)}", True,
            START_MARKER_COLOR)
        self.screen.blit(label, (8, self.canvas_height - 20))

    def _draw_completion_banner(self) -> None:
        if self.frame >= self.completion_until:
            return
        banner_height = 84
        top = self.canvas_height // 2 - banner_height // 2
        banner = pygame.Surface((self.canvas_width, banner_height),
                                pygame.SRCALPHA)
        banner.fill((7, 14, 22, 218))
        pygame.draw.line(banner, FINISH_MARKER_COLOR,
                         (0, 1), (self.canvas_width, 1), 3)
        pygame.draw.line(banner, START_MARKER_COLOR,
                         (0, banner_height - 2),
                         (self.canvas_width, banner_height - 2), 3)
        title = self.banner_font.render("LEVEL COMPLETE!", True, ACCENT)
        subtitle = self.small_font.render(
            "Sonic returned to the green start marker", True, TEXT)
        banner.blit(title, title.get_rect(
            center=(self.canvas_width // 2, 31)))
        banner.blit(subtitle, subtitle.get_rect(
            center=(self.canvas_width // 2, 66)))
        self.screen.blit(banner, (0, top))

    @staticmethod
    def _ellipsize(text: str, font: pygame.font.Font,
                   width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        shortened = text
        while shortened and font.size("..." + shortened)[0] > width:
            shortened = shortened[1:]
        return "..." + shortened

    def _draw_file_dialog(self) -> None:
        assert self.file_dialog in ("save", "load")
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 178))
        self.screen.blit(shade, (0, 0))

        rect = self.dialog_rect
        pygame.draw.rect(self.screen, (20, 28, 38), rect, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, rect, 2, border_radius=8)
        left = rect.x + 24
        title = "SAVE LEVEL" if self.file_dialog == "save" else "LOAD LEVEL"
        self.screen.blit(self.title_font.render(title, True, TEXT),
                         (left, rect.y + 18))
        directory = self._ellipsize(
            f"Folder: {self.levels_dir}", self.small_font, rect.width - 48)
        self.screen.blit(self.small_font.render(directory, True, MUTED),
                         (left, rect.y + 55))

        self.screen.blit(self.small_font.render(
            "LEVEL NAME  (.json is added automatically)", True, MUTED),
            (left, rect.y + 84))
        pygame.draw.rect(self.screen, (7, 13, 21), self.dialog_input_rect,
                         border_radius=4)
        pygame.draw.rect(self.screen, ACCENT, self.dialog_input_rect, 2,
                         border_radius=4)
        visible_name = self.dialog_name
        available = self.dialog_input_rect.width - 20
        while (visible_name and
               self.font.size(visible_name)[0] > available):
            visible_name = visible_name[1:]
        name_surface = self.font.render(visible_name, True, TEXT)
        name_position = (self.dialog_input_rect.x + 10,
                         self.dialog_input_rect.centery - name_surface.get_height() // 2)
        if self.dialog_select_all and visible_name:
            selected_rect = name_surface.get_rect(topleft=name_position).inflate(4, 4)
            pygame.draw.rect(self.screen, (38, 83, 126), selected_rect)
        self.screen.blit(name_surface, name_position)
        if (not self.dialog_select_all and
                (pygame.time.get_ticks() // 500) % 2 == 0):
            cursor_x = name_position[0] + name_surface.get_width() + 1
            pygame.draw.line(
                self.screen, TEXT,
                (cursor_x, self.dialog_input_rect.y + 8),
                (cursor_x, self.dialog_input_rect.bottom - 8), 2)

        count = len(self.dialog_files)
        self.screen.blit(self.small_font.render(
            f"SAVED LEVELS  ({count})", True, MUTED),
            (left, rect.y + 160))
        pygame.draw.rect(self.screen, (8, 14, 22), self.dialog_list_rect)
        pygame.draw.rect(self.screen, PANEL_EDGE, self.dialog_list_rect, 1)
        mouse = pygame.mouse.get_pos()
        if not self.dialog_files:
            self.screen.blit(self.font.render(
                "No saved levels yet.", True, MUTED),
                (self.dialog_list_rect.x + 14,
                 self.dialog_list_rect.y + 16))
        visible_files = self.dialog_files[
            self.dialog_scroll:self.dialog_scroll + self.dialog_rows]
        levels_root = self.levels_dir.resolve()
        current = self.level_path.resolve() if self.level_path is not None else None
        for row, path in enumerate(visible_files):
            row_rect = pygame.Rect(
                self.dialog_list_rect.x + 1,
                self.dialog_list_rect.y + row * 30 + 1,
                self.dialog_list_rect.width - 2, 29)
            selected = path == self.dialog_selected
            hovered = row_rect.collidepoint(mouse)
            if selected or hovered:
                color = (43, 69, 92) if selected else (28, 39, 51)
                pygame.draw.rect(self.screen, color, row_rect)
            suffix = ""
            if path.parent.resolve() != levels_root:
                suffix += "  [legacy]"
            if path == current:
                suffix += "  [current]"
            label = self._ellipsize(
                path.name + suffix, self.font, row_rect.width - 24)
            self.screen.blit(self.font.render(
                label, True, ACCENT if selected else TEXT),
                (row_rect.x + 10, row_rect.y + 5))

        if len(self.dialog_files) > self.dialog_rows:
            track = pygame.Rect(self.dialog_list_rect.right - 7,
                                self.dialog_list_rect.y + 3, 4,
                                self.dialog_list_rect.height - 6)
            pygame.draw.rect(self.screen, (29, 40, 52), track)
            thumb_height = max(
                20, track.height * self.dialog_rows // len(self.dialog_files))
            maximum = len(self.dialog_files) - self.dialog_rows
            thumb_y = (track.y +
                       (track.height - thumb_height) * self.dialog_scroll // maximum)
            pygame.draw.rect(self.screen, MUTED,
                             (track.x, thumb_y, track.width, thumb_height))

        if self.dialog_error:
            status, status_color = self.dialog_error, (255, 112, 112)
        elif self.file_dialog == "save":
            try:
                normalize_level_filename(self.dialog_name)
            except ValueError as error:
                status, status_color = str(error), MUTED
            else:
                try:
                    find_level_file(self.levels_dir, self.dialog_name)
                except FileNotFoundError:
                    exists = False
                except OSError as error:
                    exists = None
                    status = f"Cannot read save directory: {error}"
                    status_color = (255, 112, 112)
                else:
                    exists = True
                if exists is True:
                    status = "This filename exists; Save will overwrite it."
                    status_color = COLLISION_SIDES
                elif exists is False:
                    status = "A new saved level file will be created."
                    status_color = COLLISION_TOP
        else:
            status = "Select a filename, then Load (or double-click it)."
            status_color = MUTED
        status = self._ellipsize(status, self.small_font, rect.width - 48)
        self.screen.blit(self.small_font.render(status, True, status_color),
                         (left, rect.y + 439))

        def draw_dialog_button(button: pygame.Rect, label: str,
                               primary: bool = False) -> None:
            hovered = button.collidepoint(mouse)
            if primary:
                color = (131, 108, 25) if hovered else (94, 80, 25)
            else:
                color = (53, 66, 80) if hovered else (36, 46, 58)
            pygame.draw.rect(self.screen, color, button, border_radius=4)
            pygame.draw.rect(self.screen, ACCENT if primary else PANEL_EDGE,
                             button, 2, border_radius=4)
            label_surface = self.font.render(label, True, TEXT)
            self.screen.blit(label_surface,
                             label_surface.get_rect(center=button.center))

        draw_dialog_button(self.dialog_cancel_rect, "CANCEL")
        draw_dialog_button(self.dialog_confirm_rect, title.split()[0], True)
        help_text = "Enter confirms  |  Esc cancels  |  Up/Down selects"
        self.screen.blit(self.small_font.render(help_text, True, MUTED),
                         (left, rect.y + 520))

    def draw(self) -> None:
        ram = self.runtime.env.get_ram()
        image = self.renderer.render(self.level, ram, self.view_options())
        source = pygame.image.frombuffer(
            image.tobytes(), (NATIVE_WIDTH, NATIVE_HEIGHT), "RGB").convert()
        pygame.transform.scale(source,
                               (self.canvas_width, self.canvas_height),
                               self.screen.subsurface(
                                   (0, 0, self.canvas_width, self.canvas_height)))
        if self.canvas_height < self.window_height:
            pygame.draw.rect(self.screen, COLLISION_BACKGROUND,
                             (0, self.canvas_height, self.canvas_width,
                              self.window_height - self.canvas_height))

        hover = self.tile_from_mouse(pygame.mouse.get_pos())
        if self.editor_tab == "terrain":
            self._draw_terrain_overlay()
        elif self._placing_sprites():
            self._draw_sprite_overlay()
        elif self._capturing():
            self._draw_capture_overlay()
        elif self.editor_tab == "visual" and hover is not None:
            camera_x, camera_y = self.runtime.camera()
            color = (ACCENT if self.visual_material == MATERIAL_AUTO
                     else MATERIALS[self.visual_material].mask_color)
            pygame.draw.rect(
                self.screen, color,
                ((hover[0] * 16 - camera_x) * self.scale,
                 (hover[1] * 16 - camera_y) * self.scale,
                 16 * self.scale, 16 * self.scale),
                3 if self.visual_foreground else 2)
        elif self.editor_tab == "tiles" and hover is not None:
            camera_x, camera_y = self.runtime.camera()
            ghost = (safe_stamp_pattern(self.stamp_mode)
                     if self.stamp_mode else ((0, 0, self.brush_word),))
            for dx, dy, word in ghost:
                tile_x, tile_y = hover[0] + dx, hover[1] + dy
                if not (0 <= tile_x < WORLD_TILES_X and
                        0 <= tile_y < WORLD_TILES_Y):
                    continue
                hx = (tile_x * 16 - camera_x) * self.scale
                hy = (tile_y * 16 - camera_y) * self.scale
                solidity = word & 0x6000
                color = (self.assets.color_for(solidity)
                         if solidity else ACCENT)
                pygame.draw.rect(
                    self.screen, color,
                    (hx, hy, 16 * self.scale, 16 * self.scale), 2)
        self._draw_marker_overlay()
        self._draw_completion_banner()
        self._draw_panel()
        if self.file_dialog is not None:
            self._draw_file_dialog()
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self.handle_events()
            if not self.paused and self.file_dialog is None:
                self.runtime.step(self.held)
                if self.runtime.respawned:
                    self.finish_armed = False
                    self.finish_cooldown = 20
                    self.notify("SONIC RESET")
                self._update_level_completion()
            self.draw()
            self.frame += 1
            if self.max_frames and self.frame >= self.max_frames:
                self.running = False
            if not self.headless:
                self.clock.tick(FPS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level-file", type=Path,
                        help="specific JSON level to open at startup")
    parser.add_argument("--levels-dir", type=Path,
                        default=DEFAULT_LEVEL_DIRECTORY,
                        help="directory for named level saves")
    parser.add_argument("--textures", type=Path,
                        default=DEFAULT_TEXTURE_DIRECTORY,
                        help="folder of <material>.png artwork replacing the "
                             "procedural placeholders")
    parser.add_argument("--sprites", type=Path,
                        default=DEFAULT_SPRITE_DIRECTORY,
                        help="folder of decoration images for the Visual tab")
    parser.add_argument("--visualizations", type=Path,
                        default=DEFAULT_VISUALIZATION_DIRECTORY,
                        help="folder the Capture tool writes template PNGs to")
    parser.add_argument("--export-textures", type=Path, metavar="DIR",
                        help="write each material's semantic mask, "
                             "placeholder art, and metadata, then exit")
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), default=3,
                        help="game canvas scale (default: 3)")
    parser.add_argument("--headless", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, default=0,
                        help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.export_textures is not None:
        pygame.init()
        try:
            written = export_material_assets(
                SemanticTheme(), args.export_textures)
        finally:
            pygame.quit()
        print(f"Wrote {len(written)} files to "
              f"{args.export_textures.expanduser().resolve()}")
        print("Texture the .mask.png files, save the results as "
              "<material>.png beside them, then pass --textures that folder.")
        return

    levels_dir = args.levels_dir.expanduser().resolve()
    if args.level_file is not None:
        level_path = args.level_file.expanduser().resolve()
    else:
        try:
            named_levels = list_level_files(levels_dir)
        except OSError:
            named_levels = []
        if named_levels:
            level_path = max(
                named_levels, key=lambda path: path.stat().st_mtime_ns)
        elif DEFAULT_LEVEL_FILE.exists():
            level_path = DEFAULT_LEVEL_FILE.resolve()
        else:
            level_path = None

    if level_path is not None and level_path.exists():
        level = EditableLevel.load(level_path)
        print(f"Loaded {level_path}")
    else:
        level = EditableLevel.with_ground()
        print("New level")
    print(f"Named level folder: {levels_dir}")
    print("Booting the Sonic 1 physics engine...")
    runtime = SonicRuntime()
    frames = 0
    try:
        assets = CollisionAssets(runtime.rom)
        app = MakerApp(runtime, assets, level, level_path,
                       scale=args.scale, headless=args.headless,
                       max_frames=args.max_frames, levels_dir=levels_dir,
                       texture_dir=args.textures, sprite_dir=args.sprites,
                       visualization_dir=args.visualizations)
        app.run()
        frames = app.frame
    finally:
        runtime.close()
        pygame.quit()
    if args.headless:
        print(f"Headless smoke complete: {frames} frames")


if __name__ == "__main__":
    main()
