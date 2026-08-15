"""Core and live-engine checks for the interactive Sonic collision editor."""
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

import sonic_maker as maker


@pytest.fixture(scope="session", autouse=True)
def _pygame_ready():
    maker.pygame.init()
    yield
    maker.pygame.quit()


@pytest.fixture(scope="module")
def assets():
    """Collision masks straight from the ROM, without booting the emulator."""
    import stable_retro as retro
    path = pathlib.Path(retro.data.get_original_romfile_path(maker.GAME))
    return maker.CollisionAssets(path.read_bytes())


@pytest.fixture(scope="module")
def theme():
    return maker.SemanticTheme()


def fake_ram(camera_x: int = 0, camera_y: int = 0) -> np.ndarray:
    """A blank WRAM image with only the camera set; no sprites are drawn."""
    ram = np.zeros(0x10000, dtype=np.uint8)
    for address, value in ((maker.RAM_CAMERA_X, camera_x),
                           (maker.RAM_CAMERA_Y, camera_y)):
        ram[address] = value & 0xFF
        ram[address + 1] = (value >> 8) & 0xFF
    return ram


def test_tile_word_round_trip():
    word = maker.encode_tile(0x50, maker.SOLID_SIDES,
                             flip_x=True, flip_y=True)
    assert word == 0x5850
    assert maker.decode_tile(word) == (
        0x50, maker.SOLID_SIDES, True, True
    )
    assert maker.encode_tile(0, maker.SOLID_ALL) == 0
    assert maker.decode_tile(0) == (0, maker.SOLID_NONE, False, False)
    with pytest.raises(ValueError):
        maker.encode_tile(256)
    with pytest.raises(ValueError):
        maker.validate_tile_word(0x6000)


def test_chunk_word_offsets_cover_private_chunk_buffer():
    assert maker.chunk_word_offset(0, 0) == 0x0000
    assert maker.chunk_word_offset(15, 15) == 0x01FE
    assert maker.chunk_word_offset(16, 0) == 0x0200
    assert maker.chunk_word_offset(255, 63) == 0x7FFE
    with pytest.raises(IndexError):
        maker.chunk_word_offset(256, 0)


def test_default_level_has_flat_surface_and_solid_filler():
    level = maker.EditableLevel.with_ground()
    surface = maker.decode_tile(level.word_at(10, maker.GROUND_TILE_Y))
    filler = maker.decode_tile(level.word_at(10, maker.GROUND_TILE_Y + 1))
    assert surface == (0xFF, maker.SOLID_ALL, False, False)
    assert filler == (0xFF, maker.SOLID_ALL, False, False)
    assert level.word_at(10, maker.GROUND_TILE_Y - 1) == 0
    assert (level.spawn_x, level.spawn_y) == (maker.SPAWN_X, maker.SPAWN_Y)
    assert level.start_marker == (maker.SPAWN_X, maker.GROUND_Y)
    assert level.finish_marker == (maker.FINISH_X, maker.FINISH_Y)


def test_marker_positions_round_trip_and_version_one_levels_migrate():
    level = maker.EditableLevel.with_ground()
    level.set_start_marker(321.4, 700.6)
    level.set_finish_marker(987.2, 654.8)
    payload = level.to_dict()
    loaded = maker.EditableLevel.from_dict(payload)
    assert loaded.start_marker == (321, 701)
    assert (loaded.spawn_x, loaded.spawn_y) == (
        321, 701 - maker.MARKER_FOOT_OFFSET
    )
    assert loaded.finish_marker == (987, 655)

    payload["version"] = 1
    del payload["finish"]
    migrated = maker.EditableLevel.from_dict(payload)
    assert migrated.start_marker == (321, 701)
    assert migrated.finish_marker == (maker.FINISH_X, maker.FINISH_Y)


def test_finish_marker_trigger_uses_sonics_feet_and_flag_height():
    finish_x, finish_y = 500, 700
    assert maker.finish_marker_reached(500, 680, finish_x, finish_y)
    assert maker.finish_marker_reached(
        500, 680 - maker.FINISH_HEIGHT, finish_x, finish_y)
    assert not maker.finish_marker_reached(
        500 + maker.FINISH_HALF_WIDTH + 1, 680, finish_x, finish_y)
    assert not maker.finish_marker_reached(
        500, 680 - maker.FINISH_HEIGHT - 1, finish_x, finish_y)


def test_freehand_surface_is_smoothed_contiguous_and_safe_for_floor_masks():
    x_values, y_values = maker.freehand_surface([
        (96, 896), (220, 830), (350, 842), (480, 896),
    ])
    assert (x_values[0], x_values[-1]) == (96, 480)
    assert np.all(np.diff(x_values) == 1)
    assert np.max(np.abs(np.diff(y_values))) <= 1.000001
    assert np.min(y_values) < 850


def test_spline_surface_passes_through_endpoints_and_limits_steepness():
    x_values, y_values = maker.spline_surface([
        (96, 896), (288, 816), (512, 896),
    ])
    assert (x_values[0], x_values[-1]) == (96, 512)
    assert y_values[0] == pytest.approx(896)
    assert y_values[-1] == pytest.approx(896)
    assert np.min(y_values) <= 820
    assert np.max(np.abs(np.diff(y_values))) <= 1.000001


def test_generated_columns_have_top_only_surface_full_backing_and_undo():
    level = maker.EditableLevel.with_ground()
    original = level.cells.copy()
    columns = [
        maker.TerrainColumn(20, 50, 0x50),
        maker.TerrainColumn(21, 49, 0x50, flip_x=True),
    ]
    edits = maker.apply_terrain_columns(level, columns)
    assert maker.decode_tile(level.word_at(20, 50)) == (
        0x50, maker.SOLID_TOP, False, False
    )
    assert maker.decode_tile(level.word_at(21, 49)) == (
        0x50, maker.SOLID_TOP, True, False
    )
    assert maker.decode_tile(level.word_at(20, 51)) == (
        0xFF, maker.SOLID_ALL, False, False
    )
    assert level.word_at(20, 49) == 0

    history = maker.EditHistory()
    history.record(edits)
    history.undo(level)
    assert level.cells == original


def test_sparse_json_round_trip(tmp_path):
    level = maker.EditableLevel.with_ground()
    custom = maker.encode_tile(0x50, maker.SOLID_TOP, flip_x=True)
    level.set_word(22, 40, custom)
    path = tmp_path / "level.json"
    level.save(path)
    loaded = maker.EditableLevel.load(path)
    assert loaded.cells == level.cells
    assert loaded.to_dict() == level.to_dict()
    payload = json.loads(path.read_text())
    assert payload["format"] == maker.FORMAT_NAME
    assert [22, 40, custom] in payload["tiles"]


def test_named_level_files_are_safe_listed_and_overwritten(tmp_path):
    levels_dir = tmp_path / "sonic_maker_levels"
    first = maker.EditableLevel.with_ground()
    path, replaced = maker.save_named_level(
        first, levels_dir, "Green Hill Test")
    assert path == levels_dir / "Green Hill Test.json"
    assert replaced is False

    second = maker.EditableLevel.with_ground()
    custom = maker.encode_tile(0x50, maker.SOLID_TOP)
    second.set_word(22, 40, custom)
    same_path, replaced = maker.save_named_level(
        second, levels_dir, "green hill test.JSON")
    assert same_path == path
    assert replaced is True
    assert maker.EditableLevel.load(path).word_at(22, 40) == custom

    legacy = tmp_path / "old_single_save.json"
    first.save(legacy)
    files = maker.list_level_files(levels_dir, [legacy])
    assert [file.name for file in files] == [
        "Green Hill Test.json", "old_single_save.json"
    ]
    assert maker.find_level_file(
        levels_dir, "old_single_save", [legacy]) == legacy


@pytest.mark.parametrize("name", (
    "", ".hidden", "../escape", "folder/name", "bad*name", "name.",
))
def test_invalid_named_level_filenames_are_rejected(name):
    with pytest.raises(ValueError):
        maker.normalize_level_filename(name)


def test_history_groups_a_stroke_for_undo_and_redo():
    level = maker.EditableLevel()
    history = maker.EditHistory()
    word = maker.encode_tile(0x08, maker.SOLID_TOP)
    history.record([
        maker.Edit(1, 2, level.set_word(1, 2, word), word),
        maker.Edit(2, 2, level.set_word(2, 2, word), word),
    ])
    assert {(x, y, value) for x, y, value in history.undo(level)} == {
        (1, 2, 0), (2, 2, 0)
    }
    assert level.word_at(1, 2) == 0
    assert {(x, y, value) for x, y, value in history.redo(level)} == {
        (1, 2, word), (2, 2, word)
    }
    assert level.word_at(2, 2) == word


def test_line_cells_fills_fast_drag_gaps():
    assert list(maker.line_cells((1, 1), (5, 1))) == [
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    diagonal = list(maker.line_cells((0, 0), (3, 5)))
    assert diagonal[0] == (0, 0)
    assert diagonal[-1] == (3, 5)
    assert len(diagonal) == 6


def test_safe_ramp_stamps_include_top_only_surface_and_full_backing():
    right = maker.safe_stamp_pattern(maker.STAMP_45_RIGHT)
    right_cells = {(dx, dy): maker.decode_tile(word)
                   for dx, dy, word in right}
    assert right_cells[(0, 0)] == (
        0x50, maker.SOLID_TOP, False, False
    )
    assert right_cells[(0, 1)] == (
        0xFF, maker.SOLID_ALL, False, False
    )
    assert right_cells[(1, 0)][0] == 0x0F

    left = maker.safe_stamp_pattern(maker.STAMP_45_LEFT)
    left_cells = {(dx, dy): maker.decode_tile(word)
                  for dx, dy, word in left}
    assert left_cells[(0, 0)] == (
        0x50, maker.SOLID_TOP, True, False
    )
    assert (-2, 0) in left_cells

    gentle = maker.safe_stamp_pattern(maker.STAMP_GENTLE_RIGHT)
    gentle_ids = [maker.decode_tile(word)[0]
                  for dx, dy, word in gentle if dy == 0]
    assert gentle_ids == [0x18, 0x19, 0x1A, 0x1B, 0x0F, 0x0F]


def test_material_overrides_never_reach_the_collision_words():
    level = maker.EditableLevel.with_ground()
    words = level.cells.copy()
    assert level.set_visual(4, 56, "platform",
                            maker.VISUAL_FLAG_FOREGROUND) is None
    assert level.visual_at(4, 56) == ("platform", maker.VISUAL_FLAG_FOREGROUND)
    assert level.cells == words

    assert level.set_visual(4, 56, maker.MATERIAL_AUTO) == (
        "platform", maker.VISUAL_FLAG_FOREGROUND
    )
    assert level.visual_at(4, 56) is None
    with pytest.raises(ValueError):
        level.set_visual(4, 56, "lava")
    with pytest.raises(ValueError):
        level.set_visual(4, 56, "grass", 0x80)
    with pytest.raises(IndexError):
        level.set_visual(maker.WORLD_TILES_X, 56, "grass")


def test_visual_layer_round_trips_and_older_saves_get_the_auto_style(tmp_path):
    level = maker.EditableLevel.with_ground()
    level.set_visual(3, 56, "sand")
    level.set_visual(4, 56, "decor", maker.VISUAL_FLAG_FOREGROUND)
    path = tmp_path / "level.json"
    level.save(path)

    payload = json.loads(path.read_text())
    assert payload["version"] == 3
    assert payload["visual"]["theme"] == maker.DEFAULT_THEME
    assert payload["visual"]["cells"] == [
        [3, 56, "sand", 0], [4, 56, "decor", maker.VISUAL_FLAG_FOREGROUND]
    ]
    loaded = maker.EditableLevel.load(path)
    assert loaded.visual == level.visual
    assert loaded.theme == level.theme

    # A version-2 save has no artwork block at all, so it falls back to the
    # automatic style while keeping every collision word.
    payload["version"] = 2
    legacy = maker.EditableLevel.from_dict(payload)
    assert legacy.visual == {}
    assert legacy.theme == maker.DEFAULT_THEME
    assert legacy.cells == level.cells

    payload["version"] = 3
    payload["visual"]["cells"] = [[3, 56, "sand"]]
    with pytest.raises(ValueError):
        maker.EditableLevel.from_dict(payload)


def test_history_undoes_collision_and_material_edits_as_one_stroke():
    level = maker.EditableLevel.with_ground()
    history = maker.EditHistory()
    word = maker.encode_tile(0x08, maker.SOLID_TOP)
    history.record([
        maker.Edit(1, 2, level.set_word(1, 2, word), word),
        maker.VisualEdit(1, 2, level.set_visual(1, 2, "rock"), ("rock", 0)),
    ])
    assert len(history.peek_undo()) == 2

    # Only the collision half is handed back, because only that half has to be
    # mirrored into live WRAM.
    assert history.undo(level) == [(1, 2, 0)]
    assert level.word_at(1, 2) == 0
    assert level.visual_at(1, 2) is None
    assert history.redo(level) == [(1, 2, word)]
    assert level.visual_at(1, 2) == ("rock", 0)


def test_chunk_invalidation_covers_the_render_margin():
    assert maker.chunks_touching(8, 40) == ((0, 2),)
    # A block on a chunk seam also feeds its neighbour's margin.
    assert set(maker.chunks_touching(16, 40)) == {(0, 2), (1, 2)}
    # Depth flows downward, so the rows above a chunk feed it too.
    assert (2, 3) in maker.chunks_touching(32, 47)
    assert all(0 <= x < maker.CHUNKS_X and 0 <= y < maker.CHUNKS_Y
               for x, y in maker.chunks_touching(0, 0))


def test_auto_material_style_follows_depth_below_the_surface():
    solid = np.zeros((64, 8), dtype=bool)
    solid[10:, :] = True
    walls = np.zeros_like(solid)
    depth = maker.VisualTerrain._depth(solid)
    index = maker.VisualTerrain._auto_materials(solid, walls, depth)

    assert index[9, 4] == 0
    assert index[10, 4] == maker.MATERIAL_INDEX["grass"]
    assert index[10 + maker.GRASS_DEPTH - 1, 4] == maker.MATERIAL_INDEX["grass"]
    assert index[10 + maker.GRASS_DEPTH, 4] == maker.MATERIAL_INDEX["soil"]
    assert index[10 + maker.ROCK_DEPTH - 1, 4] == maker.MATERIAL_INDEX["rock"]

    # A sides-solid block is wall material the whole way through.
    walls[10:, 0] = True
    index = maker.VisualTerrain._auto_materials(solid, walls, depth)
    assert index[10, 0] == maker.MATERIAL_INDEX["rock"]
    assert index[10, 4] == maker.MATERIAL_INDEX["grass"]


def test_terrain_artwork_is_clipped_to_the_collision_mask(assets, theme):
    level = maker.EditableLevel.with_ground()
    terrain = maker.VisualTerrain(assets, theme)
    back, front = terrain.layers(level, 0, 3)
    surface = maker.GROUND_Y - 3 * maker.CHUNK_PIXELS

    assert front is None
    assert back[:surface, :, 3].max() == 0
    assert back[surface:, :, 3].min() == 255

    # Erasing the collision erases the artwork above it, and nothing else.
    for tile_y in range(maker.GROUND_TILE_Y, maker.WORLD_TILES_Y):
        level.set_word(4, tile_y, 0)
    terrain.invalidate(4, maker.GROUND_TILE_Y)
    back, _ = terrain.layers(level, 0, 3)
    assert back[surface:, 64:80, 3].max() == 0
    assert back[surface:, 96:112, 3].min() == 255


def test_overlay_materials_paint_where_there_is_no_collision(assets, theme):
    level = maker.EditableLevel.with_ground()
    level.set_visual(2, 40, "water")
    level.set_visual(3, 40, "decor", maker.VISUAL_FLAG_FOREGROUND)
    terrain = maker.VisualTerrain(assets, theme)
    back, front = terrain.layers(level, 0, 2)
    row = 40 * 16 - 2 * maker.CHUNK_PIXELS

    assert level.word_at(2, 40) == 0 and level.word_at(3, 40) == 0
    assert back[row:row + 16, 32:48, 3].max() == maker.MATERIALS["water"].alpha
    assert front[row:row + 16, 32:48, 3].max() == 0
    assert front[row:row + 16, 48:64, 3].max() > 0
    assert back[row:row + 16, 48:64, 3].max() == 0


def test_layers_toggle_independently_and_leave_the_level_alone(assets, theme):
    level = maker.EditableLevel.with_ground()
    renderer = maker.LevelRenderer(assets, theme)
    ram = fake_ram(camera_y=784)
    words = level.cells.copy()

    both = renderer.render(level, ram, maker.ViewOptions(show_grid=False))
    art_only = renderer.render(level, ram, maker.ViewOptions(
        show_collision=False, show_grid=False))
    collision_only = renderer.render(level, ram, maker.ViewOptions(
        show_visual=False, show_grid=False))

    assert level.cells == words
    surface = maker.GROUND_Y - 784 + 2
    assert tuple(collision_only[surface, 40]) == maker.COLLISION_ALL
    assert tuple(collision_only[0, 0]) == maker.COLLISION_BACKGROUND
    # With artwork on, the sky replaces the void and the overlay only tints.
    assert tuple(art_only[0, 0]) != maker.COLLISION_BACKGROUND
    assert not np.array_equal(both, art_only)
    assert not np.array_equal(both, collision_only)


def test_material_export_writes_masks_art_and_metadata(theme, tmp_path):
    written = maker.export_material_assets(theme, tmp_path / "tex")
    names = {path.name for path in written}
    assert {"grass.mask.png", "grass.png", "materials.json"} <= names

    payload = json.loads((tmp_path / "tex" / "materials.json").read_text())
    assert payload["tile"] == maker.TEXTURE_SHEET
    grass = next(item for item in payload["materials"]
                 if item["key"] == "grass")
    assert grass["mask_hex"] == "#00ff00"
    assert grass["clipped_to_collision"] is True

    # Exported art is exactly what a replacement drop-in has to look like.
    reloaded = maker.SemanticTheme(texture_dir=tmp_path / "tex")
    assert set(reloaded.replaced) == set(maker.MATERIALS)
    assert np.array_equal(reloaded.sheets["grass"], theme.sheets["grass"])


def write_flattened_sprite(path):
    """A JPEG-style asset with no alpha channel at all.

    Its "transparency" is a checkerboard baked into the pixels, and the
    subject encloses a white square that a naive brightness key would punch a
    hole through.
    """
    rows, columns = np.mgrid[0:64, 0:64]
    checker = ((rows // 8) + (columns // 8)) % 2 == 0
    image = np.repeat(
        np.where(checker, 255, 214).astype(np.uint8)[..., None], 3, axis=2)
    image[16:48, 16:48] = (40, 160, 60)      # the subject
    image[28:36, 28:36] = (255, 255, 255)    # an enclosed highlight
    surface = maker.pygame.image.frombuffer(
        np.ascontiguousarray(image).tobytes(), (64, 64), "RGB")
    assert surface.get_masks()[3] == 0
    maker.pygame.image.save(surface, str(path))


def write_rgba_sprite(path, image):
    surface = maker.pygame.image.frombuffer(
        np.ascontiguousarray(image).tobytes(), image.shape[1::-1], "RGBA")
    maker.pygame.image.save(surface, str(path))


def test_placed_sprite_geometry_is_anchored_at_its_base():
    sprite = maker.PlacedSprite("tree", 100, 200, 40, 60)
    assert sprite.bounds == (80, 140, 40, 60)
    assert sprite.front is False
    assert sprite.contains(100, 199) and sprite.contains(80, 140)
    assert not sprite.contains(100, 200)      # the base row is exclusive
    assert not sprite.contains(79, 180)

    level = maker.EditableLevel.with_ground()
    assert level.add_sprite(sprite) == 0
    assert level.add_sprite(sprite) == 1
    for broken in (
            maker.PlacedSprite("", 10, 10, 40, 60),
            maker.PlacedSprite("tree", -1, 10, 40, 60),
            maker.PlacedSprite("tree", 10, 10, 40, maker.SPRITE_MAX_SIZE + 1),
            maker.PlacedSprite("tree", 10, 10, 4, 60),
            maker.PlacedSprite("tree", 10, 10, 40, 60, flags=0x80)):
        with pytest.raises(ValueError):
            level.add_sprite(broken)


def test_sprites_round_trip_and_are_absent_from_older_saves(tmp_path):
    level = maker.EditableLevel.with_ground()
    level.add_sprite(maker.PlacedSprite("tree", 300, 896, 64, 80))
    level.add_sprite(maker.PlacedSprite(
        "tree", 500, 880, 32, 40, maker.VISUAL_FLAG_FOREGROUND))
    path = tmp_path / "level.json"
    level.save(path)

    payload = json.loads(path.read_text())
    assert payload["visual"]["sprites"] == [
        {"art": "tree", "x": 300, "y": 896, "width": 64, "height": 80,
         "flags": 0},
        {"art": "tree", "x": 500, "y": 880, "width": 32, "height": 40,
         "flags": maker.VISUAL_FLAG_FOREGROUND},
    ]
    loaded = maker.EditableLevel.load(path)
    assert list(loaded.sprites.values()) == list(level.sprites.values())
    assert loaded.cells == level.cells

    payload["version"] = 2
    assert maker.EditableLevel.from_dict(payload).sprites == {}

    payload["version"] = 3
    payload["visual"]["sprites"] = [{"art": "tree", "x": 1, "y": 2}]
    with pytest.raises(ValueError):
        maker.EditableLevel.from_dict(payload)


def test_history_undoes_sprite_placement_moves_and_removal():
    level = maker.EditableLevel.with_ground()
    history = maker.EditHistory()
    first = maker.PlacedSprite("tree", 300, 896, 64, 80)
    sprite_id = level.add_sprite(first)
    history.record([maker.SpriteEdit(sprite_id, None, first)])

    moved = maker.replace(first, x=380, width=96, height=120)
    history.record([maker.SpriteEdit(
        sprite_id, level.apply_sprite(sprite_id, moved), moved)])
    # Sprites never reach WRAM, so no collision changes come back.
    assert history.undo(level) == []
    assert level.sprites[sprite_id] == first
    assert history.undo(level) == []
    assert level.sprites == {}
    history.redo(level)
    history.redo(level)
    assert level.sprites[sprite_id] == moved


def test_sprite_library_keys_flattened_art_but_keeps_enclosed_highlights(
        tmp_path):
    write_flattened_sprite(tmp_path / "prop.png")
    (tmp_path / "notes.txt").write_text("ignored")
    library = maker.SpriteLibrary(tmp_path)

    assert library.names == ("prop",)
    assert library.failed == ()
    source = library.source("prop")
    # The checkerboard margin is gone and the art has been cropped to it.
    assert source.shape[:2] == (32, 32)
    assert source[..., 3].min() == 255
    assert tuple(source[0, 0, :3]) == (40, 160, 60)
    assert tuple(source[16, 16, :3]) == (255, 255, 255)

    scaled = library.scaled("prop", 16, 16)
    assert scaled.shape == (16, 16, 4)
    assert scaled is library.scaled("prop", 16, 16)
    assert library.scaled("missing", 16, 16) is None
    assert library.aspect("prop") == pytest.approx(1.0)
    assert library.thumbnail("prop", 40, 20).get_size() == (20, 20)


def test_sprite_library_honours_real_transparency(tmp_path):
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[8:24, 8:24] = (255, 255, 255, 255)      # a white subject, no key
    write_rgba_sprite(tmp_path / "white.png", image)

    source = maker.SpriteLibrary(tmp_path).source("white")
    assert source.shape[:2] == (16, 16)
    assert source[..., 3].min() == 255
    assert tuple(source[8, 8, :3]) == (255, 255, 255)


def test_a_fully_opaque_alpha_channel_is_taken_at_its_word(tmp_path):
    """The file declaring transparency settles it, even when none is used.

    A rectangular tile is legitimate artwork; guessing a background out of it
    from pixel brightness would eat the whole sprite.
    """
    image = np.zeros((24, 24, 4), dtype=np.uint8)
    image[..., :3] = (250, 250, 250)
    image[..., 3] = 255
    write_rgba_sprite(tmp_path / "slab.png", image)

    source = maker.SpriteLibrary(tmp_path).source("slab")
    assert source.shape == (24, 24, 4)
    assert source[..., 3].min() == 255
    assert tuple(source[12, 12, :3]) == (250, 250, 250)


def test_page_left_inside_a_cutout_mask_is_trimmed(tmp_path):
    """A mask cut wider than the art keeps a ring of the page it came from."""
    rows, columns = np.mgrid[0:40, 0:40]
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    image[..., :3] = 246                                   # the page colour
    image[(np.abs(rows - 20) <= 10) & (np.abs(columns - 20) <= 10), 3] = 255
    image[(np.abs(rows - 20) <= 8) & (np.abs(columns - 20) <= 8), :3] = (
        30, 120, 40)
    write_rgba_sprite(tmp_path / "cut.png", image)

    source = maker.SpriteLibrary(tmp_path).source("cut")
    # The two-pixel ring of page is gone; only the artwork is left.
    assert source.shape[:2] == (17, 17)
    assert source[..., 3].min() == 255
    assert source[..., :3].max() <= 130


def test_a_dark_outline_is_never_mistaken_for_a_page(tmp_path):
    """Transparency whose colour was zeroed is not a page to trim against."""
    rows, columns = np.mgrid[0:24, 0:24]
    image = np.zeros((24, 24, 4), dtype=np.uint8)
    image[(np.abs(rows - 12) <= 6) & (np.abs(columns - 12) <= 6)] = (
        20, 20, 20, 255)
    image[(np.abs(rows - 12) <= 4) & (np.abs(columns - 12) <= 4), :3] = (
        220, 60, 60)
    write_rgba_sprite(tmp_path / "outlined.png", image)

    source = maker.SpriteLibrary(tmp_path).source("outlined")
    assert source.shape[:2] == (13, 13)
    assert tuple(source[0, 0, :3]) == (20, 20, 20)


def test_white_matted_edges_lose_the_halo_but_keep_their_alpha(tmp_path):
    """Soft edges composed against white are recoloured, never re-cut."""
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[4:12, 4:12] = (20, 140, 40, 255)
    image[3, 4:12] = (255, 255, 255, 90)      # an unpremultiplied soft edge
    write_rgba_sprite(tmp_path / "soft.png", image)

    source = maker.SpriteLibrary(tmp_path).source("soft")
    assert source.shape[:2] == (9, 8)
    assert source[0, 4, 3] == 90              # the declared alpha survives
    assert tuple(source[0, 4, :3]) == (20, 140, 40)


def test_sprites_render_on_the_chosen_side_of_sonic(assets, theme, tmp_path):
    write_flattened_sprite(tmp_path / "prop.png")
    library = maker.SpriteLibrary(tmp_path)
    renderer = maker.LevelRenderer(assets, theme, library)
    level = maker.EditableLevel.with_ground()
    ram = fake_ram(camera_y=768)
    view = maker.ViewOptions(show_collision=False, show_grid=False)

    empty = renderer.render(level, ram, view)
    behind_id = level.add_sprite(maker.PlacedSprite("prop", 64, 850, 32, 32))
    behind = renderer.render(level, ram, view)
    level.apply_sprite(behind_id, maker.replace(
        level.sprites[behind_id], flags=maker.VISUAL_FLAG_FOREGROUND))
    front = renderer.render(level, ram, view)

    # The sprite's top-left corner lands here: bounds are (48, 818, 32, 32)
    # and the camera starts at world row 768.
    sample = (818 - 768 + 4, 48 + 4)
    assert tuple(behind[sample]) == (40, 160, 60)
    # Same pixels either way with no Sonic on screen, but both differ from bare
    # terrain, and the artwork toggle hides them.
    assert np.array_equal(behind, front)
    assert not np.array_equal(behind, empty)
    hidden = renderer.render(level, ram, maker.ViewOptions(
        show_visual=False, show_grid=False))
    assert tuple(hidden[sample]) != (40, 160, 60)


def test_capture_edges_snap_to_blocks_and_never_invert():
    region = maker.CaptureRegion.around(100, 200, 300, 150)
    assert (region.left, region.top) == (96, 192)
    assert (region.right, region.bottom) == (400, 352)
    assert (region.width, region.height) == (304, 160)

    assert region.with_edge("left", 150).left == 144
    assert region.with_edge("left", 150, snap=False).left == 150
    assert region.with_edge("top", -80).top == 0
    assert region.with_edge("right", maker.WORLD_WIDTH + 99).right == \
        maker.WORLD_WIDTH

    # An edge pushed past its opposite number stops short of it instead of
    # turning the rectangle inside out.
    assert region.with_edge("left", 9999).left == \
        region.right - maker.CAPTURE_MIN_SIZE
    assert region.with_edge("bottom", -9999).bottom == \
        region.top + maker.CAPTURE_MIN_SIZE

    with pytest.raises(ValueError):
        region.edge("middle")
    with pytest.raises(ValueError):
        maker.CaptureRegion(10, 10, 10, 40).validated()
    with pytest.raises(ValueError):
        maker.CaptureRegion(-1, 10, 40, 40).validated()


def test_capture_region_round_trips_with_the_level(tmp_path):
    level = maker.EditableLevel.with_ground()
    assert level.capture is None
    level.capture = maker.CaptureRegion(64, 784, 320, 896)
    path = tmp_path / "framed.json"
    level.save(path)

    payload = json.loads(path.read_text())
    assert payload["visual"]["capture"] == {
        "left": 64, "top": 784, "right": 320, "bottom": 896
    }
    assert maker.EditableLevel.load(path).capture == level.capture

    payload["version"] = 2
    assert maker.EditableLevel.from_dict(payload).capture is None

    payload["version"] = 3
    payload["visual"]["capture"] = {"left": 0, "top": 0}
    with pytest.raises(ValueError):
        maker.EditableLevel.from_dict(payload)


def test_render_region_matches_the_live_view(assets, theme):
    """The export path and the screen path must agree pixel for pixel."""
    level = maker.EditableLevel.with_ground()
    level.set_visual(6, 56, "sand")
    renderer = maker.LevelRenderer(assets, theme)
    ram = fake_ram(camera_x=128, camera_y=768)      # no sprites, so no Sonic

    live = renderer.render(level, ram)
    region = renderer.render_region(
        level, 128, 768, maker.NATIVE_WIDTH, maker.NATIVE_HEIGHT)
    assert np.array_equal(live, region)

    with pytest.raises(ValueError):
        renderer.render_region(level, 0, 0, 0, 32)


def test_visualization_export_writes_the_region_one_to_one(
        assets, theme, tmp_path):
    level = maker.EditableLevel.with_ground()
    renderer = maker.LevelRenderer(assets, theme)
    region = maker.CaptureRegion(64, 784, 320, 896)
    view = maker.ViewOptions(show_collision=False, show_grid=True)

    path = maker.export_visualization(
        renderer, level, region, tmp_path / "vis", "green hill", view)
    assert path.name == "green hill_64x784_256x112.png"
    surface = maker.pygame.image.load(str(path))
    assert surface.get_size() == (region.width, region.height)

    written = maker.pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    assert np.array_equal(written, renderer.render_region(
        level, region.left, region.top, region.width, region.height, view))

    # Names stay usable whatever the level was called.
    assert maker.visualization_filename("a/b*c", region) == \
        "a_b_c_64x784_256x112.png"
    assert maker.visualization_filename("", region).startswith("untitled_")


def test_live_engine_uses_edited_collision_map():
    runtime = maker.SonicRuntime()
    level = maker.EditableLevel.with_ground()
    try:
        runtime.install(level)
        for _ in range(3):
            runtime.step(set())
        grounded = runtime.player_info()
        assert grounded["airborne"] is False
        assert grounded["angle"] == 0
        assert grounded["y"] == maker.SPAWN_Y

        for _ in range(30):
            runtime.step({maker.pygame.K_RIGHT})
        assert runtime.player_info()["x"] > maker.SPAWN_X

        runtime.reset_player()
        runtime.step(set())
        for _ in range(4):
            runtime.step({maker.pygame.K_SPACE})
        jumping = runtime.player_info()
        assert jumping["airborne"] is True
        assert jumping["yvel"] < 0
        runtime.reset_player()
        runtime.step(set())

        # Sonic's two floor sensors sit in these adjacent columns at spawn.
        # Removing the entire columns must make the ROM's own routine put him
        # in the air; no editor-side collision calculation participates.
        for x in (9, 10):
            for y in range(maker.GROUND_TILE_Y, maker.WORLD_TILES_Y):
                level.set_word(x, y, 0)
                runtime.write_tile(x, y, 0)
        for _ in range(4):
            runtime.step(set())
        falling = runtime.player_info()
        assert falling["airborne"] is True
        assert falling["yvel"] > 0
    finally:
        runtime.close()


def test_live_engine_climbs_an_edited_native_slope():
    level = maker.EditableLevel.with_ground()
    # Stamp exactly where a user clicks: one grid cell above the starter
    # floor.  The helper replaces the obscuring floor tile underneath and
    # creates a short upper ledge.
    for dx, dy, word in maker.safe_stamp_pattern(maker.STAMP_45_RIGHT):
        level.set_word(12 + dx, 55 + dy, word)

    runtime = maker.SonicRuntime()
    try:
        runtime.install(level)
        angles, heights, pushing_frames = set(), [], 0
        for _ in range(75):
            runtime.step({maker.pygame.K_RIGHT})
            info = runtime.player_info()
            angles.add(info["angle"])
            heights.append(info["y"])
            status = runtime.memory.extract(
                maker.RAM_BASE + maker.RAM_SONIC + 0x22, "|u1")
            pushing_frames += bool(status & 0x20)
        assert 0xE0 in angles
        assert min(heights) < maker.SPAWN_Y
        assert pushing_frames == 0
        assert runtime.player_info()["airborne"] is False
    finally:
        runtime.close()


def test_live_engine_stays_grounded_across_fitted_terrain_hill():
    runtime = maker.SonicRuntime()
    try:
        assets = maker.CollisionAssets(runtime.rom)
        x_values = np.arange(96.0, 1025.0)
        phase = (x_values - x_values[0]) / (x_values[-1] - x_values[0])
        y_values = maker.GROUND_Y - 64 * np.sin(np.pi * phase) ** 2

        columns = maker.TerrainFitter(assets).fit(x_values, y_values)
        level = maker.EditableLevel.with_ground()
        maker.apply_terrain_columns(level, columns)
        runtime.install(level)

        records = []
        for _ in range(270):
            runtime.step({maker.pygame.K_RIGHT})
            records.append(runtime.player_info())

        assert len(columns) == 59
        assert records[-1]["x"] > 1000
        assert min(record["y"] for record in records) < maker.SPAWN_Y - 50
        assert any(record["angle"] not in (0, 0xFF) for record in records)
        assert not any(record["airborne"] for record in records)
    finally:
        runtime.close()


def test_visual_tab_paints_materials_without_disturbing_physics(tmp_path):
    runtime = maker.SonicRuntime()
    try:
        level = maker.EditableLevel.with_ground()
        app = maker.MakerApp(
            runtime, maker.CollisionAssets(runtime.rom), level, None,
            scale=2, headless=True, levels_dir=tmp_path / "levels")
        app._set_editor_tab("visual")
        app.visual_material = "platform"
        words = level.cells.copy()
        camera_x, camera_y = runtime.camera()

        def screen(tile_x, tile_y):
            return (int((tile_x * 16 + 8 - camera_x) * app.scale),
                    int((tile_y * 16 + 8 - camera_y) * app.scale))

        row = maker.GROUND_TILE_Y
        app._start_stroke(1, screen(11, row))
        app._continue_stroke(screen(14, row))
        app._finish_stroke()
        assert set(level.visual) == {(x, row) for x in range(11, 15)}
        assert all(value == ("platform", 0) for value in level.visual.values())
        assert level.cells == words

        app.undo()
        assert level.visual == {}
        assert app.message == "UNDO"
        app.redo()
        assert len(level.visual) == 4

        # Right-click restores the automatic material for that block only.
        app._start_stroke(3, screen(12, row))
        app._finish_stroke()
        assert (12, row) not in level.visual
        assert len(level.visual) == 3
        assert level.cells == words

        # A painted level saves and reloads as version 3.
        path, _ = maker.save_named_level(level, app.levels_dir, "painted")
        assert maker.EditableLevel.load(path).visual == level.visual

        app.toggle_collision_layer()
        app.toggle_visual_layer()
        assert app.view_options() == maker.ViewOptions(
            show_visual=False, show_collision=False,
            collision_alpha=app.collision_alpha, show_grid=True)
        app.draw()
    finally:
        runtime.close()


def test_view_and_material_panel_buttons_are_wired(tmp_path):
    runtime = maker.SonicRuntime()
    try:
        app = maker.MakerApp(
            runtime, maker.CollisionAssets(runtime.rom),
            maker.EditableLevel.with_ground(), None,
            scale=2, headless=True, levels_dir=tmp_path / "levels")

        def click(name):
            assert app._button_click(app.buttons[name].center), name

        click("tab_visual")
        assert app.editor_tab == "visual"
        click("material_sand")
        assert app.visual_material == "sand"
        click("material_auto")
        assert app.visual_material == maker.MATERIAL_AUTO

        click("visual_foreground")
        assert app.visual_foreground is True
        click("visual_outline")
        assert app.collision_outline is True

        first = app.collision_alpha
        seen = {first}
        for _ in range(len(maker.COLLISION_ALPHAS) - 1):
            click("visual_opacity")
            seen.add(app.collision_alpha)
        assert seen == set(maker.COLLISION_ALPHAS)
        click("visual_opacity")
        assert app.collision_alpha == first

        click("view_collision")
        assert app.show_collision is False
        click("view_visual")
        assert app.show_visual is False
        click("visual_reload")
        assert app.theme.replaced == ()

        click("visual_mode_sprites")
        assert app.visual_mode == "sprites"
        click("sprite_plane")
        assert app.sprite_front is True
        # These need a selection and say so rather than doing anything.
        for name in ("sprite_delete", "sprite_smaller", "sprite_bigger"):
            click(name)
            assert app.message == "SELECT A SPRITE FIRST", name
        app.draw()

        click("visual_mode_capture")
        assert app.visual_mode == "capture"
        click("capture_view")
        assert app.level.capture is not None
        for name, attribute in (("capture_snap", "capture_snap"),
                                ("capture_grid", "capture_grid"),
                                ("capture_collision", "capture_collision")):
            before = getattr(app, attribute)
            click(name)
            assert getattr(app, attribute) is not before, name
        for edge in maker.CAPTURE_EDGES:
            click(f"capture_{edge}")
            assert app.capture_edge == edge
        click("capture_sequence")
        assert app.capture_edge == "left"
        assert app.capture_queue == maker.CAPTURE_EDGES[1:]
        app.draw()

        click("visual_mode_materials")
        assert app.visual_mode == "materials"
        app.draw()
    finally:
        runtime.close()


def test_sprite_tools_place_move_scale_and_choose_a_plane(tmp_path):
    write_flattened_sprite(tmp_path / "prop.png")
    runtime = maker.SonicRuntime()
    try:
        level = maker.EditableLevel.with_ground()
        app = maker.MakerApp(
            runtime, maker.CollisionAssets(runtime.rom), level, None,
            scale=2, headless=True, levels_dir=tmp_path / "levels",
            sprite_dir=tmp_path)
        app._set_editor_tab("visual")
        app._set_visual_mode("sprites")
        assert app.sprite_choice == "prop"
        words = level.cells.copy()
        camera_x, camera_y = runtime.camera()

        def screen(world_x, world_y):
            return (int((world_x - camera_x) * app.scale),
                    int((world_y - camera_y) * app.scale))

        # Click empty space to drop the chosen sprite on its base.
        app._sprite_mouse_down(1, screen(120, maker.GROUND_Y))
        app._sprite_mouse_up(1)
        sprite_id = app.selected_sprite
        placed = level.sprites[sprite_id]
        assert (placed.art, placed.x, placed.y) == ("prop", 120,
                                                    maker.GROUND_Y)
        assert placed.height == maker.SPRITE_DEFAULT_HEIGHT
        assert level.cells == words

        # Dragging the body moves it; the grip resizes it from the base.
        app._sprite_mouse_down(1, screen(120, maker.GROUND_Y - 20))
        app._sprite_mouse_motion(screen(150, maker.GROUND_Y - 20))
        app._sprite_mouse_up(1)
        assert level.sprites[sprite_id].x == 150

        grip = app._sprite_handle(level.sprites[sprite_id])
        app._sprite_mouse_down(1, screen(*grip))
        assert app.sprite_drag == "scale"
        app._sprite_mouse_motion(screen(grip[0] + 20, grip[1]))
        app._sprite_mouse_up(1)
        grown = level.sprites[sprite_id]
        assert grown.width > placed.width
        assert grown.y == maker.GROUND_Y      # still planted on its base

        # Undo walks back the resize, then the move, then the placement.
        app.undo()
        assert level.sprites[sprite_id].width == placed.width
        app.undo()
        assert level.sprites[sprite_id].x == 120
        app.undo()
        assert level.sprites == {}
        app.redo()
        assert len(level.sprites) == 1
        assert level.cells == words

        app._select_sprite(sprite_id)
        app._set_selected_sprite_plane(True)
        assert level.sprites[sprite_id].front is True
        app._scale_selected_sprite(maker.SPRITE_STEP)
        assert level.sprites[sprite_id].height > placed.height

        # A saved level keeps the sprite, and a right-click removes it.
        path, _ = maker.save_named_level(level, app.levels_dir, "props")
        assert list(maker.EditableLevel.load(path).sprites.values()) == \
            list(level.sprites.values())
        sprite = level.sprites[sprite_id]
        app._sprite_mouse_down(3, screen(sprite.x, sprite.y - 8))
        assert level.sprites == {}
        assert app.selected_sprite is None
        assert level.cells == words
        app.draw()
    finally:
        runtime.close()


def test_capture_tool_walks_the_four_edges_then_exports(tmp_path):
    runtime = maker.SonicRuntime()
    try:
        level = maker.EditableLevel.with_ground()
        app = maker.MakerApp(
            runtime, maker.CollisionAssets(runtime.rom), level, None,
            scale=2, headless=True, levels_dir=tmp_path / "levels",
            visualization_dir=tmp_path / "vis")
        app._set_editor_tab("visual")
        app._set_visual_mode("capture")
        assert app._capturing()

        assert level.capture is None
        app.export_capture()
        assert app.message == "SET A CAPTURE REGION FIRST"

        camera_x, camera_y = runtime.camera()

        def screen(world_x, world_y):
            return (int((world_x - camera_x) * app.scale),
                    int((world_y - camera_y) * app.scale))

        app.start_capture_sequence()
        for edge, point in (("left", (60, 800)), ("right", (300, 800)),
                            ("top", (0, 790)), ("bottom", (0, 900))):
            assert app.capture_edge == edge
            app.draw()                      # the guides must render mid-flow
            app._capture_mouse_down(1, screen(*point))
        assert app.capture_edge is None

        region = level.capture
        assert (region.left, region.right) == (64, 304)
        assert (region.top, region.bottom) == (784, 896)

        app.export_capture()
        path = app.last_capture
        assert path.parent == (tmp_path / "vis").resolve()
        assert maker.pygame.image.load(str(path)).get_size() == (
            region.width, region.height)

        # Re-arming one edge leaves the other three alone.
        app.arm_capture_edge("top")
        app._capture_mouse_down(1, screen(0, 830))
        assert level.capture == maker.replace(region, top=832)
        assert app.capture_edge is None

        # A right-click abandons an edge without moving it.
        app.arm_capture_edge("left")
        app._capture_mouse_down(3, screen(0, 800))
        assert app.capture_edge is None
        assert level.capture.left == 64

        # The frame is part of the level, so it survives a save and reload.
        saved, _ = maker.save_named_level(level, app.levels_dir, "framed")
        assert maker.EditableLevel.load(saved).capture == level.capture
        app.draw()
    finally:
        runtime.close()


def test_marker_editor_drag_and_level_completion_return_to_start(tmp_path):
    runtime = maker.SonicRuntime()
    try:
        level = maker.EditableLevel.with_ground()
        app = maker.MakerApp(
            runtime, maker.CollisionAssets(runtime.rom), level, None,
            scale=2, headless=True, levels_dir=tmp_path / "levels")
        app._set_editor_tab("markers")

        camera_x, camera_y = runtime.camera()
        finish_base = (280, maker.GROUND_Y)
        finish_screen = (
            int((finish_base[0] - camera_x) * app.scale),
            int((finish_base[1] - camera_y) * app.scale),
        )
        app.marker_tool = "finish"
        app._marker_mouse_down(1, finish_screen)
        app._marker_mouse_up(1)
        assert level.finish_marker == finish_base

        # Grabbing the pole above its base and dragging horizontally retains
        # the vertical base position rather than jumping to the mouse cursor.
        grab = (finish_screen[0], finish_screen[1] - 30 * app.scale)
        app._marker_mouse_down(1, grab)
        app._marker_mouse_motion((grab[0] + 20 * app.scale, grab[1]))
        app._marker_mouse_up(1)
        assert level.finish_marker == (300, maker.GROUND_Y)

        start_base = (140, maker.GROUND_Y)
        start_screen = (
            int((start_base[0] - camera_x) * app.scale),
            int((start_base[1] - camera_y) * app.scale),
        )
        app.marker_tool = "start"
        app._marker_mouse_down(1, start_screen)
        app._marker_mouse_up(1)
        assert level.start_marker == start_base

        completed = False
        for frame in range(120):
            app.frame = frame
            runtime.step({maker.pygame.K_RIGHT})
            app._update_level_completion()
            if app.completion_until:
                completed = True
                break
        assert completed
        assert app.message == "LEVEL COMPLETE!"
        info = runtime.player_info()
        assert (info["x"], info["y"]) == (level.spawn_x, level.spawn_y)
    finally:
        runtime.close()
