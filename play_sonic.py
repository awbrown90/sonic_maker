"""Play Sonic the Hedgehog with the keyboard.

Usage:
    play-sonic.py [--level ZONE-ACT]

Levels are given as zone-act in game order, e.g. 1-1 is Green Hill Act 1 and
3-2 is Spring Yard Act 2. The full retro state name works too.

Controls:
    LEFT / RIGHT arrows   move
    UP or SPACE           jump
    DOWN                  duck / roll
    A                     toggle terrain collision map
    C                     save checkpoint
    V                     load checkpoint
    ESC                   quit
"""
import argparse

import numpy as np
import stable_retro as retro
import pygame

SCALE = 3
FPS = 60

# how long an on-screen notice stays up, in frames
MSG_FRAMES = 45

GAME = 'SonicTheHedgehog-Genesis-v0'

# Sonic 1's terrain is assembled from 256x256 chunks containing 16x16 tiles.
# The chunk mappings and level layout below are the live, decompressed copies in
# Genesis RAM. Each tile then indexes a signed, 16-column height mask in ROM.
RAM_CHUNKS = 0x0000
RAM_LAYOUT = 0xA400
RAM_SONIC_GFX = 0xC800
RAM_SONIC = 0xD000
RAM_SPRITE_COUNT = 0xF62C
RAM_CAMERA_X = 0xF700
RAM_CAMERA_Y = 0xF704
RAM_COLLISION_INDEX = 0xF796
RAM_SPRITES = 0xF800
RAM_WATER_PALETTE = 0xFA80
RAM_DRY_PALETTE = 0xFB00
ROM_COLLISION_HEIGHTS = 0x62A00

SONIC_TILE = 0x780
SONIC_TILE_COUNT = 0x17
MAX_SPRITES = 0x50

COLLISION_BACKGROUND = (4, 10, 20)
COLLISION_GRID = (15, 30, 44)
COLLISION_CHUNK_GRID = (39, 68, 91)
COLLISION_TOP = (34, 197, 94)
COLLISION_ALL = (28, 126, 214)
COLLISION_SIDES = (238, 154, 44)

# Zones in the order the game plays them, so zone-act codes match the manual.
# This is not alphabetical order - Marble is zone 2 but sorts seventh.
ZONE_ORDER = [
    'GreenHillZone',
    'MarbleZone',
    'SpringYardZone',
    'LabyrinthZone',
    'StarLightZone',
    'ScrapBrainZone',
]

# Map 'zone-act' onto the savestates that ship with the game's retro data.
# Only states that actually exist are offered: Scrap Brain has no Act 3 state,
# so 6-3 is absent rather than a level that fails to load.
_STATES = set(retro.data.list_states(GAME))
LEVELS = {f"{z}-{a}": f"{zone}.Act{a}"
          for z, zone in enumerate(ZONE_ORDER, 1)
          for a in (1, 2, 3)
          if f"{zone}.Act{a}" in _STATES}


def level_arg(value):
    """Resolve a 'zone-act' code, or a full state name, to a state name."""
    if value in LEVELS:
        return LEVELS[value]
    if value in _STATES:
        return value
    raise argparse.ArgumentTypeError(
        f"invalid level {value!r}; choose from {', '.join(LEVELS)} "
        f"(or a full state name such as {ZONE_ORDER[0]}.Act1)")


def action_from_keys(held, buttons):
    """Map the set of held pygame keys to a Genesis controller action."""
    action = [0] * len(buttons)
    if pygame.K_LEFT in held:
        action[buttons.index('LEFT')] = 1
    if pygame.K_RIGHT in held:
        action[buttons.index('RIGHT')] = 1
    if pygame.K_DOWN in held:
        action[buttons.index('DOWN')] = 1
    if pygame.K_UP in held or pygame.K_SPACE in held:
        action[buttons.index('B')] = 1  # B = jump / spin
    return action


def reset_obs(env):
    obs = env.reset()
    return obs[0] if isinstance(obs, tuple) else obs


class CollisionMapRenderer:
    """Render Sonic 1's live tile collision data instead of its artwork."""

    def __init__(self, rom):
        collision_end = ROM_COLLISION_HEIGHTS + (256 * 16)
        if (len(rom) < collision_end or
                rom[ROM_COLLISION_HEIGHTS:ROM_COLLISION_HEIGHTS + 16]
                != bytes(16) or
                rom[ROM_COLLISION_HEIGHTS + 16:
                    ROM_COLLISION_HEIGHTS + 32] != bytes([1]) * 16):
            raise ValueError("unsupported Sonic 1 ROM revision")

        self.rom = rom
        self.masks = np.zeros((256, 4, 16, 16), dtype=bool)
        for collision_id in range(1, 256):
            start = ROM_COLLISION_HEIGHTS + collision_id * 16
            heights = np.frombuffer(rom, dtype=np.int8, count=16,
                                    offset=start)
            mask = self.masks[collision_id, 0]
            for x, height in enumerate(heights):
                height = int(height)
                if height > 0:
                    mask[16 - height:, x] = True
                elif height < 0:
                    mask[:-height, x] = True
            self.masks[collision_id, 1] = mask[:, ::-1]
            self.masks[collision_id, 2] = mask[::-1, :]
            self.masks[collision_id, 3] = mask[::-1, ::-1]

    @staticmethod
    def _u8(ram, address):
        # stable-retro exposes Genesis RAM in host-word order, so byte
        # addresses within every 16-bit word are reversed.
        return int(ram[address ^ 1])

    @staticmethod
    def _u16(ram, address):
        return int(ram[address]) | (int(ram[address + 1]) << 8)

    @classmethod
    def _pointer(cls, ram, address):
        return (cls._u16(ram, address) << 16) | cls._u16(ram, address + 2)

    @staticmethod
    def _genesis_rgb(value):
        """Convert a Genesis 3-bit-per-channel colour to retro's RGB888."""
        red = (value >> 1) & 7
        green = (value >> 5) & 7
        blue = (value >> 9) & 7
        red = ((red << 2) | (red >> 2)) << 3
        green = ((green << 3) | (green >> 1)) << 2
        blue = ((blue << 2) | (blue >> 2)) << 3
        return red, green, blue

    def _palette(self, ram, address, line):
        start = address + line * 32
        return [self._genesis_rgb(self._u16(ram, start + index * 2))
                for index in range(16)]

    def _sonic_tiles(self, ram):
        """Decode the 23 live 4bpp tiles DMA'd to Sonic's VRAM slot."""
        tiles = np.zeros((SONIC_TILE_COUNT, 8, 8), dtype=np.uint8)
        for tile in range(SONIC_TILE_COUNT):
            tile_address = RAM_SONIC_GFX + tile * 32
            for y in range(8):
                for pair in range(4):
                    packed = self._u8(ram, tile_address + y * 4 + pair)
                    tiles[tile, y, pair * 2] = packed >> 4
                    tiles[tile, y, pair * 2 + 1] = packed & 0xF
        return tiles

    def _draw_sonic(self, image, ram):
        """Draw Sonic alone from the live VDP sprite table and tile buffer."""
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

        # Lower-numbered VDP sprites have priority, so composite the list in
        # reverse order. Most Sonic frames use non-overlapping pieces anyway.
        for x, y, tiles_wide, tiles_high, first_tile, tile_word in reversed(pieces):
            indices = np.zeros((tiles_high * 8, tiles_wide * 8),
                               dtype=np.uint8)
            # Multi-cell Genesis sprites number cells down each column first.
            for tile_x in range(tiles_wide):
                for tile_y in range(tiles_high):
                    source = first_tile + tile_x * tiles_high + tile_y
                    if source >= SONIC_TILE_COUNT:
                        continue
                    indices[tile_y * 8:(tile_y + 1) * 8,
                            tile_x * 8:(tile_x + 1) * 8] = tiles[source]
            if tile_word & 0x0800:
                indices = indices[:, ::-1]
            if tile_word & 0x1000:
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

    def render(self, obs, ram):
        """Return an RGB frame of the visible collision tiles and Sonic."""
        height, width = obs.shape[:2]
        image = np.empty_like(obs)
        image[:] = COLLISION_BACKGROUND
        camera_x = self._u16(ram, RAM_CAMERA_X)
        camera_y = self._u16(ram, RAM_CAMERA_Y)
        collision_index = self._pointer(ram, RAM_COLLISION_INDEX)
        behind_loop = self._u8(ram, RAM_SONIC + 1) & 0x40

        first_x = camera_x & ~0xF
        first_y = camera_y & ~0xF
        for world_y in range(first_y, camera_y + height + 15, 16):
            if world_y < 0:
                continue
            screen_y = world_y - camera_y
            for world_x in range(first_x, camera_x + width + 15, 16):
                if world_x < 0:
                    continue
                screen_x = world_x - camera_x
                layout_address = (RAM_LAYOUT + ((world_y >> 8) & 7) * 0x80
                                  + ((world_x >> 8) & 0x7F))
                chunk = self._u8(ram, layout_address)
                if chunk == 0:
                    continue
                chunk_number = chunk & 0x7F
                if chunk_number == 0:
                    continue
                # GHZ's loop marks a layout cell with bit 7. Sonic selects an
                # alternate collision chunk while he is behind that loop.
                if (chunk & 0x80 and behind_loop and chunk_number == 0x28):
                    chunk_number = 0x51
                mapping = (RAM_CHUNKS + (chunk_number - 1) * 0x200
                           + ((world_y & 0xFF) >> 4) * 0x20
                           + ((world_x & 0xFF) >> 4) * 2)
                tile = self._u16(ram, mapping)
                solidity = tile & 0x6000
                block = tile & 0x7FF
                index_address = collision_index + block
                if (not solidity or not block or
                        not (0 <= index_address < len(self.rom))):
                    continue
                collision_id = self.rom[index_address]
                if collision_id == 0:
                    continue
                flip = ((tile >> 11) & 1) | (((tile >> 12) & 1) << 1)
                mask = self.masks[collision_id, flip]
                if solidity == 0x6000:
                    color = COLLISION_ALL
                elif solidity & 0x2000:
                    color = COLLISION_TOP
                else:
                    color = COLLISION_SIDES

                x0, x1 = max(0, screen_x), min(width, screen_x + 16)
                y0, y1 = max(0, screen_y), min(height, screen_y + 16)
                if x0 >= x1 or y0 >= y1:
                    continue
                visible_mask = mask[y0 - screen_y:y1 - screen_y,
                                    x0 - screen_x:x1 - screen_x]
                region = image[y0:y1, x0:x1]
                region[visible_mask] = color

        # The fine grid exposes the 16x16 lookups; brighter lines mark the
        # much larger 256x256 chunks referenced by the level layout.
        image[:, (-camera_x) % 16::16] = COLLISION_GRID
        image[(-camera_y) % 16::16, :] = COLLISION_GRID
        image[:, (-camera_x) % 256::256] = COLLISION_CHUNK_GRID
        image[(-camera_y) % 256::256, :] = COLLISION_CHUNK_GRID
        self._draw_sonic(image, ram)
        return image


def draw_message(screen, font, text):
    """Draw text across the top of the frame, outlined so it stays readable."""
    label = font.render(text, True, (255, 255, 255))
    x = (screen.get_width() - label.get_width()) // 2
    y = 8 * SCALE
    shadow = font.render(text, True, (0, 0, 0))
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        screen.blit(shadow, (x + dx, y + dy))
    screen.blit(label, (x, y))


def main():
    parser = argparse.ArgumentParser(
        description="Play Sonic the Hedgehog with the keyboard.")
    parser.add_argument('--level', default='1-1', type=level_arg,
                        metavar='ZONE-ACT',
                        help="level to load as zone-act in game order, one "
                             f"of: {', '.join(LEVELS)} (default: 1-1, "
                             "Green Hill Act 1)")
    args = parser.parse_args()

    print(__doc__)
    title = f"Sonic - {args.level}"
    # render_mode=None: we draw frames ourselves with pygame (the default
    # "human" mode would auto-open a pyglet window that clashes with pygame)
    env = retro.make(GAME, args.level, render_mode=None)
    obs = reset_obs(env)
    rom_path = retro.data.get_original_romfile_path(GAME)
    with open(rom_path, 'rb') as rom_file:
        collision_renderer = CollisionMapRenderer(rom_file.read())

    pygame.init()
    h, w = obs.shape[:2]
    screen = pygame.display.set_mode((w * SCALE, h * SCALE))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 12 * SCALE)

    held = set()
    info = {}
    frame = 0
    message = None
    message_until = 0
    checkpoint = None
    terrain_view = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    terrain_view = not terrain_view
                    message = ("COLLISION MAP" if terrain_view
                               else "NORMAL VIEW")
                    message_until = frame + MSG_FRAMES
                # C / V are one-shot actions, not held buttons. retro
                # serialises the whole emulator state, so a checkpoint is
                # independent of the savestate reset() restarts the level from.
                elif event.key == pygame.K_c:
                    checkpoint = env.em.get_state()
                    message, message_until = "CHECKPOINT", frame + MSG_FRAMES
                elif event.key == pygame.K_v:
                    if checkpoint is not None:
                        env.em.set_state(checkpoint)
                        message = "LOADED"
                    else:
                        message = "NO CHECKPOINT"
                    message_until = frame + MSG_FRAMES
                held.add(event.key)
            elif event.type == pygame.KEYUP:
                held.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held.clear()

        out = env.step(action_from_keys(held, env.buttons))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
        if done:
            obs = reset_obs(env)

        display_obs = (collision_renderer.render(obs, env.get_ram())
                       if terrain_view else obs)
        surf = pygame.image.frombuffer(
            display_obs.tobytes(), (w, h), 'RGB').convert()
        pygame.transform.scale(surf, screen.get_size(), screen)
        if message and frame < message_until:
            draw_message(screen, font, message)
        elif message:
            message = None
        pygame.display.flip()

        frame += 1
        if frame % 15 == 0:
            view_name = "collision map" if terrain_view else "normal"
            pygame.display.set_caption(
                f"{title} - x={info.get('x', 0)} "
                f"rings={info.get('rings', 0)} lives={info.get('lives', 3)} "
                f"- {view_name}")
        clock.tick(FPS)

    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
