"""In-memory frames and a reused recording overlay for the ViZDoom player.

The checkpoint scores images by path. A sentinel path stays in the request,
and the real pixels are handed to its image opener from memory. The recording
chrome is drawn once per decision; each later tic only replaces the game
image and the status line.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

FRAME_KEY = "/intern-decision-frame.png"
TOP, STATUS_H, DECISION_H, FOOTER_H = 56, 28, 70, 22


class FrameStore:
    """Serve one screen to the checkpoint opener without writing a file."""

    def __init__(self, image_module):
        self.key = FRAME_KEY
        self._image = None
        original = image_module.open

        def open_image(path, *args, **kwargs):
            if str(path) == self.key and self._image is not None:
                # The checkpoint closes the image it opens. Return a copy.
                return self._image.copy()
            return original(path, *args, **kwargs)

        image_module.open = open_image

    def bind(self, rgb) -> str:
        self._image = Image.fromarray(rgb, mode="RGB")
        return self.key


class Overlay:
    """Static chrome for one decision. ``stamp`` fills the game picture and the live counts."""

    def __init__(self, spec: dict, episode: int, decision: dict | None, note: str, width: int, game_height: int, font):
        self.width = width
        self.episode = episode
        self.top = TOP
        self.game_height = game_height
        self.status_h = STATUS_H
        self.show_ammo = spec["press_field"] == "attack"
        height = TOP + game_height + STATUS_H + DECISION_H + FOOTER_H
        self.canvas = Image.new("RGB", (width, height), (10, 16, 28))
        draw = ImageDraw.Draw(self.canvas)
        title_font = font(22)
        small_font = font(14)
        draw.text((16, 8), spec["title"], fill=(244, 246, 250), font=title_font)
        draw.text((16, 34), spec["subtitle"], fill=(186, 196, 210), font=small_font)
        y = TOP + game_height + STATUS_H
        if decision is None:
            turn_text, press_text = "TURN: …", f"{spec['press_field'].upper()}: …"
            turn_conf = press_conf = ""
        else:
            turn_text = f"TURN: {decision['turn'].upper()}"
            press_text = f"{spec['press_field'].upper()}: {decision['press_label'].upper()}"
            turn_conf = f"Model confidence {decision['turn_confidence'] * 100:.0f}%"
            press_conf = f"Model confidence {decision['press_confidence'] * 100:.0f}%"
        draw.text((16, y + 8), turn_text, fill=(64, 220, 196), font=title_font)
        draw.text((width // 2, y + 8), press_text, fill=(64, 220, 196), font=title_font)
        draw.text((16, y + 40), turn_conf, fill=(210, 220, 230), font=small_font)
        draw.text((width // 2, y + 40), press_conf, fill=(210, 220, 230), font=small_font)
        draw.text((16, height - 18), note, fill=(150, 162, 176), font=small_font)
        self._body = font(16)
        self._status_y = TOP + game_height

    def stamp(self, game_rgb, hud: dict):
        game = Image.fromarray(game_rgb, mode="RGB")
        if game.size != (self.width, self.game_height):
            raise ValueError(f"frame {game.size} does not match {(self.width, self.game_height)}")
        self.canvas.paste(game, (0, self.top))
        draw = ImageDraw.Draw(self.canvas)
        y = self._status_y
        draw.rectangle((0, y, self.width, y + self.status_h), fill=(16, 42, 64))
        status = f"Episode {self.episode}    Kills {hud.get('kills', 0)}    Health {hud.get('health', 0)}"
        if self.show_ammo and "ammo" in hud:
            status += f"    Ammo {hud['ammo']}"
        draw.text((16, y + 5), status, fill=(232, 238, 244), font=self._body)
        return np.asarray(self.canvas)
