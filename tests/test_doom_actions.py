"""Button mapping and request shape for the ViZDoom player. No game, no weights."""

import unittest

from src.doom.schema import action_vector, build_request, parse_answers

try:
    from src.doom.present import FRAME_KEY, FrameStore, Overlay
except ImportError:  # system interpreters used for the schema tests may lack Pillow
    FrameStore = Overlay = None
    FRAME_KEY = ""


DEFEND = ("TURN_LEFT", "TURN_RIGHT", "ATTACK")
HEALTH = ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD")


class ActionVectorTest(unittest.TestCase):
    def test_defend_combines_turn_and_attack(self):
        self.assertEqual(action_vector(DEFEND, "left", True), [1, 0, 1])
        self.assertEqual(action_vector(DEFEND, "right", False), [0, 1, 0])
        self.assertEqual(action_vector(DEFEND, "stay", True), [0, 0, 1])

    def test_health_combines_turn_and_move(self):
        self.assertEqual(action_vector(HEALTH, "stay", True), [0, 0, 1])
        self.assertEqual(action_vector(HEALTH, "right", False), [0, 1, 0])

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            action_vector(("TURN_LEFT", "MOVE_BACK"), "left", True)

    def test_request_is_image_plus_two_fields(self):
        request = build_request("defend", "/tmp/frame.png")
        self.assertEqual(request["images"], ["/tmp/frame.png"])
        self.assertEqual(list(request["questions"]), ["turn", "attack"])
        self.assertNotIn("health", request["state"])
        with_hud = build_request("health", "frame.png", {"health": 80, "kills": 0})
        self.assertEqual(with_hud["state"]["health"], 80)
        self.assertIn("acid", with_hud["state"]["task"])

    def test_parse_uses_probability_argmax(self):
        answers = {
            "turn": {"probabilities": {"left": 0.2, "right": 0.5, "stay": 0.3}},
            "attack": {"probabilities": {"no": 0.66, "yes": 0.34}},
        }
        parsed = parse_answers(answers, "attack")
        self.assertEqual(parsed["turn"], "right")
        self.assertFalse(parsed["pressed"])
        self.assertAlmostEqual(parsed["press_confidence"], 0.66)


@unittest.skipUnless(FrameStore is not None, "Pillow is required")
class FramePathTest(unittest.TestCase):
    def test_opener_serves_memory_and_survives_close(self):
        class Module:
            def open(self, path, *args, **kwargs):
                raise AssertionError(f"real opener called for {path}")

        module = Module()
        store = FrameStore(module)
        rgb = __import__("numpy").zeros((4, 6, 3), dtype="uint8")
        rgb[0, 0] = (9, 8, 7)
        self.assertEqual(store.bind(rgb), FRAME_KEY)
        opened = module.open(FRAME_KEY)
        self.assertEqual(opened.getpixel((0, 0)), (9, 8, 7))
        opened.close()
        self.assertEqual(module.open(FRAME_KEY).getpixel((0, 0)), (9, 8, 7))
        with self.assertRaises(AssertionError):
            module.open("/other.png")

    def test_overlay_stamp_shape(self):
        import numpy as np
        from PIL import ImageFont

        spec = {"title": "Defend the line", "subtitle": "Hold.", "press_field": "attack"}
        decision = {"turn": "stay", "press_label": "yes", "turn_confidence": 0.8, "press_confidence": 0.6}
        overlay = Overlay(spec, 1, decision, "note", 8, 4, lambda _size: ImageFont.load_default())
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        stamped = overlay.stamp(frame, {"kills": 1, "health": 90, "ammo": 10})
        self.assertEqual(stamped.shape, (4 + 56 + 28 + 70 + 22, 8, 3))
        self.assertEqual(overlay.stamp(frame, {"kills": 2, "health": 80, "ammo": 9}).shape, stamped.shape)


if __name__ == "__main__":
    unittest.main()
