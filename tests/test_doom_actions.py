"""Button mapping and request shape for the ViZDoom player. No game, no weights."""

import unittest

from src.doom.schema import action_vector, build_request, parse_answers


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


if __name__ == "__main__":
    unittest.main()
