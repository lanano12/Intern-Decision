"""Decision requests and ViZDoom button maps for the two recorded Doom demos.

The published clips show an image-driven policy. Each step answers two fields
in one forward pass: a three-way turn, and a yes/no button (attack or move).
The original instruction text was not released, so the wording here is the
schema those overlays imply, not a recovered training prompt.
"""

from __future__ import annotations

PRESS_BUTTONS = {"ATTACK": "attack", "MOVE_FORWARD": "move"}


SCENARIOS = {
    "defend": {
        "config": "defend_the_line.cfg",
        "title": "Defend the line",
        "subtitle": "Hold position against waves of enemies.",
        "press_field": "attack",
        "expected_buttons": ("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
        "state": {
            "task": "Defend the line. Hold position and shoot enemies coming down the hall.",
        },
        "questions": {
            "turn": {
                "type": "choice",
                "instructions": "Which way should the player turn?",
                "criteria": {
                    "left": "Turn left.",
                    "right": "Turn right.",
                    "stay": "Do not turn.",
                },
            },
            "attack": {
                "type": "noul",
                "instructions": "Should the player fire?",
                "criteria": {
                    "yes": "An enemy is in front of the gun.",
                    "no": "No enemy is in front of the gun.",
                },
            },
        },
    },
    "health": {
        "config": "health_gathering_supreme.cfg",
        "title": "Health gathering: supreme",
        "subtitle": "Navigate the arena and collect health packs.",
        "press_field": "move",
        "expected_buttons": ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
        "state": {
            "task": "Health gathering. The green floor is acid. Walk onto medkits, the white boxes with a green cross.",
        },
        "questions": {
            "turn": {
                "type": "choice",
                "instructions": "Which way should the player turn?",
                "criteria": {
                    "left": "Turn left, toward a medkit or an open path.",
                    "right": "Turn right, toward a medkit or an open path.",
                    "stay": "Do not turn. Forward is already the right direction.",
                },
            },
            "move": {
                "type": "noul",
                "instructions": "Should the player walk forward?",
                "criteria": {
                    "yes": "A medkit or open floor is ahead.",
                    "no": "A wall is immediately ahead, or the player should turn first.",
                },
            },
        },
    },
}


def button_name(button) -> str:
    return str(button).split(".")[-1]


def action_vector(button_names, turn: str, pressed: bool) -> list[int]:
    """Map one turn label and one yes/no press onto the scenario's button order."""
    if turn not in {"left", "right", "stay"}:
        raise ValueError(f"turn must be left, right, or stay, got {turn!r}")
    known = {spec["expected_buttons"] for spec in SCENARIOS.values()}
    if tuple(button_names) not in known:
        raise ValueError(f"unsupported ViZDoom buttons: {list(button_names)}")
    vector = []
    for name in button_names:
        if name == "TURN_LEFT":
            vector.append(int(turn == "left"))
        elif name == "TURN_RIGHT":
            vector.append(int(turn == "right"))
        elif name in PRESS_BUTTONS:
            vector.append(int(pressed))
        else:
            raise ValueError(f"unsupported ViZDoom button {name}")
    return vector


def chosen_label(answer: dict) -> str:
    probabilities = answer["probabilities"]
    return min(probabilities, key=lambda label: (-probabilities[label], label))


def parse_answers(answers: dict, press_field: str) -> dict:
    if "turn" not in answers or press_field not in answers:
        raise ValueError(f"response is missing turn or {press_field}")
    turn = chosen_label(answers["turn"])
    press = chosen_label(answers[press_field])
    if turn not in {"left", "right", "stay"}:
        raise ValueError(f"turn decision {turn!r} is not left, right, or stay")
    if press not in {"yes", "no"}:
        raise ValueError(f"{press_field} decision {press!r} is not yes or no")
    return {
        "turn": turn,
        "pressed": press == "yes",
        "press_label": press,
        "turn_confidence": answers["turn"]["probabilities"][turn],
        "press_confidence": answers[press_field]["probabilities"][press],
    }


def build_request(scenario: str, image_path: str, hud: dict | None = None) -> dict:
    spec = SCENARIOS[scenario]
    state = dict(spec["state"])
    if hud:
        state.update(hud)
    return {
        "state": state,
        "images": [image_path],
        "questions": spec["questions"],
    }
