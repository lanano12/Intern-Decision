"""Play the recorded Doom demos: ViZDoom frames in, Intern-Decision buttons out.

The two scenarios match the README clips.

  defend   defend_the_line           turn left/right/stay + attack
  health   health_gathering_supreme  turn left/right/stay + move

One checkpoint forward pass scores both fields. The game then holds that
button mask for ``--frame-skip`` tics (default 4, the usual ViZDoom step).
Inference time is not part of the saved video, same as the published clips.

Requires the ViZDoom Python package and a local Intern-Decision checkpoint
directory (tokenizer, processor, and weights). The 4B weights are
``internlm/Intern-Decision-4B``. The player loads ``inference.py`` from that
directory so the checkpoint's own temperature is applied.

  .venv-doom/bin/python -m src.doom.play --scenario both
  .venv-doom/bin/python -m src.doom.play --scenario defend --full-episode
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.doom.schema import SCENARIOS, action_vector, build_request, button_name, parse_answers

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "Intern-Decision-4B"


def load_engine(checkpoint: Path, device: str, dtype: str, attn: str):
    module_path = checkpoint / "inference.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint} has no inference.py. Download internlm/Intern-Decision-4B into this directory."
        )
    spec = importlib.util.spec_from_file_location("intern_decision_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DecisionEngine(
        checkpoint=str(checkpoint),
        device=device,
        dtype=dtype,
        attn_implementation=attn,
    )


def screen_hwc(buffer) -> np.ndarray:
    frame = np.array(buffer, copy=True)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def read_hud(state, names: list[str]) -> dict:
    values = getattr(state, "game_variables", None)
    if values is None:
        return {}
    hud = {}
    for name, value in zip(names, list(values)):
        hud[name] = int(value)
    return hud


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


class Recorder:
    """Stream composited RGB frames to an H.264 file at the Doom tic rate."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 35):
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.path = path
        self.size = size

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1] or frame.shape[2] != 3:
            raise ValueError(f"frame {frame.shape} does not match {self.size}")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code} while writing {self.path}")


def composite(game_rgb: np.ndarray, spec: dict, episode: int, hud: dict, decision: dict | None, note: str) -> np.ndarray:
    game = Image.fromarray(game_rgb, mode="RGB")
    width = game.width
    top, status_h, decision_h, footer_h = 56, 28, 70, 22
    canvas = Image.new("RGB", (width, top + game.height + status_h + decision_h + footer_h), (10, 16, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    body_font = load_font(16)
    small_font = load_font(14)
    draw.text((16, 8), spec["title"], fill=(244, 246, 250), font=title_font)
    draw.text((16, 34), spec["subtitle"], fill=(186, 196, 210), font=small_font)
    canvas.paste(game, (0, top))
    y = top + game.height
    draw.rectangle((0, y, width, y + status_h), fill=(16, 42, 64))
    status = f"Episode {episode}    Kills {hud.get('kills', 0)}    Health {hud.get('health', 0)}"
    if "ammo" in hud:
        status += f"    Ammo {hud['ammo']}"
    draw.text((16, y + 5), status, fill=(232, 238, 244), font=body_font)
    y += status_h
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
    draw.text((16, canvas.height - 18), note, fill=(150, 162, 176), font=small_font)
    return np.asarray(canvas)


def open_game(config_path: Path, visible: bool, sound: bool, seed: int | None):
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(str(config_path))
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_window_visible(visible)
    game.set_sound_enabled(sound)
    variables = [vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.AMMO2]
    game.set_available_game_variables(variables)
    if seed is not None:
        game.set_seed(seed)
    try:
        game.init()
    except Exception:
        if not visible:
            raise
        game.set_window_visible(False)
        game.init()
        visible = False
    buttons = tuple(button_name(button) for button in game.get_available_buttons())
    return game, buttons, visible


def scripted_action(scenario: str, step: int) -> dict:
    """A button-basher used to prove the game loop without loading weights."""
    if scenario == "defend":
        turn = "right" if step % 8 < 4 else "left"
        return {"turn": turn, "pressed": True, "press_label": "yes", "turn_confidence": 1.0, "press_confidence": 1.0}
    turn = "stay" if step % 6 else "right"
    return {"turn": turn, "pressed": True, "press_label": "yes", "turn_confidence": 1.0, "press_confidence": 1.0}


def play_episode(game, scenario: str, buttons, engine, args, episode: int, recorder, trace) -> dict:
    spec = SCENARIOS[scenario]
    frame_dir = Path(args.output) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"{scenario}.png"
    note = "Scripted buttons | game-time playback" if engine is None else (
        "Image-driven Intern-Decision | game-time playback; inference pauses omitted"
    )
    game.new_episode()
    total_reward = 0.0
    decisions = 0
    infer_ms = []
    last = None
    variable_names = ["health", "kills", "ammo"]

    while not game.is_episode_finished():
        if args.max_decisions and decisions >= args.max_decisions:
            break
        state = game.get_state()
        if state is None or state.screen_buffer is None:
            break
        hud = read_hud(state, variable_names)
        screen = screen_hwc(state.screen_buffer)
        Image.fromarray(screen, mode="RGB").save(frame_path)
        started = time.perf_counter()
        if engine is None:
            decision = scripted_action(scenario, decisions)
            elapsed = (time.perf_counter() - started) * 1000
        else:
            request = build_request(scenario, str(frame_path), hud if args.hud_state else None)
            result = engine.predict(request)
            decision = parse_answers(result["answers"], spec["press_field"])
            elapsed = float(result.get("timing", {}).get("inference_ms", (time.perf_counter() - started) * 1000))
        infer_ms.append(elapsed)
        vector = action_vector(buttons, decision["turn"], decision["pressed"])
        decisions += 1
        record = {
            "scenario": scenario,
            "episode": episode,
            "decision": decisions,
            "hud": hud,
            "turn": decision["turn"],
            "press_field": spec["press_field"],
            "press": decision["press_label"],
            "turn_confidence": round(decision["turn_confidence"], 4),
            "press_confidence": round(decision["press_confidence"], 4),
            "action": vector,
            "inference_ms": round(elapsed, 2),
        }
        print(
            f"{scenario} #{decisions:03d}  turn={decision['turn']:5s} {decision['turn_confidence'] * 100:4.0f}%"
            f"  {spec['press_field']}={decision['press_label']:3s} {decision['press_confidence'] * 100:4.0f}%"
            f"  hp={hud.get('health', '?'):>3}  kills={hud.get('kills', '?'):>3}  {elapsed:7.0f} ms",
            flush=True,
        )
        for _ in range(args.frame_skip):
            if game.is_episode_finished():
                break
            total_reward += float(game.make_action(vector, 1))
            held = game.get_state()
            if recorder is not None and held is not None and held.screen_buffer is not None:
                held_hud = read_hud(held, variable_names)
                recorder.write(composite(screen_hwc(held.screen_buffer), spec, episode, held_hud, decision, note))
        record["reward_so_far"] = total_reward
        trace.write(json.dumps(record) + "\n")
        trace.flush()
        last = record

    summary = {
        "scenario": scenario,
        "episode": episode,
        "decisions": decisions,
        "reward": total_reward,
        "final": last,
        "mean_inference_ms": round(sum(infer_ms) / len(infer_ms), 2) if infer_ms else None,
    }
    return summary


def run(args):
    import vizdoom as vzd

    args.output.mkdir(parents=True, exist_ok=True)
    engine = None
    if args.policy == "model":
        print(f"Loading {args.checkpoint} on {args.device} ({args.dtype})", flush=True)
        engine = load_engine(args.checkpoint, args.device, args.dtype, args.attn)
    scenarios = list(SCENARIOS) if args.scenario == "both" else [args.scenario]
    summaries = []
    for scenario in scenarios:
        spec = SCENARIOS[scenario]
        config_path = Path(vzd.scenarios_path) / spec["config"]
        game, buttons, visible = open_game(config_path, not args.headless, args.sound, args.seed)
        if buttons != spec["expected_buttons"]:
            game.close()
            raise RuntimeError(f"{spec['config']} buttons {buttons} != {spec['expected_buttons']}")
        print(
            f"{spec['title']}: buttons={buttons} window={'on' if visible else 'off'} "
            f"skip={args.frame_skip} decisions={args.max_decisions or 'episode'}",
            flush=True,
        )
        video_path = args.output / f"{scenario}.mp4"
        recorder = None if args.no_video else Recorder(video_path, (640, 56 + 480 + 28 + 70 + 22))
        trace_path = args.output / f"{scenario}.jsonl"
        try:
            with trace_path.open("w", encoding="utf-8") as trace:
                summary = play_episode(game, scenario, buttons, engine, args, 1, recorder, trace)
            summaries.append(summary)
        finally:
            if recorder is not None:
                recorder.close()
            game.close()
        print(
            f"{scenario} done: decisions={summary['decisions']} reward={summary['reward']:.1f} "
            f"mean_inference_ms={summary['mean_inference_ms']}",
            flush=True,
        )
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("defend", "health", "both"), default="both")
    parser.add_argument("--policy", choices=("model", "scripted"), default="model")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn", default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=160, help="Cap decisions per episode. 0 plays until Doom ends the episode.")
    parser.add_argument("--full-episode", action="store_true", help="Ignore --max-decisions and play until timeout or death.")
    parser.add_argument("--hud-state", action="store_true", help="Also pass health, kills, and ammo in the JSON state.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sound", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.full_episode:
        args.max_decisions = 0
    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")
    if args.output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.output = ROOT / "outputs" / "doom" / f"{args.scenario}-{args.policy}-{stamp}"
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
