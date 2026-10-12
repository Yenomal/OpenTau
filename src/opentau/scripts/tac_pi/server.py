"""OpenTau server speaking the existing UniVTAC/OpenPI WebSocket protocol."""

import argparse
import asyncio
import logging
import sys
import traceback
from dataclasses import asdict
from pprint import pformat
from typing import Any

import numpy as np
import torch

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("The UniVTAC server requires the `websockets` package.") from exc

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.policies.utils import maybe_compile_sample_actions, to_dtype_preserving_siglip_float32
from opentau.scripts.tac_pi.protocol import Packer, packb, unpackb
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import INFERENCE_STATE_DTYPE, auto_torch_device, init_logging

logger = logging.getLogger(__name__)

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8008
STATE_DIM = 9
ACTION_DIM = 9
PROMPT_MODE = "fixed"
FIXED_PROMPT = "Empty"
TORCH_COMPILE = False
WARMUP = True
PROTOCOL = "univtac-openpi-v1"


def _parse_server_args() -> None:
    """Parse server-only flags and remove them before OpenTau parses the config."""
    global SERVER_HOST, SERVER_PORT, STATE_DIM, ACTION_DIM
    global PROMPT_MODE, FIXED_PROMPT, TORCH_COMPILE, WARMUP

    def _bool(value: str) -> bool:
        return value.lower() in {"1", "true", "yes", "y"}

    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--server_host", default=None)
    arg_parser.add_argument("--server_port", type=int, default=None)
    arg_parser.add_argument("--state_dim", type=int, default=None)
    arg_parser.add_argument("--action_dim", type=int, default=None)
    arg_parser.add_argument("--prompt_mode", choices=("empty", "client", "fixed"), default=None)
    arg_parser.add_argument("--fixed_prompt", default=None)
    arg_parser.add_argument("--torch_compile", default=None)
    arg_parser.add_argument("--warmup", default=None)
    args, remaining = arg_parser.parse_known_args(sys.argv[1:])

    if args.server_host is not None:
        SERVER_HOST = args.server_host
    if args.server_port is not None:
        SERVER_PORT = args.server_port
    if args.state_dim is not None:
        STATE_DIM = args.state_dim
    if args.action_dim is not None:
        ACTION_DIM = args.action_dim
    if args.prompt_mode is not None:
        PROMPT_MODE = args.prompt_mode
    if args.fixed_prompt is not None:
        FIXED_PROMPT = args.fixed_prompt
    if args.torch_compile is not None:
        TORCH_COMPILE = _bool(args.torch_compile)
    if args.warmup is not None:
        WARMUP = _bool(args.warmup)
    if STATE_DIM != 9 or ACTION_DIM != 9:
        raise ValueError("The existing UniVTAC client requires 9-D state and 9-D model actions")
    sys.argv = [sys.argv[0], *remaining]


def _prompt_for_request(prompt: Any) -> str:
    raw_prompt = "" if prompt is None else str(prompt)
    if PROMPT_MODE == "empty":
        return ""
    if PROMPT_MODE == "fixed":
        return FIXED_PROMPT
    return raw_prompt.strip()


def _as_image_tensor(image: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"UniVTAC image must be HWC RGB, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.array(array, dtype=np.uint8, order="C", copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype).div_(255.0)


def build_batch(
    cfg: TrainPipelineConfig,
    payload: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | list[str]]:
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError("UniVTAC request must contain an 'images' dictionary")
    for key in ("cam_side", "cam_wrist"):
        if key not in images:
            raise ValueError(f"UniVTAC request is missing images[{key!r}]")

    if "state" not in payload:
        raise ValueError("UniVTAC request is missing 'state'")
    state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"UniVTAC state must have exactly {STATE_DIM} values, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("UniVTAC state contains non-finite values")
    state_width = int(cfg.max_state_dim)
    if state_width < STATE_DIM:
        raise ValueError(f"OpenTau max_state_dim={state_width} is smaller than UniVTAC state dim={STATE_DIM}")
    padded_state = np.zeros((state_width,), dtype=np.float32)
    padded_state[:STATE_DIM] = state

    return {
        "camera0": _as_image_tensor(images["cam_side"], device, dtype),
        "camera1": _as_image_tensor(images["cam_wrist"], device, dtype),
        "state": torch.from_numpy(padded_state[None]).to(device=device, dtype=INFERENCE_STATE_DTYPE),
        "prompt": [_prompt_for_request(payload.get("prompt", ""))],
        "img_is_pad": torch.zeros((1, 2), dtype=torch.bool, device=device),
    }


class UniVTACPolicyRunner:
    """Load an OpenTau policy and expose one full action chunk per request."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg
        self.device = auto_torch_device()
        self.dtype = torch.bfloat16
        if cfg.policy.type != "pi05":
            raise ValueError(f"UniVTAC baseline server expects policy.type='pi05', got {cfg.policy.type!r}")
        if not cfg.policy.pretrained_path:
            raise ValueError("Set policy.pretrained_path to a trained OpenTau checkpoint")
        if cfg.num_cams != 2:
            raise ValueError(f"UniVTAC server expects num_cams=2, got {cfg.num_cams}")
        if cfg.policy.chunk_size != 50:
            raise ValueError(
                f"UniVTAC server expects a 50-step checkpoint, got chunk_size={cfg.policy.chunk_size}"
            )
        if cfg.seed is not None:
            set_seed(cfg.seed)

        policy_class = get_policy_class(cfg.policy.type)
        logger.info("Loading OpenTau %s policy from %s", cfg.policy.type, cfg.policy.pretrained_path)
        self.policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
        to_dtype_preserving_siglip_float32(self.policy, device=self.device, dtype=self.dtype)
        self.policy.eval()

        if TORCH_COMPILE:
            self.policy.model.sample_actions = maybe_compile_sample_actions(
                self.policy, self.policy.model.sample_actions, device_hint=self.device
            )
        self.policy.reset()

        if WARMUP:
            self._warmup()

    def _warmup(self) -> None:
        height, width = self.cfg.resolution
        payload = {
            "images": {
                "cam_side": np.zeros((height, width, 3), dtype=np.uint8),
                "cam_wrist": np.zeros((height, width, 3), dtype=np.uint8),
            },
            "state": np.zeros((STATE_DIM,), dtype=np.float32),
            "prompt": "",
        }
        with torch.inference_mode():
            actions = self.policy.sample_actions(build_batch(self.cfg, payload, self.device, self.dtype))
        self._validate_actions(actions)
        self.policy.reset()
        logger.info("OpenTau UniVTAC warmup complete; output shape=%s", tuple(actions.shape))

    @staticmethod
    def _validate_actions(actions: torch.Tensor) -> np.ndarray:
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"OpenTau policy must return shape (1, T, D), got {tuple(actions.shape)}")
        if actions.shape[1] != 50 or actions.shape[2] < ACTION_DIM:
            raise ValueError(
                f"OpenTau policy returned shape {tuple(actions.shape)}, expected (1, 50, D >= {ACTION_DIM})"
            )
        result = actions[0, :, :ACTION_DIM].to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(result).all():
            raise ValueError("OpenTau policy returned non-finite actions")
        return np.asarray(result, dtype=np.float32)

    def infer(self, payload: dict[str, Any]) -> np.ndarray:
        with torch.inference_mode():
            actions = self.policy.sample_actions(build_batch(self.cfg, payload, self.device, self.dtype))
        result = self._validate_actions(actions)
        return result


def _response(actions: np.ndarray) -> bytes:
    return packb({"actions": np.asarray(actions, dtype=np.float32)})


async def _handler(websocket: Any, runner: UniVTACPolicyRunner) -> None:
    packer = Packer()
    await websocket.send(
        packer.pack(
            {
                "protocol": PROTOCOL,
                "backend": "opentau-pytorch",
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "action_steps": 50,
                "camera_keys": ["cam_side", "cam_wrist"],
            }
        )
    )
    logger.info("UniVTAC client connected: %s", getattr(websocket, "remote_address", None))
    runner.policy.reset()
    try:
        async for message in websocket:
            try:
                payload = unpackb(message)
                if not isinstance(payload, dict):
                    raise ValueError("UniVTAC request must be a dictionary")
                await websocket.send(_response(runner.infer(payload)))
            except Exception:
                logger.exception("UniVTAC inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="OpenTau inference failed")
                return
    except websockets.exceptions.ConnectionClosed:
        logger.info("UniVTAC client disconnected")
    finally:
        runner.policy.reset()


async def _serve(runner: UniVTACPolicyRunner) -> None:
    async with websockets.serve(
        lambda websocket: _handler(websocket, runner),
        SERVER_HOST,
        SERVER_PORT,
        compression=None,
        max_size=None,
        ping_timeout=None,
    ):
        logger.info("OpenTau UniVTAC server listening on ws://%s:%d", SERVER_HOST, SERVER_PORT)
        await asyncio.Future()


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    logger.info(
        "Starting OpenTau UniVTAC server with config:\n%s\nserver options: host=%s port=%s prompt_mode=%s compile=%s warmup=%s",
        pformat(asdict(cfg)),
        SERVER_HOST,
        SERVER_PORT,
        PROMPT_MODE,
        TORCH_COMPILE,
        WARMUP,
    )
    asyncio.run(_serve(UniVTACPolicyRunner(cfg)))


if __name__ == "__main__":
    _parse_server_args()
    init_logging()
    main()
