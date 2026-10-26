"""OpenPI server that accepts explicit per-inference Gaussian action noise."""
import dataclasses
import logging
import socket

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as train_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint: str
    port: int = 8000


class ExplicitNoisePolicy:
    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        obs = dict(obs)
        noise = np.asarray(obs.pop("_policy_noise"), dtype=np.float32)
        if noise.shape != (10, 32) or not np.isfinite(noise).all():
            raise ValueError(f"invalid explicit noise: {noise.shape}")
        return self._policy.infer(obs, noise=noise)


def main(args: Args):
    policy = policy_config.create_trained_policy(
        train_config.get_config(args.config), args.checkpoint,
        sample_kwargs={"num_steps": 10})
    wrapped = ExplicitNoisePolicy(policy)
    host = socket.gethostname()
    logging.info("Creating seeded server (host=%s, config=%s, checkpoint=%s)",
                 host, args.config, args.checkpoint)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port,
        metadata={"config": args.config, "checkpoint": args.checkpoint,
                  "explicit_noise": True, "flow_steps": 10},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
