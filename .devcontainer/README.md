# Agent development container

Open the repository in a Dev Container-compatible editor and choose **Reopen in
Container**. The image provides Python 3.12, `uv`, Node.js 20, Rust, build tools,
FFmpeg, and an editable Moshi Python installation.

The container is intentionally a constrained agent workspace:

- it runs commands as the non-root `vscode` user;
- Linux capabilities are dropped and privilege escalation is disabled;
- only this repository and the Moshi model cache are bind-mounted from the host;
- the host Docker socket and host home directory are not mounted.

Agents can change the repository because that is the purpose of the workspace
mount. They can write model files under `/mnt/data/huggingface_moshi/`, and can
access the network and NVIDIA GPUs for dependency downloads, model downloads,
and testing. Treat the container as a strong reduction in host exposure, not as
a security boundary for hostile code.

The container requires a host with Docker, NVIDIA Container Toolkit, and an
available NVIDIA GPU. Compose is deliberately not available through the host
Docker socket.

Before opening the container for the first time, make sure the cache exists and
is writable by your host user:

```bash
mkdir -p /mnt/data/huggingface_moshi
```

The first container creation installs Moshi and its CUDA/PyTorch dependencies,
so it can take some time. The container starts the model and customized web UI
automatically through `start-server.sh`. To launch it manually in the
foreground:

```bash
./start-server.sh
```

The launcher builds `client/dist` and serves it from the same Moshi process as
the `/api/chat` WebSocket. Port `8998` is published on the Docker host; port
`5173` is reserved for optional frontend-only development and is not part of
the normal launch path. Another LAN computer can use
`http://<docker-host-lan-ip>:8998`.

For browser microphone access, place trusted `cert.pem` and `key.pem` files in a
directory and run `MOSHI_SSL_DIR=/path/to/certs ./start-server.sh`. Then use
`https://<docker-host-lan-name-or-ip>:8998`. The issuing local CA must be trusted
by the browser computer.
