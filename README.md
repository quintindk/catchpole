# Catchpole

A LiteLLM-proxy-based hybrid router that decides, per request, whether to
serve a completion from a **local** model (LM Studio) or a **cloud** model
(GitHub Copilot). Simple requests stay local; complex ones get escalated.

The custom provider `catchpole/auto` is a thin `CustomLLM` that:

1. Asks the local model to classify the incoming request as `LOCAL` or `CLOUD`.
2. Forwards the original request to the chosen backend via LiteLLM.

## Layout

| File | Purpose |
| --- | --- |
| `catchpole.py` | The custom LiteLLM provider (router logic). |
| `litellm_config.yaml` | LiteLLM proxy model list + provider registration. |
| `docker-compose.yaml` | Runs the LiteLLM proxy with the router mounted in. |
| `requirements.txt` | Python deps (for running the router outside Docker). |
| `.env.example` | Template for required secrets. |

## Setup

1. Copy the env template and fill in your secrets:

   ```sh
   cp .env.example .env
   # edit .env: set LM_STUDIO_API_KEY and (optionally) LITELLM_MASTER_KEY
   ```

2. Make sure LM Studio is running on the host at `http://localhost:1234/v1`
   and is serving the model referenced by `CATCHPOLE_LOCAL_MODEL`
   (default `openai/google/gemma-4-26b-a4b`).

3. Bring it up:

   ```sh
   docker compose up -d
   ```

   The proxy listens on `http://localhost:4000`.

4. First time only — authorize the GitHub Copilot provider inside the
   container so the OAuth token gets cached in the `copilot-auth` volume.

## Usage

Point any OpenAI-compatible client at `http://localhost:4000` with the
`LITELLM_MASTER_KEY` as the API key, and use `catchpole/auto` as the model:

```sh
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "catchpole/auto",
    "messages": [{"role": "user", "content": "write a fizzbuzz in python"}]
  }'
```

You can also target a specific model directly (e.g.
`github_copilot/claude-sonnet-4.6`, `lm_studio/local`) — see
`litellm_config.yaml` for the full list. `lm_studio/local` is an alias
that resolves to whatever model id you set in `CATCHPOLE_LOCAL_MODEL`.

## Configuration

Environment variables consumed by the router (set in `.env` /
`docker-compose.yaml`):

| Var | Default | Description |
| --- | --- | --- |
| `LITELLM_MASTER_KEY` | `sk-catchpole-dev` | Auth key for the proxy. |
| `LM_STUDIO_API_KEY` | — | API key for LM Studio. |
| `CATCHPOLE_LOCAL_API_BASE` | `http://localhost:1234/v1` | LM Studio endpoint. |
| `CATCHPOLE_LOCAL_MODEL` | `openai/google/gemma-4-26b-a4b` | Local model id. |
| `CATCHPOLE_ROUTER_MODEL` | `$CATCHPOLE_LOCAL_MODEL` | Model used to make routing decisions. |
| `CATCHPOLE_CLOUD_MODEL` | `github_copilot/claude-sonnet-4.6` | Cloud fallback model. |
