# WAN access through Cloudflare Tunnel

Use Cloudflare Tunnel for remote browser access to the macOS Operator. Do **not** bind the Operator directly to a public IP and do not port-forward TCP 8765 on the router.

## Security boundary

The intended topology is:

```text
Remote browser
    |
    | HTTPS + Cloudflare Access
    v
Cloudflare edge
    |
    | outbound Cloudflare Tunnel
    v
cloudflared on the Mac
    |
    | http://127.0.0.1:8765
    v
CARLA Operator / Garage
```

Keep the Operator on loopback:

```dotenv
CARLA_OPERATOR_BIND='127.0.0.1'
CARLA_OPERATOR_PORT='8765'
```

Cloudflare Access is required for the public hostname. The Operator's existing browser bootstrap token is an application request token, not a WAN identity boundary.

## Prerequisites

- a Cloudflare account
- a domain managed by Cloudflare for a normal public-hostname Tunnel
- `cloudflared` installed on the Mac
- the Operator already working locally at `http://127.0.0.1:8765/`

Install `cloudflared` on macOS:

```bash
brew install cloudflared
```

## Create a named tunnel

Authenticate and create a tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create carla-garage
```

The create command prints the Tunnel UUID and creates a credentials JSON under `~/.cloudflared/`.

Copy the repository example:

```bash
mkdir -p ~/.cloudflared
cp deploy/cloudflared/config.example.yml ~/.cloudflared/config.yml
```

Edit `~/.cloudflared/config.yml` and replace:

- `<TUNNEL-UUID>` with the created Tunnel UUID
- `<YOUR-USER>` with the macOS username
- `garage.example.com` with the selected public hostname

Create the DNS route:

```bash
cloudflared tunnel route dns carla-garage garage.example.com
```

Run it interactively first:

```bash
cloudflared tunnel run carla-garage
```

Then open:

```text
https://garage.example.com/
```

## Require Cloudflare Access

Before treating the hostname as usable WAN access, create a Cloudflare Zero Trust Access application for the complete hostname.

Dashboard path:

```text
Zero Trust -> Access controls -> Applications -> Create new application
```

Choose **Self-hosted and private**, add the public hostname, and configure an **Allow** policy for only the identities that should reach the Garage. Access applications deny unmatched users by default.

For a stronger origin boundary, enable the Tunnel option that protects the published application with Access / validates the Access token before forwarding the request to the local origin.

Do not add a Bypass policy for the Garage hostname.

## Run at login or boot on macOS

After the named tunnel and `~/.cloudflared/config.yml` are working:

```bash
cloudflared service install
```

This installs a per-user launch agent and starts the tunnel when that user logs in. To install it as a boot-time system daemon instead, use the root/service setup documented by Cloudflare and place the corresponding configuration under `/etc/cloudflared`.

## Start the Garage

Run the Operator normally and keep it on loopback:

```bash
uv run carla-operator-ui --open-browser
```

No `0.0.0.0` bind is needed for Cloudflare Tunnel.

## Streaming limitation

The current browser video transport is MJPEG (`multipart/x-mixed-replace`). Cloudflare documents that Tunnel HTTP responses are buffered by default unless the response is `text/event-stream`. Therefore WAN MJPEG should currently be treated as best-effort monitoring, not as a verified low-latency driving transport.

Remote manual/model actuation over WAN is not release-qualified. Use LAN/local driving for safety-sensitive control until the project gains and validates a WAN-friendly streaming/control transport (for example a dedicated WebSocket/WebRTC path) with measured latency, stale-frame behavior, and fail-closed control tests.

## Never expose these directly

Do not publish or port-forward:

- CARLA RPC `2000`
- Traffic Manager `8000`
- World Worker `8766`
- Operator `8765` directly from the router

Only the Cloudflare-protected HTTP hostname should be reachable from the Internet.
