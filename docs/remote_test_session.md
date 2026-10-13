# Remote test session

**English** · [فارسی](remote_test_session_fa.md)

Run CARLA on a Windows server and test it together from another computer:
you in a browser, and a coding agent (or any script) through `carla-remote`.
There are two ways in. Use the LAN path whenever both computers share a
network; use the Cloudflare path when they do not.

| | LAN path | Cloudflare path |
| --- | --- | --- |
| Where the web app runs | The other computer (e.g. the Mac) | The Windows server |
| What crosses the network | The authenticated Worker, on the LAN only | HTTPS to your Cloudflare hostname, behind Cloudflare Access |
| Exposed to the internet | Nothing | Only the web app, and only to identities you allow |
| Camera quality | Full live MJPEG | Snapshots and best-effort video (Cloudflare buffers streams) |

The World Worker (8766) and CARLA (2000–2002) are never exposed to the
internet in either path.

## 1. Prepare the Windows server (both paths)

1. Install and set up as in the [user guide](user_guide.md#windows):
   `restore.ps1 … --role local`, then **Setup → Install / Repair**.
2. In **Run**, press **Start all** and wait for CARLA, Worker and Web to show
   **Ready**.

## 2a. LAN path

On the Windows server:

1. **Setup →** enable **Share Worker with another computer on this LAN**,
   choose the server's LAN address, press **Allow LAN in Windows Firewall…**,
   **Save settings**, then **Export connection**.
2. Copy the connection file to the other computer privately (it contains the
   Worker token).

On the other computer (e.g. the Mac):

1. Open the launcher, **Setup →** *Web app — connect to another CARLA
   computer*, enter the file path, **Import connection**, **Check connection**.
2. **Run → Start web app.** The console is at `http://127.0.0.1:8765/`.

`carla-remote` then works with no extra settings:

```bash
uv run carla-remote status
```

## 2b. Cloudflare path

You need a domain managed by Cloudflare. Everything below happens on the
Windows server unless it says otherwise.

### Create the tunnel

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create carla-garage
```

Create `%USERPROFILE%\.cloudflared\config.yml` (replace the UUID, user name
and hostname):

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: C:\Users\<YOU>\.cloudflared\<TUNNEL-UUID>.json

ingress:
  - hostname: carla.example.com
    service: http://127.0.0.1:8765
  - service: http_status:404
```

```powershell
cloudflared tunnel route dns carla-garage carla.example.com
cloudflared tunnel run carla-garage
```

The web app reads this file at startup and accepts `carla.example.com`
automatically because it points at the web app's loopback port. Restart the
web app (**Stop**, **Start all**) after changing the hostname. To run the
tunnel as a Windows service instead, run `cloudflared service install` and copy
`config.yml` and the credentials JSON to
`C:\Windows\System32\config\systemprofile\.cloudflared\`; the web app also
reads that location.

### Protect it with Cloudflare Access (required)

In the Cloudflare dashboard, **Zero Trust → Access controls → Applications →
Create new application → Self-hosted**, hostname `carla.example.com`, with two
policies:

1. **Allow**: *Emails* equal to your own address. This is you in the browser.
2. **Service Auth**: include the service token created under **Zero Trust →
   Access controls → Service credentials → Service Tokens** (for example
   `carla-remote-test`). Copy its Client ID and Client Secret when they are
   shown; the secret is displayed once.

Never add a **Bypass** policy. Without Access, anyone who finds the hostname
could drive the simulator and start research jobs.

### Give `carla-remote` the token (on the testing computer)

Create a private file; never paste the secret into a chat, issue or commit:

```bash
mkdir -p ~/.config/carla-vision
cat > ~/.config/carla-vision/remote.env <<'EOF'
CARLA_REMOTE_URL=https://carla.example.com
CF_ACCESS_CLIENT_ID=<client id>.access
CF_ACCESS_CLIENT_SECRET=<client secret>
EOF
chmod 600 ~/.config/carla-vision/remote.env
```

`carla-remote` refuses a file other users can read, sends the credentials only
to an `https://` URL, and never prints the web app's own request token.

## 3. Test together

Tell the agent the server is up. It can then run:

| Command | What it does |
| --- | --- |
| `uv run carla-remote status` | CARLA version and map, Worker, Garage, Drive and the last live check |
| `uv run carla-remote snapshot --view garage --out garage.jpg` | Saves the current Garage camera frame |
| `uv run carla-remote snapshot --view drive --out drive.jpg` | Saves the current driving camera frame |
| `uv run carla-remote live-check` | Runs the live CARLA check and prints the result (reloads Town10HD and drives a test car for about a minute) |

You can watch the same session in your browser at the console URL. Remote
driving over Cloudflare is for testing, not a release-qualified control path;
video is buffered and may lag.

## 4. After testing

- Stop the tunnel (`Ctrl+C`, or stop the `cloudflared` service).
- In Cloudflare, revoke or rotate the service token when the test period ends.
- Delete `~/.config/carla-vision/remote.env` on the testing computer if it is
  no longer needed.
