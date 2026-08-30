# Run the Operator on a macOS LAN

The Operator defaults to loopback. To expose the Garage to trusted devices on the same private network, bind it explicitly to `0.0.0.0` or to the Mac's private LAN address.

## Recommended `.env.local`

```dotenv
CARLA_OPERATOR_BIND='0.0.0.0'
CARLA_OPERATOR_PORT='8765'
```

Keep the CARLA and optional World Worker values in the same `.env.local` as usual, then start:

```bash
uv run carla-operator-ui --open-browser
```

Find the Mac's Wi-Fi address:

```bash
ipconfig getifaddr en0
```

If Wi-Fi is not `en0`, inspect active interfaces with:

```bash
networksetup -listallhardwareports
```

From another device on the same LAN, open:

```text
http://<mac-lan-ip>:8765/
```

Example:

```text
http://192.168.1.25:8765/
```

The Operator accepts loopback, wildcard binds, RFC1918 IPv4, IPv4 link-local, IPv6 ULA, and IPv6 link-local addresses. Public/global IP binds and arbitrary hostnames are rejected.

This remains a trusted-LAN research service, not an internet-facing or multi-user application. Do not configure router port forwarding for port 8765. If macOS asks whether Python may accept incoming connections, allow it only on the trusted local network you intend to use.
