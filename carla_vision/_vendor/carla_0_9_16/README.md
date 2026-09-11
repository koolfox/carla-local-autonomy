# Official CARLA navigation agents

Unmodified BehaviorAgent, routing/controller helpers and their transitive source
dependencies from CARLA **0.9.16**, commit
`294096eb1c38eabf246e4f3a9cdab704e33a7f4c`:
https://github.com/carla-simulator/carla/tree/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/PythonAPI/carla/agents

MIT license: see `LICENSE`. Copyright belongs to the upstream authors.
This small source dependency is shipped because the CARLA wheel alone may not
provide the top-level `agents` package. No code is downloaded at runtime.
An already installed `agents` package takes precedence. The fallback is only
for the supported 0.9.16 runtime, not arbitrary UE5 development builds.

Dependencies: matching `carla`, `numpy`, `networkx`, `shapely`. Keep upstream
files unchanged; application integration belongs in `native/agent_support.py`.
When updating CARLA support, update this pinned source and its license together.
