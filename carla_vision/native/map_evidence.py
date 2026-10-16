"""Read-only map identity for recorded research trajectories (no simulator import)."""

from __future__ import annotations

import hashlib
from typing import Any


def record_map_evidence(world: Any, writer: Any, episode_id: str) -> dict[str, Any]:
    """Retain the map description actually returned by the attached world.

    Older adapters/fakes without to_opendrive remain usable for non-geometric
    research. Missing evidence is explicit and cannot pass the trajectory audit.
    An available but failing API is not silently downgraded to an identity.
    """
    map_object = world.get_map()
    getter = getattr(map_object, "to_opendrive", None)
    if not callable(getter):
        return {"map_evidence_available": False, "opendrive_sha256": None}
    document = getter()
    if not isinstance(document, str) or not document.strip():
        raise ValueError("CARLA map returned an empty OpenDRIVE description")
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
    writer.add_auxiliary_json(
        f"maps/{episode_id}.opendrive.json",
        {
            "schema_version": "1.0",
            "map_name": str(map_object.name),
            "opendrive_sha256": digest,
            "opendrive": document,
            "source": "world.get_map().to_opendrive",
            "synthetic": False,
        },
        role="captured_map_description",
    )
    return {"map_evidence_available": True, "opendrive_sha256": digest}
