"""Quality-aware shot scheduling (Sec. 3.2, Fig. 2/3).

For each recurring entity, the shot with the highest predicted reference quality
qsrc(s, e) becomes its source shot; directed edges source -> other shots containing
the entity constrain generation order. Conflicting constraints can create cycles;
they are resolved by admitting constraints greedily in the order
(entity priority desc, predicted quality gain desc, narrative proximity asc) and
skipping any edge that would close a cycle — this implements the paper's
"preserve the constraints that better serve reference construction".
The paper leaves "quality gain" undefined; we use
gain(e) = qsrc(source, e) - qsrc(second-best shot, e), i.e. how much reference
quality would be lost if the source shot could not go first.
Topological sort breaks ties by narrative id, keeping the order as close to
narrative order as the constraints allow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import SchedulerConfig
from ..schema import ParsedScript

log = logging.getLogger("groundshot.sched")


@dataclass
class EntityConstraint:
    entity_id: str
    priority: int
    source_shot: int          # narrative shot id
    dependent_shots: list[int]
    gain: float


def build_constraints(script: ParsedScript, cfg: SchedulerConfig) -> list[EntityConstraint]:
    constraints = []
    for eid, ent in script.entities.items():
        appearances = [(s.shot_id, s.ref_quality.get(eid, 0.0))
                       for s in script.shots if eid in s.entity_ids]
        if len(appearances) < 2:
            continue  # nothing to reorder for single-appearance entities
        ranked = sorted(appearances, key=lambda x: (-x[1], x[0]))
        src, best_q = ranked[0]
        gain = best_q - ranked[1][1]
        deps = [sid for sid, _ in appearances if sid != src]
        # Skip reorderings that would not help: source already first in narrative
        # order needs no constraint edges to precede later shots — but edges are
        # still recorded so topo-sort respects them if other constraints reorder.
        constraints.append(EntityConstraint(eid, ent.priority, src, deps, gain))
    return constraints


def schedule(script: ParsedScript, cfg: SchedulerConfig) -> list[int]:
    """Return the generation order as a list of narrative shot ids."""
    n = len(script.shots)
    ids = [s.shot_id for s in script.shots]
    constraints = build_constraints(script, cfg)

    # Greedy edge admission in constraint-preference order.
    order_key = lambda c: (-c.priority, -c.gain, abs(c.source_shot - min(c.dependent_shots)))
    adj: dict[int, set[int]] = {i: set() for i in ids}

    def creates_cycle(u: int, v: int) -> bool:
        # would adding u->v create a cycle? i.e. is u reachable from v?
        stack, seen = [v], set()
        while stack:
            x = stack.pop()
            if x == u:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x])
        return False

    kept, dropped = 0, 0
    for c in sorted(constraints, key=order_key):
        for dep in c.dependent_shots:
            if creates_cycle(c.source_shot, dep):
                dropped += 1
                continue
            adj[c.source_shot].add(dep)
            kept += 1
    if dropped:
        log.info("cycle resolution dropped %d/%d constraint edges", dropped, kept + dropped)

    # Kahn topological sort, ready set ordered by narrative id (stay close to narrative).
    indeg = {i: 0 for i in ids}
    for u in adj:
        for v in adj[u]:
            indeg[v] += 1
    ready = sorted(i for i in ids if indeg[i] == 0)
    order: list[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
        ready.sort()
    assert len(order) == n, "topological sort failed (cycle left in graph)"
    log.info("generation order: %s", order)
    return order


def source_shots(script: ParsedScript, cfg: SchedulerConfig) -> dict[str, int]:
    """entity_id -> its selected reference-source shot (for risk-gated verification)."""
    return {c.entity_id: c.source_shot for c in build_constraints(script, cfg)}
