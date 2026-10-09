"""Privacy-preserving object removal and topology repair for 3D scene graphs.

Given a ConceptGraphs-style scene graph -- objects with 3D bounding boxes and
directed edges carrying one of a fixed set of qualitative spatial relations --
this module removes a target object and repairs the resulting graph so that the
object is neither present nor mentioned, while the spatial relations among the
remaining objects are preserved.

Repair combines three mechanisms:

  * Edge redirection (`redirect_around`): relations that passed through the
    removed node are recovered by *composing* the two incident relations via a
    fixed composition table (`COMPOSE`), yielding a directly applicable relation
    between the two neighbours. The composition is complement-consistent and fully 
    reversible (see check_reversibility), and composing a relation with itself returns it. 
    The picked relation is the least specific one compatible with both input relations; 
    alternatives are listed in COMPOSE. Inferred edges are validated against a per-relation 
    distance ceiling derived from the scene's own geometry (`get_metrics`,
    `threshold_from_metrics`).

  * Support cascading / lowering (`remove_with_cascade`): objects that rested on
    the removed object are handled by one of two policies -- `remove`
    (cascade-delete them, matching the physical removal of the object including all objects on 
    top of it) or `lower` (keep them and drop them into the vacated position so the scene stays 
    physically plausible). Support stacks are moved rigidly (`lower_subtree`).

  * Residue scrubbing (`scrub_caption_mentions`, `downgrade_stale_above`):
    captions that name the removed object are replaced by a generic 'A/An <tag>.', 
    and relations that lowering has rendered unverifiable are conservatively softened.

Coordinate convention: the vertical axis is index `Y = 1` of `bbox_center`, and
it points DOWN (a smaller/more-negative y is higher). "Bottom face" is therefore
`center_y + extent/2`. All geometric reasoning assumes the bounding boxes are
axis-aligned with this world frame.

The graph is a `networkx.DiGraph`; each relation is stored once in a single
direction and read in either direction via `rel_dir` (flipping through
`COMPLEMENT`), to allow spatial reasoning regardless of edge orientation.

Usage
-----
As a library:
    from remove_object import run_removal
    summary = run_removal("objects.json", "edges.json", target=<object_id>,
                          out_objects="objects_after_edit.json",
                          out_edges="edges_after_edit.json",
                          dependent_policy=<policy>)

As a command-line tool:
    python remove_object.py objects.json edges.json <object_id> \\
        --policy <policy> --out-objects out_objects.json --out-edges out_edges.json

where <object_id> is the integer `id` of the object to remove, and <policy>
is one of:
    remove  -- redirect the removed object's relations and cascade-remove
               objects that rested on it (physically faithful removal).
    lower   -- redirect relations but KEEP dependent objects, lowering them
               into the vacated position instead of deleting them.
    naive   -- baseline: networkx's built-in node deletion only, with no
               redirection, cascading, lowering, or caption scrubbing.

To verify the composition table's reversibility property (a self-test, runs no
removal):
    python remove_object.py --check-table
"""

import argparse 
import json
import sys

import networkx as nx
import math

import re
import time


Y = 1  # index of the vertical component in bbox_center (y, pointing DOWN)

RELATIONS = ["on top of", "under", "next to", "in front of", "behind", "above", "below"]

COMPLEMENT = {
    "on top of": "under",
    "under": "on top of",
    "in front of": "behind",
    "behind": "in front of",
    "next to": "next to",
    "above": "below",
    "below": "above"
}

# Composition table for edge redirection.
#   COMPOSE[r1][r2] -> ordered list of candidate relations for the new edge
#   n1 -> n2, where r1 = rel(n1 -> n0) and r2 = rel(n0 -> n2) are the two
#   relations incident to the removed node n0.
#
# The current picking policy (`pick_primary`) always uses the first candidate;
# further entries reserve alternatives for custom pick policies.
# The table is fully reversible: composing a pair in either direction yields
# complementary relations (verified by check_reversibility), so a redirected
# edge is well-defined independent of which neighbour is taken as the source.
COMPOSE = {
    "next to": {
        "next to":     ["next to"],
        "under":       ["next to", "under"],
        "on top of":   ["next to", "on top of"],
        "in front of": ["next to", "in front of"],
        "behind":      ["next to", "behind"],
        "above":       ["next to", "above"],
        "below":       ["next to", "below"]
    },
    "under": {
        "next to":     ["next to"],
        "under":       ["under"],
        "on top of":   ["next to"],            
        "in front of": ["in front of"],
        "behind":      ["behind"],
        "above":       ["next to"],
        "below":       ["below"]
    },
    "on top of": {
        "next to":     ["next to"],            
        "under":       ["next to", "under", "on top of"],  
        "on top of":   ["on top of"],
        "in front of": ["in front of"],        
        "behind":      ["behind"],             
        "above":       ["above"],
        "below":       ["next to"]
    },
    "in front of": {
        "next to":     ["next to", "in front of"],
        "under":       ["in front of"],
        "on top of":   ["in front of"],
        "in front of": ["in front of"],
        "behind":      ["next to", "in front of", "behind"],
        "above":       ["next to", "above"],
        "below":       ["next to", "below"]
    },
    "behind": {
        "next to":     ["next to", "behind"],
        "under":       ["behind"],
        "on top of":   ["behind"],
        "in front of": ["next to", "in front of", "behind"],
        "behind":      ["behind"],
        "above":       ["next to", "above"],
        "below":       ["next to", "below"]
    },
    "above": {
            "next to":     ["next to", "above"],
            "under":       ["next to"],
            "on top of":   ["above"],
            "in front of": ["next to", "above"],
            "behind":      ["next to", "above"],
            "above":       ["above"],
            "below":       ["next to"]
        },
    "below": {
            "next to":     ["next to"],
            "under":       ["below"],
            "on top of":   ["next to"],
            "in front of": ["next to", "below"],
            "behind":      ["next to", "below"],
            "above":       ["next to"],
            "below":       ["below"]
        },
}

def rel_dir(G, src, dst):
    """Relationship from src -> dst, using whichever edge is stored.
    Prefers a real src->dst edge; else flips a stored dst->src edge via COMPLEMENT."""
    if G.has_edge(src, dst):
        return G.edges[src, dst]["relationship"]
    if G.has_edge(dst, src):
        return COMPLEMENT[G.edges[dst, src]["relationship"]]
    return None

def edge_id_dir(G, a, b):
    """edge_id of the edge between a and b in whichever direction it is stored (-1 if none)."""
    if G.has_edge(a, b):
        return G.edges[a, b]["edge_id"]
    if G.has_edge(b, a):
        return G.edges[b, a]["edge_id"]
    return -1

def load_json(path):
    """Load a JSON file, exiting with a clear message if it can't be read."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Could not parse JSON in {path}: {e}")

def _decimals(x):
    """Decimals in the shortest repr of float x (None for exponent notation, e.g. 1e-05)."""
    s = repr(float(x))
    if "e" in s or "E" in s:
        return None
    return len(s.split(".")[1]) if "." in s else 0

def build_scene_graph(objects, edges, add_missing_nodes=False):
    """Build a directed scene graph from ConceptGraphs objects and edges dicts.

    Nodes are keyed by integer object id; all other object fields become node
    attributes. Edges are added object_1_id -> object_2_id with their relation
    and metadata. Every original element is tagged `original_graph=True` so
    derived (redirected/lowered) elements can later be distinguished. The input's 
    coordinate precision (max decimals over all bbox_center components) is stored 
    as G.graph['coord_decimals'].

    Parameters
    ----------
    objects, edges : dict
        The "object_N"/"edge_N"-keyed dicts as loaded from the JSON files.
    add_missing_nodes : bool
        If False (default), an edge referencing an id not present in `objects`
        is skipped and recorded rather than silently creating a bare node.

    Returns
    -------
    (G, skipped, overwritten, max_node_id, max_edge_id)
        G            : the DiGraph.
        skipped      : edges dropped for referencing a missing object id.
        overwritten  : (u, v) pairs that collapsed onto an existing directed
                       edge (a DiGraph keeps one edge per ordered pair).
        max_node_id, max_edge_id : the largest ids seen, for allocating new ones.
    """
    
    G = nx.DiGraph()
    max_node_id = 0
    max_edge_id = 0

    for obj in objects.values():
        node_id = obj["id"]
        if node_id > max_node_id:
            max_node_id = node_id
        attrs = {k: v for k, v in obj.items() if k != "id"}
        attrs["original_graph"] = True 
        G.add_node(node_id, **attrs)

    # precision the input coordinates are written with, so lowered positions can be
    # rounded to match it (see lower_node); None if there is nothing to measure
    coords = [v for n in G.nodes for v in (G.nodes[n].get("bbox_center") or [])]
    decs = [d for d in map(_decimals, coords) if d is not None]
    G.graph["coord_decimals"] = max(decs) if decs else None

    skipped = []      # edges referencing a missing object id
    overwritten = []  # duplicate (u, v) pairs collapsed by the DiGraph

    for key, edge in edges.items():
        u = edge["object_1_id"]
        v = edge["object_2_id"]

        eid = edge.get("edge_id")
        if eid is not None and eid > max_edge_id:
            max_edge_id = eid

        if not add_missing_nodes and (u not in G or v not in G):
            skipped.append((key, u, v))
            continue

        if G.has_edge(u, v):
            overwritten.append((key, u, v))

        G.add_edge(
            u,
            v,
            edge_id=edge.get("edge_id"),
            edge_description=edge.get("edge_description"),
            relationship=edge.get("relationship"),
            num_detections=edge.get("num_detections"),
            original_graph=True,
        )

    return G, skipped, overwritten, max_node_id, max_edge_id

# the exact object fields present in the Before graph -- ONLY these are emitted
OBJECT_FIELDS = ("id", "object_tag", "object_caption",
                 "bbox_extent", "bbox_center", "bbox_volume")


def write_scene_graph(G, objects_path, edges_path, include_derived=True):
    """Inverse of build_scene_graph: write G back to objects/edges JSON files.

    Emits ONLY the fields that exist in the original Before schema
    (OBJECT_FIELDS for objects; the fixed edge fields below), so no internal
    bookkeeping attribute can leak into the output.
    """
    # ----- objects: emit only the whitelisted schema fields, id first -----
    objects = {}
    for i, (node_id, data) in enumerate(sorted(G.nodes(data=True)), start=1):
        if not include_derived and not data.get("original_graph", False):
            continue
        rec = {}
        for k in OBJECT_FIELDS:
            if k == "id":
                rec["id"] = node_id                 # the node key, not an attribute
            elif k in data:
                rec[k] = data[k]
        objects[f"object_{i}"] = rec

    # ----- edges: already a fixed field set -----
    edges = {}
    for i, (u, v, data) in enumerate(
            sorted(G.edges(data=True), key=lambda e: e[2].get("edge_id", -1)), start=1):
        if not include_derived and not data.get("original_graph", False):
            continue
        edges[f"edge_{i}"] = {
            "edge_id": data.get("edge_id"),
            "edge_description": data.get("edge_description"),
            "num_detections": data.get("num_detections"),
            "object_1_id": u,
            "object_1_tag": G.nodes[u].get("object_tag"),
            "object_2_id": v,
            "object_2_tag": G.nodes[v].get("object_tag"),
            "relationship": data.get("relationship"),
        }

    with open(objects_path, "w") as f:
        json.dump(objects, f, indent=2)
    with open(edges_path, "w") as f:
        json.dump(edges, f, indent=2)
    return objects, edges


def naive_removal(G, target): 
    """Baseline: delete the node and its incident edges only (networkx default).
    No redirection, no cascade, no lowering, no caption scrubbing.
    Used as the comparison baseline.
    """
    G.remove_node(target)
    return {"removed": [target], "steps": []}

def get_metrics(G):
    """Per-relation distance statistics between connected objects' bbox centers.

    For each relation in RELATIONS, computes the Euclidean distance between the
    bbox_center of the two nodes joined by every edge carrying that relation or its complement, and
    aggregates into mean, count, min, and max. Each edge counts for its relation 
    and its complement, so complementary relations share one distance statistic.

    Reads node attribute ``bbox_center`` (a 3D point) and edge attribute
    ``relationship``. Edges whose relationship is not in RELATIONS are ignored, and
    an edge is skipped if either endpoint lacks a bbox_center, so ``count`` reflects
    only contributing edges.

    Returns {relation: {"mean", "count", "min_value", "max_value"}}. For a relation
    with no usable edges, count is 0 and mean/min_value/max_value are None (rather
    than 0.0, which would read as a real zero-distance measurement).

    These feed threshold_from_metrics, which turns a chosen stat into a per-relation
    distance ceiling for validating redirected edges.
    """
    stats = {r: {"sum": 0.0, "count": 0, "min_value": None, "max_value": None}
             for r in RELATIONS}
    for u, v, d in G.edges(data=True):
        rel = d.get("relationship")
        if rel not in stats:
            continue
        c1 = G.nodes[u].get("bbox_center")
        c2 = G.nodes[v].get("bbox_center")
        if c1 is None or c2 is None:
            continue
        dist = math.dist(c1, c2)
        for r in {rel, COMPLEMENT[rel]}:   # "A below B" is "B above A": same pair, same distance
            s = stats[r]
            s["sum"] += dist
            s["count"] += 1
            s["min_value"] = dist if s["min_value"] is None else min(s["min_value"], dist)
            s["max_value"] = dist if s["max_value"] is None else max(s["max_value"], dist)
    out = {}
    for rel, s in stats.items():
        out[rel] = {
            "mean": (s["sum"] / s["count"] if s["count"] else None),
            "count": s["count"],
            "min_value": s["min_value"],
            "max_value": s["max_value"],
        }
    return out

def threshold_from_metrics(metrics, stat="mean", scale=1.2, custom=None):
    """Build a callable relation -> max allowed distance.
    A relation never observed in the scene (nor its complement) falls back to the largest
    value of stat over all observed relations. Returns None (no ceiling) only if no relation
    is observed at all, or if a custom dict omits the relation.

    stat: "mean" | "max" | "min" | "custom". For "custom", `custom` is a flat number
    or a {relation: number} dict and `metrics` is ignored. `scale` multiplies the result.
    """
    if stat == "custom":
        if custom is None:
            raise ValueError("stat='custom' requires a `custom` value or dict")
        def max_distance(relation):
            val = custom.get(relation) if isinstance(custom, dict) else custom
            return None if val is None else val * scale
        return max_distance
 
    key = {"mean": "mean", "max": "max_value", "min": "min_value"}[stat]
    observed = [m[key] for m in metrics.values() if m and m.get(key) is not None]
    fallback = max(observed) if observed else None   # for relations absent from the scene
    def max_distance(relation):
        m = metrics.get(relation)
        val = m.get(key) if m else None
        if val is None:
            val = fallback
        return None if val is None else val * scale
    return max_distance

def pick_primary(cands, c1, c2):
    """Default picker: always take the table's primary (first) candidate."""
    return cands[0]

# optionally add further policies for picking table entry

def check_reversibility(pick=pick_primary, verbose=True):
    """Check whether COMPOSE is reversible under the reverse-edge convention.

    For a neighbour pair around n0, the forward path n1->n0->n2 yields
    COMPOSE[r1][r2] and the reverse path n2->n0->n1 yields
    COMPOSE[COMPLEMENT[r2]][COMPLEMENT[r1]]. The two are edges in opposite
    directions, so they are consistent iff they are complements of each other:

        COMPLEMENT[ pick(COMPOSE[r1][r2]) ] == pick(COMPOSE[cr2][cr1])

    where cr1 = COMPLEMENT[r1], cr2 = COMPLEMENT[r2].

    `pick` selects the definitive relation from a candidate list; it is called
    as pick(cands, None, None) since reversibility does not depend on geometry.
    Returns True if the property holds for every (r1, r2) pair, else False and
    (if verbose) prints each offending pair.

    Runnable from the CLI via --check-table.
    """
    ok = True
    for r1 in COMPOSE:
        for r2 in COMPOSE[r1]:
            cr1, cr2 = COMPLEMENT[r1], COMPLEMENT[r2]

            fwd_cands = COMPOSE.get(r1, {}).get(r2)
            rev_cands = COMPOSE.get(cr2, {}).get(cr1)
            if not fwd_cands or not rev_cands:
                ok = False
                if verbose:
                    print(f"MISSING cell: ({r1}, {r2}) or its reverse "
                          f"({cr2}, {cr1}) has no candidates")
                continue

            fwd = pick(fwd_cands, None, None)          # relation n1 -> n2
            rev = pick(rev_cands, None, None)          # relation n2 -> n1

            if COMPLEMENT[fwd] != rev:
                ok = False
                if verbose:
                    print(f"NOT reversible: r1={r1!r}, r2={r2!r}  "
                          f"| forward(n1->n2)={fwd!r}  reverse(n2->n1)={rev!r}  "
                          f"| expected reverse={COMPLEMENT[fwd]!r}")
    if verbose:
        print("reversible" if ok else "NOT fully reversible")
    return ok

def downgrade_stale_above(G, lowered_nodes):
    """Conservatively soften 'above' relations invalidated by lowering.

    When an object is lowered, an 'above' relation to a neighbour that did NOT
    move with it may no longer hold (the drop can bring it to or below that
    neighbour's level). Such edges are downgraded to 'next to' -- the weakest
    in-vocabulary relation -- in place, whichever direction they are stored,
    and flagged geometry_downgraded=True; the edge_description is regenerated 
    ('<tag> next to <tag>') so it cannot contradict the new relation. The edge is kept 
    (it already passed the redirection distance check); only the relation is weakened.

    Only 'above' is affected: 'on top of'/'under'/'below' and the horizontal
    relations are preserved, because lowering drops an object into the removed
    object's vacated resting position, which keeps resting/contact and
    below-relations valid; a co-moved neighbour (in `lowered_nodes`) also keeps
    its relation and is skipped. This holds under the default drop-to-vacated-
    position lowering (not `ground_y`).

    Returns the list of (lowered_id, neighbour_id) pairs downgraded.
    """
    lowered = set(lowered_nodes)
    downgraded = []
    for L in lowered:
        neighbours = (set(G.successors(L)) | set(G.predecessors(L))) - {L}
        for M in neighbours:
            if M in lowered:
                continue                      # co-moved (e.g. cup on vase) -> relation preserved
            if rel_dir(G, L, M) == "above":   # L is above M (either storage direction)
                # rewrite the stored edge, whichever way it points
                u, v = (L, M) if G.has_edge(L, M) else (M, L)   # M->L is stored as "below"
                G.edges[u, v]["relationship"] = "next to"
                G.edges[u, v]["edge_description"] = (
                    f"{G.nodes[u].get('object_tag', u)} next to {G.nodes[v].get('object_tag', v)}")
                G.edges[u, v]["geometry_downgraded"] = True
                downgraded.append((L, M))
    return downgraded



def redirect_around(G, n0, exclude=frozenset(), max_distance=None, pick=pick_primary,
                    limit=1, summary=None):
    """Add redirected edges among the neighbours of n0; does NOT remove n0.

    For each source neighbour n1 (excluding `exclude`), the `limit` closest
    other neighbours not already connected to n1 are considered. For each such
    n2, the relation of the new edge n1 -> n2 is obtained by composing
    rel(n1 -> n0) and rel(n0 -> n2) via COMPOSE and applying `pick`. If a
    `max_distance` callable is given, the candidate edge is rejected when the
    two objects' centres are farther apart than the ceiling for that relation.

    Relations are read via `rel_dir`, so either stored edge direction is
    handled. New edges are tagged `original_graph=False`, `redirected=True`,
    and carry `via`, `source_edges`, and the full `candidates` list as
    provenance. Distances are read from the current geometry, so under the
    lowering policy the check reflects already-lowered positions.

    Parameters
    ----------
    exclude : set of node ids never used as an edge endpoint (e.g. dependents
        being cascade-removed).
    limit : maximum number of new edges added per source neighbour (default 1;
        a conservative cap that keeps the repaired graph as sparse as real
        scene graphs).
    summary : an existing summary dict to append to, or None to create one.

    Returns the summary dict with keys `added`, `skipped_distance`,
    `skipped_existing`, `missing_edge`.
    """
    if summary is None:
        summary = {"target": n0, "added": [], "skipped_distance": [],
                    "skipped_existing": [], "missing_edge": []}
    existing_ids = [d.get("edge_id", -1) for _, _, d in G.edges(data=True)]
    next_edge_id = (max(existing_ids) + 1) if existing_ids else 0
    neighbors = (set(G.successors(n0)) | set(G.predecessors(n0))) - {n0} - set(exclude)

    for n1 in sorted(neighbors):
        c1 = G.nodes[n1].get("bbox_center")
    
        # candidate targets: other neighbours NOT already connected to n1
        candidates_n2 = []
        for n2 in sorted(neighbors):
            if n2 == n1:
                continue
            if G.has_edge(n1, n2) or G.has_edge(n2, n1):        # already connected
                continue
            c2 = G.nodes[n2].get("bbox_center")
            if c1 is None or c2 is None:
                d = float("inf")                                # no geometry -> deprioritise
            else:
                d = math.dist(c1, c2)
            candidates_n2.append((d, n2))

        # X closest, nearest first
        candidates_n2.sort(key=lambda t: t[0])
        chosen = candidates_n2[:limit]                          # at most limit per source

        for _, n2 in chosen:
            r1 = rel_dir(G, n1, n0)
            r2 = rel_dir(G, n0, n2)
            if r1 is None or r2 is None:
                summary["missing_edge"].append((n1, n2)); continue
            cands = COMPOSE.get(r1, {}).get(r2)
            if not cands:
                summary["missing_edge"].append((n1, n2)); continue
            c2 = G.nodes[n2].get("bbox_center")
            rel = pick(cands, c1, c2)

            if max_distance is not None:
                thr = max_distance(rel)
                if thr is not None and c1 is not None and c2 is not None:
                    if math.dist(c1, c2) > thr:
                        summary["skipped_distance"].append((n1, n2, rel, math.dist(c1, c2), thr)); continue

            tag1 = G.nodes[n1].get("object_tag", n1)
            tag2 = G.nodes[n2].get("object_tag", n2)
            G.add_edge(n1, n2, edge_id=next_edge_id, relationship=rel,
                    edge_description=f"{tag1} {rel} {tag2}", num_detections=1,
                    original_graph=False, redirected=True, via=n0,
                    source_edges=[edge_id_dir(G, n1, n0), edge_id_dir(G, n0, n2)],
                    candidates=list(cands))
            summary["added"].append((n1, n2, rel, next_edge_id))
            next_edge_id += 1
    
    return summary



def remove_with_cascade(G, target, max_distance=None, pick=pick_primary,
                        support_rel="on top of",
                        dependent_policy="remove",     # "remove" | "lower"
                        extent_index=1, ground_y=None):
    """Remove `target` and repair the graph, cascading through the support stack.

    Processes a worklist starting at `target`. For each node it (1) finds its
    dependents -- every node resting on it via `support_rel`, read directionally
    with `rel_dir`; (2) applies `dependent_policy` (`remove` deletes them and
    queues them for their own removal; `lower` keeps and lowers them into the
    vacated position); (3) redirects the surviving neighbours among themselves, never involving a 
    node scheduled for removal (redirect_around); (4) under `lower`, conservatively downgrades any now-
    unverifiable 'above' relation of a lowered object (`downgrade_stale_above`);
    (5) scrubs captions naming the node; and (6) removes the node.

    Note: every object resting on a node is treated as a dependent, assuming objects
    only have one main support.

    Parameters
    ----------
    support_rel : the relation that defines "resting on" (default "on top of").
    dependent_policy : "remove" | "lower".
    extent_index, ground_y : geometry parameters passed to the lowering step.

    Returns {"removed": sorted ids, "steps": [per-node summary dicts]}.
    """
    if target not in G:
        raise KeyError(f"{target} not in graph")

    removed, queued, worklist, steps = set(), {target}, [target], []

    while worklist:
        n = worklist.pop()
        if n not in G or n in removed:
            continue

        # ---- find dependents: every node resting on n (single-support assumption) ----
        dependents = []
    
        # every node resting "on top of" n is a dependent
        for d in (set(G.successors(n)) | set(G.predecessors(n))) - {n}:
            if rel_dir(G, d, n) == support_rel: # also catches a stored "n under d"
                dependents.append(d)

        # ---- apply the policy to those dependents ----
        # "remove"  -> to_remove = dependents, nothing lowered
        # "lower"-> to_remove = [],         dependents lowered in place
        to_remove, geom = handle_dependents(
            G, n, dependents, dependent_policy,
            extent_index=extent_index, ground_y=ground_y, support_rel=support_rel,
        )

        # ---- exclude every node scheduled for removal (this step's AND pending ones) ----
        summary = redirect_around(G, n, exclude=set(to_remove) | queued,
                                  max_distance=max_distance, pick=pick)
        summary["target"] = n
        summary["cascaded_dependents"] = list(to_remove)
        summary["lowered"] = geom["lowered"]           
        # conservatively downgrade now-unverifiable 'above' relations of lowered objects
        lowered_ids = [nid for nid, _drop in geom["lowered"]]
        summary["above_downgraded"] = downgrade_stale_above(G, lowered_ids)    
        summary["caption_scrubbed"] = scrub_caption_mentions(G, n)
        steps.append(summary)

        G.remove_node(n)
        removed.add(n)

        # ---- queue to_remove, not dependents ----
        for d in to_remove:
            if d not in removed and d not in queued:
                queued.add(d)
                worklist.append(d)
    return {"removed": sorted(removed), "steps": steps}



def handle_dependents(G, n0, dependents, policy, extent_index=1,
                      ground_y=None, support_rel="on top of"):
    """Apply the dependent policy to the objects resting on n0.

    policy="remove": returns (dependents, {...}) so the caller cascade-removes
    them; nothing is moved.
    policy="lower" : keeps the dependents, lowering each -- together with the
    sub-stack resting on it (`lower_subtree`) -- so its bottom lands in n0's
    vacated position (`drop_amount`). Lowered nodes are tagged
    `unsupported=True` and `position_adjusted`. Returns ([], info).

    Returns
    -------
    (to_remove, info)
        to_remove : node ids the caller should delete and queue (empty under
                    the lower policy).
        info      : {"lowered": [(id, drop), ...]}.
    """
    if policy == "remove":
        return list(dependents), {"lowered": []}
    if policy != "lower":
        raise ValueError(f"unknown dependent policy {policy!r}; use 'remove' or 'lower'")

    lowered = []
    seen = set()                                              
    for d in dependents:
        if d in seen:                                         
            continue
        drop = drop_amount(G, d, n0, extent_index, ground_y)
        if drop is not None and drop > 0:
            moved = lower_subtree(G, d, drop, extent_index, support_rel, seen)   
            lowered.extend((m, round(drop, 3)) for m in moved)                   
        else:
            seen.add(d)
            G.nodes[d]["position_adjusted"] = False
        G.nodes[d]["unsupported"] = True
    return [], {"lowered": lowered}


def drop_amount(G, d, n0, extent_index=1, ground_y=None):
    """How far to lower dependent d when its support n0 is removed.

    1. If ground_y is given: drop d so its BOTTOM face rests at ground_y.
    2. Else: drop d so its BOTTOM face aligns with n0's previous BOTTOM face
       (the plane the removed object rested on). Correct even when d sits
       inside n0's bounding box.
    Returns None if geometry is unavailable.
    """
    cd = G.nodes[d].get("bbox_center")
    hd = vertical_size(G, d, extent_index)
    if cd is None or hd is None:
        return None
    bottom_d = cd[Y] + hd / 2.0          # d's bottom face (largest y)

    if ground_y is not None:
        return ground_y - bottom_d

    # align d's bottom to n0's previous bottom
    cn = G.nodes[n0].get("bbox_center")
    hn = vertical_size(G, n0, extent_index)
    if cn is None or hn is None:
        return None
    bottom_n0 = cn[Y] + hn / 2.0         # n0's bottom face
    return bottom_n0 - bottom_d

def vertical_size(G, n, extent_index=1):
    """Vertical size of node n's bbox, or None if unknown."""
    ext = G.nodes[n].get("bbox_extent")
    if not ext or extent_index >= len(ext):
        return None
    return ext[extent_index]

def lower_subtree(G, root, drop, extent_index=1, support_rel="on top of", seen=None):
    """Lower `root` and, transitively, every node resting on it by the SAME drop.

    When a supported object falls by `drop`, whatever sits on it must fall by the
    same amount to stay put, so the whole stack moves rigidly. Returns the list of
    nodes lowered; `seen` guards against cycles and double-lowering.
    """
    if seen is None:
        seen = set()
    lowered_nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        lower_node(G, node, drop, extent_index)
        lowered_nodes.append(node)
        for w in (set(G.successors(node)) | set(G.predecessors(node))) - {node}:
            if w not in seen and rel_dir(G, w, node) == support_rel:
                stack.append(w)          # w rests on node -> it moves with node
    return lowered_nodes

def lower_node(G, n, drop, extent_index=1):
    """Move node n DOWN by `drop` (y points down, so add), rounded to
    `G.graph['coord_decimals']` when set. In place."""
    c = G.nodes[n].get("bbox_center")
    if c is None or drop is None:
        return None
    new_c = list(c)
    new_y = c[Y] + drop
    k = G.graph.get("coord_decimals")                # match the input's precision (no float tails)
    new_c[Y] = round(new_y, k) if k is not None else new_y
    G.nodes[n]["bbox_center"] = new_c
    G.nodes[n]["lowered_by"] = round(G.nodes[n].get("lowered_by", 0.0) + drop, 6)
    G.nodes[n]["position_adjusted"] = True
    return new_c


def scrub_caption_mentions(G, removed_id):
    """Replace the caption of any OTHER surviving node that names the removed node's tag.

    For every node whose object_caption contains removed_id's object_tag, replace the whole caption 
    with 'A/An {own object_tag}', keeping a trailing period if the original had one. A node's own tag is ignored 
    when matching, so objects of the same (or a containing) category, such as another 
    chair or a 'coffee table', are not scrubbed for naming themselves. Must be called while
    removed_id is still in G (it reads removed_id's tag). Returns scrubbed ids.
    """
    removed_tag = G.nodes[removed_id].get("object_tag")
    if not removed_tag:
        return []
    pat = re.compile(r"\b" + re.escape(removed_tag) + r"\b", re.IGNORECASE)
    scrubbed = []
    for m in G.nodes:
        if m == removed_id:
            continue
        cap = G.nodes[m].get("object_caption")
        own = G.nodes[m].get("object_tag") or "object"
        if cap and pat.search(re.sub(r"\b" + re.escape(own) + r"\b", "", cap, flags=re.IGNORECASE)):
            art = "An" if own[:1].lower() in "aeiou" else "A"
            G.nodes[m]["object_caption"] = f"{art} {own}{'.' if cap.rstrip().endswith('.') else ''}"
            scrubbed.append(m)
    return scrubbed


def run_removal(objects_path, edges_path, target, out_objects, out_edges,
                dependent_policy="remove", threshold_stat="mean", threshold_scale=1.2,
                extent_index=1, ground_y=None):
    """Load a scene, remove `target`, write the repaired scene, and return a summary.

    Convenience entry point that wires the full pipeline together: load the JSON,
    build the graph, derive per-relation distance thresholds, run the chosen
    removal policy, and write the result back to JSON. Prints nothing (so it is
    safe to call in a loop / from other scripts); all human-facing output is the
    caller's responsibility.

    Parameters
    ----------
    objects_path, edges_path : str or Path
        Input objects / edges JSON files.
    target : int
        Integer id of the object to remove.
    out_objects, out_edges : str or Path
        Output paths for the repaired objects / edges JSON.
    dependent_policy : {"remove", "lower", "naive"}
        "remove" -- redirect relations and cascade-remove dependent objects;
        "lower"  -- redirect relations but keep dependents, lowering them into
                    the vacated position;
        "naive"  -- baseline: plain node deletion only (no redirection, cascade,
                    lowering, or caption scrubbing).
    threshold_stat : {"mean", "max", "min"}
        Which per-relation distance statistic forms the ceiling for validating
        redirected edges.
    threshold_scale : float
        Multiplier applied to that ceiling.
    extent_index, ground_y : geometry parameters passed to the lowering step
        (used only under the "lower" policy).

    Returns
    -------
    dict
        The removal summary, extended with elapsed_seconds (the removal itself), 
        elapsed_total_seconds (graph construction + threshold derivation + removal; 
        both exclude file I/O) and `skipped_edges` / `overwritten_edges`
        from graph construction.
    """
    if dependent_policy not in ("remove", "lower", "naive"):
        raise ValueError(f"unknown dependent_policy {dependent_policy!r}; "
                         "use 'remove', 'lower' or 'naive'")
    objects, edges = load_json(objects_path), load_json(edges_path)
    total_start = time.perf_counter()                # timer 2: build + thresholds + removal
    G, skipped, overwritten, _, _ = build_scene_graph(objects, edges)
    metrics = get_metrics(G)
    max_distance = threshold_from_metrics(metrics, stat=threshold_stat, scale=threshold_scale)

    start = time.perf_counter()                      # timer: wraps only the algorithm
    if dependent_policy == "naive":
        result = naive_removal(G, target)
    else:
        result = remove_with_cascade(G, target, max_distance=max_distance,
                                     dependent_policy=dependent_policy,
                                     extent_index=extent_index, ground_y=ground_y)
    result["elapsed_seconds"] = time.perf_counter() - start   # stop right after
    result["elapsed_total_seconds"] = time.perf_counter() - total_start

    write_scene_graph(G, out_objects, out_edges)     # write is outside the timed span
    result["skipped_edges"], result["overwritten_edges"] = skipped, overwritten
    return result


def main():
    p = argparse.ArgumentParser(
        description="Privacy-preserving scene-graph object removal.",
        epilog="Run 'python remove_object.py --check-table' to verify the reversibility "
               "of the composition table; this runs no removal.")
    p.add_argument("objects", help="objects JSON file of the input scene graph")
    p.add_argument("edges", help="edges JSON file of the input scene graph")
    p.add_argument("target", type=int, help="integer id of the object to remove")
    p.add_argument("--policy", choices=["remove", "lower", "naive"], default="remove",
                   help="remove: redirect relations and remove dependent objects (default); "
                        "lower: redirect relations and lower dependent objects; "
                        "naive: delete the node and its edges only")
    p.add_argument("--out-objects", default="objects_after_edit.json",
                   help="output objects file (default: %(default)s)")
    p.add_argument("--out-edges", default="edges_after_edit.json",
                   help="output edges file (default: %(default)s)")
    p.add_argument("--threshold-stat", choices=["mean", "max", "min"], default="mean",
                    help="distance statistic of the input relations used as ceiling "
                            "for redirected relations (default: %(default)s)")
    p.add_argument("--threshold-scale", type=float, default=1.2,
                    help="factor applied to the ceiling (default: %(default)s)")
    p.add_argument("--ground-y", type=float, default=None,
                   help="y coordinate at which lowered objects come to rest (lower policy); "
                        "by default they are lowered to the former bottom face of the "
                        "removed object")
    p.add_argument("--summary", default=None,
                   help="also write the removal summary to this JSON file; it names the "
                        "removed objects, so do not pass it on with the edited graph")
    args = p.parse_args()

    try:
        result = run_removal(args.objects, args.edges, args.target,
                             args.out_objects, args.out_edges,
                             dependent_policy=args.policy,
                             threshold_stat=args.threshold_stat,
                             threshold_scale=args.threshold_scale,
                             ground_y=args.ground_y)
    except (KeyError, nx.NetworkXError):
        if any(o.get("id") == args.target for o in load_json(args.objects).values()):
            raise                                    # some other fault: show it unchanged
        sys.exit(f"No object with id {args.target} in {args.objects}; nothing was written.")

    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(result, f, indent=2)

    print(f"Removed {len(result['removed'])} object(s): {result['removed']}")
    print(f"Algorithm time: {result['elapsed_seconds']*1000:.2f} ms")
    print(f"Wrote {args.out_objects} and {args.out_edges}")
    if args.summary:
        print(f"Wrote {args.summary}")


if __name__ == "__main__":
    # `--check-table` runs the composition-table reversibility self-test and exits;
    # any other invocation runs the normal removal CLI (see main / module docstring).
    if "--check-table" in sys.argv:
        check_reversibility()
    else:
        main()