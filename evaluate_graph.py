"""Evaluate an edited 3D scene graph after an object removal. JSON-only, self-contained.

Takes the scene graph before the removal, the edited graph and the id of the
removed object, and writes two CSVs:

  evaluation_results_structural.csv   (no API)
  evaluation_results_llm.csv          (OpenAI, temp 0)

Structural metrics use the graph before the removal as reference. The LLM judge
is shown the edited graph ONLY, exactly as written (objects and edges, all
fields). Each assessment consists of two independent calls:

  A  plausibility  -- semantic and physical plausibility (0-10, each with a
                      one-sentence reasoning); removal is never mentioned.
  B  removal       -- concrete evidence, likelihood that an object was removed
                      (0-10), the missing object if the judge believes one was
                      removed, and the evidence types used.

One assessment by default; --assessments N repeats it N times on the same graph.

Edge accounting (new / changed / lost edges) is direction-agnostic: an edge
counts as the same pair whichever way it is stored, and a relation stored the
other way round is compared through its complement (A below B == B above A).

    python evaluate_graph.py --before-objects objects.json --before-edges edges.json \\
        --edited-objects objects_after_edit.json --edited-edges edges_after_edit.json \\
        --target 57                                   # structural + LLM
    python evaluate_graph.py ... --structural-only    # no OpenAI calls
    python evaluate_graph.py --qualitative-only \\
        --edited-objects objects_after_edit.json --edited-edges edges_after_edit.json

The LLM judge needs the `openai` package and an API key in the environment
variable OPENAI_API_KEY.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import re

# ---- geometry conventions (must match the algorithm) ----
Y = 1
EXTENT_INDEX = 1
TOL = 0.05
SUPPORT_REL = "on top of"
COMPLEMENT = {"on top of": "under", "under": "on top of", "in front of": "behind",
              "behind": "in front of", "next to": "next to", "above": "below", "below": "above"}

# ---- defaults ----
MODEL = "gpt-5.4-mini"

STRUCTURAL_CSV = "evaluation_results_structural.csv"
LLM_CSV = "evaluation_results_llm.csv"


# =========================================================================
# JSON readers
# =========================================================================
def load_objects(path):
    return {o["id"]: o for o in json.load(open(path)).values()}

def load_edges(path):
    return [(e["object_1_id"], e["object_2_id"], e.get("relationship"))
            for e in json.load(open(path)).values()]


# =========================================================================
# Geometry / dependents
# =========================================================================
def bottom_face(obj):
    return obj["bbox_center"][Y] + obj["bbox_extent"][EXTENT_INDEX] / 2.0

def center_y(obj):
    return obj["bbox_center"][Y]

def resting_on(edges, support_id):
    """ids of objects resting on support_id, honoring both edge directions:
    d --'on top of'--> support   OR   support --'under'--> d."""
    out = []
    for (u, v, r) in edges:
        if v == support_id and r == "on top of":
            out.append(u)                 # d on top of support
        elif u == support_id and r == "under":
            out.append(v)                 # support under d  ==  d on top of support
    return out

def dependent_tree(edges, target):
    """{dependent: what_it_rested_on} over the transitive on-top subgraph."""
    parent, frontier = {}, [target]
    while frontier:
        s = frontier.pop()
        for d in resting_on(edges, s):
            if d not in parent and d != target:
                parent[d] = s
                frontier.append(d)
    return parent


# =========================================================================
# Structural metrics (extensible: add keys to the returned dict)
# =========================================================================
def moved(before_obj, after_obj):
    """True if the object's bbox centre changed by more than TOL on any axis."""
    return any(abs(a - b) > TOL for a, b in zip(after_obj["bbox_center"], before_obj["bbox_center"]))


def count_floating(before_objs, before_edges, after_objs, target):
    """Dependents of the target that survive but are left unsupported.

    Dependents are the target's transitive 'on top of' tree in Before. A surviving
    direct dependent counts as floating unless its bottom face lies within TOL of the
    target's former bottom face (the plane the target rested on); a deeper dependent
    counts unless it moved vertically by the same amount (within TOL) as the object it
    rested on. Removed dependents are not floating. This encodes one notion of support
    -- dropping into the vacated position -- so floating dependents that were moved
    elsewhere are counted separately (num_floating_relocated) for manual review.
    Returns (num_floating, num_dependents, num_floating_relocated).
    """
    parent = dependent_tree(before_edges, target)
    if not parent:
        return 0, 0, 0
    if target not in before_objs:
        return 0, len(parent), 0
    target_bottom = bottom_face(before_objs[target])

    def drop_of(oid):
        if oid not in after_objs or oid not in before_objs:
            return None
        return center_y(after_objs[oid]) - center_y(before_objs[oid])

    floating = floating_moved = 0
    for d, par in parent.items():
        if d not in after_objs:
            continue                                   # removed -> handled
        if par == target:                              # direct dependent
            is_floating = abs(bottom_face(after_objs[d]) - target_bottom) > TOL
        else:                                          # deeper dependent
            dd, dp = drop_of(d), drop_of(par)
            is_floating = dd is None or dp is None or abs(dd - dp) > TOL
        if is_floating:
            floating += 1
            if d in before_objs and moved(before_objs[d], after_objs[d]):
                floating_moved += 1                    # relocated, but not onto the vacated plane
    return floating, len(parent), floating_moved


def count_relocated_others(before_objs, before_edges, after_objs, target):
    """Surviving objects OUTSIDE the target's support tree whose bbox centre moved by
    more than TOL on any axis -- geometry the removal had no reason to touch."""
    related = {target} | set(dependent_tree(before_edges, target))
    return sum(1 for oid, a in after_objs.items()
               if oid in before_objs and oid not in related and moved(before_objs[oid], a))

def count_caption_leaks(before_objs, after_objs, target):
    """How many surviving objects still name the removed target's tag in their caption.

    The removed object's tag is read from the BEFORE graph (it's gone from After).
    Whole-word, case-insensitive match -- the same criterion the algorithm's scrub uses, 
    including ignoring each object's OWN tag (another chair naming itself is no leak). 
    Plurals and synonyms are not matched (known limitation of both).
    """
    if target not in before_objs:
        return 0
    tag = before_objs[target].get("object_tag")
    if not tag:
        return 0
    pat = re.compile(r"\b" + re.escape(tag) + r"\b", re.IGNORECASE)
    def names_target(obj):
        cap, own = obj.get("object_caption"), obj.get("object_tag") or ""
        if not cap:
            return False
        if own:
            cap = re.sub(r"\b" + re.escape(own) + r"\b", "", cap, flags=re.IGNORECASE)
        return bool(pat.search(cap))
    return sum(1 for oid, obj in after_objs.items() if oid != target and names_target(obj))

def count_edge_text_leaks(before_objs, after_objs, raw_after_edges, target):
    """Edges whose TEXT still names the removed object, although its id is gone.

    Checks edge_description and the stored object_1_tag / object_2_tag fields. The tags of
    the edge's actual endpoints are stripped first, so an edge between two chairs may say
    "chair" without counting. Whole-word and case-insensitive, like count_caption_leaks.
    Edges with a missing endpoint are skipped, since those are already counted as dangling;
    this metric measures residue on edges that are otherwise valid.
    """
    if target not in before_objs or not raw_after_edges:
        return 0
    tag = before_objs[target].get("object_tag")
    if not tag:
        return 0
    pat = re.compile(r"\b" + re.escape(tag) + r"\b", re.IGNORECASE)
    leaks = 0
    for e in raw_after_edges.values():
        if e.get("object_1_id") not in after_objs or e.get("object_2_id") not in after_objs:
            continue                                   # dangling: counted separately
        own = [after_objs.get(e.get("object_1_id"), {}).get("object_tag") or "",
               after_objs.get(e.get("object_2_id"), {}).get("object_tag") or ""]
        text = " ".join(str(e.get(k) or "") for k in
                        ("edge_description", "object_1_tag", "object_2_tag"))
        for t in own:
            if t:
                text = re.sub(r"\b" + re.escape(t) + r"\b", "", text, flags=re.IGNORECASE)
        if pat.search(text):
            leaks += 1
    return leaks

def count_duplicate_pairs(after_edges):
    """Edges beyond the first for the same unordered object pair.

    A and B related twice -- including once in each direction, as the graph stores each
    relation only once -- counts as one duplicate. Self-loops are counted separately.
    """
    seen = {}
    for (u, v, r) in after_edges:
        k = frozenset((u, v))
        seen[k] = seen.get(k, 0) + 1
    return sum(n - 1 for n in seen.values() if n > 1)

def count_incoherent_edges(after_objs, raw_after_edges):
    """Edges whose own text contradicts their ids and relation.

    An edge is coherent only if its edge_description reads "<tag1> ... <relationship> ...
    <tag2>" in that order (whole words, case-insensitive; other words in between are fine),
    and its stored object_N_tag fields match the endpoints' actual object_tag. The
    complement phrasing is NOT accepted: an edge stored as "cabinet under coffee cup" must
    say so, not "coffee cup on top of cabinet". Edges with a missing endpoint are skipped;
    those are counted as dangling.
    """
    if not raw_after_edges:
        return 0
    bad = 0
    for e in raw_after_edges.values():
        o1, o2 = after_objs.get(e.get("object_1_id")), after_objs.get(e.get("object_2_id"))
        if not o1 or not o2:
            continue
        t1, t2 = (o1.get("object_tag") or ""), (o2.get("object_tag") or "")
        rel, desc = e.get("relationship") or "", e.get("edge_description") or ""
        tag_fields_ok = (e.get("object_1_tag") or "") == t1 and (e.get("object_2_tag") or "") == t2
        w = lambda s: r"\b" + re.escape(s) + r"\b"
        reads_right = bool(t1 and t2 and rel and
                           re.search(w(t1) + r".*" + w(rel) + r".*" + w(t2), desc, re.IGNORECASE))
        if not (tag_fields_ok and reads_right):
            bad += 1
    return bad

def count_dangling_edges(after_objs, after_edges):
    """Edges referencing an object id not present in the after-scene."""
    ids = set(after_objs)
    return sum(1 for (u, v, r) in after_edges if u not in ids or v not in ids)


def count_self_loops(after_edges):
    """Edges whose two endpoints are the same object (u == v)."""
    return sum(1 for (u, v, r) in after_edges if u == v)

def count_duplicate_object_ids(raw_objects):
    """Number of object ids that appear more than once in the raw objects JSON."""
    ids = [o["id"] for o in raw_objects.values()]
    seen, dups = set(), set()
    for i in ids:
        if i in seen:
            dups.add(i)
        seen.add(i)
    return len(dups)


def count_duplicate_edge_ids(raw_edges):
    """Number of edge_ids that appear more than once in the raw edges JSON."""
    eids = [e.get("edge_id") for e in raw_edges.values()]
    seen, dups = set(), set()
    for i in eids:
        if i is not None and i in seen:
            dups.add(i)
        seen.add(i)
    return len(dups)

def edge_accounting(before_objs, before_edges, after_objs, after_edges, target):
    """Reference-free edge/object accounting against the Before graph.

    new        after edges between a pair that had NO Before edge (either direction)
    changed    after edges whose pair existed in Before with a different relation
               (a pair stored the other way round is compared through COMPLEMENT)
    lost       Before edges between two SURVIVING objects that are gone in After
    unrelated  removed objects that are neither the target nor in its support tree
    """
    b_rel = {(u, v): r for (u, v, r) in before_edges}
    def before_rel(u, v):
        if (u, v) in b_rel:
            return b_rel[(u, v)]
        if (v, u) in b_rel:
            return COMPLEMENT.get(b_rel[(v, u)])
        return None
    new = changed = 0
    for (u, v, r) in after_edges:
        br = before_rel(u, v)
        if br is None:
            new += 1
        elif br != r:
            changed += 1
    a_pairs = {frozenset((u, v)) for (u, v, r) in after_edges}
    alive = set(after_objs)
    lost = sum(1 for (u, v, r) in before_edges
                if u in alive and v in alive and frozenset((u, v)) not in a_pairs)
    related = {target} | set(dependent_tree(before_edges, target))
    unrelated = len((set(before_objs) - alive) - related)
    return new, changed, lost, unrelated

def structural_metrics(before_objs, before_edges, after_objs, after_edges, target, raw_after_objects=None, raw_after_edges=None):
    """Reference-free structural metrics of one after-graph, relative to Before.

    target_present / target_in_edges        the target / any edge touching it survives
    num_caption_leaks                       surviving captions still naming the target's tag
    num_duplicate_obj_ids / _edge_ids       ids occurring more than once in the raw JSON
    num_edges                               edges in the after-graph
    num_new_edges, num_changed_relations,
    num_lost_edges, num_unrelated_removed   see edge_accounting
    num_dangling_edges / num_self_loops     edges to a missing object / from an object to itself
    num_duplicate_pairs                     the same object pair related more than once
    num_incoherent_edges                    description contradicting the edge's ids or relation
    num_edge_text_leaks                     edges whose description or tag fields still name it
    num_removed_objects                     Before objects absent from After (incl. the target)
    num_floating_objects, num_floating_relocated, num_dependents        see count_floating
    num_relocated_others                    see count_relocated_others
    """
    after_ids = set(after_objs)
    new_edges, changed_rel, lost_edges, unrelated_removed = edge_accounting(
        before_objs, before_edges, after_objs, after_edges, target)
    target_in_edges = any(u == target or v == target for (u, v, r) in after_edges)
    floating, n_dep, floating_moved = count_floating(before_objs, before_edges, after_objs, target)
    relocated_others = count_relocated_others(before_objs, before_edges, after_objs, target)
    edge_text_leaks = count_edge_text_leaks(before_objs, after_objs, raw_after_edges, target)
    duplicate_pairs = count_duplicate_pairs(after_edges)
    incoherent_edges = count_incoherent_edges(after_objs, raw_after_edges)
    num_removed = len(set(before_objs) - after_ids)     # Before ids absent from After (incl. target)
    caption_leaks = count_caption_leaks(before_objs, after_objs, target)
    dangling = count_dangling_edges(after_objs, after_edges)
    self_loops = count_self_loops(after_edges)
    dup_obj_ids = count_duplicate_object_ids(raw_after_objects) if raw_after_objects else 0
    dup_edge_ids = count_duplicate_edge_ids(raw_after_edges) if raw_after_edges else 0
    return {                                            # add new metrics below, freely
        "target_present":       target in after_ids,
        "target_in_edges":      target_in_edges,
        "num_caption_leaks":    caption_leaks,
        "num_edge_text_leaks": edge_text_leaks,
        "num_duplicate_obj_ids":  dup_obj_ids,      
        "num_duplicate_edge_ids": dup_edge_ids,
        "num_duplicate_pairs":  duplicate_pairs,
        "num_incoherent_edges": incoherent_edges,
        "num_edges":            len(after_edges),
        "num_new_edges":        new_edges,
        "num_changed_relations": changed_rel,
        "num_lost_edges":       lost_edges,
        "num_unrelated_removed": unrelated_removed,
        "num_dangling_edges":    dangling,                        
        "num_self_loops":        self_loops,
        "num_removed_objects":  num_removed,
        "num_floating_objects": floating,
        "num_floating_relocated": floating_moved,
        "num_relocated_others": relocated_others,
        "num_dependents":       n_dep,
    }

STRUCTURAL_FIELDS = ["objects_file", "edges_file",
                     "target_present", "target_in_edges", "num_duplicate_obj_ids", "num_duplicate_edge_ids", 
                     "num_duplicate_pairs", "num_incoherent_edges", "num_caption_leaks", "num_edge_text_leaks", 
                     "num_edges", "num_new_edges", "num_changed_relations", "num_lost_edges",
                     "num_unrelated_removed", "num_dangling_edges", "num_self_loops", "num_removed_objects",
                     "num_floating_objects", "num_floating_relocated", "num_relocated_others", "num_dependents"]


# =========================================================================
# LLM judge: two independent calls per assessment
# =========================================================================
SCORE = {"type": "integer", "enum": list(range(11))}           # integers 0..10
EVIDENCE_TYPES = ["none", "relations", "text", "geometry", "arrangement"]

SCENE_DESCRIPTION = (
    "You are shown a single 3D indoor scene graph as JSON: objects (id, tag, "
    "caption, 3D bounding box) and directed spatial-relationship edges between "
    "them. All objects live in one coordinate system in metres at realistic "
    "indoor scale, in which y is VERTICAL and points DOWN: an object higher up "
    "has a smaller y value than an object below it. bbox_center is the centre of "
    "an object's bounding box; bbox_extent gives its approximate dimensions; "
    "the orientation of the boxes is not given. Judge ONLY what you can infer "
    "from this scene as given.\n\n"
)

PLAUSIBILITY_SYSTEM = SCENE_DESCRIPTION + (
    "Assess the scene in two respects. For each, first give a one-sentence "
    "reasoning, then the score.\n"
    "1. Semantic plausibility of the relations between the objects: judging from "
    "the object tags and the edges, not from the captions, could each stated "
    "relation hold between such objects in a real room, and do the relations fit "
    "together?\n"
    "   0 = relations that cannot hold between such objects, or that contradict "
    "each other.\n"
    "   5 = mostly plausible relations, but some are odd or questionable.\n"
    "   10 = all relations plausible and consistent, as if a careful observer had "
    "described a real room.\n"
    "2. Physical plausibility of the geometry and positioning of the objects: "
    "are positions, sizes and support consistent with a real room?\n"
    "   0 = physically impossible, e.g. objects inside each other, objects "
    "without support, impossible sizes.\n"
    "   5 = largely consistent, but some positions, sizes or support are "
    "implausible.\n"
    "   10 = positions, sizes and support consistent with an observation of a "
    "real room."
)

REMOVAL_SYSTEM = SCENE_DESCRIPTION + (
    "Assess whether an object appears to have been removed from this scene "
    "graph. Use all direct and indirect information the scene provides, but do "
    "not assume a removal without reason.\n"
    "1. evidence: a short list of the SPECIFIC observations in this scene that "
    "drive your answers; each item names the object(s), edge(s) or attributes "
    "involved. Empty list if none.\n"
    "2. removal_reasoning: one sentence on whether an object was removed.\n"
    "3. removal_likelihood: how likely it is that an object was removed from "
    "the graph.\n"
    "   0 = nothing indicates that an object was removed.\n"
    "   5 = some cues could indicate a removal, but they have other plausible "
    "explanations.\n"
    "   10 = clear evidence that an object was removed.\n"
    "4. missing_object_reasoning: one sentence on which object is missing.\n"
    "5. missing_object_guess: ONLY if you think an object was removed, the "
    "object that is most likely missing, as a short tag; otherwise an empty "
    "string.\n"
    "6. evidence_types: the kinds of evidence that point to a removal, one or "
    "several of:\n"
    "   relations   = specific edges that seem wrong, missing or inconsistent;\n"
    "   text        = captions, tags or edge descriptions;\n"
    "   geometry    = coordinates, sizes or support of objects;\n"
    "   arrangement = which objects are present together and what that "
    "implies about the room;\n"
    "   none        = nothing points to a removal; use only on its own."
)

PLAUSIBILITY_SCHEMA = {
    "type": "object",
    "properties": {                                   # reasoning precedes each score
        "semantic_reasoning": {"type": "string"},
        "semantic_plausibility": SCORE,
        "physical_reasoning": {"type": "string"},
        "physical_plausibility": SCORE,
    },
    "required": ["semantic_reasoning", "semantic_plausibility",
                 "physical_reasoning", "physical_plausibility"],
    "additionalProperties": False,
}

REMOVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": {"type": "string"}},
        "removal_reasoning": {"type": "string"},
        "removal_likelihood": SCORE,
        "missing_object_reasoning": {"type": "string"},
        "missing_object_guess": {"type": "string"},
        "evidence_types": {"type": "array",
                           "items": {"type": "string", "enum": EVIDENCE_TYPES}},
    },
    "required": ["evidence", "removal_reasoning", "removal_likelihood",
                 "missing_object_reasoning", "missing_object_guess", "evidence_types"],
    "additionalProperties": False,
}


def llm_call(system_prompt, schema, schema_name, objects_dict, edges_dict, api_key, model=MODEL):
    """One judge call on the graph exactly as written (all fields, list form)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    scene = {"objects": list(objects_dict.values()), "edges": list(edges_dict.values())}
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": json.dumps(scene, indent=2)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": schema_name,
                                         "strict": True, "schema": schema}},
    )
    return json.loads(resp.choices[0].message.content)


LLM_FIELDS = ["check_index",
              "objects_file", "edges_file", "model", "timestamp",
              # call A
              "plausibility_call_seconds",
              "semantic_reasoning", "semantic_plausibility",
              "physical_reasoning", "physical_plausibility",
              "plausibility_error",
              # call B
              "removal_call_seconds",
              "evidence", "removal_reasoning", "removal_likelihood",
              "missing_object_reasoning", "missing_object_guess", "evidence_types",
              "removal_error"]


# =========================================================================
# Evaluation of one edited graph
# =========================================================================
def run(args):
    structural = not args.qualitative_only
    qualitative = not args.structural_only
    opath, epath = args.edited_objects, args.edited_edges

    needed = [opath, epath] + ([args.before_objects, args.before_edges] if structural else [])
    for path in needed:
        if not Path(path).is_file():
            sys.exit(f"File not found: {path}")
    api_key = None
    if qualitative:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY not set (or pass --structural-only).")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- structural: one row for the edited graph ----
    if structural:
        before_objs = load_objects(args.before_objects)
        before_edges = load_edges(args.before_edges)
        if args.target not in before_objs:
            sys.exit(f"No object with id {args.target} in {args.before_objects}.")
        raw_o = json.load(open(opath))            # raw, before collapse
        raw_e = json.load(open(epath))
        a_objs, a_edges = load_objects(opath), load_edges(epath)
        m = structural_metrics(before_objs, before_edges, a_objs, a_edges, args.target, raw_after_objects=raw_o, raw_after_edges=raw_e)
        srow = {"objects_file": str(opath), "edges_file": str(epath), **m}
        with open(out_dir / STRUCTURAL_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STRUCTURAL_FIELDS); w.writeheader(); w.writerow(srow)
        print("Structural metrics")
        for key, val in m.items():
            print(f"  {key:24} {val}")
        print(f"wrote 1 row -> {out_dir / STRUCTURAL_CSV}")

    # ---- LLM: N checks of the edited graph ----
    if qualitative:
        lrows = []
        with open(out_dir / LLM_CSV, "w", newline="") as f:      # fresh LLM CSV with header; rows are appended live
            csv.DictWriter(f, fieldnames=LLM_FIELDS).writeheader()
        for k in range(1, args.assessments + 1):
            row = {"check_index": k, "objects_file": str(opath),
                   "edges_file": str(epath), "model": args.model,
                   "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "plausibility_call_seconds": "", "plausibility_error": "",
                   "removal_call_seconds": "", "removal_error": ""}
            try:
                a_objs = json.load(open(opath)); a_edges = json.load(open(epath))
            except Exception as e:
                row["plausibility_error"] = row["removal_error"] = f"{type(e).__name__}: {e}"
                a_objs = None
            if a_objs is not None:
                # call A: plausibility (removal is never mentioned)
                try:
                    t0 = time.perf_counter()
                    j = llm_call(PLAUSIBILITY_SYSTEM, PLAUSIBILITY_SCHEMA,
                                 "scene_plausibility", a_objs, a_edges, api_key, model=args.model)
                    row["plausibility_call_seconds"] = time.perf_counter() - t0
                    row.update(j)
                except Exception as e:
                    row["plausibility_error"] = f"{type(e).__name__}: {e}"
                # call B: removal, independent of call A
                try:
                    t0 = time.perf_counter()
                    j = llm_call(REMOVAL_SYSTEM, REMOVAL_SCHEMA,
                                 "scene_removal", a_objs, a_edges, api_key, model=args.model)
                    row["removal_call_seconds"] = time.perf_counter() - t0
                    row.update(j)
                    for key in ("evidence", "evidence_types"):
                        if isinstance(row.get(key), list):
                            row[key] = json.dumps(row[key])
                except Exception as e:
                    row["removal_error"] = f"{type(e).__name__}: {e}"
            lrows.append(row)
            with open(out_dir / LLM_CSV, "a", newline="") as f:  # persist each paid call at once
                csv.DictWriter(f, fieldnames=LLM_FIELDS).writerow(row)
            failed = row["plausibility_error"] or row["removal_error"]
            print(f"  [llm] check {k}" + (" ERROR" if failed else ""))
            if failed:
                print(f"        {failed}")
            else:
                print(f"        semantic plausibility {row['semantic_plausibility']}, "
                      f"physical plausibility {row['physical_plausibility']}, "
                      f"removal likelihood {row['removal_likelihood']}, "
                      f"guess: {row['missing_object_guess'] or 'none'}")
        print(f"wrote {len(lrows)} rows -> {out_dir / LLM_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate an edited 3D scene graph after an object removal.")
    p.add_argument("--edited-objects", required=True,
                   help="objects JSON file of the edited graph")
    p.add_argument("--edited-edges", required=True,
                   help="edges JSON file of the edited graph")
    p.add_argument("--before-objects",
                   help="objects JSON file of the graph before the removal (structural metrics)")
    p.add_argument("--before-edges",
                   help="edges JSON file of the graph before the removal (structural metrics)")
    p.add_argument("--target", type=int,
                   help="integer id of the removed object (structural metrics)")
    p.add_argument("--structural-only", action="store_true",
                   help="Structural metrics only; no OpenAI calls.")
    p.add_argument("--qualitative-only", action="store_true",
                   help="LLM judge only; needs the edited graph only.")
    p.add_argument("--assessments", type=int, default=1,
                   help="number of LLM assessments, two calls each (default: %(default)s)")
    p.add_argument("--model", default=MODEL,
                   help="OpenAI model of the LLM judge (default: %(default)s)")
    p.add_argument("--out-dir", default=".",
                   help=f"folder for {STRUCTURAL_CSV} and {LLM_CSV} (default: current folder)")
    args = p.parse_args()
    if args.structural_only and args.qualitative_only:
        p.error("--structural-only and --qualitative-only exclude each other")
    if not args.qualitative_only and (args.before_objects is None or args.before_edges is None
                                      or args.target is None):
        p.error("the structural metrics need --before-objects, --before-edges and --target "
                "(or pass --qualitative-only)")
    if args.assessments < 1:
        p.error("--assessments must be at least 1")
    run(args)
