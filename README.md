# Object Erasure and Topology Repair in 3D Scene Graphs

Two command-line tools for 3D scene graphs in the JSON format written by
[ConceptGraphs](https://github.com/concept-graphs/concept-graphs):

- `remove_object.py` removes an object from a scene graph and repairs the graph, so that
  the object is neither present nor mentioned and the spatial relations among the
  remaining objects are preserved.
- `evaluate_graph.py` evaluates an edited graph against the graph before the removal,
  without needing a reference graph of the edited scene.

The removal is deterministic and needs no training, no model and no network access. Only
the optional qualitative assessment in `evaluate_graph.py` calls a language model.

This repository contains the code of the bachelor's thesis *Object Erasure and Topology
Repair in 3D Scene Graphs* (ETH Zürich, 2026).

## Installation

Python 3.11 or later is required. `remove_object.py` needs `networkx`;
`evaluate_graph.py` needs `openai`, and only for the qualitative assessment.

```
conda env create -f environment.yml
conda activate object_removal
```

Without conda: `pip install networkx==3.6.1 openai==2.45.0`.

## Quick start

```
mkdir -p out

# remove the dining table (id 41) together with the objects resting on it
python remove_object.py scenario_before_graphs/TableWithVase/objects.json \
    scenario_before_graphs/TableWithVase/edges.json 41 \
    --out-objects out/objects.json --out-edges out/edges.json

# evaluate the edited graph against the original one (no API calls)
python evaluate_graph.py \
    --before-objects scenario_before_graphs/TableWithVase/objects.json \
    --before-edges scenario_before_graphs/TableWithVase/edges.json \
    --edited-objects out/objects.json --edited-edges out/edges.json \
    --target 41 --structural-only --out-dir out
```

## Input format

Both tools read a scene graph as two JSON files (the values below are illustrative):

```
objects.json   {"object_1": {"id": 41, "object_tag": "dining table",
                             "object_caption": "...", "bbox_center": [x, y, z],
                             "bbox_extent": [x, y, z], "bbox_volume": 0.5}, ...}
edges.json     {"edge_1": {"edge_id": 0, "object_1_id": 7, "object_2_id": 41,
                           "object_1_tag": "vase", "object_2_tag": "dining table",
                           "relationship": "on top of",
                           "edge_description": "...", "num_detections": 3}, ...}
```

`relationship` is one of `on top of`, `under`, `next to`, `in front of`, `behind`,
`above`, `below`. An edge reads "object 1 `relationship` object 2". The object to remove
is given by its `id`. The edited graph is written in the same format.

The input is not validated: a missing file, invalid JSON or an unknown target id is
reported, but other faults in the input can lead to errors or wrong results.

## Removing an object

```
python remove_object.py OBJECTS EDGES TARGET_ID [options]
python remove_object.py --help
```

| Option | Meaning |
| --- | --- |
| `--policy remove` | Repair the relations and also remove the objects resting on the target (default). |
| `--policy lower` | Repair the relations, keep the objects resting on the target and lower them into the position it vacated. |
| `--policy naive` | Delete the object and its relations only, without any repair. |
| `--out-objects`, `--out-edges` | Output files (default: `objects_after_edit.json`, `edges_after_edit.json` in the current folder). |
| `--summary FILE` | Also write a summary of the removal to a JSON file. |
| `--threshold-stat`, `--threshold-scale` | Distance ceiling for new relations, see below (default: `mean` and `1.2`). |
| `--ground-y Y` | Height at which lowered objects come to rest (`lower` policy). Default: the former bottom face of the removed object. |
| `--check-table` | Only verify that the composition table is reversible; runs no removal. |


What the repair does:

- **Relations.** Two objects that were each related to the removed object receive a direct
  relation, obtained by composing their two relations with a fixed table (for example,
  A `next to` X and X `below` B gives A `next to` B). At most one such relation is added
  per neighbour, and none between objects that are already related.
- **Distance ceiling.** A new relation is added only if the two objects are no farther
  apart than a ceiling for that relation. The ceiling is a statistic of the distances
  between the objects that carry this relation in the input (`--threshold-stat`: `max`,
  `mean` or `min`) multiplied by a factor (`--threshold-scale`). In small graphs with few 
  relations of a kind, the mean is a tight ceiling;
  `--threshold-stat max` or a larger `--threshold-scale` then admits more relations.
- **Objects resting on the target** are those related to it by `on top of` or `under`.
  They are removed with it or lowered, depending on the policy; objects resting on them
  are treated the same way.
- **Text.** Captions of other objects that name the removed object by its tag are
  replaced by a generic caption of the form "A <tag>.".

Existing output files are overwritten.

The summary written with `--summary` lists the removed objects and the steps taken. It
names what was removed, so do not pass it on together with the edited graph.

From Python:

```python
from remove_object import run_removal, check_reversibility

summary = run_removal("objects.json", "edges.json", target=41,
                      out_objects="out/objects.json", out_edges="out/edges.json",
                      dependent_policy="lower")
check_reversibility()          # self-test of the composition table
```

## Evaluating an edited graph

```
python evaluate_graph.py --before-objects FILE --before-edges FILE \
    --edited-objects FILE --edited-edges FILE --target ID [options]
python evaluate_graph.py --help
```

| Option | Meaning |
| --- | --- |
| (default) | Structural metrics and qualitative assessment. |
| `--structural-only` | Structural metrics only; no API calls. |
| `--qualitative-only` | Qualitative assessment only; needs the edited graph only. |
| `--assessments N` | Number of qualitative assessments (default: 1). Each makes two API calls. |
| `--model NAME` | OpenAI model of the qualitative assessment (default: `gpt-5.4-mini`). |
| `--out-dir DIR` | Folder for the result files (default: current folder). |

The results are printed and written to `evaluation_results_structural.csv` and
`evaluation_results_llm.csv`. Existing result files are overwritten.

### Structural metrics

Computed from the two graphs alone. Apart from the last four, lower is better.

| Column | Meaning |
| --- | --- |
| `target_present`, `target_in_edges` | The removed object, or a relation that refers to it, is still in the edited graph. |
| `num_caption_leaks` | Captions of remaining objects that still name the removed object's tag. |
| `num_edge_text_leaks` | Relations between remaining objects whose description or tag fields still name it. |
| `num_dangling_edges`, `num_self_loops` | Relations to an object that does not exist, or from an object to itself. |
| `num_duplicate_obj_ids`, `num_duplicate_edge_ids` | Identifiers that occur more than once. |
| `num_duplicate_pairs` | Pairs of objects related more than once. |
| `num_incoherent_edges` | Relations whose description or tag fields contradict their endpoints or their relation. |
| `num_changed_relations`, `num_lost_edges` | Relations between remaining objects that were changed or lost. |
| `num_unrelated_removed` | Removed objects that are neither the target nor rest on it. |
| `num_floating_objects` | Objects that rested on the target and are left without support. |
| `num_floating_relocated` | Of those, the ones that were moved somewhere else. |
| `num_relocated_others` | Objects that did not rest on the target but were moved. |
| `num_dependents` | Objects that rested on the target, directly or through other objects. |
| `num_removed_objects` | Objects removed, including the target. |
| `num_new_edges` | Relations between pairs of objects that were not related before. |
| `num_edges` | Relations in the edited graph. |

### Qualitative assessment

A language model is shown the edited graph only, in two independent calls. The first asks
for the semantic and the physical plausibility of the scene (0 to 10 each, with a short
reasoning) and does not mention a removal. The second asks for evidence that an object was
removed, for the likelihood of a removal (0 to 10), for a guess at the missing object, and
for the kinds of evidence used (`relations`, `text`, `geometry`, `arrangement` or `none`).
All answers, the duration of each call and any error are written to the CSV file.

This needs the `openai` package and an API key in the environment variable
`OPENAI_API_KEY`. The assessment is not deterministic: repeated assessments of the same
graph can differ, which `--assessments N` makes visible.

## Example graphs

The folder `scenario_before_graphs` contains five small scenes, one folder each with
`objects.json` and `edges.json`:

| Scene | Target id | Target | Objects resting on it |
| --- | --- | --- | --- |
| `Cabinet` | 57 | cabinet | 2 |
| `Chair` | 29 | folded chair | 0 |
| `CoffeeMaker` | 12 | coffee maker | 1 |
| `SofaWithPainting` | 228 | couch | 2 |
| `TableWithVase` | 41 | dining table | 1 |

The scenes were built with [ConceptGraphs](https://github.com/concept-graphs/concept-graphs)
(branch `ali-dev`) from the
[living-room capture, sofa sequence](https://huggingface.co/datasets/simkoc/Remove360/tree/main/living-room/sofa/train),
of the [Remove360](https://huggingface.co/datasets/simkoc/Remove360) dataset and then edited
by hand: objects and relations were added, some positions were adjusted, and some relations
were relabelled. They are test scenes and not unmodified output of that pipeline. If you use them, please also cite
[ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) and
[Remove360](https://arxiv.org/abs/2508.11431).

## Assumptions and limitations

The tools do not detect violations of these assumptions; the result can then be wrong
without an error.

- **Coordinates.** All coordinates share one frame and one unit. The second coordinate
  (index 1, y) is vertical and points **down**. The second side length of a bounding box
  is its vertical size.
- **One relation per pair.** Two objects are connected by at most one edge, stored in one
  direction.
- **Support.** An object rests on another only if the graph says so with an `on top of` or
  `under` edge; support is not deduced from positions.
- **Lowering.** An object is lowered vertically into the position of the removed object.
  This is a geometric heuristic: collisions, overlap and stability are not checked.
- **Distances.** The distances between related objects in the input are taken to indicate
  how far apart two objects may be for a relation to hold.
- **Mentions.** Mentions of the removed object are recognised by its tag as a whole word.
  Synonyms, paraphrases and plurals are not recognised, neither by the removal nor by the
  evaluation.
- **Evaluation.** Positions are compared with a tolerance of 0.05 coordinate units. The
  prompt of the qualitative assessment states that coordinates are in metres at indoor
  scale and that y points down; for graphs with other conventions the assessment is not
  meaningful.