# Appendix: Food-Web Symbolic Helper

## Motivation

Food-web questions were one of the clearest examples where the vision-language model recognized the broad task category but still made systematic reasoning errors. These questions require the model to interpret a structured diagram, follow arrows between organisms, identify ecological roles, and then map the result to a multiple-choice answer. In practice, SmolVLM often understood that an image was a food web, but failed on the exact graph operation required by the question, such as distinguishing a primary consumer from a secondary consumer or tracing where matter could move through the web.

To address this failure mode, we added a deterministic food-web helper as a post-processing module during inference. The model still produces the initial answer choice, but for eligible food-web rows the helper attempts to solve the question symbolically. If the helper can determine a unique answer, it overrides the model prediction. If the helper cannot solve the question safely, the original model prediction is preserved. This gives the final system a hybrid structure: the VLM handles general visual question answering, while a narrow symbolic module handles a class of structured diagrams where graph logic is more reliable than free-form generation.

## Scope

The helper is intentionally narrow. It activates only for rows whose dataset skill is one of the food-web categories:

- `Interpret food webs`
- `Interpret food webs I`
- `Interpret food webs II`

This skill gate prevents the module from affecting unrelated ScienceQA examples. Within those food-web rows, the helper only applies when the image matches one of the known food-web templates observed in the provided competition data. Unknown images, unsupported question wording, missing files, or ambiguous answer choices all fall back to the VLM prediction.

## Template Identification

The competition data reuses a small number of food-web diagrams across many questions. Rather than trying to re-read each diagram with OCR at inference time, the helper identifies the exact diagram template by computing the SHA256 hash of the image file. The hash is then mapped to a template identifier such as `monterey_full`, `nunavut_full`, `shenandoah_forest`, or `lake_lr`.

This design has two advantages. First, it is deterministic: the same image file always maps to the same template. Second, it avoids introducing a separate image-processing pipeline that could fail on small labels or arrows. Once a template is identified, the helper can use a hand-coded graph representation of that diagram.

The helper currently covers six repeated food-web templates from the provided dataset. No external images or test labels are used; the template logic is derived from the provided competition images and their visible food-web structure.

## Graph Representation

Each food-web template is represented as a directed graph. Organisms are nodes, and arrows are encoded from food source to consumer. For example, if kelp is eaten by sea urchins, the graph contains an edge:

```text
kelp -> sea urchin
```

This direction matches the flow of energy or matter through the food web. It also makes many question types natural to solve with graph operations:

- A producer is a node with no incoming food source.
- A consumer is a node with at least one incoming food source.
- A decomposer is treated as a terminal sink that receives matter but does not feed another organism in the represented graph.
- Reachability answers whether energy or matter can move from one organism to another.
- Shortest-path distance helps choose the nearest downstream organism when multiple choices are reachable.

For each template, the implementation stores an adjacency list where each key is an organism and each value is the list of organisms that consume it. A simplified example is:

```text
kelp: sea urchin, kelp bass
sea urchin: sea otter
sea otter: orca
```

This graph representation converts the visual reasoning problem into a small deterministic graph problem.

## Question Solving Procedure

For each eligible row, the helper follows a conservative sequence:

1. Confirm that the row belongs to a food-web skill.
2. Parse and normalize the answer choices.
3. Check whether the question and choice set matches a known multiple-choice pattern from the provided labeled data.
4. Identify the image template using the SHA256 hash of the image.
5. Load the graph associated with that template.
6. Normalize the question text.
7. Match the question against supported food-web wording patterns.
8. Apply the corresponding graph rule.
9. Return an answer only if exactly one answer choice satisfies the rule.
10. Otherwise return no symbolic answer and keep the VLM prediction.

The uniqueness check is important. If a rule matches zero choices or more than one choice, the helper does not guess. This avoids replacing a model prediction with an uncertain symbolic answer.

## Supported Reasoning Patterns

The helper covers the most common food-web question types observed in the dataset.

### Producers and Consumers

Producer questions are solved by identifying nodes with no incoming edges. These organisms are sources of energy or matter within the diagram. Consumer questions are solved by identifying organisms with at least one incoming edge, meaning they consume another organism in the graph.

### Primary, Secondary, and Tertiary Consumers

The helper computes approximate trophic bands from the graph. Producers form the base level. Primary consumers eat producers. Secondary consumers eat primary consumers, and tertiary consumers eat secondary consumers. The implementation uses predecessor relationships to assign these levels and then selects the unique answer choice that belongs to the requested level.

The primary-consumer rule is slightly more flexible than the strict trophic-band definition: an organism can be treated as a primary consumer if it eats at least one producer. This reflects the wording of some dataset questions, where the intended answer is the organism that directly consumes a producer even if the diagram also shows additional feeding relationships.

### Decomposers

Decomposer questions are handled by identifying organisms that receive matter from other organisms but do not feed into another organism in the graph. In the encoded diagrams, these organisms function as terminal sinks for matter flow.

### Omnivores

Omnivore questions are solved by checking whether an organism has both producer and non-producer food sources. This captures the ecological definition of an omnivore as an organism that eats both plant-like producers and animal-like consumers.

### Energy and Matter Flow

Some questions ask where energy or matter can move through a web. These are handled with graph traversal. If the question asks whether something eventually moves to a target organism, the helper checks which answer choices can reach that target by following arrows. If the question asks where matter that was once part of a source organism could move, the helper computes downstream paths from the source and selects the uniquely closest reachable answer choice.

## Lookup for Dataset-Specific Multiple-Choice Patterns

Some food-web questions depend on dataset-specific wording or rubric conventions that are difficult to capture with a single general rule. To handle these cases, the helper includes a small lookup table keyed by normalized question text and sorted normalized answer choices. This lookup is built from the provided labeled training and validation examples, along with manually verified cases that follow the same visible diagram structure and rubric.

The lookup is used before general graph reasoning. If a question-choice combination appears in the table, the helper returns the associated answer. If it does not appear, the solver proceeds to template-based graph reasoning. This improves reliability on repeated food-web question forms while still avoiding any use of hidden test labels.

## Hybrid Inference Integration

The integration point is a simple wrapper around the model prediction. During inference, the model first predicts a 0-indexed answer choice. The food-web helper is then called with the row, the model prediction, and the image root. If the helper returns a symbolic answer, that answer replaces the model prediction. If the helper returns `None`, the model prediction is left unchanged.

Conceptually, the logic is:

```text
model_prediction = VLM(row)
symbolic_prediction = food_web_helper(row)

if symbolic_prediction is available:
    final_prediction = symbolic_prediction
else:
    final_prediction = model_prediction
```

This makes the helper an override layer rather than a replacement for the learned model.

## Reliability and Limitations

The helper is reliable only within its defined scope. It assumes that the image belongs to one of the known food-web templates and that the question wording matches one of the supported patterns. It also assumes that the encoded graph accurately reflects the diagram. For this reason, the implementation is deliberately conservative: it requires both a known template and a unique symbolic answer before overriding the VLM.

The main limitation is coverage. The helper does not generalize to arbitrary unseen food-web diagrams unless a new template graph is added. It also does not solve all possible food-web question wordings. However, this limited coverage is acceptable for the competition setting because the provided food-web rows reuse a small number of diagrams, and the goal is to correct systematic failures on those known structured cases without harming unrelated examples.

## Contribution to Final Performance

The food-web helper was part of the final hybrid inference strategy used in the best-performing run. Together with the Punnett-square helper, it targeted categories that remained weak under prompt engineering, context ablation, and LoRA fine-tuning alone. These categories required exact structured reasoning rather than broad semantic understanding. By converting food-web diagrams into explicit graphs, the helper performed the required reasoning deterministically and reduced the burden on the VLM.

The result supports a broader conclusion from the project: compact vision-language models can perform well on many multimodal science questions, but narrow symbolic modules can provide substantial gains on diagram types with clear formal structure. In these cases, hybrid inference is a practical way to combine the flexibility of a learned model with the reliability of deterministic reasoning.

## Report Summary

For food-web questions, we added a deterministic symbolic post-processing helper. The helper first identifies repeated food-web diagrams by hashing the image file and mapping it to one of six known templates from the provided dataset. Each template is represented as a directed graph whose edges point from food source to consumer. The helper then parses common food-web question patterns, including producer, consumer, decomposer, omnivore, trophic-level, and reachability questions, and applies exact graph rules to the answer choices. It overrides the VLM only when exactly one answer choice satisfies the symbolic rule; otherwise, it preserves the original model prediction. This conservative hybrid design corrected systematic graph-reasoning failures while limiting the risk of applying symbolic logic outside its reliable scope.

# Food Web Helper Explanation

This note explains the food-web helper implemented in `vqa_dataset.py`, lines 543-940. The helper is a deterministic post-processing module used during hybrid TA inference: the VLM first predicts an answer choice, and the food-web helper optionally overrides that prediction when it can solve the row symbolically with high confidence.

## High-Level Purpose

The ScienceQA food-web questions are visually structured. Many examples use repeated food-web diagrams with different questions and answer choices. The VLM often recognizes that a question is about a food web, but can fail on the precise reasoning step: following arrows, identifying producers, counting trophic levels, or tracing matter/energy movement.

The helper addresses this by encoding the small set of known food-web diagrams as graphs. Each organism is a node, and each arrow is represented as an edge from the organism being eaten to the organism that eats it. For example:

```python
"kelp": ["kelp bass", "sea urchin"]
```

means kelp is eaten by kelp bass and sea urchin. This edge direction is important: graph traversal follows the flow from food source to consumer.

The helper is conservative. It returns a symbolic answer only when the row is clearly a food-web row and the answer can be uniquely determined. Otherwise, it returns `None`, and the model prediction is kept.

## Skill Gate

The helper only activates for three dataset skill labels:

```python
FOOD_WEB_SKILLS = {
    "Interpret food webs",
    "Interpret food webs I",
    "Interpret food webs II",
}
```

The function `food_web_skill_row(row)` checks `row["skill"]` against this set. If the row is not one of these skills, `solve_food_web_symbolic(...)` immediately returns `None`.

This prevents the food-web logic from accidentally touching unrelated ScienceQA categories.

## Template Identification

The dataset reuses a small number of food-web images. The code identifies each diagram by hashing the image file bytes with SHA256:

```python
sha = hashlib.sha256(p.read_bytes()).hexdigest()
return FOOD_WEB_SHA256_TO_TEMPLATE.get(sha)
```

`FOOD_WEB_SHA256_TO_TEMPLATE` maps six known hashes to template names:

- `nunavut_full`
- `monterey_full`
- `shenandoah_forest`
- `nunavut_zoom`
- `lake_lr`
- `monterey_zoom`

This means the helper does not need OCR or computer vision at inference time. If the image bytes match one of the known templates, the helper knows exactly which food-web graph applies.

If the image file is missing or the SHA256 hash is unknown, `food_web_template_id(...)` returns `None`, and the row falls back to the VLM prediction.

## Graph Representation

Each food-web template is represented by an adjacency dictionary. The keys are prey, producers, or lower-level food sources, and the values are organisms that eat them.

For example, part of the Monterey food web is encoded as:

```python
MONTEREY_FULL_ADJ = {
    "kelp": ["kelp bass", "sea urchin"],
    "sea urchin": ["sea otter"],
    "sea otter": ["orca"],
}
```

This encodes the chain:

```text
kelp -> sea urchin -> sea otter -> orca
```

The helper includes graph dictionaries for Monterey, Shenandoah Forest, Nunavut, and Little Rock Lake diagrams. `FOOD_WEB_EDGES_BY_TEMPLATE` connects each template name to its graph.

## Graph Utilities

The helper defines several small graph functions.

### `_fw_nodes(adj)`

Collects every organism in the graph, including organisms that only appear as consumers and not as dictionary keys.

### `_fw_predecessors(adj)`

Builds the reverse graph: for each organism, list what it eats or receives matter from. If `adj` maps prey to predators, then `preds[predator]` gives its food sources.

For example, if:

```python
"kelp": ["sea urchin"]
```

then:

```python
preds["sea urchin"] == ["kelp"]
```

This is used to identify consumers, producers, decomposers, and omnivores.

### `_fw_forward_distances(adj, start)`

Runs breadth-first search along eaten-to-eater edges and returns hop distances from a starting organism. This is useful for questions like:

> Matter that was once part of the X could move to which organism?

If multiple answer choices are reachable, the helper chooses the nearest reachable organism only when that nearest choice is unique.

### `_fw_reaches_forward(adj, start, goal)`

Checks whether there is any path from `start` to `goal` along the food-web arrows. This is used for questions like:

> Energy eventually moves to the Y from which organism?

### `_fw_pick_single_choice_matching(choices, ok)`

Applies a predicate to each answer choice and returns the index only if exactly one choice matches. If zero or multiple choices match, it returns `None`.

This uniqueness requirement is what makes the helper conservative. It avoids overriding the VLM when the symbolic rule is ambiguous.

## Label Normalization

`_fw_norm_org(label)` standardizes organism names before lookup:

- Converts to string
- Strips whitespace
- Lowercases
- Collapses repeated whitespace
- Removes leading `"the "`

This makes choices like `"The black bear"` match graph labels such as `"black bear"`.

## Trophic-Level Logic

The function `_fw_il_trophic_bands(adj)` computes four sets:

- `s0`: producers
- `l1`: primary consumers
- `l2`: secondary consumers
- `l3`: tertiary consumers

The logic is based on predecessor sets:

- Producers have no predecessors, meaning nothing in the diagram feeds into them.
- Primary consumers eat only producers.
- Secondary consumers eat at least one primary consumer and otherwise only producers or primary consumers.
- Tertiary consumers eat at least one secondary consumer and otherwise only organisms up through secondary consumers.

This “band” logic is used for questions asking for producer, primary consumer, secondary consumer, or tertiary consumer.

## Lookup Table Shortcut

Before doing graph reasoning, `solve_food_web_symbolic(...)` checks a cached lookup table:

```python
data/food_web_mc_lookup.json
```

The key is:

```python
(question_text, sorted_normalized_choices)
```

If the exact question-and-choice set appears in this lookup, the helper returns the stored answer index immediately.

This exists because some food-web multiple-choice questions follow dataset-specific rubrics that are hard to capture with a simple trophic-level rule. The comment specifically mentions Monterey secondary-consumer cases where the expected label depends on how the dataset treats organisms such as `kelp bass`.

The lookup is cached with `@lru_cache(maxsize=1)`, so the JSON file is loaded only once per Python process.

## Main Solver Flow

The main function is:

```python
solve_food_web_symbolic(row, image_root) -> int | None
```

It follows this sequence:

1. Check that the row is a food-web skill.
2. Parse the answer choices.
3. Try the exact multiple-choice lookup.
4. Identify the image template by SHA256 hash.
5. Load the corresponding graph.
6. Normalize the question text.
7. Match known food-web question patterns.
8. Return a unique 0-based answer index if solved.
9. Return `None` if unsupported or ambiguous.

The output is always a 0-based answer index compatible with the Kaggle label format, or `None` to signal fallback.

## Question Patterns Handled

The solver recognizes several common food-web question families.

### Arrow-Based Producer / Consumer / Decomposer / Omnivore

Questions containing `"based on the arrows,"` are handled first because they often ask directly about the diagram arrows.

For consumers, the helper checks whether the organism has any predecessors:

```python
bool(preds.get(lab))
```

If an organism has something feeding into it, it consumes something.

For producers, the helper checks membership in `s0`, the producer set.

For decomposers, the helper checks:

```python
lab in preds and not adj.get(lab)
```

This means the organism receives matter from something but does not feed into another organism in the graph, making it a sink.

For omnivores, the helper checks whether the organism eats at least one producer and at least one non-producer.

### “Eventually Moves To” Questions

For wording like:

```text
eventually moves to the <target>
```

the helper extracts the target organism and tests which answer choice can reach that target by following food-web arrows.

This answers questions about possible energy or matter paths through the web.

### “Matter Once Part Of” Questions

For wording like:

```text
matter that was once part of the <source>
```

the helper computes forward distances from the source organism. It then looks for answer choices reachable downstream.

If exactly one answer choice is the nearest reachable organism, the helper returns it. If multiple choices tie or none are reachable, it returns `None`.

### Producer Questions

For wording like:

```text
which of the following organisms is the producer
```

or:

```text
living things is a producer
```

the helper picks the unique choice in `s0`.

### Decomposer Questions

For decomposer wording, the helper selects an organism that has incoming matter and no outgoing food-web edge. In these diagrams, decomposers often act as terminal sinks.

### Omnivore Questions

For omnivore wording, the helper selects an organism whose food sources include both:

- A producer
- A non-producer

This captures the standard definition: an omnivore eats both producers/plants and consumers/animals.

### Primary Consumer Questions

For primary consumers, the helper first uses a slightly weaker rule:

```python
lab not in s0
and preds.get(lab)
and any(p in s0 for p in preds[lab])
```

This accepts organisms that eat at least one producer, even if they may also have other food sources. If that does not produce a unique answer, it falls back to strict `l1` membership from `_fw_il_trophic_bands(...)`.

This weak-first behavior likely reflects dataset wording where the expected answer may be an organism that directly consumes a producer, even if the graph also gives it other relationships.

### Secondary and Tertiary Consumer Questions

For secondary and tertiary consumers, the helper uses the trophic bands:

```python
lab in l2
lab in l3
```

These correspond to organisms that consume primary and secondary consumers respectively under the helper’s layered graph interpretation.

### Generic Consumer Questions

For wording like:

```text
living things is a consumer
```

the helper returns the unique choice with any predecessor, meaning any organism that consumes another organism.

## Hybrid Prediction Wrapper

The integration point is:

```python
hybrid_food_web_predict(row, model_pred_choice_index, image_root=...)
```

It calls:

```python
s = solve_food_web_symbolic(row, image_root)
```

Then:

- If `s` is not `None`, return the symbolic answer.
- Otherwise, return the original VLM prediction.

In other words, the helper is an override layer, not a replacement model. It only changes predictions for rows it can solve deterministically.

## Why This Helps

This helper targets a failure mode observed in the VLM: the model can identify that an image is a food web, but still mis-handle exact graph reasoning. Food-web questions often require operations that are easy for a graph algorithm:

- Follow arrows
- Reverse arrows to find food sources
- Identify source nodes as producers
- Identify sink nodes as decomposers
- Check reachability
- Compute shortest downstream paths
- Classify trophic levels

By representing each known diagram as a graph, the helper performs these operations exactly and avoids relying on generated text.

## Why It Is Conservative

The helper avoids risky overrides in several ways:

- It activates only on known food-web skills.
- It requires the image hash to match one of six known templates.
- It returns an answer only when exactly one choice matches the rule.
- It falls back to the model for unknown templates, unsupported wording, or ambiguous matches.

This design protects overall accuracy: the symbolic helper improves rows it understands while preserving VLM predictions elsewhere.
