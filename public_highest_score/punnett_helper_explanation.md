# Appendix: Punnett-Square Symbolic Helper

## Motivation

Punnett-square questions were another category where the model often recognized the general topic but failed on the exact procedural reasoning step. These examples require two separate abilities. First, the system must read the parent gametes from the margins of the Punnett-square image. Second, it must combine those alleles into offspring genotypes, count the requested genotype or phenotype class, and map that count to the correct multiple-choice answer.

SmolVLM was not consistently reliable at performing the second step implicitly through direct answer generation. Even when it appeared to understand that a question involved recessive traits, heterozygous offspring, or phenotype probabilities, it could still choose the wrong fraction or ratio. The Punnett helper addresses this by separating the visual reading step from the symbolic genetics step. Once the four parent gametes are available, the helper deterministically constructs the Punnett grid and solves supported question forms.

As with the food-web helper, this module is used as a conservative hybrid inference layer. The VLM produces an initial answer. The Punnett helper attempts to compute a symbolic answer only for eligible genetics rows and only when the required parent alleles are available. If the helper succeeds, its answer replaces the model prediction. If it cannot solve the row, the original VLM prediction is preserved.

## Scope

The helper is designed for two-by-two Punnett-square questions. It detects relevant rows using both explicit skill/question text and broader genetics keywords such as:

- `punnett`
- `offspring`
- `homozygous`
- `heterozygous`
- `dominant`
- `recessive`
- `genotype`
- `phenotype`
- `cross`
- `probability`
- `ratio`

This broad gate is useful because not every relevant row is guaranteed to use identical metadata wording. However, the helper still requires allele inputs before it can perform any symbolic override. If no parent gametes are available, it returns the model prediction unchanged.

## Information Sources

The Punnett helper uses two kinds of information from the provided example.

First, it reads the text hint to identify the dominant and recessive alleles and, when available, the associated trait phrases. The dataset hint often contains wording such as:

```text
The allele for [dominant trait] (A) is dominant over the allele for [recessive trait] (a).
```

or:

```text
The allele for [recessive trait] (a) is recessive to the allele for [dominant trait] (A).
```

From these sentences, the helper extracts the dominant trait, recessive trait, dominant allele letter, and recessive allele letter. This text tells the solver how to interpret genotype and phenotype questions, but it does not provide the two parent genotypes.

Second, the helper needs the parent gametes from the image margins: two alleles across the top and two alleles down the left side. These can come from either a manually prepared allele override file or from a VLM transcript that explicitly reports the axes in a structured format such as:

```text
TOP: A a
LEFT: A a
```

This design reflects the real bottleneck in the task. The symbolic module can compute the answer exactly after the axes are known, but the axes themselves must still be read from the image or supplied as an override derived from the provided diagram.

## Allele and Genotype Normalization

The helper normalizes each offspring genotype so that uppercase dominant alleles appear before lowercase recessive alleles. For example:

```text
aA -> Aa
Bb -> Bb
```

This normalization makes downstream counting stable. Without it, the same biological genotype could appear in two string forms depending on whether the allele came from the top parent or the left parent.

The helper then constructs the two-by-two Punnett grid by combining each left-margin allele with each top-margin allele. If the top alleles are `A` and `a`, and the left alleles are `A` and `a`, the generated offspring genotypes are:

```text
AA, Aa, Aa, aa
```

This corresponds to the standard row-major Punnett-square construction.

## Supported Reasoning Patterns

After constructing the four offspring genotypes, the helper solves several common question types.

### Homozygous and Heterozygous Questions

For genotype-count questions, the helper counts how many of the four offspring boxes match the requested class:

- Homozygous dominant: both alleles are uppercase, such as `AA`.
- Homozygous recessive: both alleles are lowercase, such as `aa`.
- Heterozygous: the two alleles differ, such as `Aa`.

The resulting count is always between zero and four. The helper then maps that count to a matching answer choice.

### Dominant and Recessive Phenotype Questions

For phenotype questions, the helper uses the dominance relationship from the hint. A dominant phenotype appears whenever at least one allele is uppercase. A recessive phenotype appears only when both alleles are lowercase.

For example, with offspring genotypes:

```text
AA, Aa, Aa, aa
```

the dominant phenotype count is three and the recessive phenotype count is one.

### Probability Questions

Many Punnett questions ask for the probability that an offspring will have or will not have a particular trait. The helper parses the asked trait phrase from the question and compares it against the dominant and recessive trait phrases extracted from the hint.

If the question asks for the dominant phenotype, the helper uses the dominant phenotype count. If it asks for the recessive phenotype, it uses the homozygous recessive count. If the question is negated, such as asking for the probability that offspring will not have a trait, the helper complements the count by subtracting it from four.

The final count is then converted to a fraction or percentage answer choice.

### Ratio Questions

Some questions ask for an expected ratio, such as the ratio of offspring with one phenotype to offspring with another phenotype. The helper parses the two clauses on either side of the ratio, maps each clause to a phenotype count, and then matches the ordered ratio against the answer choices.

For example, if the two counts are three and one, the helper matches answer choices equivalent to:

```text
3:1
```

The ratio matcher treats equivalent ratios as the same when appropriate, using cross multiplication rather than relying only on exact string equality.

## Mapping Counts to Answer Choices

The competition answers are multiple-choice strings rather than raw counts. The helper therefore converts computed counts into the format used by the answer choices.

For probability and fraction questions, the helper supports common equivalent forms:

- `0/4`, `0`, `0%`
- `1/4`, `25%`, `0.25`
- `2/4`, `1/2`, `50%`, `0.5`
- `3/4`, `75%`, `0.75`
- `4/4`, `1`, `100%`, `1.0`

For ratio questions, it parses numeric ratio strings such as `3:1` and compares them mathematically. This allows the solver to match the answer even if formatting varies slightly across choices.

## Hybrid Inference Integration

The integration point is the hybrid Punnett prediction wrapper. During inference, the model first predicts an answer choice. The helper then attempts to obtain the Punnett axes:

1. If an allele override is available, it uses that.
2. Otherwise, it tries to parse `TOP:` and `LEFT:` lines from a VLM transcript.
3. If neither source provides all four alleles, it keeps the model prediction.

Once the axes are available, the helper verifies that the row is a Punnett or genetics question, constructs the Punnett grid, solves the supported question pattern, and returns a symbolic answer if one is found.

Conceptually, the flow is:

```text
model_prediction = VLM(row)
axes = manual_override or parsed_VLM_axes

if axes are available and row is Punnett-related:
    symbolic_prediction = solve_punnett_grid(row, axes)
else:
    symbolic_prediction = unavailable

if symbolic_prediction is available:
    final_prediction = symbolic_prediction
else:
    final_prediction = model_prediction
```

This keeps the helper narrowly scoped. It does not override predictions simply because a row mentions genetics; it must have the four parent gametes and a supported question form.

## Reliability and Limitations

The helper is deterministic once the parent gametes are known. Its main advantage is that it performs the formal Punnett-square calculation exactly, avoiding errors from implicit model reasoning. This is especially useful for questions involving fractions, percentages, phenotype probabilities, and expected ratios.

The main limitation is that the helper does not independently solve the visual axis-reading problem. It relies on either a manually supplied allele override or a structured VLM transcript that correctly identifies the top and left margin alleles. If the axes are wrong, the symbolic calculation will also be wrong. This makes the helper most reliable when the parent gametes have been read accurately from the provided diagram.

The helper also supports only the common two-by-two Punnett-square patterns present in the dataset. It is not a general genetics engine. More complex inheritance patterns, larger crosses, incomplete dominance, sex-linked traits, or unusual wording would require additional logic.

## Contribution to Final Performance

The Punnett helper was included in the final hybrid strategy because Punnett-square questions remained weak under model-only inference. The VLM could often identify the topic and sometimes produce plausible reasoning, but it was inconsistent at converting the grid into exact genotype or phenotype counts. By moving the counting and answer-choice matching into a symbolic module, the system reduced this source of error.

Together with the food-web helper, the Punnett module illustrates the value of targeted symbolic reasoning for structured science diagrams. The learned VLM remains responsible for broad multimodal understanding, while deterministic helpers correct narrow categories where the required reasoning follows explicit rules. This hybrid approach was important for improving the final submission score while staying within the competition's trainable-parameter and compute constraints.

