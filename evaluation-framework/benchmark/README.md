# Retrieval benchmark evidence, revision 2

The original 27 questions and types are retained. Their 152 evidence units describe
individual facts, with PDF quotations and reviewed mappings to the frozen
103-concept corpus. No generated answers were used as ground truth. Descriptions
and bodies both count as OKF content because retrieval indexes both.

## Evidence fields

- `id`: unique within the question; IDs changed in this revision.
- `fact`: the precise claim and its required qualifiers.
- `relevance`: 3 essential, 2 supporting, or 1 contextual.
- `derived_from`: previous evidence IDs, or an empty list for new facts.
- `pdf_anchors`: filename, one-based physical PDF pages, and supporting quotation.
  A quotation can include context beyond its target fact.
- Each anchor's `required_quotes` lists the complete source excerpts needed to
  establish its fact, in source order. Context not needed for that fact can be
  omitted between excerpts. All excerpts must occur in the same retrieved record.
- `pdf_anchor_mode: "any"`: each anchor independently establishes the same fact.
- `okf_concept_ids`: ONLY concepts individually supporting the complete fact.
- `okf_support`: full/partial judgments, verbatim excerpts, and reasons for partial
  support. The excerpts can come from concept descriptions or bodies.
- `coverage_note`: required explanation when no concept fully supports the fact.

A complete classification set or list of materials is one relationship: all
members named by `fact` must be supported for full credit. Independent
relationships are split. Resource classifications have separate units for each
basis; each waterway has separate route and length units; missing canal details
have their own units.

## Partial and combined support

| Situation | Rule |
|---|---|
| Complete fact supported | Full binary evidence credit. |
| A season is named but required months are missing | Partial annotation; no full credit. |
| No concept preserves the fact | Empty full-support IDs; keep the PDF fact in the denominator. |
| Several anchors establish the same fact | Alternatives: any complete anchor suffices. |
| Several independent facts are required | Separate units; combine coverage across retrieved results. |
| Multiple fragments are needed for one fact | No fragment is full. Record partial support; joint-fragment entailment is outside this revision. |
| Multiple concepts fully support the same fact | Full alternatives; credit the fact once. |
| Only topic, filename, or page overlaps | No evidence credit. |

Joint-anchor (`all`) units are rejected by the validator and matcher. Combined requirements
are represented as separate facts, preventing one waterway or planning stage
from earning credit for an entire answer. Full and partial labels are editorial
judgments, not generated relevance scores. Lexical checks verify excerpt
locations, not semantic entailment. Partial support never enters full-support IDs.

## Frozen inputs and validation

`evidence_audit.json` records the benchmark hash, all 103 reviewed concept hashes,
all nine PDF hashes, and full-support mapping changes per question. The 55 facts
without full OKF coverage remain in the benchmark to expose conversion loss.
The corpus was not edited or regenerated; its unrelated factual errors and lack
of original stored provenance are not certified by this audit.

The resource-planning description preserves national-plan alignment even though
its body omits it. The iron-ore concept does not preserve the export or slurry
pipeline facts, so those full mappings were removed. Steelmaking material and
ratio mappings now include the mineral-based-industries concept where supported.

Quotes retain extraction spelling and artifacts, including the repeated `tankas`
tokens actually emitted by the PDF text layer. Normalization ignores Unicode
presentation, whitespace, and formatting punctuation. Numeric ratio, fraction,
decimal, and percent meaning is preserved: `4:2:1` does not match `42:1`, and
`No.1` does not match `No.10`. Words, negations and numbers are not discarded.
Required excerpts may explicitly omit repeated extraction tokens, such as
duplicated `tankas`, while retaining all the words needed for the fact.

Run from the repository root in the converter environment (which has PyMuPDF):

```powershell
.\pdf-to-okf-auto\.venv\Scripts\python.exe -B evaluation-framework/scripts/validate_benchmark.py
```

This checks evidence structure, full/partial consistency, concept excerpt
locations, quotations on cited PDF pages, and frozen fingerprints. Deliberate
benchmark/corpus changes need annotation review and an updated snapshot together;
do not overwrite hashes merely to suppress drift errors. Source files must be
available locally at the configured paths.

## Relevance matching and scoring status

PDF matching now requires the correct document, all cited physical pages, and
every required excerpt in order within the retrieved text. For legacy anchors
without `required_quotes`, the entire anchor quotation is required. Empty or
invalid provenance/text fails closed. A long partial quote does not get full
credit. Partial records are not concatenated into a complete fact.

OKF matching requires an audited full-support ID and all of that annotation's
support excerpts in the record text. IDs alone and partial annotations cannot
earn full credit. Hybrid results are judged by their underlying PDF/OKF type.

This is conservative lexical matching, not semantic entailment: paraphrases,
missing context, and extraction differences beyond normalization may be false
negatives. There are no keyword-overlap thresholds, fuzzy substring scores, or
LLM judgments. Keep the selected excerpts under review against actual chunks.

Old result files used different evidence units and matching rules. Do not compare
their scores with newly evaluated results. Metric definitions still require the
separate review described in the project assessment. Successful annotation or
matching tests do not certify all evaluation metrics. No experiment report was
rewritten by this change.

This remains a pilot: nine factual, nine conceptual, and nine multi-hop-labelled
questions over seven chapter PDFs. The other two PDFs were considered through
their concepts but have no benchmark questions. These labels do not establish
cross-document multi-hop coverage. All questions are answerable from the PDFs;
missing OKF facts do not make them unanswerable corpus questions.
