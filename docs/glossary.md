# Glossary

This page defines the project terms. Each term has one meaning in all the documentation.

| Term | Meaning |
| --- | --- |
| Golden set | A set of sentences with labels that a human assigns. |
| Sentence | One unit of text from a source. |
| Relevance label | The label `yes` or `no`. It shows if a sentence is relevant to land use. |
| Source | A public Hugging Face dataset. The project uses two sources: Wikipedia and website text. |
| Candidate | A sentence that the annotator can label. |
| Candidate bank | The stored pool of candidates. The V2 bank holds 512 records (256 per source). |
| Annotator | The person who assigns the labels. |
| Rater | A person or a model that assigns labels in an agreement study. |
| Annotation | A label that an annotator assigns to a sentence. |
| Adjudication | The manual decision on a sentence where the raters disagree. |
| Disagreement | A sentence where the raters assign different labels. |
| Benchmark | A labeled CSV file that you use to measure a model. |
| H3 | A hexagonal grid system for geographic cells. |
| Cell | One H3 area. The project selects one candidate for each cell. |
| Maximin spacing | A selection method. It makes the smallest distance between selected cells as large as possible. |
| Stratification | The division of the data into groups, for example by geographic cell. |
| Quota | The number of records that the project needs for one source or one label. |
| Reserve candidate | A candidate that replaces another candidate when a quota is not complete. |
| Round | One numbered LLM evaluation run. A round is never overwritten. |
| Prompt | The text instructions that the project sends to a model. |
| Manifest | A file that records the inputs, the outputs, and the hashes of a round. |
| Hash | A short code that identifies the exact content of a file. |
| Cohen's kappa | A measure of agreement between two raters. |
| Fleiss' kappa | A measure of agreement between three or more raters. |
| Release | An immutable git tag. A tagged benchmark file is never edited. |
| Seagate drive | The external project drive. It holds the `results/` and `state/` directories. |
| Dataset card | The Hugging Face page that describes a dataset and its licenses. |
| Quality gauntlet | The full set of automated quality checks. |
