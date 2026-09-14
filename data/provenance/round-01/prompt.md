Classify every row in the attached CSV. Use only the `sentence` column as the TARGET SENTENCE. Apply the following prompt independently to each row, replacing `{}` with that row's sentence.
Return a downloadable CSV with exactly the same 158 rows, in the same order, preserving every original column and value exactly. Add exactly one column named `llm_label`. Each value must be exactly lowercase `yes` or `no`. Do not add explanations, markdown, code fences, or other columns. Do not overwrite or infer any human annotation. Do not use web search.
Prompt:
Classify whether the TARGET SENTENCE contains information about the target place that could help characterize its land use, land cover, or geographic environment from remote sensing, either directly or through observable proxies.
Return exactly one token: yes or no.
Answer yes for information about vegetation, agriculture, forests, water, soil or surface, terrain, buildings, settlements, infrastructure, transport networks, mining, managed land, or other human or natural features with a spatial or remotely detectable signature.
Answer no for information only about history, administration, people, events, demographics, economy, navigation, or activities with no meaningful land-use, land-cover, or remotely detectable implication.
Output only the lowercase token yes or no.
TARGET SENTENCE: {}
