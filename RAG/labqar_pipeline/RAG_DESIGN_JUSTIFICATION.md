# RAG design justification

This documents the technical choices for the RAG part of the project and
why each one fits the data, not just that it satisfies a checklist item.

## Data characteristics that drove every choice below

- **Corpus size: 550 rows.** Small by RAG standards. This rules out
  anything justified mainly by *scale* (distributed vector DBs, large
  hosted embedding APIs, approximate-NN indexes) — none of that pays for
  itself here.
- **Fully structured, low-noise source.** Every LabQAR row already has
  clean `parameter / specimen / gender / age_group / category / condition /
  reference_type` fields (see Step 1). That is a much stronger signal than
  raw free text, and it's *free* to use — no embedding needed to read a
  field that's already typed and clean.
- **Context changes correctness, not just relevance.** The same test name
  (e.g. Cholesterol) legitimately maps to different numeric bounds
  depending on specimen/gender/age/condition/reference_type (see Step 2's
  docstring). A retriever that ignores this can retrieve a *plausible* but
  *wrong* row — worse than retrieving nothing, because it's a wrong lab
  interpretation with high confidence.
- **Two different query patterns exist in this project**, and they need
  different retrieval strategies:
  1. The production path: a fine-tuned extraction model produces clean
     structured fields from a lab panel.
  2. A free-text path: a question typed in natural language, with no
     structured fields at all (what the assignment's RAG requirements are
     actually testing).

## Choice 1 — Exact structured match stays the primary retriever

`step3_retriever.py`'s exact-match-on-metadata approach is kept as the
first-line strategy for query pattern (1) above. Justification: when the
caller already has clean fields, exact match is 100% precise (verified in
`step5_evaluate.py`: 100% retrieval accuracy) and free (no embedding call,
no vector search, no latency). An embedding-based approach would be *worse*
here, not just unnecessary — it introduces a small but real chance of
retrieving a *near*-match instead of the *exact* match a clean structured
query deserves.

## Choice 2 — Embeddings + a vector DB for the free-text path

For query pattern (2), exact match has literally no mechanism to work at
all, and the old lexical fuzzy fallback (character n-gram TF-IDF) only
catches *substring* overlap — it fails on genuine paraphrase with no shared
n-grams. This is where dense embeddings earn their cost: `step3b_embeddings.py`
encodes each corpus document once, and `step3d_hybrid_retriever.py` embeds
the free-text query at request time and does nearest-neighbor search
(`step3c_vector_store.py`).

- **Embedding model** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  CPU-friendly, no API key/cost). Given the corpus is 550 short, single-
  sentence documents, a small local model is a better fit than a large
  hosted embedding API: at this scale, API latency/cost would dominate for
  no measurable accuracy gain, and reproducibility (running the same
  pipeline offline, in a notebook, without secrets) matters more than
  squeezing out the last point of embedding quality.
- **Vector database** — Chroma, embedded/in-process, persisted to disk. At
  550 documents there's no scaling problem to solve, so a managed/hosted
  vector DB (Pinecone, Weaviate Cloud, etc.) would add an external service
  dependency for zero benefit. Chroma's native metadata `where` filtering
  is used so semantic search can still be narrowed by known context (e.g.
  gender, specimen) exactly the way the exact-match retriever is, keeping
  the "context changes correctness" property intact even on the semantic
  path.

## Choice 3 — Hybrid routing, not "embeddings everywhere"

`step3d_hybrid_retriever.py` routes structured queries to exact match and
free-text queries to semantic search, falling back from one to the other
rather than picking a single strategy globally. This follows directly from
having two different query patterns in the same project (see above) — a
single retrieval strategy would be measurably worse for at least one of
them, which is exactly what `step5c_evaluate_rag.py` empirically shows:
exact match is unbeatable (100%) on structured queries; only semantic
search has a chance on paraphrased free text.

## Choice 4 — Evaluating retrieval on a *hard* query set, not just the easy one

An early version of the free-text eval set embedded the literal parameter
name in every query template (e.g. *"what is a normal **Cholesterol**
level?"*). That's not a fair test of semantic understanding — a lexical
method wins on it for the wrong reason (it's matching the substring, not
the meaning), which is visible directly in the measured results: the raw
lexical TF-IDF baseline actually *beat* the offline embedding fallback on
that set. `step5b_generate_eval_queries.py` therefore also includes a small
hand-written **hard set** with genuine synonyms and no literal overlap
("liver enzyme SGPT test" → ALT, "good cholesterol" → HDL, etc.) — this is
the set that actually tests what embeddings are for, and it's evaluated
separately in `step5c_evaluate_rag.py` rather than averaged together with
the easy set, so the numbers don't hide which capability they're measuring.

## Choice 5 — Generation stays constrained and gets its own evaluation

`step6_llm_explain.py` never lets the LLM decide the Low/Normal/High
verdict — that stays a deterministic calculation (`step4`). The LLM's only
job is to turn an already-decided verdict plus retrieved context into a
readable sentence. This constraint is what makes automatic generation
evaluation tractable without a human-labeled reference set: `step5c`
checks (a) verdict-consistency (does the text avoid contradicting the
given status) and (b) numeric groundedness (does every number in the text
correspond to a retrieved bound or the measured value, rather than being
invented). Both are meaningful precisely because the generation task is
narrow — a similarity-to-reference metric (BLEU/ROUGE) would be meaningless
here since there's no gold explanation text to compare against.

`step6b_freetext_rag_qa.py` adds the more open-ended, classic
retrieve-then-generate pattern for genuinely free-text questions, kept
separate from step6 because it has a different failure mode (the model can
fail to stay grounded in retrieved context, not just fail to be numerically
accurate) and is evaluated differently as a result.

## Known limitation, honestly stated

This project environment has no internet access, so all the numbers
produced by `step3b`/`step3c`/`step5c` in this repo's own test runs used an
offline TF-IDF+SVD fallback embedder and an in-memory cosine-similarity
fallback vector store, not the real `sentence-transformers` + `chromadb`
stack. That fallback is itself lexical under the hood, so it *understates*
the real benefit of semantic embeddings, especially on the hard/paraphrase
query set. Re-run `step3b_embeddings.py`, `step3c_vector_store.py`, and
`step5c_evaluate_rag.py` in Colab (which has internet) after
`pip install sentence-transformers chromadb` to get the numbers that
should actually go in your report — the code doesn't change, only the
backend each file auto-selects.
