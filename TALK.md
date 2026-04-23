# Clustering: The Good, The Bad and The Beautiful — talk outline

**Venue:** ODSC AI East 2026 · 30-min virtual slot · targeting ≈23 min of talk + 5-7 min Q&A buffer

**Structure:** the unsupervised-learning pipeline is the navigational backbone and sits on screen across the talk with the current stage highlighted. The **ASK → THE GOOD → THE BAD → THE BEAUTIFUL** arc honors the title's order: you see the working pipeline first, then what would have broken it, then what becomes possible once it works across modalities.

```
raw data  →  encoding  →  embeddings  →  dim reduction  →  clustering  →  clusters  →  labels  →  actions
```

Seth's version of this slide lives in `slides/pipeline.*`.

**Running dataset:** top 5,000 movies by TMDB vote count. Overviews for text, posters for image. Title nods to *The Bad and the Beautiful* (1952).

---

## 0 · ASK — open (2 min)

- **Cold open (30s).** Poster palindrome animation (`animation_posters.mp4`) full-screen. No voiceover for the first five seconds.
- **Frame the ask (45s).** *"I have 5,000 movies. I want to know what themes live in this catalog — the real ones, not the genre tags a studio's marketing team typed in — and I want a visual I can show an exec in 30 seconds. That's the whole ask."*
- **Set the expectation (45s).** *"I'll walk one pipeline end-to-end. It's the same pipeline you'd use for customer reviews, support tickets, product descriptions — anything text-ish. We'll start with the pipeline working well (the Good), then I'll show you what would have gone wrong at each stage (the Bad), and then we'll see what becomes possible once the pipeline runs clean (the Beautiful)."* Show the pipeline slide. This is the map.

---

## 1 · THE GOOD — walk the pipeline (12 min)

*Notebook: `01_the_good.ipynb`. One dataset, one pipeline, stage by stage.*

### Raw data (45s)
- TMDB, 5,000 films by vote count. Top-10 slide: Interstellar, Inception, Dark Knight, Avatar… *"This is the messy-metadata-plus-long-tail case most analysts face — some English, some foreign-language, titles spanning 120 years."*

### Encoding + embeddings (1.5 min)
- sentence-transformers, `all-MiniLM-L6-v2`, 384-D. One line on a slide: `model.encode(overviews)`. *"This is the boring-but-everything stage — your clustering can only be as good as your embedding."*
- *"Two years ago we were bag-of-words and TF-IDF. Now we encode a sentence's meaning. Every downstream step inherits that lift."*

### Dim reduction — UMAP (2 min)
- Play `animation_text.mp4`. Palindrome sweep of `n_neighbors` 2 → 60 → 2 with clusters morphing on screen. Talk over it.
- *"Two knobs really matter: `n_neighbors` (local vs. global) and `min_dist` (how tight clusters are). Notice I'm sweeping — never set them by memory."*
- *"Two projections from the same data: 5-D for the clusterer to breathe in, 2-D for visualization."*

### Clustering — HDBSCAN (2 min)
- *"No k. Density-based. Noise labelled `-1` by design."*
- Show the numbers: **47 natural clusters, 48.5% noise.** *"Half the catalog doesn't fit a tight theme. That's honest. Those are your unusual films — the data you want to investigate, not ignore. k-means would have forced every one of them into a bucket and called it done."*

### Clusters (30s)
- Show the 2-D scatter at the coarsest layer of `map.html` — no labels yet. *"Forty-seven clusters, visible structure, no names. Now the real work."*

### Labels — the pre/post-LLM pivot (3 min) 🎬
*The emotional centerpiece. Slow down here.*

- **BERTopic's native labels.** `0_planet_earth_space_alien`, `3_queen_prince_princess_king`, `4_heist_bank_drug_police`. *"This is what I was shipping in 2023. They're not wrong — they're just ugly. You can't put them in front of a stakeholder."*
- **The same clusters, relabeled by Claude.** **Space sci-fi · Royalty & fairy tales · Neo-noir crime.** *"Nothing in the pipeline got smarter. The labeling layer did. That's the whole story of the last two years for anyone doing this work."*
- **Live demo: the 5-layer hierarchy.** `map.html`. Zoom out: 5 coarse themes (*Superheroes & fantasy worlds · Crime, spies & capers · Human drama & family · Space & sea adventures · Horror & survival thrillers*). Zoom in: 47 specific ones (*Jurassic dinosaurs · James Bond · Batman saga · Vampire romance*). Search "Inception", watch it highlight. Drag the release-year filter to the 2010s, watch the emphasis shift.

### Actions (45s)
- *"Now the pipeline pays off: you can cross-reference, segment, recommend, brief a stakeholder."*
- Genre-purity validation: **0.38 vs. 0.19 baseline** — ~2× the signal of chance.
- *"That's The Good. Now let me show you every way it could have broken."*

---

## 2 · THE BAD — what would have gone wrong (4 min)

*Notebook: `02_the_bad.ipynb`. You've seen the pipeline work. Each beat rewinds to a stage you just saw and shows how a shortcut breaks it. Brisk pacing — every failure mode is one idea and one visual.*

- **Transition (20s).** *"The pipeline you just saw protects against four failure modes that bite every clustering project. Let me show them quickly — these are the guardrails, not the villains."*

- **Skipping dim reduction → distance compression (45s).** Stage: *dim reduction*. Show the one-number story: raw 384-D nearest/farthest ratio **0.47**, UMAP 5-D ratio **0.05**. *"At 384 dimensions, nearest and farthest neighbors sit in the same narrow band. Density-based clustering can't tell them apart. Skipping UMAP is the most common mistake I see."*

- **Parameter choices *become* the story (1 min).** Stage: *clustering*. *Inception*'s cluster-mates at `k=5`: Shawshank, Django, Deadpool. At `k=80`: Fight Club, Kingsman, Matrix Reloaded. *"Same data, different knob, different story. If the story changes with a knob turn, the story isn't about the data. Sweep, don't set."*

- **Clusters from structureless data (45s).** Stage: *clustering*. Gaussian noise, `k=8`, k-means returns eight clean-looking clusters. *"Clustering algorithms never tell you whether structure exists. They return a partition regardless. If your only evidence for structure is 'the algorithm gave me 8 groups,' you have no evidence."*

- **Instability across runs (30s).** Stage: *clustering*. Same data, three seeds, k-means on raw 384-D: pairwise **ARI ≈ 0.55**. *"Cluster identity shifts just by re-running. If your downstream decisions depend on 'cluster 3 is romantic comedies,' you need to check that identity is stable."*

- **Wrap (20s).** Back to the pipeline slide. *"Four failures, four stages. The pipeline you saw in The Good protects against all of them — UMAP fixes distance compression, HDBSCAN removes the k-question, noise-as-label handles structureless outliers, and a sensitivity sweep beats a single 'best' config."*

---

## 3 · THE BEAUTIFUL — swap the encoder (6 min)

*Notebook: `03_the_beautiful.ipynb`. Same pipeline, different modality. This is what the title's "Beautiful" points at.*

- **Same stages, CLIP instead of MiniLM (1 min).** Flash the pipeline slide again — only the *encoding* stage changed. *"If you can swap the encoder, you can cluster any modality. That's the big idea."* Code diff: `SentenceTransformer("clip-ViT-B-32").encode([Image.open(p) for p in posters])`.

- **The 2-D poster constellation (1.5 min).** Open `map_posters.html`. Audience sees clusters of horror covers, pastel animation, minimalist indie posters. *"Each point is the actual poster. Zoom in — read titles. Zoom out — see the shape of the catalog."*

- **The 3-D cosmos (1 min).** Open `cosmos.html`. Let it auto-rotate. Grab the camera, zoom into a neighborhood.

- **The disagreement insight (1.5 min).** *"Here's what text clustering and image clustering *disagree* on."* Announce the number: **text ARI vs. poster ARI = 0.035.** *"That's essentially uncorrelated. The way a film reads in its overview tells you almost nothing about how its poster looks. And the disagreement is information — those are the films whose marketing and narrative diverge."* Spotlight: Avatar, Coco, Up, Wonder Woman land in clean text clusters but as poster-noise.

- **Wrap the act (30s).** *"The same pipeline on a different encoder gave us a second lens on the same 5,000 films. Multimodal isn't harder — it's a second lens."*

---

## 4 · Takeaways (2 min)

One slide. Five bullets. Spoken briskly.

1. **Always reduce before you cluster.** Raw embeddings lie about distance.
2. **HDBSCAN over k-means for high-dim embeddings.** Noise is a valid label.
3. **Sweep parameters, don't set them.** If the story changes with a knob turn, the story isn't about the data.
4. **LLMs moved labels from `planet_earth_space` to `Space sci-fi`.** The pipeline didn't get smarter; the labeling layer did.
5. **Same pipeline, different encoder = multimodal.** Disagreements between modalities are where the interesting questions live.

**Closing line:** *"The knobs won't find the structure for you. You find the structure — the knobs only decide how cleanly you can show it."*

---

## Q&A buffer (5-7 min)

Anticipated questions, <60s each:

- *"Why not k-means?"* — *"I still use it when clusters are spherical and I know k. For sentence embeddings in 384-D, it's the wrong default."*
- *"How do I pick `min_cluster_size` for HDBSCAN?"* — Sweep 10-20-50-100. Structure that survives across settings is more trustworthy. Default to `min_samples=5`.
- *"Can I do this on 10M documents?"* — Yes: batch-encode with a GPU, use faiss for ANN, UMAP scales surprisingly well. BERTopic has a `vectorizer_model` for memory-efficient c-TF-IDF.
- *"What about silhouette / gap statistic / k-selection?"* — Useful signals, not oracles. HDBSCAN sidesteps the question.
- *"Which LLM for labeling?"* — Any frontier model. I used Claude Haiku for speed + cost — the 155 labels in this demo cost under \$0.50.
- *"Can I get the code?"* — `github.com/splevine/clustering-good-bad-beautiful`. Colab badges on every notebook.

---

## Artifact cheat-sheet

| Moment | Open / play |
| --- | --- |
| Cold open | `animation_posters.mp4` full-screen |
| Pipeline nav | `slides/pipeline.*` |
| UMAP knob story | `animation_text.mp4` |
| Labels pivot — before | `map_classic.html` (keyword version) |
| Labels pivot — after | `map.html` (Claude version) — the primary interactive demo |
| The Bad — distance compression | `02_the_bad.ipynb` section 3 |
| The Bad — parameter sensitivity | `02_the_bad.ipynb` section 4 |
| Visual payoff | `map_posters.html` |
| The 3D moment | `cosmos.html` |
| Disagreement detail | `03_the_beautiful.ipynb` section 6 |

## Pre-talk checklist

- [ ] Cache warm: all HTML + MP4 artifacts load in the presentation browser once before going live.
- [ ] Local backup: have `map.html`, `map_classic.html`, `map_posters.html`, `cosmos.html`, both MP4s on disk in case Pages flakes.
- [ ] Rehearse the 3-min "labels" beat — the memorable pivot and the emotional centerpiece. Make sure the BERTopic-default vs. Claude comparison lands crisply.
- [ ] Rehearse the Bad → Beautiful transition. Don't let the pitfalls section deflate momentum; it's a quick guardrail recap, not a detour.
- [ ] One silent practice run against a timer; aim for 23 min with two natural breath points where you could cut 30-60s if needed.

---

## Why this structure (rationale)

- **Title-faithful order.** The Good, The Bad, The Beautiful — that's the title, so the talk opens with the working pipeline instead of leading with failures. Audience gets something to use before they're shown what to avoid.
- **Pitfalls as guardrails, not despair.** Because The Bad comes *after* the pipeline works, each failure mode lands as "here's the safety rail you just relied on," not as "everything is broken." Shorter, more useful, less negative.
- **K-means demoted from villain.** It appears twice as "the wrong default for this case," never as the thing to avoid everywhere.
- **The labels beat is still the emotional centerpiece** — right in the middle of The Good, at minute 9-10 when attention is highest.
- **The multimodal story closes strong.** Poster constellation + 3D cosmos + text-vs-poster ARI = 0.035 is visual, memorable, and leaves the audience with something to remember long after the Q&A.
