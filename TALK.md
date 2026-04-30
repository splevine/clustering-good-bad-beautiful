# Clustering: The Good, The Bad and The Beautiful — talk track

**Venue:** ODSC AI East 2026 · 30-min slot · ~20 min talk + 5-10 min Q&A.
**Running dataset:** TMDB top 5,000 films (5K core), 100K scale-up for the WWII reveal. Title nods to *The Bad and the Beautiful* (1952).

**Flow.** Three chained HTML decks, all driven by `space` / `←` / `esc`:

```
open.html  →  slides.html  →  close.html
   5 scenes      22 scenes        5 scenes
   ~3 min        ~16 min          ~2:30
```

`space` at the last scene of each file fades to black and loads the next. `←` at scene 1 fades back to the previous file's last scene. `esc` from anywhere returns to `index.html`.

**Pipeline backbone** (persistent nav at the top of every `slides.html` stage slide):

```
raw data → encoding → dim reduction → clustering → labels → actions
```

---

## 0 · Open (`open.html` — 2 min)

### S1 · "What's in here?" (20s)
- *"In machine learning, there is one question you can't answer with a prediction."*
- Mega type lands: **What's in here?**

### S2 · Dots cluster (45s — auto)
- 420 gray dots scatter across the screen, then migrate into 8 color-coded Gaussian clusters.
- Caption 1: *"5,000 films. No labels. No schema."*
- Caption 2 (after the dots land): *"Clustering answers the question no one has labeled yet."*

### S3 · The anecdote (45s — **space-advanced**, four beats)
1. *"I work on finding structure in customer experience."*
2. *"Hundreds of clients have handed us piles of text — tickets, reviews, survey responses — no labels, asking 'what's in here?'"*
3. *"I can't show you client data. Same pipeline — UMAP, HDBSCAN, BERTopic — pointed at 5,000 movies. 47 clusters."*
4. (mono, muted) **`new_young_life_family`** — "*one of them, verbatim.*"

### S4 · Labels pivot (auto-flip after 2.2s)
- Before: BERTopic shipped me **`new_young_life_family`**
- After: Claude labelled it **Coming-of-age drama**
- Kicker: *"The clustering didn't get smarter. **The labelling layer did.**"*

### S5 · Pipeline cascade (30s)
- Line: *"Clustering isn't a button. It's a pipeline."*
- Six boxes cascade in, then highlight in sequence.
- Final: *"Let's walk this."* — press space to enter `slides.html`.

---

## 1 · The deck (`slides.html` — ~16 min, 22 scenes)

> Persistent ODSC East 2026 logo on the title (s1) only. Pipeline nav lit on the active stage for every stage slide.

### S1 · Title (~10s)
- **The Art of Clustering** · The Good. The Bad. The Beautiful.

### S2 · Hook (30s)
- *"Run clustering twice. Get two different answers."*
- *"Clustering doesn't **discover** structure. It **creates** it."*

### S3 · Pipeline (30s)
- 6-stage pipeline diagram, neutral.
- *"Every step is a decision about what structure you're **allowed** to see."*

### S4 · Raw data — triptych (30s)
- Stage: **raw data** · *"What signal exists?"*
- **Good:** 5,000 films from TMDB · 120 years · long-tail metadata
- **Bad:** junk in, junk out · clean before you cluster
- **Beautiful:** multimodal — overviews *and* posters

### S5 · Encoding — triptych (45s)
- Stage: **encoding** · *"What 'similar' means."*
- **Good:** sentence-transformers · `all-MiniLM-L6-v2` · 384-D
- **Bad:** TF-IDF / bag of words · *plane crash ≠ airplane disaster*
- **Beautiful:** CLIP encodes text and images into the **same** 512-D space → average per film → fused embedding (notebook section 7)

### S6 · UMAP knob sweep — Good (45s)
- Stage: **dim reduction**
- Two videos auto-loop side-by-side: `animation_text.mp4` (`n_neighbors` 2 → 60 → 2) and `animation_text_min_dist.mp4` (`min_dist` 0.0 → 0.8 → 0.0). 2,000 movie overviews, colored by primary genre.
- *"Two knobs really matter: `n_neighbors` (local vs global structure) and `min_dist` (how tight clusters sit in the projection). Notice I'm sweeping — never set them by memory."*

### S7 · Distance compression — Bad (45s · **MANIM**)
- Stage: **dim reduction**
- Plays `slides/distance_compression.mp4` (a numpy-driven Manim render of pairwise distance distributions concentrating as `d` grows from 2 to 384).
- Stats: **0.47** raw 384-D ratio · **0.05** UMAP 5-D ratio
- *"Skip dim reduction → density-based clustering has nothing to grip."*

### S8 · HDBSCAN — Good (45s)
- Stage: **clustering** · *"No `k`. Density-based."*
- **47 natural clusters · 48.5% noise.** *"Half doesn't fit a tight theme. That's honest, not a failure."*

### S9 · *Inception*'s neighbors — Bad (1 min)
- Stage: **clustering** · k-means, two settings of `k`.
- `k=5`: Shawshank · Django · Deadpool · `k=80`: Fight Club · Kingsman · Matrix Reloaded
- *"Same data. Different knob. Different story. **If the story changes with a knob turn, the story isn't about the data.**"*

### S9b · Topics over time — bridge (1 min · embedded iframe)
- Stage: **clustering · over time**
- Shows `topics_100k_over_time.html` inline. *"BERTopic ships a temporal view: topic prevalence across decades."*
- Tease: *"Hover the line for 'WWII & Nazi Germany.' Where does it start?"* (live-hover the line — it extends back into the early 1900s).
- This sets up the WWII reveal.

### S10 · WWII setup — Beautiful (30s)
- Stage: **clustering**
- *"Claude labelled this cluster '**WWII & Nazi Germany.**'"*
- Four canonical members: Dunkirk (2017), Schindler's List (1993), The Bridge on the River Kwai (1957), Casablanca (1942).
- *"Looks right."*

### S10b · WWII reveal — the punch (1 min)
- *"Now look at the rest of it."*
- Earlier members: Grand Illusion (1937) · Triumph of the Will (1935) · Westfront 1918 (1930) · Wings (1927) · The Big Parade (1925) · I Accuse (1919) · **Red Cross Ambulance on Battlefield (1900)**.
- **134** films before WWII · **97** before Hitler took power · **22** before WWI even began.
- *"It isn't a WWII cluster. It's a **war-shaped** cluster."*
- *"Clustering captures meaning — not metadata."*

### S11 · Two paths — Distinction (1.5 min · **MANIM**)
- Plays `slides/use_cases.mp4`. Central dataset forks: top path re-clusters three ways (EXPLORATION · *instability is signal*); bottom path clusters once → classifier → labels (LABELING · *consistency matters*).
- Voiceover ask: *"Are you clustering to **understand** your data, or to **generate labels** for a system?"*
- Punchline (in the video): *"Clustering is sketching. Classification is building."*

### S12 · Sketch → Blueprint (45s · **MANIM**)
- Plays `slides/sketch_blueprint.mp4`. Hand-drawn sketch on the left, precision blueprint grid on the right.
- *"Clustering → discover structure. Classification → scale it."*

### S13 · Labels pivot — emotional centerpiece (1.5 min · **space-advanced**)
- Stage: **labels**
- Before (mono, muted): `0_planet_earth_space_alien` · `3_queen_prince_princess_king` · `4_heist_bank_drug_police`
- Press space → cross-fade to **Claude relabel**: Space sci-fi · Royalty & fairy tales · Neo-noir crime
- *"Nothing in the pipeline got smarter. **The labelling layer did.**"*

### S13b · Live `map.html` — the lift at scale (1 min — speaker-paced)
- Stage: **labels · the lift at scale**
- Embedded `map.html`: 5K films, 5 zoom levels (BERTopic hierarchy 80→40→20→10→5).
- Demo moves: zoom out → 5 coarse themes (*Superheroes & fantasy worlds · Crime, spies & capers · Human drama & family · Space & sea adventures · Horror & survival thrillers*); zoom in → 47 specific (*Vampire romance · Slasher horror · Space sci-fi*); search "Inception"; drag the year filter into the 2010s.
- *"This is the same lift, applied across the whole catalog. 5,000 films, 47 specific themes, 5 super-themes — labelled by Claude in under \$0.50."*

### S14 · Genre purity — Good (30s)
- Stage: **actions**
- **0.38 vs 0.19 baseline** — ~2× the signal of chance.
- *"And these clusters were never told what genre is."*

### S15 · Cluster → classify — Beautiful (30s)
- Stage: **actions**
- *"Clustering finds the structure. Classification — trained on the labels you wrote — applies it to the next 10 million rows."*
- *"**Cluster to discover** · → · **classify to deploy.**"*

### S15b · Poster constellation — visual payoff (45s)
- Stage: **encoding** · *"Same pipeline. CLIP instead of MiniLM."*
- Embedded `map_posters.html`: each point is the actual poster at its UMAP coordinate.
- Speaker zooms in to read titles, zooms out to see the catalog's shape.
- *"Multimodal isn't harder. It's a second lens."*

### S15c · 3D cosmos — visual finale (30s)
- Embedded `cosmos.html`: 1,000 films in 3D, auto-rotating until you grab the camera.
- Let it run for 8-10 seconds. Then: *"Catalogs become constellations."*
- This is the most arresting visual in the deck — sets up the closing line.

### S16 · Closing line
- *"You're not discovering structure."*
- **"You're *designing* a perspective."**

### S17 · Final 4 bullets (space-advanced, one at a time)
1. Clusters ≠ truth
2. Instability is signal
3. Exploration ≠ labeling
4. The goal is decisions

Last space → fade → load `close.html`.

---

## 2 · Close (`close.html` — ~2:30)

### S1 · Montage (45s — auto-cycling 4 clips)
- `animation_text.mp4` → "Sweep, don't set."
- `animation_posters.mp4` → "Swap the encoder."
- `map_static_100k.png` → "The labelling layer got smarter."
- `cosmos.html` (iframe) → "Catalogs become constellations."

### S2 · Box quote (30s)
- *"All models are wrong, **but some are useful.**"* — George E. P. Box, 1976.

### S3 · Takeaway (30s)
- *"Clustering is a pipeline, not a button."*
- *"And every stage has an LLM-era upgrade you might not be using yet."*

### S3b · Built with — open source shoutout (30s, staggered fade-in)
- 16 library tiles fade in with a staggered cascade: Hugging Face, PyTorch, scikit-learn, Anthropic Claude · pandas, NumPy, JupyterLab, TMDB · UMAP, HDBSCAN, BERTopic, datamapplot · Plotly, Matplotlib, Manim, EVoC.
- *"None of this exists without these. Standing on the shoulders of giants."*

### S4 · Thanks + QR (30s + Q&A)
- **Thank you.** · Seth Levine · Director of AI Innovation, Contentsquare.
- QR code → `github.com/splevine/clustering-good-bad-beautiful`.
- Easter egg: press **`c`** for a poster swarm — only deploy if Q&A is friendly and time allows.

---

## Pre-flight checklist

- [ ] Open all four pages in the presentation browser once to warm the cache and confirm the prefetched chain works (`open.html` → `slides.html` → `close.html`).
- [ ] Confirm `slides/distance_compression.mp4`, `slides/use_cases.mp4`, `slides/sketch_blueprint.mp4` autoplay (browser may require a user-gesture first — clicking once into the page satisfies this).
- [ ] Confirm `topics_100k_over_time.html` iframe loads on slide 9b and the WWII line is hoverable.
- [ ] Verify `esc` returns to `index.html` from each file.
- [ ] Have local copies of every `.html` and `.mp4` on disk in case GitHub Pages flakes.
- [ ] Rehearse the three space-advanced moments: open S3 (4 beats), slides S13 (labels pivot), slides S17 (4 bullets).
- [ ] Time-trial: aim for 19-20 min so there's a 5-10 min Q&A buffer.

## Q&A anticipations (≤ 60s each)

- *"Why not k-means?"* — *"Still useful when clusters are spherical and `k` is known. For sentence embeddings in 384-D, it's the wrong default."*
- *"How do I pick `min_cluster_size` for HDBSCAN?"* — *"Sweep 10/20/50/100. Structure that survives across settings is more trustworthy. Default `min_samples=5`."*
- *"Can I do this on 10M documents?"* — *"Yes — batch-encode on GPU, faiss for ANN, UMAP scales surprisingly well. BERTopic has memory-efficient `vectorizer_model`."*
- *"What about silhouette / gap statistic?"* — *"Useful signals, not oracles. HDBSCAN sidesteps `k`-selection."*
- *"Which LLM for labeling?"* — *"Any frontier model. Used Claude Haiku for speed and cost — 155 labels under \$0.50."*
- *"Why does the map show 'Minor subtopics'?"* — *"datamapplot's default for HDBSCAN-noise points that fall inside a labeled parent region. It's the library being honest about points that don't fit a tight subtopic."*
- *"Where's the code?"* — `github.com/splevine/clustering-good-bad-beautiful` · Colab badges on every notebook.

## Artifact cheat-sheet

| Moment | File | Scene / Path |
| --- | --- | --- |
| Cold open | `open.html` | s1 — *"What's in here?"* |
| Dots cluster | `open.html` | s2 |
| Anecdote (space-advance) | `open.html` | s3 |
| Labels-pivot teaser (auto) | `open.html` | s4 |
| Pipeline appears | `open.html` | s5 |
| Hook · "two answers" | `slides.html` | s2 |
| Pipeline overview | `slides.html` | s3 |
| Distance compression Manim | `slides.html` | s7 → `slides/distance_compression.mp4` |
| Inception cluster-mates | `slides.html` | s9 |
| **Topics over time bridge** | `slides.html` | **s9b** → `topics_100k_over_time.html` |
| **WWII reveal** | `slides.html` | **s10 + s10b** |
| Two-paths Manim | `slides.html` | s11 → `slides/use_cases.mp4` |
| Sketch → Blueprint Manim | `slides.html` | s12 → `slides/sketch_blueprint.mp4` |
| Labels pivot (full · space-flip) | `slides.html` | s13 |
| **Live `map.html` demo** | `slides.html` | **s13b** → `map.html` (5-layer hierarchy) |
| **Poster constellation** | `slides.html` | **s15b** → `map_posters.html` |
| **3D cosmos** | `slides.html` | **s15c** → `cosmos.html` |
| Closing line | `slides.html` | s16 |
| Final bullets (space-advance) | `slides.html` | s17 |
| Montage | `close.html` | s1 |
| Thanks + QR | `close.html` | s4 |

## Why this structure

- **The pipeline is the spine.** Every slide between Title and Closing lives at one of six stages — the audience always knows where they are. The persistent nav reinforces it.
- **Good / Bad / Beautiful per stage, not as three monolithic acts.** The audience sees both the lift *and* the failure mode for the same stage, in immediate juxtaposition.
- **The Distinction (s11 + s12) sits at the centerpiece.** Exploration vs labeling is the new earned moment — it reframes everything that came before it. The two Manim animations buy you the visual weight to land it.
- **The WWII reveal is the audience-favorite "wait, what?"** The numbers are real (1063-film cluster spans 1900–2026; 134 pre-WWII members). Use the topics-over-time bridge (s9b) to plant the seed before the punch.
- **Closing is short, visual, and quotable.** Box quote → "pipeline, not a button" → QR. Audience leaves with one line and one URL.
