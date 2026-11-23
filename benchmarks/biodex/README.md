Instructions for running biodex pipeline:

Setup:
```
conda create lotus-bench python=3.10 -y
conda activate lotus-bench
pip install -r requirements.txt
```

Run the biodex tester file which includes the following two programs:
- Join only
- JoinAndRerank (which uses follows the join with listwise re-ranking, resembling the orignal logic in https://arxiv.org/pdf/2401.12178, and comparble to the docetl baseline)
Both are configured with gpt-4o-mini as the LLM and text-embeddings-3-small as the retreiver model

```
python biodex_tester.py
```

This will run both the Join and JoinAndRerank programs and output summary metrics for each, including RP@K and total_latency, in the directory biodex_results.
Note that the latency for the JoinAndRerank pipeline measures the ranking step only, reusing the Join outputs as intermediate results, so latency numbers from the Join should be added to get the latency for the full JoinAndRerank pipeline.


You can find example results I ran earlier in biodex_results/Mar-17-25, which gave the following for the full JoinAndRerank 

```
+-------------------------+--------+---------+-------------+
|                         |   RP@5 |   RP@10 | Latency (s) |
+=========================+========+=========+=============+
| Lotus JoinAndRerank     | .2669  |  .2722  |  566.55     |
+-------------------------+--------+---------+-------------+
```

## BioLORD article-level negative pruning

Run these commands from `benchmarks/biodex` in the existing LOTUS environment.
Install the benchmark loader with `pip install datasets` if needed. The runner
uses LOTUS's `SentenceTransformersRM` and the existing pairwise early-stop executor.
Set `REACTION_SET=liver` (default) or `haematologic`, and use the same `PROMPT`,
`LLM_MODEL`, `LLM_API_BASE`, `LLM_BATCH`, and `EARLY_WORKERS` as the team's control.

```bash
# Score once on validation and test; neither command calls the LLM.
python semijoin_pruning.py score --split validation --output validation_scores.json
python semijoin_pruning.py calibrate validation_scores.json --output thresholds.json
python semijoin_pruning.py score --split test --n-articles 250 --output test_scores.json

# Experiment 5: one live run with the conservative validation threshold.
python semijoin_pruning.py run test_scores.json --calibration thresholds.json \
  --policy conservative --output-dir pruning_results/conservative

# Evaluate the second threshold from that completed run, with zero new LLM calls.
python semijoin_pruning.py replay test_scores.json --calibration thresholds.json \
  --policy retain_95 --source-dir pruning_results/conservative --output retain_95.json
```

The proxy embeds each article's existing first 8,000 characters and each reaction
label once, normalizes the vectors, and takes their dot product (cosine similarity).
BioLORD's native maximum input length is recorded in the score file (currently
128 tokens). This initial implementation uses that document prefix, so the control
must use the same representation and scores. `--device cuda` selects GPU embedding.

An article is pruned only when its **maximum** reaction similarity is **strictly
below** the threshold. Ties are retained. Retained articles evaluate all candidate
labels in descending score order, stopping at the first accepted LLM pair.
Invalid scores retain the article with its original label order.

Calibration uses existing validation reaction annotations. `conservative` retains
all validation positives. `retain_95` allows at most the integer number of misses
consistent with 95% validation-positive retention. With few positives the two
thresholds may coincide. Neither setting guarantees recall on new articles.
`--threshold NUMBER` can also supply a preselected cutoff for run or replay.

Results include every original article, including zero-call pruned negatives and
pruned positives counted as false negatives. `summary.json` reports article-level
precision, recall, F1, positive retention, pruned-positive count, LLM calls, and
mean first accepted LLM witness rank. `articles.json` also records the first gold
witness rank, which can differ from the first LLM-accepted witness.

Reusing a run directory requires an identical configuration. Replay requires a
completed run with the same score file and enough evaluated articles for the new
threshold. In particular, it can evaluate a more aggressive threshold from the
conservative run. It refuses to invent results for previously pruned articles.
Replay labels calls as replayed logical pair evaluations and records its source
manifest. Live results count logical pair evaluations, following the team baseline;
provider retries are outside that counter. `estimated_embedding_plus_run_time_s`
sums model initialization/embedding/scoring time and the current run through result
assembly. Dataset loading in the score command and output serialization are outside
that estimate. Resumed runs mark this estimate unavailable; reissuing a completed
run preserves its original summary. Measure command wall time separately on the
server when comparing end-to-end runtime. Each run directory permits one writer,
and a final incomplete checkpoint record is recovered after an interrupted job.

The shared pairwise predicate and parser follow the team's existing baseline.
For comparison with experiment 4, share the score file and document preprocessing
and compare the same model, prompt, label panel, input articles, and concurrency.
