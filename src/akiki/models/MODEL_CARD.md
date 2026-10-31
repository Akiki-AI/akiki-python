# Akiki VAD — release model, seed 1730

Release ID: `teacher-only623-v2-s1730`. Bundled by `akiki==0.1.3`.

CNN + GRU128, **210,561 parameters**, full-waveform FP32 ONNX with explicit
recurrent state. Finite mono float32 audio at 16,000 Hz; **256 new samples per
output (16 ms)** with 320 past context samples. Fixed threshold 0.5. Reset state
between independent recordings. Incomplete trailing frames are left pending.

## Training protocol

The student starts from fresh random weights, seed **1730**. It uses the frozen
Silero **6.2.3** teacher-label cache: 357,672 training recordings (993.772 hours)
and 17,540 validation recordings (28.262 hours). The source audio passes the
existing commercial licence allowlist. Source groups and frozen benchmark audio
hashes are excluded across the relevant splits. Cross-corpus speaker identity
has not been exhaustively established.

Targets are the independent teacher's native **32 ms soft probabilities**. Each
target supervises the mean of two causal 16 ms student probabilities using stable
binary cross-entropy. These targets do not provide manual 16 ms boundary accuracy.
The cache retains 58 documented instrumental recordings with explicit zero
targets. Already prepared labelled mixtures remain in the corpus.

There is **no AMI/ICSI training or validation, TEN or Silero v5 supervision,
pretrained student initialization, or online acoustic augmentation**, including
synthetic silence substitution. Random crop sampling and the existing source
balance are retained. The mathematical frontend is fixed.

- 48 windows × 2,048 updates = **98,304 optimizer updates**, batch size 64.
- Sequence: 256 × 16 ms = 4.096 s; burn-in: 128 × 16 ms = 2.048 s.
- AdamW: betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01; gradient clip 5.
- Per-update cosine LR **0.001 → 0.000002 over 40 windows**, followed by eight
  windows at 0.000002. No early stopping.
- Original teacher-only validation: 0.5 × mean base-category recording BCE plus
  0.5 × mean expanded-source recording BCE. Recordings are averaged within each
  category/source. No benchmark scores enter this checkpoint rule.
- Deterministic FP32 on physical GPU 1 (RTX PRO 6000 Blackwell); AMP and TF32 off.

**Window 31** is the validation-selected checkpoint (BCE
0.081766859076); the run still completed all 48 windows.
The other seeds were 1729 and 1731. The user designated seed 1730 for release
**after reviewing the benchmark results**. This seed choice must not be described
as independent of benchmark inspection.

The frozen recipe is included in `training-recipe.json`. Exact code snapshots,
input hashes, environment, histories and the other seeds are retained in the
source experiment. On the training host, reproduce this seed with:

```bash
bash training/teacher-only623-v2/run.sh \
  --seed 1730 --output experiments/reproduce-release-s1730
```

That command requires the existing repository, local data caches and recorded
runtime; it does not download or retag data. Byte-identical reproducibility is
scoped to the matching recorded execution environment.

## Recorded evaluation

| Benchmark | AP | F1 at 0.5 | Precision at recall ≥95% |
|---|---:|---:|---:|
| TEN public test set | 98.0187% | 92.6053% | 90.6487% |
| VoxConverse | 99.3045% | 94.9869% | 96.1582% |

TEN: 30 clips / 8,179 common 32 ms frames. VoxConverse: 48 recordings / 999,894
common 32 ms frames. AP is exact pooled average precision; P95 uses the original
0.01 threshold grid. Native 16 ms probabilities are averaged in pairs against the
unchanged 32 ms references. No timing shift or smoothing is applied. Each seed
was scored after training, with the schedule frozen before any seed was scored.
These are repeatedly inspected development comparisons, not untouched test sets.
No new latency claim is made for this release.

## Provenance and licence

Code and Akiki weights: **Apache-2.0**, matching the repository licence. The
independent Silero teacher is MIT licensed; its notice is retained in
`SILERO-MIT.txt`. No teacher weights or source audio are included in this model
package. Source datasets retain their own licences and attribution; the private
corpus index pins the prepared data and offline tagging code:
https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-corpus-index .

Teacher SHA256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

Training manifest SHA256: `69bec79c13ba9021f7b7a604bbef626fdc67131c218a7a7e55239855a87009b8`.

Target cache SHA256: `26faa6b6c3c078a3ed7f10c2af60234c2d476d29caf94ed02d7398e2fae7631b`.

Training checkpoint SHA256: `4b7d904c14f154cc7f39b78eb4c8cde0156de2a8ef8671650d0e05b12c518d73`.

ONNX SHA256: `0fab56bc4ea6725ec881c35973e04813a4963dc5bbf4bf5ccfe8fd7cf468c6f5`.
