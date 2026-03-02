# EEG Sleep Stage Classifier

## Goal
Subject/device-agnostic single-channel EEG sleep stage classifier for in-ear earbuds.

## Architecture (locked)
**EpochEncoder** (1.08M params, 5-branch):
- Small CNN (k=50, 3 deep conv): spindles, K-complexes (8-45 Hz) -> 128d
- Medium CNN (k=150, 3 deep conv): theta/alpha morphology (4-12 Hz) -> 128d
- Large CNN (k=400, 3 deep conv): delta/SWA (0.5-4 Hz) -> 128d
- SpectralBranch: 18 fixed features -> MLP -> 128d
- STFTBranch: STFT(256,128) -> patch embed -> 1-layer transformer (4-head) -> mean pool -> 128d
- GroupNorm in CNN, LayerNorm on branch outputs
- Output: 640d concatenated features (128*3 + 128 + 128)

**SleepStageNet** (~5.3M params, 4.2M trainable):
- EpochEncoder (frozen) -> spectral + STFT (both mean pooled) = 256d, CNN branches skipped
- 2-layer BiGRU (input=256d, hidden=384, bidirectional)
- Shared MLP head: Dropout -> LayerNorm(768) -> Linear(768,128) -> GELU -> Linear(128,5)
- CRF on top of logits
- seq_len=200, stride_train=50, stride_val=200

## Training Pipeline
```
# Step 1: Encoder (reconstruction decoders, fully unsupervised -- no stage labels)
python train_encoder.py --exp-name v12

# Step 2: Downstream (frozen encoder, BiGRU + CRF only, batch_size=48)
python train_model.py --freeze-epochs 999

# Step 3: Evaluate
python eval_embeddings.py
python eval_per_dataset.py
```
Encoder uses AMP fp16 + cudnn.benchmark (1.32x speedup). Downstream uses fp32 (CRF log-sum-exp overflows in fp16).

## Encoder Metrics
- val_knn: kNN Cohen's kappa (5-fold, k=5) -- primary encoder quality metric, directly comparable to downstream kappa
- val_knn_f1: kNN macro F1 -- secondary, tracks per-stage balance
- val_silhouette: stage silhouette score -- cluster quality
- val_ch: Calinski-Harabasz index
- silhouette(dataset): domain invariance (want ~0)

## Experiment Isolation
All training/eval scripts support `--exp-name <name>` (default: "default").
Checkpoints in `checkpoints/{encoder|sleep_model}/<exp_name>/`, logs in `logs/{encoder|sleep_model}/<exp_name>/`.

## Best Checkpoints
- final: test kappa=0.768, F1=0.777, N1 F1=0.530, acc=0.832 (simplified, mean pooled)
  Checkpoint: checkpoints/sleep_model/final/epoch=11-val_kappa=0.774.ckpt
  Per-class F1: W=0.937, N1=0.530, N2=0.823, N3=0.751, REM=0.844
- Encoder: checkpoints/encoder/v12/epoch=49-val_loss=0.627.ckpt

## Data
17 datasets, 2293 subjects, ~2.67M epochs, 5 device types, 128Hz, 30s epochs (3840 samples)
Preprocessing: bandpass 0.3-45Hz, z-normalize per recording, cache as HDF5
Quality checks: drop flat-line (std<1e-6), extreme artifact (>5% samples >20 sigma) epochs
Sanitize caches: `python data_pipeline.py --sanitize` (or `--audit` for dry run)

## Proven Dead Ends
- Learnable filterbank: worse at scale (fixed AASM bands stronger prior)
- Branch dropout / RevIN / InstanceNorm: harmful (strips amplitude info, kappa ceiling 0.49)
- Two-view contrastive / temperature tuning: identical at full scale
- Semi-Markov CRF: no benefit over linear CRF
- EOG estimation from ear-EEG: r=0.42 ceiling, needs r>0.7
- Bilateral fusion: data advantage (2295 subj) > signal advantage (83 bilateral subj)
- Focal loss / ArcFace / center loss: consistently harmful
- Label smoothing: harmful with CRF (CRF already smooths)
- Dataset removal (drop 4 worst): val_kappa -0.008, encoder quality collapses
- TTA: net negative with CRF (noise before Viterbi hurts)
- Gap features (10 SWA/spindle/K-complex features): kappa -0.001 full scale; BiGRU+CRF already captures these
- Multi-token (tokens_per_epoch=8): -0.005 full scale
- BiMamba: -0.016 kappa, 2.6x slower; GRU sufficient at seq_len=200
- Recon loss in downstream: wash at full scale; belongs in encoder pretraining
- Dual-view encoder (1D+2D CNNs): -0.016 vs 4-branch encoder
- Phase decoder: encoder branches destroy STFT phase via stride+pool
- DANN: z-normalized recon already domain-invariant; DANN fights useful signal
- Projection layer (Linear 448->256 + L2 norm): contrastive vestige, lossy bottleneck
- SE attention: downstream kappa identical to 4 decimals; 101K wasted params
- Band decoders (theta/alpha/sigma from fused): kappa +0.003 / N1 F1 -0.018 full scale; per-branch decoders already sufficient
- AMP fp16 in downstream: CRF log-sum-exp overflows, loss diverges
- Trajectory dynamics aux loss: next-embed prediction head on BiGRU; net -0.003 kappa
- Position embedding (window-relative scalar): +0.002 kappa wash; BiGRU encodes position implicitly
- VICReg covariance regularizer (encoder): improves embedding structure but test kappa wash; bottleneck is BiGRU+CRF capacity
- Encoder fine-tuning (downstream, LR=1e-5): test wash; encoder quality not N1 bottleneck
- Symmetric cross-entropy loss (RCE): test kappa -0.004
- Multi-scale BiGRU (coarse+fine): test kappa -0.005; fine BiGRU already captures full-sequence context
- Learnable branch weights: test kappa -0.004; GRU input layer already does implicit weighting
- Secondary branch GRU: test kappa -0.001; main GRU still processes noisy CNN dims
- Per-stage heads on full 640d: test kappa -0.004; CNN noise in GRU output limits specialization
- Full 640d downstream (CNN branches): CNN branches device-specific, hurt N1; spectral-only +0.005 kappa
- Sub-epoch segments >3 (n_segments=5,6): test kappa -0.002; fewer samples per segment reduces spectral resolution; 3 segments (10s) matches alpha dropout timescale
- CRF transition priors (AASM init): test kappa +0.001 wash; CRF learns correct transitions from data within a few epochs
- Cross-epoch conv (residual Conv1d over N1Aux): test kappa -0.003; BiGRU already captures temporal context; conv adds noise
- gru384 + CRF prior combo: test kappa -0.003 vs gru384 alone, N1 F1 -0.013; CRF priors don't stack with capacity
- N1AuxFeatures: +0.004 kappa / +0.021 N1 F1, but 170K effective params (24K module + 147K GRU widening 256->320); duplicates STFT computation; removed in simplification
- Per-stage heads (5x [768->64->1]): 246K params, hurts N1 F1 -0.019 vs shared head; redundant LayerNorm+Dropout per stage; shared 768->64->5 head is both smaller and better
- Directional STFT pool + delta features: N1 F1 -0.043; lossy projections (512->256) destroyed information, epoch-to-epoch deltas are noise-dominated (artifacts, impedance), frozen transformer tokens lack temporal ordering after self-attention
- No CRF: N1 F1 +0.011 but kappa -0.009, N2 F1 -0.014; CRF's N1 penalty reduced to -0.006 with attention pooling (was -0.020)
- AttentionPool on STFT: training collapse at epoch 12 (attention weight saturation in softmax); temperature scaling defers but doesn't prevent; mean pooling is stable and equivalent

## Reconstruction Decoders (training-only, discarded after encoder training)
| Decoder | Input | Target | Output | Params |
|---------|-------|--------|--------|--------|
| SpectrogramDecoder | fused 640d | log1p(STFT power) | (129, 31) | 1.15M |
| WaveformDecoder | fused 640d | z-normed waveform, 4x ds | 960 | 506K |
| BranchDecoder (delta) | large CNN 128d | bandpass 0.5-4 Hz signed, 8x ds | 480 | 65K |
| BranchDecoder (hfreq) | small CNN 128d | bandpass 8-45 Hz envelope, 8x ds | 480 | 65K |
| BranchDecoder (theta) | medium CNN 128d | bandpass 4-8 Hz signed, 8x ds | 480 | 65K |
| BranchDecoder (slowwave) | large CNN 128d | bandpass 0.5-1 Hz signed, 8x ds | 480 | 65K |
| BranchDecoder (sawtooth) | medium CNN 128d | bandpass 2-6 Hz signed, 8x ds | 480 | 65K |

## Codebase (10 files)
| File | Purpose |
|------|---------|
| config.py | Paths, constants, 17-dataset registry |
| data_pipeline.py | EDF/SET -> HDF5 preprocessing + caching |
| readers.py | 10 dataset readers (BIDS + external) |
| readers_extra.py | 6 additional readers |
| dataset.py | PyTorch Dataset + Lightning DataModule |
| model.py | EpochEncoder + SleepStageNet + recon decoders |
| train_encoder.py | Self-supervised encoder training (reconstruction, no labels) |
| train_model.py | Frozen encoder + BiGRU + shared head + CRF downstream |
| eval_embeddings.py | Embedding quality metrics |
| eval_per_dataset.py | Per-dataset/device evaluation |
