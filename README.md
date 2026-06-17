# OFDM Bistatic ISAC — V2I Communication + Forward-Scatter Sensing

A single OFDM waveform that simultaneously **communicates** (V2I text payload) and
**senses** (moving-object detection via line-of-sight disruption) on commodity
SDR hardware — a proof-of-concept for **Integrated Sensing and Communication
(ISAC)** toward 6G.

> **Main script:** `ofdm_isac_bistatic_isac_demo.py` (Python 3, UHD).
> Hardware: USRP B210 (TX) + LibreSDR B210-mini (RX), fc = 5.9 GHz, FS = 40 MS/s.

---

## What this does

One OFDM frame carries both functions at once:

- **Communication:** a V2I packet (`"STEI"` + counter + CRC16) is QPSK-modulated on
  the 8 center subcarriers. Recovered text + packet-reception-rate (PRR) are shown
  live.
- **Sensing:** the remaining subcarriers + channel estimate drive a
  **forward-scatter detector** that flags a moving object when it disrupts the
  TX→RX line-of-sight (amplitude/phase high-pass + adaptive MAD-CFAR).

A live 6-panel dashboard (`--plot`) shows comm constellation, recovered-text log
with PRR, EVM trend, a channel range-profile (context), and the **JCAS detection
panel** that lights up ("ADA OBJEK") when someone crosses the link.

---

## Results / Proof

Validated static-vs-walk, identical tuning (`k=3.5`, `min_score=1.2`), no-plot,
1519 frames each, post-warmup:

| Metric | Static (empty room) | Walk (person moving) | Separation |
|---|---:|---:|---:|
| Comm PRR (CRC OK) | 100% | 100% | comm intact |
| Object-detection rate | **0.07%** (1 burst) | **2.50%** (6 bursts) | ~37× |
| LoS amplitude disruption \|amp_hp\| (95-pct) | 0.055 dB | 1.782 dB | **~32×** |
| Max JCAS score | 2.7 | 8.7 | — |

**Headline:** communication runs at 100% PRR while the sensor stays quiet in an
empty room (0.07% false alarms) and fires cleanly when a person crosses the link
(32× amplitude contrast). The detector discriminates motion from stillness — a
random detector would fire equally in both.

> Official numbers come from **no-plot** runs. Live plotting (`--plot`) competes for
> CPU/GIL, raising overflow (OVF) and inflating the static false-alarm floor
> (0.07% → 0.82%); use it for **demos**, not for **measurements**.

---

## What we learned

1. **Pick the sensing modality that fits the geometry.** Classical CIR range/echo
   detection *fails* in a small room: range resolution (≈4.8 m) and a ~5.6 m blind
   zone are larger than the target's bistatic delta-range (<2 m), so the target
   never appears as an echo. The "echoes" at 7.5/9.38 m are blind-zone-edge clutter
   artifacts, not targets. **Forward-scatter (LoS-disruption) sensing is the right
   tool here** — it is independent of range resolution and the blind zone.
2. **The LoS amplitude metric must be broadband (`avg_amp`), not the CIR peak
   (`max(cir_mag)`).** The CIR-peak metric "bin-hops" onto rising multipath and
   masks the very shadowing event we want to detect.
3. **Comparisons must be apples-to-apples** (same tuning, run length, and plot
   setting), and **warm-up frames must be excluded** from statistics (an empty
   history produces meaningless score spikes).
4. **The robust discriminator is the amplitude-disruption magnitude (`amp_hp`),
   not the raw detection percentage** — the latter is sensitive to overflow timing.
5. **Don't present artifacts as detections.** The range/echo output is kept as
   visual *context* only; the detection *claim* comes solely from the JCAS panel.
6. **RX architecture:** single-thread receive with a large recv chunk
   (`RECV_CHUNK = 65536`) keeps overflow ~4% at 40 MS/s; a Python producer-consumer
   queue failed (GIL contention, mass drops).

---

## Known limitation — geometry / antenna positioning

The current bistatic geometry is **not yet optimized**: a short TX–RX baseline
(~55 cm) in a 4×4 m room yields a very small target delta-path, which is what kills
range-based sensing. Forward-scatter detection works around this, but the setup is
not arranged so the target cleanly **crosses the TX–RX line-of-sight**.
Consequently, the present claim is limited to **binary detection of a moving
object** (present / absent). **Ranging and velocity are not yet claimed.** Fixing
antenna placement/baseline so targets traverse the LoS is the top item of future
work.

---

## Quick start

```bash
# ISAC demo with live 6-panel dashboard (final tuning)
python3 ofdm_isac_bistatic_isac_demo.py --phase1b --plot --fs 40e6 --frames 3000 \
    --tx-gain 90 --rx-gain 76 --sense-threshold-k 3.5 --sense-min-score 1.2 \
    --log-csv run.csv

# Clean measurement run (no plot → trustworthy false-alarm floor)
python3 ofdm_isac_bistatic_isac_demo.py --phase1b --fs 40e6 --frames 1500 \
    --tx-gain 90 --rx-gain 76 --sense-threshold-k 3.5 --sense-min-score 1.2 \
    --log-csv static.csv

# Find the highest feasible sample rate (no --phase1b)
python3 ofdm_isac_bistatic_isac_demo.py --fs-sweep \
    --fs-candidates 5e6,10e6,20e6,30e6,40e6 --frames-per-fs 200 --tx-gain 80 --rx-gain 70

# AWGN self-test (no hardware)
python3 ofdm_isac_bistatic_isac_demo.py --simulate --fs 40e6 --phase1b
```

Key flags: `--no-jcas` (comm only), `--sense-threshold-k` (lower = more sensitive),
`--load-clutter` / `--save-clutter` (skip 300-frame calibration).

---

## Signal design (summary)

- Frame: `STF | LTF1 | LTF2 | 26×DATA` = 2320 samples, Nfft = 64, CP = 16.
- 46 data subcarriers: 8 comm (QPSK) + 38 sense (BPSK known) + 4 pilots.
- V2I packet: ASCII text (32b) + counter (8b) + CRC16-CCITT (16b) = 56 bit.
- Sync: Schmidl-Cox + LTF matched-filter fine timing; CFO well within B210 budget.
- Sensing: forward-scatter detector (amplitude/phase high-pass, MAD-CFAR);
  CIR range-profile retained as context.

---

## Repository contents (suggested)

```
ofdm_isac_bistatic_isac_demo.py   # main ISAC script (Python, UHD)
CONTEXT_TRANSFER.md               # engineering handoff / continuation notes
README.md                         # this file
data/                             # static_k35.csv, walk_k35.csv (proof)
```

> Note: the code is **Python**, not MATLAB. Rename references accordingly if your
> repo plan listed MATLAB.

---

## Roadmap

- [ ] Optimize bistatic geometry / antenna placement (target crosses LoS).
- [ ] ROC curve (Pd vs Pfa) with ground-truth crossings.
- [ ] Reduce overflow < 5% reliably → enable Doppler/velocity estimation.
- [ ] Range-Doppler Map (RDM) once geometry/bandwidth allow.
- [ ] AoA / multi-target (antenna array).
- [ ] Align with 3GPP ISAC use-cases (TR 22.837, Rel-19 → 6G Rel-21).

---

## Acknowledgements

Developed as an ISAC proof-of-concept on USRP B210 / LibreSDR hardware. Sensing
claim = forward-scatter LoS-disruption detection; range/echo output is contextual.
