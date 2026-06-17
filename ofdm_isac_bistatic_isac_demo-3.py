#!/usr/bin/env python3
r"""
ofdm_isac_bistatic_isac_demo.py — ISAC demo: V2I 'STEI' comm + JCAS forward-scatter sensing
  Base = adv(c) (comm/text dashboard + range v1.4) + ForwardScatterDetector (Phase 1F).
  KLAIM DETEKSI = JCAS forward-scatter (amp/phase LoS, MAD-CFAR), pakai avg_amp.
  Range/echo = KONTEKS visual saja (artefak blind-zone di ruangan kecil, bukan klaim).
Status:
  Phase 1A — IMPLEMENTED  : FS sweep mode, tentukan max FS hardware-feasible
  Phase 1B — IMPLEMENTED  : V2I "STEI" comm payload (QPSK center 8 SC)
  Phase 1C — IMPLEMENTED  : Range estimation v1.4 (CIR + delta-range, in-band sync,
                            background subtraction + hard geometric limit)
  Phase 1D — IMPLEMENTED  : Real-time matplotlib live plot (--plot flag)
  Phase 1E — PENDING      : Lab validation tests (moving target scenarios)
Frame structure (Nfft=64 fixed, FS-scalable):
  [STF | LTF1 | LTF2 | DATA × 26]
  Total samples = 240 + 26*80 = 2320
Phase 0/1A frame layout (default):
  All DATA_REL (46 SC) → BPSK known sequence (deterministic seed=42)
  Pilot SC (4)         → 1+0j (CPE reference)
Phase 1B frame layout (--phase1b):
  COMM_SC  (8 center)  → QPSK packet "STEI"+ctr+CRC16 + random pad (416 bit cap)
  SENSE_SC (38)        → BPSK known (sense reference, seed=123)
  Pilot SC (4)         → 1+0j
  TX cycles 256 frames (counter 0..255), each ~50 μs @ FS=40 MHz
V2I packet (56 bit):
  [ASCII text 32b | counter 8b | CRC16-CCITT 16b]
Phase 1C range estimation (v1.4):
  CIR = IFFT(Hanning_window(H_est), 4× zero-pad) → 256-bin
  Range bin = c / (FS × oversample_factor)
  @ FS=40 MHz, 4× pad → 1.875 m/bin (interpolated; theoretical δR = 4.8 m)
  Detect: direct path (max) + echoes (threshold 14 dB di atas noise floor
          FISIK dari CIR mentah; background subtraction clutter map;
          hard geometric limit max_range_m; sidelobe skip)
  Output: list of (delta_range_m, peak_db) per frame
CFO unambiguous range = fs/N = fs/64 (Schmidl-Cox dengan delay=32):
  @  2 MHz : ±31.25 kHz
  @ 40 MHz : ±625 kHz
B210 ±2 ppm @ 5.9 GHz → CFO actual ±11.8 kHz. Aman semua FS.
Usage:
  # Phase 1A — FS sweep (sudah validated, fs_winner = 40 MHz untuk hardware ini)
  python3 ofdm_isac_bistatic.py --fs-sweep \
      --fs-candidates 5e6,10e6,20e6,30e6,40e6 \
      --frames-per-fs 200 --tx-gain 80 --rx-gain 70
  # Phase 1B+1C — V2I "STEI" comm + range estimation @ FS_winner
  python3 ofdm_isac_bistatic.py --phase1b --fs 40e6 --frames 500 \
      --tx-gain 90 --rx-gain 76 --text "STEI"
  # Phase 1B+1C+1D — Tambah live plot
  python3 ofdm_isac_bistatic.py --phase1b --plot --fs 40e6 --frames 500 \
      --tx-gain 90 --rx-gain 76
  # Plot dengan echo threshold lebih sensitif (default 14 dB)
  python3 ofdm_isac_bistatic.py --phase1b --plot --fs 40e6 --frames 500 \
      --tx-gain 90 --rx-gain 76 --echo-threshold-db 8
  # Phase 1B dengan custom CSV log
  python3 ofdm_isac_bistatic.py --phase1b --fs 40e6 --frames 1000 \
      --tx-gain 90 --rx-gain 76 --log-csv lab_test_1.csv
  # Single FS run Phase 0-style (no comm split)
  python3 ofdm_isac_bistatic.py --fs 40e6 --frames 100 \
      --tx-gain 90 --rx-gain 76
  # AWGN self-test (no hardware)
  python3 ofdm_isac_bistatic.py --simulate --fs 20e6
  python3 ofdm_isac_bistatic.py --simulate --fs 40e6 --phase1b
"""
import multiprocessing as mp
import threading
import time
import os
import sys
import argparse
import csv
import json
import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════════
# HARDWARE CONFIG (sama dengan Phase 0)
# ═══════════════════════════════════════════════════════════════════
TX_SERIAL    = "000000037"
RX_SERIAL    = "HQHGTFH"
TX_IMAGE_DIR = "/home/telmat/uhd_images/asli"
RX_IMAGE_DIR = "/home/telmat/uhd_images/libre"
TX_ANT       = "TX/RX"
RX_ANT       = "TX/RX"
FC           = 5.9e9


def _find_fpga(directory):
    import glob
    for name in ("usrp_b210_fpga.bin", "usrp_b210_fpga.bit",
                 "usrp_b200_fpga.bin", "usrp_b200_fpga.bit"):
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            return os.path.abspath(full)
    for pat in ("*b210*.bin", "*b210*.bit", "*b200*.bin", "*b200*.bit"):
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return os.path.abspath(hits[0])
    return None


TX_FPGA = _find_fpga(TX_IMAGE_DIR) if os.path.isdir(TX_IMAGE_DIR) else None
RX_FPGA = _find_fpga(RX_IMAGE_DIR) if os.path.isdir(RX_IMAGE_DIR) else None


# ═══════════════════════════════════════════════════════════════════
# OFDM PARAMETERS — Nfft fixed, FS variable (set via init_params)
# ═══════════════════════════════════════════════════════════════════
NSC     = 64
NCP     = 16
LSYM    = NSC + NCP            # 80
NSYM    = 26
N_HALF  = NSC // 2             # 32 = S&C delay

DATA_REL  = [i for i in range(-NSC // 2, NSC // 2)
             if 1 <= abs(i) <= 25 and i not in (-21, -7, 7, 21)]
PILOT_REL = [-21, -7, 7, 21]
N_DATA    = len(DATA_REL)
ACTIVE_REL = sorted(set(DATA_REL + PILOT_REL))
N_ACTIVE   = len(ACTIVE_REL)

STF_OFF   = 0
LTF1_OFF  = LSYM
LTF2_OFF  = 2 * LSYM
DATA_OFF  = 3 * LSYM
FRAME_LEN = DATA_OFF + NSYM * LSYM   # 2320

# ── Runtime-variable globals (set by init_params) ─────────────────
FS           = None
BW           = None
MAX_CFO      = None
KNOWN_BITS   = None
STF          = None
X_STF        = None
LTF_SYM      = None
X_LTF        = None
LTF_TD       = None
LTF_TD_NO_CP = None
TX_FRAME     = None

# ═══════════════════════════════════════════════════════════════════
# PHASE 1B/1C — V2I COMM + SENSING
# ═══════════════════════════════════════════════════════════════════
# Subcarrier split:
#   COMM_SC  = 8 center SC (avoid DC, |k|≤4)  → QPSK packet payload
#   SENSE_SC = sisanya dari DATA_REL          → BPSK known (sense reference)
#   PILOT    = [-21, -7, 7, 21]               → 1+0j
COMM_SC  = [-4, -3, -2, -1, 1, 2, 3, 4]
SENSE_SC = sorted([sc for sc in DATA_REL if sc not in COMM_SC])
N_COMM   = len(COMM_SC)        # 8
N_SENSE  = len(SENSE_SC)       # 38
COMM_BITS_PER_FRAME  = N_COMM * NSYM * 2   # 8 × 26 × 2 = 416 bit/frame (QPSK)
SENSE_BITS_PER_FRAME = N_SENSE * NSYM      # 38 × 26 = 988 bit/frame (BPSK)

# V2I packet format: [text 32b | counter 8b | CRC16 16b] = 56 bit
PKT_TEXT_BITS  = 32
PKT_CTR_BITS   = 8
PKT_CRC_BITS   = 16
PKT_TOTAL_BITS = PKT_TEXT_BITS + PKT_CTR_BITS + PKT_CRC_BITS  # 56

N_FRAMES_CYCLE = 256   # TX cycles through 256 pre-built frames (counter 0..255)
CIR_OVERSAMPLE = 4     # IFFT zero-pad factor untuk smooth CIR (Phase 1C)
CIR_NFFT       = NSC * CIR_OVERSAMPLE  # 256

# Phase 1B/1C runtime globals
PHASE1B_MODE       = False
TX_TEXT            = "STEI"
SENSE_KNOWN_BITS   = None
TX_FRAMES_P1B      = None         # list of N_FRAMES_CYCLE pre-built frames
LTF_FREQ_WINDOW    = None         # Hanning window for CIR sidelobe suppression


# ─────────────────────────────────────────────────────────────────
# CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflect, no xorout)
# ─────────────────────────────────────────────────────────────────
def crc16_ccitt(data_bytes):
    crc = 0xFFFF
    for b in data_bytes:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_packet(text, counter):
    """Build 56-bit packet: ASCII(4) + counter(1) + CRC16(2)."""
    text_bytes = text.encode('ascii', errors='replace')[:4].ljust(4, b'\x00')
    ctr_byte = bytes([counter & 0xFF])
    payload = text_bytes + ctr_byte           # 5 bytes = 40 bit
    crc = crc16_ccitt(payload)
    crc_bytes = bytes([(crc >> 8) & 0xFF, crc & 0xFF])
    full_bytes = payload + crc_bytes          # 7 bytes = 56 bit
    bits = np.zeros(56, dtype=np.uint8)
    for i, byte in enumerate(full_bytes):
        for j in range(8):
            bits[i*8 + j] = (byte >> (7 - j)) & 1
    return bits


def decode_packet(bits56):
    """Decode 56-bit array → (text, counter, crc_ok)."""
    if len(bits56) < 56:
        return ("????", 0, False)
    bytes_arr = bytearray(7)
    for i in range(7):
        b = 0
        for j in range(8):
            b = (b << 1) | int(bits56[i*8 + j])
        bytes_arr[i] = b
    text = bytes_arr[:4].decode('ascii', errors='replace')
    counter = bytes_arr[4]
    rx_crc = (bytes_arr[5] << 8) | bytes_arr[6]
    expected = crc16_ccitt(bytes(bytes_arr[:5]))
    return (text, counter, rx_crc == expected)


# ─────────────────────────────────────────────────────────────────
# QPSK map/demap (Gray-coded, unit power)
# ─────────────────────────────────────────────────────────────────
# bit pair (b0,b1):  (0,0)->+1+1j, (0,1)->+1-1j, (1,0)->-1+1j, (1,1)->-1-1j  / sqrt(2)
QPSK_TABLE = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex) / np.sqrt(2)


def qpsk_map(bits_pairs):
    """bits_pairs: shape (N, 2) → N complex symbols."""
    idx = bits_pairs[:, 0] * 2 + bits_pairs[:, 1]
    return QPSK_TABLE[idx]


def qpsk_demap(symbols):
    """N complex → shape (N, 2) bits via hard sign decision."""
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:, 0] = (np.real(symbols) < 0).astype(np.uint8)
    bits[:, 1] = (np.imag(symbols) < 0).astype(np.uint8)
    return bits


def _build_phase1b_frame(comm_payload_bits_416, sense_known_bits_988):
    """Construct Phase 1B time-domain frame.

    comm_payload_bits_416: 416 bit (QPSK on 8 center SC × 26 sym × 2 bit)
    sense_known_bits_988:  988 bit (BPSK on 38 sense SC × 26 sym × 1 bit)
    """
    # QPSK comm symbols
    comm_pairs = comm_payload_bits_416.reshape(-1, 2)         # (208, 2)
    comm_syms = qpsk_map(comm_pairs).reshape(NSYM, N_COMM)    # (26, 8)

    # BPSK sense symbols
    sense_syms = (1.0 - 2.0 * sense_known_bits_988.astype(float)).astype(complex)
    sense_syms = sense_syms.reshape(NSYM, N_SENSE)            # (26, 38)

    parts = [STF, LTF_SYM, LTF_SYM]
    for m in range(NSYM):
        f = np.zeros(NSC, dtype=complex)
        for i, sc in enumerate(COMM_SC):
            f[sc % NSC] = comm_syms[m, i]
        for i, sc in enumerate(SENSE_SC):
            f[sc % NSC] = sense_syms[m, i]
        for p in PILOT_REL:
            f[p % NSC] = 1.0 + 0j
        td = np.fft.ifft(f) * np.sqrt(NSC)
        parts.append(np.concatenate([td[-NCP:], td]).astype(np.complex64))
    return np.concatenate(parts).astype(np.complex64)


def _build_all_phase1b_frames(text):
    """Pre-build N_FRAMES_CYCLE frames (counter 0..255). Returns list."""
    np.random.seed(123)  # different seed dari Phase 0 KNOWN_BITS
    sense_known = np.random.randint(0, 2, SENSE_BITS_PER_FRAME).astype(np.uint8)
    pad_bits = np.random.randint(0, 2, COMM_BITS_PER_FRAME - PKT_TOTAL_BITS).astype(np.uint8)

    frames = []
    for ctr in range(N_FRAMES_CYCLE):
        pkt_bits = encode_packet(text, ctr)
        # Layout: [packet 56b | random pad 360b] = 416 bit
        comm_bits = np.concatenate([pkt_bits, pad_bits])
        raw = _build_phase1b_frame(comm_bits, sense_known)
        max_amp = float(np.max(np.abs(raw)))
        frame = (raw / max_amp * 0.95).astype(np.complex64)
        frames.append(frame)
    return frames, sense_known


# ─────────────────────────────────────────────────────────────────
# Phase 1C: CIR + range estimation
# ─────────────────────────────────────────────────────────────────
def cir_from_h_est(H_est, fs, n_pad=CIR_NFFT, window=True):
    """Return oversampled CIR magnitude (linear) + bin-to-meter scale.

    Window in freq domain (Hanning over active SC) suppress sidelobe.
    Zero-pad to n_pad untuk smooth peaks (interpolation, NOT extra resolution).
    """
    H = H_est.copy()
    if window:
        # Apply Hanning over active SC only (preserve spectral mask)
        win = np.zeros(NSC, dtype=float)
        win_active = np.hanning(N_ACTIVE)
        for i, k in enumerate(ACTIVE_REL):
            win[k % NSC] = win_active[i]
        H = H * win

    # Zero-pad in freq domain (FFT-shift convention)
    H_shifted = np.fft.fftshift(H)            # DC at center NSC/2
    pad_lo = (n_pad - NSC) // 2
    pad_hi = n_pad - NSC - pad_lo
    H_padded = np.concatenate([
        np.zeros(pad_lo, dtype=complex),
        H_shifted,
        np.zeros(pad_hi, dtype=complex),
    ])
    H_back = np.fft.ifftshift(H_padded)
    cir = np.fft.ifft(H_back) * (n_pad / NSC)  # preserve amplitude

    # Each CIR bin = 1/(fs × oversample) sec → range bin = c/(fs × oversample)
    bin_to_meter = 3e8 / (fs * (n_pad / NSC))
    return np.abs(cir), bin_to_meter


def estimate_ranges(cir_mag, bin_to_meter, snr_threshold_db=14,
                    direct_skip_bins=3, max_echoes=3,
                    max_range_m=12.0, bg_cir=None):
    """Detect echo peaks dengan hard geometric limit + bg subtraction.

    PHASE 1C v1.4 — Fix cacat threshold saat background subtraction aktif.

    bg_cir       : clutter map (linear mag). None → skip background subtraction.
    max_range_m  : hard geometric limit delta path (bistatic).
                   Lab 4×4 m + TX-RX 55 cm → max realistic ≈ 12 m.
    snr_threshold_db : SNR di atas noise floor FISIK (dari CIR mentah).
    direct_skip_bins : skip N bin setelah direct (mask sidelobe Hanning).

    KENAPA noise floor dari CIR MENTAH (bukan cir_clean):
      Background subtraction menghapus CLUTTER (deterministik) tapi TIDAK
      menghapus NOISE termal. Kalau noise floor diukur dari cir_clean
      (sudah di-subtract), far region ≈ 0 → noise floor anjlok → threshold
      anjlok → residual subtraction sekecil apa pun lolos jadi echo PALSU.
      (Itu bug v1.1-v1.3: ruangan diam menghasilkan ratusan echo hantu.)
      Fix: threshold = noise FISIK + SNR. Echo nyata = peak yang (a) hilang
      dari clutter map [muncul di cir_clean] DAN (b) di atas noise fisik.

    Returns dict:
      direct_bin, direct_db, noise_floor_db, threshold_db,
      echoes=[(delta_range_m, peak_db), ...], n_echoes, bg_applied
    """
    n = len(cir_mag)

    # Direct path = global max dari CIR mentah
    direct_bin = int(np.argmax(cir_mag))
    direct_db = float(20 * np.log10(cir_mag[direct_bin] + 1e-12))

    # Background subtraction (clutter map beku)
    bg_applied = False
    if bg_cir is not None and len(bg_cir) == n:
        cir_clean = np.maximum(cir_mag - bg_cir, 1e-12)
        bg_applied = True
    else:
        cir_clean = cir_mag

    # ── Noise floor dari CIR MENTAH (FIX v1.4) ──────────────────
    # Threshold absolut relatif noise fisik — STABIL baik bg aktif/tidak.
    raw_db = 20 * np.log10(cir_mag + 1e-12)
    far_lo = max(0, direct_bin - n // 3)
    far_hi = min(n, direct_bin + n // 3)
    far_region = np.concatenate([raw_db[:far_lo], raw_db[far_hi:]])
    if len(far_region) >= 8:
        noise_floor_db = float(np.percentile(far_region, 50))
    else:
        noise_floor_db = direct_db - 30
    threshold_db = noise_floor_db + snr_threshold_db

    # CIR untuk deteksi peak: cir_clean (clutter sudah dihapus)
    cir_clean_db = 20 * np.log10(cir_clean + 1e-12)

    # HARD GEOMETRIC LIMIT
    max_bin_offset = int(np.ceil(max_range_m / bin_to_meter))
    search_start = direct_bin + direct_skip_bins
    search_end = min(n - 1, direct_bin + max_bin_offset)

    echoes = []
    for b in range(search_start + 1, search_end):
        # Peak di cir_clean (clutter-free) HARUS lewat threshold noise FISIK
        if cir_clean_db[b] > threshold_db:
            if cir_clean[b] > cir_clean[b - 1] and cir_clean[b] > cir_clean[b + 1]:
                delta_range = (b - direct_bin) * bin_to_meter
                echoes.append((float(delta_range), float(cir_clean_db[b])))

    # Keep top-K strongest, sort by range untuk display
    echoes.sort(key=lambda x: x[1], reverse=True)
    echoes = echoes[:max_echoes]
    echoes.sort(key=lambda x: x[0])

    return {
        "direct_bin": direct_bin,
        "direct_db": direct_db,
        "noise_floor_db": noise_floor_db,
        "threshold_db": threshold_db,
        "echoes": echoes,
        "n_echoes": len(echoes),
        "bg_applied": bg_applied,
    }


def _build_stf():
    """STF: only even subcarriers populated → time-domain x[n] = x[n+N/2]."""
    X = np.zeros(NSC, dtype=complex)
    even_active = [k for k in ACTIVE_REL if k % 2 == 0]
    L = len(even_active)
    n_zc = np.arange(L)
    zc = np.exp(-1j * np.pi * 5 * n_zc * (n_zc + 1) / L)
    for i, k in enumerate(even_active):
        X[k % NSC] = zc[i] * np.sqrt(2.0)
    td = np.fft.ifft(X) * np.sqrt(NSC)
    cp = td[-NCP:]
    return np.concatenate([cp, td]).astype(np.complex64), X


def _build_ltf():
    """LTF: full-band ZC, 2 identical symbols for averaging."""
    X = np.zeros(NSC, dtype=complex)
    L = N_ACTIVE
    n_zc = np.arange(L)
    zc = np.exp(-1j * np.pi * 25 * n_zc * (n_zc + 1) / L)
    for i, k in enumerate(ACTIVE_REL):
        X[k % NSC] = zc[i]
    td = np.fft.ifft(X) * np.sqrt(NSC)
    cp = td[-NCP:]
    return np.concatenate([cp, td]).astype(np.complex64), X, td


def _build_frame_signal(bits):
    """Construct time-domain frame from bits (BPSK on data SC, 1+0j on pilots)."""
    syms = (1.0 - 2.0 * bits.astype(float)).astype(complex).reshape(NSYM, N_DATA)
    parts = [STF, LTF_SYM, LTF_SYM]
    for m in range(NSYM):
        f = np.zeros(NSC, dtype=complex)
        for i, sc in enumerate(DATA_REL):
            f[sc % NSC] = syms[m, i]
        for p in PILOT_REL:
            f[p % NSC] = 1.0 + 0j
        td = np.fft.ifft(f) * np.sqrt(NSC)
        parts.append(np.concatenate([td[-NCP:], td]).astype(np.complex64))
    return np.concatenate(parts).astype(np.complex64)


def init_params(fs, phase1b=False, text="STEI"):
    """Initialize FS-dependent globals. Called once per FS (main + each TX worker).

    phase1b=True akan additionally pre-build N_FRAMES_CYCLE V2I packet frames.
    """
    global FS, BW, MAX_CFO
    global KNOWN_BITS, STF, X_STF, LTF_SYM, X_LTF, LTF_TD, LTF_TD_NO_CP, TX_FRAME
    global PHASE1B_MODE, TX_TEXT, SENSE_KNOWN_BITS, TX_FRAMES_P1B

    FS = float(fs)
    BW = FS  # Effective signal BW = sample rate (B210 sets analog BW = fs)

    # CFO threshold: scale dengan subcarrier spacing.
    # Schmidl-Cox unambiguous = fs/N. B210 ±2 ppm @ 5.9 GHz = ±11.8 kHz absolute.
    # Threshold = min(30 kHz, 30% subcarrier spacing) untuk avoid edge ambiguity & ICI.
    sc_spacing = FS / NSC
    MAX_CFO = min(30000.0, 0.3 * sc_spacing)

    np.random.seed(42)
    KNOWN_BITS = np.random.randint(0, 2, NSYM * N_DATA).astype(np.uint8)

    STF, X_STF = _build_stf()
    LTF_SYM, X_LTF, LTF_TD = _build_ltf()
    LTF_TD_NO_CP = LTF_TD.astype(np.complex64)

    raw = _build_frame_signal(KNOWN_BITS)
    max_amp = float(np.max(np.abs(raw)))
    TX_FRAME = (raw / max_amp * 0.95).astype(np.complex64)
    assert len(TX_FRAME) == FRAME_LEN

    PHASE1B_MODE = bool(phase1b)
    TX_TEXT = text
    if PHASE1B_MODE:
        TX_FRAMES_P1B, SENSE_KNOWN_BITS = _build_all_phase1b_frames(text)
    else:
        TX_FRAMES_P1B, SENSE_KNOWN_BITS = None, None


# ═══════════════════════════════════════════════════════════════════
# SYNCHRONIZATION + DEMOD (sama logic dengan Phase 0, tinggal pakai globals)
# ═══════════════════════════════════════════════════════════════════
def schmidl_cox_metric(buf):
    L = len(buf) - N_HALF
    if L < N_HALF:
        return None, None
    mult = np.conj(buf[:L]) * buf[N_HALF:N_HALF + L]
    pwr  = np.abs(buf[N_HALF:N_HALF + L]) ** 2
    P_cum = np.cumsum(mult)
    R_cum = np.cumsum(pwr)
    nw = L - N_HALF + 1
    P = P_cum[N_HALF - 1:N_HALF - 1 + nw].copy()
    R = R_cum[N_HALF - 1:N_HALF - 1 + nw].copy()
    if nw > 1:
        P[1:] -= P_cum[:nw - 1]
        R[1:] -= R_cum[:nw - 1]
    M = (np.abs(P) ** 2) / (R ** 2 + 1e-18)
    return M, P


def find_all_plateaus(M, threshold=0.7, min_width=4):
    above = M > threshold
    if not np.any(above):
        return []
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    plateaus = []
    for s, e in zip(starts, ends):
        w = int(e - s)
        if w < min_width:
            continue
        c = int((s + e - 1) // 2)
        plateaus.append((c, w, float(M[c])))
    plateaus.sort(key=lambda t: t[1] * t[2], reverse=True)
    return plateaus


def validate_sync(Y_eq_pilots, evm_db, cfo_hz, max_cfo=None):
    if max_cfo is None:
        max_cfo = MAX_CFO
    if abs(cfo_hz) > max_cfo:
        return False, f"CFO out of range ({cfo_hz:.0f} Hz)"
    if evm_db > 0:
        return False, f"EVM too high ({evm_db:.1f} dB)"
    p = np.asarray(Y_eq_pilots)
    pilot_mean = np.mean(p)
    pilot_std  = np.std(p)
    pilot_snr_db = 20 * np.log10(np.abs(pilot_mean) / (pilot_std + 1e-9))
    if pilot_snr_db < 3:
        return False, f"pilot scatter (pSNR={pilot_snr_db:.1f} dB)"
    return True, "ok"


def fine_timing_ltf(buf, coarse_start, search_radius=24):
    expected_ltf1 = coarse_start + LTF1_OFF + NCP
    lo = max(0, expected_ltf1 - search_radius)
    hi = expected_ltf1 + search_radius + len(LTF_TD_NO_CP)
    if hi > len(buf):
        return None
    seg = buf[lo:hi]
    mf = np.conj(LTF_TD_NO_CP[::-1])
    corr = np.convolve(seg, mf, mode='valid')
    if len(corr) == 0:
        return None
    peak_local = int(np.argmax(np.abs(corr)))
    refined_ltf1_start = lo + peak_local
    return refined_ltf1_start - (LTF1_OFF + NCP)


def _demod_at(buf_corr, frame_start):
    """Phase 0/1A demod: BPSK seluruh DATA_REL.
    Returns (bits, evm_db, sample_eq, pilot_eq, avg_amp, H_est)."""
    if frame_start < 0 or frame_start + FRAME_LEN > len(buf_corr):
        return None
    fb = buf_corr[frame_start:frame_start + FRAME_LEN]
    avg_amp = float(np.mean(np.abs(fb)))
    Y_ltf1 = np.fft.fft(fb[LTF1_OFF + NCP:LTF1_OFF + NCP + NSC]) / np.sqrt(NSC)
    Y_ltf2 = np.fft.fft(fb[LTF2_OFF + NCP:LTF2_OFF + NCP + NSC]) / np.sqrt(NSC)
    Y_ltf  = 0.5 * (Y_ltf1 + Y_ltf2)
    H_est = np.ones(NSC, dtype=complex)
    active_mask = np.abs(X_LTF) > 1e-9
    H_est[active_mask] = Y_ltf[active_mask] / X_LTF[active_mask]

    decoded = np.empty(NSYM * N_DATA, dtype=np.uint8)
    sample_eq_first = None
    evm_acc = 0.0
    evm_n = 0
    pilots_collected = []

    for m in range(NSYM):
        s0 = DATA_OFF + m * LSYM
        td = fb[s0 + NCP:s0 + LSYM]
        Y  = np.fft.fft(td) / np.sqrt(NSC)
        Y_eq = np.where(np.abs(H_est) > 1e-9, Y / H_est, Y)
        pilot_vals = np.array([Y_eq[p % NSC] for p in PILOT_REL])
        cpe = np.angle(np.mean(pilot_vals))
        Y_eq *= np.exp(-1j * cpe)
        pilots_collected.extend([Y_eq[p % NSC] for p in PILOT_REL])
        data_eq = np.array([Y_eq[sc % NSC] for sc in DATA_REL])
        bits = (np.real(data_eq) < 0).astype(np.uint8)
        decoded[m * N_DATA:(m + 1) * N_DATA] = bits
        ideal = np.where(bits == 0, 1.0 + 0j, -1.0 + 0j)
        evm_acc += float(np.sum(np.abs(data_eq - ideal) ** 2))
        evm_n   += N_DATA
        if m == 0:
            sample_eq_first = data_eq[:10].copy()

    evm_rms = np.sqrt(evm_acc / max(evm_n, 1))
    evm_db  = 20 * np.log10(evm_rms + 1e-12)
    return decoded, evm_db, sample_eq_first, pilots_collected, avg_amp, H_est


def _demod_at_phase1b(buf_corr, frame_start):
    """Phase 1B demod: split SC. QPSK comm + BPSK sense + H_est utk Phase 1C.

    Returns dict atau None.
    """
    if frame_start < 0 or frame_start + FRAME_LEN > len(buf_corr):
        return None
    fb = buf_corr[frame_start:frame_start + FRAME_LEN]
    avg_amp = float(np.mean(np.abs(fb)))

    # Channel estimation
    Y_ltf1 = np.fft.fft(fb[LTF1_OFF + NCP:LTF1_OFF + NCP + NSC]) / np.sqrt(NSC)
    Y_ltf2 = np.fft.fft(fb[LTF2_OFF + NCP:LTF2_OFF + NCP + NSC]) / np.sqrt(NSC)
    Y_ltf  = 0.5 * (Y_ltf1 + Y_ltf2)
    H_est = np.ones(NSC, dtype=complex)
    active_mask = np.abs(X_LTF) > 1e-9
    H_est[active_mask] = Y_ltf[active_mask] / X_LTF[active_mask]

    comm_bits  = np.empty(COMM_BITS_PER_FRAME, dtype=np.uint8)   # 416
    sense_bits = np.empty(SENSE_BITS_PER_FRAME, dtype=np.uint8)  # 988
    pilots_collected = []
    comm_evm_acc = 0.0
    comm_evm_n = 0
    sense_evm_acc = 0.0
    sense_evm_n = 0
    comm_sample_eq = None

    for m in range(NSYM):
        s0 = DATA_OFF + m * LSYM
        td = fb[s0 + NCP:s0 + LSYM]
        Y = np.fft.fft(td) / np.sqrt(NSC)
        Y_eq = np.where(np.abs(H_est) > 1e-9, Y / H_est, Y)
        # Pilot CPE
        pilot_vals = np.array([Y_eq[p % NSC] for p in PILOT_REL])
        cpe = np.angle(np.mean(pilot_vals))
        Y_eq *= np.exp(-1j * cpe)
        pilots_collected.extend([Y_eq[p % NSC] for p in PILOT_REL])

        # COMM: QPSK demap dari 8 center SC
        comm_eq = np.array([Y_eq[sc % NSC] for sc in COMM_SC])
        c_pairs = qpsk_demap(comm_eq)
        comm_bits[m * N_COMM * 2:(m + 1) * N_COMM * 2] = c_pairs.flatten()
        # Comm EVM (vs ideal QPSK)
        ideal_q = qpsk_map(c_pairs)
        comm_evm_acc += float(np.sum(np.abs(comm_eq - ideal_q) ** 2))
        comm_evm_n   += N_COMM

        # SENSE: BPSK hard slice
        sense_eq = np.array([Y_eq[sc % NSC] for sc in SENSE_SC])
        s_bits = (np.real(sense_eq) < 0).astype(np.uint8)
        sense_bits[m * N_SENSE:(m + 1) * N_SENSE] = s_bits
        ideal_s = np.where(s_bits == 0, 1.0 + 0j, -1.0 + 0j)
        sense_evm_acc += float(np.sum(np.abs(sense_eq - ideal_s) ** 2))
        sense_evm_n   += N_SENSE

        if m == 0:
            comm_sample_eq = comm_eq[:8].copy()

    comm_evm_rms = np.sqrt(comm_evm_acc / max(comm_evm_n, 1))
    comm_evm_db  = 20 * np.log10(comm_evm_rms + 1e-12)
    sense_evm_rms = np.sqrt(sense_evm_acc / max(sense_evm_n, 1))
    sense_evm_db  = 20 * np.log10(sense_evm_rms + 1e-12)

    return {
        "comm_bits": comm_bits, "sense_bits": sense_bits,
        "comm_evm_db": comm_evm_db, "sense_evm_db": sense_evm_db,
        "comm_sample_eq": comm_sample_eq,
        "pilots": pilots_collected, "avg_amp": avg_amp,
        "H_est": H_est,
    }


def sync_and_demod(buf, sc_threshold=0.7, max_candidates=3,
                   echo_threshold_db=14.0, bg_cir=None,
                   max_range_m=12.0, direct_skip_bins=3, max_echoes=3):
    """Full pipeline. Returns dict including H_est for sensing extension.

    echo_threshold_db: passed ke estimate_ranges (Phase 1C). Default 14 dB.
    bg_cir          : background CIR (linear mag, len=CIR_NFFT) untuk subtraction.
                      None → skip bg subtraction.
    max_range_m     : hard geometric limit delta path (default 12 m utk lab 4×4 m).
    direct_skip_bins: skip N bin setelah direct (mask Hanning sidelobe).
    max_echoes      : top-K echo terkuat yang di-report.
    """
    if len(buf) < FRAME_LEN + 64:
        return {"bits": None, "consume": 0}

    search_len = min(len(buf), FRAME_LEN + 256)
    M, P = schmidl_cox_metric(buf[:search_len])
    if M is None:
        return {"bits": None, "consume": 0}

    plateaus = find_all_plateaus(M, threshold=sc_threshold, min_width=3)
    if not plateaus:
        return {"bits": None, "consume": min(LSYM, len(buf) - FRAME_LEN)}

    candidates = plateaus[:max_candidates]
    best_invalid = None

    for center, width, m_peak in candidates:
        epsilon = np.angle(P[center]) / (2 * np.pi * N_HALF)
        frac_cfo_rad = 2 * np.pi * epsilon
        cfo_hz = epsilon * FS

        if abs(cfo_hz) > MAX_CFO:
            continue

        n_idx = np.arange(len(buf))
        buf_corr = buf * np.exp(-1j * frac_cfo_rad * n_idx)
        coarse_frame_start = center - NCP // 2

        frame_start = fine_timing_ltf(buf_corr, coarse_frame_start, search_radius=24)
        if frame_start is None or frame_start < 0:
            continue
        if frame_start + FRAME_LEN > len(buf):
            continue

        out = _demod_at(buf_corr, frame_start)
        if out is None:
            continue
        bits, evm_db, sample_eq, pilot_eq, avg_amp, H_est = out

        valid, reason = validate_sync(pilot_eq, evm_db, cfo_hz)
        result = {
            "bits": bits, "cfo_hz": float(cfo_hz),
            "frame_start": int(frame_start),
            "consume": int(frame_start + FRAME_LEN),
            "avg_amp": avg_amp, "evm_db": float(evm_db),
            "sample_eq": sample_eq, "m_peak": float(m_peak),
            "plateau_w": int(width), "valid": valid, "reason": reason,
            "H_est": H_est,  # untuk Phase 1C (CIR/range estimation)
        }
        # ── Phase 1B: extract comm + sense bila aktif ────────────
        if PHASE1B_MODE:
            p1b = _demod_at_phase1b(buf_corr, frame_start)
            if p1b is not None:
                # Decode V2I packet (first 56 bit)
                text, ctr, crc_ok = decode_packet(p1b["comm_bits"][:PKT_TOTAL_BITS])
                # Sense BER vs known
                if SENSE_KNOWN_BITS is not None:
                    n_sb = min(len(p1b["sense_bits"]), len(SENSE_KNOWN_BITS))
                    sense_ber = float(np.sum(
                        p1b["sense_bits"][:n_sb] != SENSE_KNOWN_BITS[:n_sb]
                    )) / max(n_sb, 1)
                else:
                    sense_ber = float('nan')
                # Range estimation (Phase 1C v1.4)
                cir_mag, bin2m = cir_from_h_est(H_est, FS)
                rng = estimate_ranges(cir_mag, bin2m,
                                      snr_threshold_db=echo_threshold_db,
                                      direct_skip_bins=direct_skip_bins,
                                      max_echoes=max_echoes,
                                      max_range_m=max_range_m,
                                      bg_cir=bg_cir)
                result.update({
                    "p1b_text": text, "p1b_counter": ctr, "p1b_crc_ok": crc_ok,
                    "p1b_comm_evm_db": float(p1b["comm_evm_db"]),
                    "p1b_sense_evm_db": float(p1b["sense_evm_db"]),
                    "p1b_sense_ber": sense_ber,
                    "p1b_comm_sample_eq": p1b["comm_sample_eq"],
                    "p1c_direct_db": rng["direct_db"],
                    "p1c_noise_floor_db": rng["noise_floor_db"],
                    "p1c_echoes": rng["echoes"],
                    "p1c_n_echoes": rng["n_echoes"],
                    "p1c_cir_mag": cir_mag,    # 256-bin CIR (untuk plot/log)
                    "p1c_bin_to_meter": bin2m,
                })
        if valid:
            return result
        if best_invalid is None:
            best_invalid = result

    if best_invalid is not None:
        best_invalid["consume"] = LSYM
        best_invalid["bits"] = None
        return best_invalid
    return {"bits": None, "consume": LSYM}


def calc_ber(rx_bits):
    n = min(len(rx_bits), len(KNOWN_BITS))
    return float(np.sum(rx_bits[:n] != KNOWN_BITS[:n])) / n if n else 0.5


# ═══════════════════════════════════════════════════════════════════
# AWGN SELF-TEST
# ═══════════════════════════════════════════════════════════════════
def run_simulation(snr_db_list=(0, 5, 10, 15, 20), n_trials=200, cfo_hz=8e3,
                   timing_offset=37):
    print(f"\n{'═' * 60}")
    print(f"  AWGN SELF-TEST   FS={FS/1e6:.1f} MHz  "
          f"(CFO={cfo_hz/1e3:.1f} kHz, timing_offset={timing_offset})")
    print(f"{'═' * 60}")
    print(f"  {'SNR(dB)':>8} | {'BER':>10} | {'EVM(dB)':>8} | {'CFO_err(Hz)':>11}")
    print(f"  {'-' * 50}")

    sig_pwr = float(np.mean(np.abs(TX_FRAME) ** 2))

    for snr_db in snr_db_list:
        ber_acc = 0; ber_n = 0; evm_acc = 0.0
        cfo_err_acc = 0.0; cfo_err_n = 0
        snr_lin = 10 ** (snr_db / 10)
        n0 = sig_pwr / snr_lin

        for _ in range(n_trials):
            pad_pre  = np.zeros(timing_offset, dtype=np.complex64)
            pad_post = np.zeros(256, dtype=np.complex64)
            tx = np.concatenate([pad_pre, TX_FRAME, pad_post])
            n_idx = np.arange(len(tx))
            tx = tx * np.exp(1j * 2 * np.pi * cfo_hz / FS * n_idx)
            noise = (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx)))
            noise = noise.astype(np.complex64) * np.sqrt(n0 / 2)
            rx = (tx + noise).astype(np.complex64)
            res = sync_and_demod(rx)
            if res["bits"] is None:
                continue
            ber_acc += int(np.sum(res["bits"] != KNOWN_BITS))
            ber_n   += len(KNOWN_BITS)
            evm_acc += res["evm_db"]
            cfo_err_acc += abs(res["cfo_hz"] - cfo_hz)
            cfo_err_n   += 1

        if ber_n == 0:
            print(f"  {snr_db:>8.0f} | {'(no sync)':>10} | {'-':>8} | {'-':>11}")
            continue
        ber = ber_acc / ber_n
        evm = evm_acc / max(cfo_err_n, 1)
        cfe = cfo_err_acc / max(cfo_err_n, 1)
        print(f"  {snr_db:>8.0f} | {ber:>10.3e} | {evm:>8.2f} | {cfe:>11.0f}")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════
# USRP INIT & WORKERS
# ═══════════════════════════════════════════════════════════════════
def _init_usrp(serial, fpga, image_dir, is_tx, gain, ant, fs):
    sys.path.append("/usr/local/lib/python3.12/site-packages")
    import uhd
    fpga_suffix = f",fpga={fpga}" if fpga else ""
    strategies = [f"serial={serial}{fpga_suffix}"] if is_tx else \
                 [f"serial={serial}{fpga_suffix}",
                  f"name=LibreSDR_B210mini{fpga_suffix}"]
    old_env = os.environ.get("UHD_IMAGES_DIR")
    os.environ["UHD_IMAGES_DIR"] = image_dir
    usrp = None
    for args in strategies:
        try:
            usrp = uhd.usrp.MultiUSRP(args)
            break
        except RuntimeError:
            time.sleep(1)
    if old_env is None:
        os.environ.pop("UHD_IMAGES_DIR", None)
    else:
        os.environ["UHD_IMAGES_DIR"] = old_env
    if usrp is None:
        raise RuntimeError(f"Device {serial} tidak ditemukan")

    if is_tx:
        usrp.set_tx_rate(fs)
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(FC), 0)
        usrp.set_tx_gain(gain, 0)
        usrp.set_tx_antenna(ant, 0)
        usrp.set_tx_bandwidth(fs, 0)
        actual_gain = float(usrp.get_tx_gain(0))
    else:
        usrp.set_rx_rate(fs)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(FC), 0)
        usrp.set_rx_gain(gain, 0)
        usrp.set_rx_antenna(ant, 0)
        usrp.set_rx_bandwidth(fs, 0)
        actual_gain = float(usrp.get_rx_gain(0))

    role = "TX (Lutetia)" if is_tx else "RX (LibreSDR)"
    if abs(actual_gain - gain) > 0.5:
        print(f"[{role}] serial={serial} | gain={gain} dB REQUESTED → "
              f"{actual_gain:.2f} dB CLAMPED (hardware max)")
        print(f"       Max valid: TX≈89.75 dB, RX≈76 dB untuk B210/AD9361")
    else:
        print(f"[{role}] serial={serial} | gain={actual_gain:.2f} dB | "
              f"fc={FC/1e9:.3f} GHz | fs={fs/1e6:.2f} MHz")
    return usrp


def init_tx(fs, gain):
    return _init_usrp(TX_SERIAL, TX_FPGA, TX_IMAGE_DIR, True, gain, TX_ANT, fs)


def init_rx(fs, gain):
    return _init_usrp(RX_SERIAL, RX_FPGA, RX_IMAGE_DIR, False, gain, RX_ANT, fs)


def tx_worker(stop_event, gain_val, fs, frame_delay_s=0.0,
              phase1b=False, text="STEI"):
    """TX worker process. Re-init params for fresh process (mp 'spawn').

    phase1b=True: cycle through TX_FRAMES_P1B (counter 0..255) per send.
    """
    init_params(fs, phase1b=phase1b, text=text)
    try:
        usrp = init_tx(fs, gain_val.value)
    except Exception as e:
        print(f"[TX] FATAL: {e}")
        return
    import uhd
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.args = "num_send_frames=1000"
    st = usrp.get_tx_stream(st_args)
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst   = False

    pad = (-FRAME_LEN) % 1024
    silence_samples = int(frame_delay_s * fs) if frame_delay_s > 0 else 0
    silence_pad = np.zeros(pad + silence_samples, dtype=np.complex64)

    # Pre-build padded frames once (avoid per-loop concat)
    if phase1b:
        padded_frames = [
            np.concatenate([f, silence_pad]).astype(np.complex64)
            for f in TX_FRAMES_P1B
        ]
        print(f"[TX] PHASE 1B: {len(padded_frames)} frame cycle | text='{text}' | "
              f"frame={FRAME_LEN}+{pad}pad samples | dur={FRAME_LEN/fs*1e6:.1f} μs")
    else:
        padded_frames = [np.concatenate([TX_FRAME, silence_pad]).astype(np.complex64)]
        print(f"[TX] frame={FRAME_LEN}+{pad}pad+{silence_samples}silence samples | "
              f"frame_dur={FRAME_LEN/fs*1e6:.1f} μs")

    last_g, ctr = gain_val.value, 0
    n_frames_total = len(padded_frames)

    while not stop_event.is_set():
        if ctr % 50 == 0:
            g = gain_val.value
            if g != last_g:
                usrp.set_tx_gain(g, 0)
                last_g = g
        st.send(padded_frames[ctr % n_frames_total], md)
        md.start_of_burst = False
        ctr += 1
    md.end_of_burst = True
    st.send(np.zeros(256, dtype=np.complex64), md)
    print("[TX] Stop.")


def amplitude_probe(rx_usrp, fs, duration_s=2.0):
    """Capture raw IQ for `duration_s` and report signal stats."""
    import uhd
    st  = rx_usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    st.issue_stream_cmd(cmd)

    chunk = np.zeros(8192, dtype=np.complex64)
    md_rx = uhd.types.RXMetadata()
    samples = []
    target = int(duration_s * fs)
    got = 0
    while got < target:
        n = st.recv(chunk, md_rx)
        if md_rx.error_code != uhd.types.RXMetadataErrorCode.none:
            continue
        samples.append(chunk[:n].copy())
        got += n
    st.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    s = np.concatenate(samples)[:target]
    mean_amp = float(np.mean(np.abs(s)))
    rms      = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    peak     = float(np.max(np.abs(s)))
    dc_re    = float(np.mean(np.real(s)))
    dc_im    = float(np.mean(np.imag(s)))
    dc_mag   = np.hypot(dc_re, dc_im)
    crest_db = 20 * np.log10(peak / (rms + 1e-12))

    NF = 4096
    psd = np.zeros(NF)
    n_seg = 0
    for i in range(0, len(s) - NF, NF):
        seg = s[i:i + NF] * np.hanning(NF)
        psd += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
        n_seg += 1
    psd /= max(n_seg, 1)
    psd_db = 10 * np.log10(psd + 1e-18)
    psd_peak = float(np.max(psd_db))
    psd_med  = float(np.median(psd_db))
    spur_dr  = psd_peak - psd_med

    print(f"\n{'─' * 60}")
    print(f"  AMPLITUDE PROBE  (capture {duration_s}s @ {fs/1e6:.1f} MS/s)")
    print(f"{'─' * 60}")
    flag_amp = '⚠ TOO WEAK' if mean_amp < 0.005 else 'ok' if mean_amp < 0.5 else '⚠ NEAR SAT'
    flag_pk  = '⚠ SATURATED' if peak > 0.95 else 'ok'
    flag_dc  = '⚠ HIGH DC' if dc_mag > 0.01 else 'ok'
    flag_sd  = '⚠ NO BAND ENERGY' if spur_dr < 6 else 'signal present' if spur_dr > 15 else 'marginal'
    print(f"  mean|x|     : {mean_amp:.5f}      ({flag_amp})")
    print(f"  RMS         : {rms:.5f}")
    print(f"  peak        : {peak:.5f}      ({flag_pk})")
    print(f"  crest       : {crest_db:.1f} dB  (OFDM expected ≈ 8–12 dB)")
    print(f"  DC offset   : {dc_mag:.5f}      ({flag_dc})")
    print(f"  PSD peak    : {psd_peak:.1f} dB")
    print(f"  PSD median  : {psd_med:.1f} dB")
    print(f"  peak/median : {spur_dr:.1f} dB     ({flag_sd})")
    print(f"{'─' * 60}\n")
    return {"mean_amp": mean_amp, "rms": rms, "peak": peak,
            "dc_mag": dc_mag, "spur_dr": spur_dr}



# ═══════════════════════════════════════════════════════════════════
# PHASE 1D — Real-time visualization (matplotlib)
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# JCAS Forward-Scatter detector (ported dari ofdm_isac_bistatic_jcas)
# Dipakai sbg KLAIM DETEKSI utama; range/echo = konteks visual saja.
# ═══════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
# Phase 1F: Forward-Scattering / LoS disruption detector
# ─────────────────────────────────────────────────────────────────
class ForwardScatterDetector:
    """Simple JCAS detector for bistatic / forward-scattering experiments.

    Input utama per decoded frame:
      - avg_amp : mean amplitude frame dari RX path
      - H_est   : channel estimate kompleks dari LTF
      - cir_mag : CIR magnitude dari Phase 1C

    Output:
      - amplitude high-pass (LoS disruption / shadowing)
      - phase high-pass + Doppler proxy dari perubahan fase antar-frame
      - adaptive threshold via rolling median + MAD (CFAR sederhana)
    """

    def __init__(self, fs_frame, ma_len=30, cfar_len=120, threshold_k=4.5,
                 min_score=1.5, amp_weight=1.0, doppler_weight=1.0):
        self.fs_frame = float(fs_frame)
        self.ma_len = int(max(3, ma_len))
        self.cfar_len = int(max(self.ma_len + 5, cfar_len))
        self.threshold_k = float(threshold_k)
        self.min_score = float(min_score)
        self.amp_weight = float(amp_weight)
        self.doppler_weight = float(doppler_weight)

        self.amp_db_hist = deque(maxlen=self.cfar_len)
        self.phase_unwrapped_hist = deque(maxlen=self.cfar_len)
        self.doppler_hist = deque(maxlen=self.cfar_len)
        self.score_hist = deque(maxlen=self.cfar_len)
        self.prev_phase = None
        self.prev_unwrapped_phase = None

    @staticmethod
    def _robust_median(x, default=0.0):
        arr = np.asarray(list(x), dtype=float)
        if arr.size == 0:
            return float(default)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        return float(np.median(arr))

    @staticmethod
    def _mad_sigma(x):
        arr = np.asarray(list(x), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return 0.0
        med = np.median(arr)
        return float(1.4826 * np.median(np.abs(arr - med)) + 1e-12)

    @staticmethod
    def _wrap_phase_delta(phi_now, phi_prev):
        return float(np.angle(np.exp(1j * (phi_now - phi_prev))))

    def update(self, avg_amp=None, H_est=None, cir_mag=None):
        # 1) LoS amplitude metric. Prefer direct CIR peak if available.
        if cir_mag is not None and len(cir_mag) > 0:
            amp_lin = float(np.max(np.asarray(cir_mag)))
        elif avg_amp is not None:
            amp_lin = float(avg_amp)
        else:
            amp_lin = 0.0
        amp_db = float(20 * np.log10(max(amp_lin, 1e-12)))

        # 2) Phase metric from mean active channel estimate.
        if H_est is not None:
            active_h = np.asarray([H_est[k % NSC] for k in ACTIVE_REL], dtype=complex)
            # Weighting by magnitude helps stabilize low-SNR subcarriers.
            mag = np.abs(active_h)
            if np.sum(mag) > 1e-12:
                los_complex = np.sum(active_h * mag) / np.sum(mag)
            else:
                los_complex = np.mean(active_h)
            phase_raw = float(np.angle(los_complex))
        else:
            phase_raw = 0.0

        if self.prev_phase is None:
            phase_unwrapped = phase_raw
            phase_delta = 0.0
        else:
            phase_delta = self._wrap_phase_delta(phase_raw, self.prev_phase)
            phase_unwrapped = self.prev_unwrapped_phase + phase_delta
        self.prev_phase = phase_raw
        self.prev_unwrapped_phase = phase_unwrapped

        # 3) Doppler proxy: f_D = Δphase/(2π) × frame_rate.
        doppler_hz = float((phase_delta / (2 * np.pi)) * self.fs_frame)

        # 4) High-pass / background removal with moving median.
        amp_bg = self._robust_median(list(self.amp_db_hist)[-self.ma_len:], amp_db)
        phase_bg = self._robust_median(list(self.phase_unwrapped_hist)[-self.ma_len:], phase_unwrapped)
        dopp_bg = self._robust_median(list(self.doppler_hist)[-self.ma_len:], doppler_hz)

        amp_hp_db = float(amp_db - amp_bg)
        phase_hp_rad = float(phase_unwrapped - phase_bg)
        doppler_hp_hz = float(doppler_hz - dopp_bg)

        # 5) Normalize using robust noise scale from history.
        amp_sigma = max(self._mad_sigma(self.amp_db_hist), 0.2)       # dB
        dopp_sigma = max(self._mad_sigma(self.doppler_hist), 2.0)     # Hz
        score = float(np.sqrt(
            (self.amp_weight * abs(amp_hp_db) / amp_sigma) ** 2 +
            (self.doppler_weight * abs(doppler_hp_hz) / dopp_sigma) ** 2
        ))

        # 6) Simple CFAR: threshold = median(score_ref) + K × MAD(score_ref).
        score_ref = list(self.score_hist)
        if len(score_ref) >= max(10, self.ma_len):
            score_floor = self._robust_median(score_ref, 0.0)
            score_sigma = self._mad_sigma(score_ref)
            threshold = max(self.min_score, float(score_floor + self.threshold_k * score_sigma))
        else:
            threshold = self.min_score
        object_detected = bool(score > threshold and len(score_ref) >= max(5, self.ma_len // 2))

        # Update histories after CFAR decision, so current event does not train threshold first.
        self.amp_db_hist.append(amp_db)
        self.phase_unwrapped_hist.append(phase_unwrapped)
        self.doppler_hist.append(doppler_hz)
        self.score_hist.append(score)

        return {
            "amp_db": amp_db,
            "amp_hp_db": amp_hp_db,
            "phase_rad": phase_raw,
            "phase_unwrapped_rad": float(phase_unwrapped),
            "phase_hp_rad": phase_hp_rad,
            "doppler_hz": doppler_hz,
            "doppler_hp_hz": doppler_hp_hz,
            "score": score,
            "threshold": float(threshold),
            "object_detected": object_detected,
            "status": "ADA_OBJEK" if object_detected else "CLEAR",
        }



def setup_live_plot(fs, max_history=200, max_range_m=12.0):
    """Dashboard ISAC: comm (text/PRR/const/EVM) + JCAS forward-scatter detector.

    Layout 2x3:
      (0,0) Range Profile (CIR)  -> KONTEKS kanal, BUKAN klaim deteksi
      (0,1) Comm constellation (QPSK)
      (0,2) JCAS score + CFAR + DOT 'ADA OBJEK'  <- KLAIM DETEKSI
      (1,0) EVM trend
      (1,1) PRR + packet text log
      (1,2) Forward-scatter LoS amplitude blip (dB)
    """
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    ax_range, ax_const, ax_score = axes[0]
    ax_evm, ax_log, ax_amp = axes[1]
    fig.suptitle(f"ISAC Bistatic Live | FS={fs/1e6:.1f} MS/s | "
                 f"V2I 'STEI' comm + JCAS forward-scatter sensing",
                 fontsize=12, fontweight='bold')

    # Range profile (KONTEKS) -- echo marker DIHAPUS (artefak blind-zone)
    bin_max_m = 3e8 / (fs * (CIR_NFFT / NSC)) * (CIR_NFFT // 2)
    range_axis = np.arange(CIR_NFFT // 2) * 3e8 / (fs * CIR_NFFT / NSC)
    line_cir, = ax_range.plot(range_axis, np.zeros(CIR_NFFT // 2), 'b-',
                              linewidth=1.0, label='CIR (current)')
    line_bg,  = ax_range.plot(range_axis, np.zeros(CIR_NFFT // 2), 'g--',
                              linewidth=0.7, alpha=0.6, label='Background median')
    line_sub, = ax_range.plot(range_axis, np.zeros(CIR_NFFT // 2), 'r-',
                              linewidth=1.2, alpha=0.8, label='CIR - background')
    direct_marker = ax_range.axvline(0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_range.set_xlabel("Delta-range from direct path (m)")
    ax_range.set_ylabel("CIR magnitude (dB)")
    ax_range.set_title("Range Profile - KONTEKS kanal (bukan klaim deteksi)")
    ax_range.set_xlim(0, min(max_range_m, bin_max_m))
    ax_range.set_ylim(-80, 5)
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(loc='upper right', fontsize=8)

    # Constellation
    scatter_const = ax_const.scatter([], [], s=20, c='cyan', alpha=0.6, edgecolors='none')
    for s in QPSK_TABLE:
        ax_const.plot(s.real, s.imag, '+', color='red', markersize=14, markeredgewidth=2)
    ax_const.set_xlim(-1.6, 1.6); ax_const.set_ylim(-1.6, 1.6); ax_const.set_aspect('equal')
    ax_const.set_xlabel("In-phase"); ax_const.set_ylabel("Quadrature")
    ax_const.set_title(f"Comm constellation (QPSK, {N_COMM} center SC)")
    ax_const.grid(True, alpha=0.3)
    ax_const.axhline(0, color='gray', linewidth=0.5)
    ax_const.axvline(0, color='gray', linewidth=0.5)

    # JCAS score + CFAR + DOT (KLAIM DETEKSI)
    line_score, = ax_score.plot([], [], 'b-', linewidth=1.2, label='JCAS score')
    line_thr,   = ax_score.plot([], [], 'r--', linewidth=1.0, label='Adaptive CFAR threshold')
    detect_scatter = ax_score.scatter([], [], s=60, marker='o', color='red',
                                      zorder=5, label='ADA OBJEK')
    status_text = ax_score.text(0.02, 0.92, 'Status: WAIT', transform=ax_score.transAxes,
                                fontsize=12, fontweight='bold')
    ax_score.set_xlabel("Frame index"); ax_score.set_ylabel("Score")
    ax_score.set_title("JCAS forward-scatter - DETEKSI OBJEK")
    ax_score.set_ylim(0, 10); ax_score.grid(True, alpha=0.3)
    ax_score.legend(loc='upper right', fontsize=8)

    # EVM trend
    line_evm_comm,  = ax_evm.plot([], [], 'b-', label='Comm EVM (QPSK)', linewidth=1.0)
    line_evm_sense, = ax_evm.plot([], [], 'g-', label='Sense EVM (BPSK)', linewidth=1.0)
    ax_evm.axhline(-10, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='QPSK threshold')
    ax_evm.set_xlabel("Frame index"); ax_evm.set_ylabel("EVM (dB)")
    ax_evm.set_title("EVM trend"); ax_evm.set_ylim(-25, 5); ax_evm.grid(True, alpha=0.3)
    ax_evm.legend(loc='lower right', fontsize=8)

    # PRR + Text log
    ax_log.axis('off')
    ax_log.text(0.02, 0.96, "Recent packets", transform=ax_log.transAxes,
                fontsize=11, fontweight='bold')
    log_text = ax_log.text(0.02, 0.85, "", transform=ax_log.transAxes,
                           fontsize=9, family='monospace', verticalalignment='top')
    prr_text = ax_log.text(0.02, 0.10, "", transform=ax_log.transAxes,
                           fontsize=11, fontweight='bold', color='green')

    # Forward-scatter amplitude blip
    line_amp, = ax_amp.plot([], [], 'm-', linewidth=1.0, label='LoS amplitude HP (dB)')
    ax_amp.axhline(0, linestyle=':', linewidth=0.8, alpha=0.5)
    ax_amp.set_xlabel("Frame index"); ax_amp.set_ylabel("Amplitude HP (dB)")
    ax_amp.set_title("Forward-scatter LoS amplitude blip")
    ax_amp.grid(True, alpha=0.3); ax_amp.legend(loc='upper right', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False); plt.pause(0.1)

    return {
        "fig": fig,
        "ax_range": ax_range, "ax_const": ax_const, "ax_score": ax_score,
        "ax_evm": ax_evm, "ax_log": ax_log, "ax_amp": ax_amp,
        "line_cir": line_cir, "line_bg": line_bg, "line_sub": line_sub,
        "direct_marker": direct_marker, "scatter_const": scatter_const,
        "line_score": line_score, "line_thr": line_thr,
        "detect_scatter": detect_scatter, "status_text": status_text,
        "line_evm_comm": line_evm_comm, "line_evm_sense": line_evm_sense,
        "log_text": log_text, "prr_text": prr_text, "line_amp": line_amp,
        "range_axis": range_axis, "max_history": max_history,
    }


def update_live_plot(handles, state, last_n_displayed):
    """Update dashboard ISAC. Range Profile = konteks; JCAS = klaim deteksi."""
    if not state["p1b_records"]:
        return last_n_displayed
    cur_n = len(state["p1b_records"])
    if cur_n == last_n_displayed:
        return cur_n
    rec_latest = state["p1b_records"][-1]

    # Range profile (KONTEKS) -- tanpa echo marker
    if state["cir_history"]:
        latest = state["cir_history"][-1]
        cir = np.maximum(latest["cir_mag"], 0)
        bin2m = latest["bin_to_meter"]
        n_half = len(cir) // 2
        cir_fwd = cir[:n_half]
        cir_db = 20 * np.log10(cir_fwd + 1e-12)
        direct_bin = int(np.argmax(cir_fwd))
        direct_db = float(cir_db[direct_bin])
        cir_shifted = cir_fwd[direct_bin:]
        delta_range_x = np.arange(len(cir_shifted)) * bin2m
        handles["line_cir"].set_data(delta_range_x,
                                     20 * np.log10(cir_shifted + 1e-12) - direct_db)
        if len(state["cir_history"]) >= 10:
            cir_stack = np.stack([np.maximum(h["cir_mag"][:n_half], 0)
                                  for h in state["cir_history"]])
            bg = np.median(cir_stack, axis=0)
            bg_shifted = bg[direct_bin:]
            handles["line_bg"].set_data(delta_range_x,
                                        20 * np.log10(bg_shifted + 1e-12) - direct_db)
            sub = np.maximum(cir_shifted - bg_shifted, 1e-12)
            handles["line_sub"].set_data(delta_range_x, 20 * np.log10(sub) - direct_db)
        else:
            handles["line_bg"].set_data([], [])
            handles["line_sub"].set_data([], [])

    # Constellation
    if rec_latest.get("comm_sample_eq") is not None:
        pts = rec_latest["comm_sample_eq"]
        handles["scatter_const"].set_offsets(
            np.column_stack([np.real(pts), np.imag(pts)]))

    # EVM trend
    n_show = min(handles["max_history"], cur_n)
    recs_show = state["p1b_records"][-n_show:]
    xs = [r["frame_idx"] for r in recs_show]
    handles["line_evm_comm"].set_data(xs, [r["comm_evm_db"] for r in recs_show])
    handles["line_evm_sense"].set_data(xs, [r["sense_evm_db"] for r in recs_show])
    if xs:
        handles["ax_evm"].set_xlim(min(xs), max(xs) + 1)

    # JCAS panel (KLAIM DETEKSI)
    score  = [r.get("jcas_score", 0.0) for r in recs_show]
    thr    = [r.get("jcas_threshold", 0.0) for r in recs_show]
    amp_hp = [r.get("jcas_amp_hp_db", 0.0) for r in recs_show]
    handles["line_score"].set_data(xs, score)
    handles["line_thr"].set_data(xs, thr)
    handles["line_amp"].set_data(xs, amp_hp)
    det_x = [r["frame_idx"] for r in recs_show if r.get("jcas_object_detected", False)]
    det_y = [r.get("jcas_score", 0.0) for r in recs_show if r.get("jcas_object_detected", False)]
    if det_x:
        handles["detect_scatter"].set_offsets(np.column_stack([det_x, det_y]))
    else:
        handles["detect_scatter"].set_offsets(np.empty((0, 2)))
    handles["status_text"].set_text(
        f"Status: {rec_latest.get('jcas_status','WAIT')} | "
        f"score={rec_latest.get('jcas_score',0.0):.2f} | "
        f"thr={rec_latest.get('jcas_threshold',0.0):.2f}")
    handles["status_text"].set_color('red' if rec_latest.get('jcas_object_detected') else 'green')
    if xs:
        for ax in (handles["ax_score"], handles["ax_amp"]):
            ax.set_xlim(min(xs), max(xs) + 1)
        finite_sc = [s for s in score if s < 100.0]   # buang spike warmup (sigma~0)
        smax = max(max(finite_sc) if finite_sc else 1.0, max(thr) if thr else 1.0, 2.0)
        handles["ax_score"].set_ylim(0, max(10.0, smax * 1.25))
        if amp_hp:
            a = max(abs(min(amp_hp)), abs(max(amp_hp)), 1.0)
            handles["ax_amp"].set_ylim(-a * 1.2, a * 1.2)

    # PRR + log
    n_total = len(state["p1b_records"])
    n_ok = state["p1b_crc_ok_count"]
    prr = n_ok / max(n_total, 1) * 100
    handles["prr_text"].set_text(
        f"PRR: {prr:.2f}%  ({n_ok}/{n_total} packets)\n"
        f"OVF: {state['ovf']}  |  Recent comm EVM: {rec_latest['comm_evm_db']:.2f} dB")
    recent = state["p1b_records"][-8:]
    log_lines = [f"{'frm':>4} {'text':>4} {'ctr':>4} {'CRC':>4} {'commEVM':>8} {'JCAS':>9}"]
    log_lines.append("-" * 46)
    for r in recent:
        crc_str = "OK" if r["crc_ok"] else "FAIL"
        log_lines.append(
            f"{r['frame_idx']:>4} {r['text']:>4} {r['counter']:>4} {crc_str:>4} "
            f"{r['comm_evm_db']:>8.2f} {r.get('jcas_status','-'):>9}")
    handles["log_text"].set_text("\n".join(log_lines))

    handles["fig"].canvas.draw_idle()
    handles["fig"].canvas.flush_events()
    return cur_n

def run_hardware(fs, n_frames=100, tx_gain=80.0, rx_gain=70.0,
                 probe_wait=10.0, frame_delay=0.0,
                 dc_offset_auto=True, probe_only=False,
                 warmup_frames=50, verbose=True,
                 phase1b=False, text="STEI", log_csv=None,
                 plot=False, echo_threshold_db=14.0, cir_history_len=30,
                 max_range_m=12.0, direct_skip_bins=3, max_echoes=3,
                 calib_frames=300, save_clutter=None, load_clutter=None,
                 jcas=True, sense_ma_len=30, sense_cfar_len=120,
                 sense_threshold_k=4.5, sense_min_score=1.5):
    """
    Single-FS hardware run. Returns metrics dict for sweep aggregation.

    phase1b=True: aktifkan V2I "STEI" comm + Phase 1C range estimation.
    plot=True   : aktifkan Phase 1D matplotlib live plot (requires phase1b).
    log_csv     : path untuk per-frame Phase 1B CSV log.
    echo_threshold_db: SNR threshold di atas noise floor (default 14 dB).
    cir_history_len  : panjang CIR history (untuk plot Phase 1D).

    Phase 1C params:
    max_range_m      : hard geometric limit delta path (default 12 m, lab 4×4 m).
    direct_skip_bins : skip N bin setelah direct (mask Hanning sidelobe, default 3).
    max_echoes       : top-K echo terkuat (default 3).
    calib_frames     : jumlah frame fase kalibrasi clutter map. Ruangan WAJIB diam.
    save_clutter     : path .npy untuk simpan clutter map hasil kalibrasi.
    load_clutter     : path .npy clutter map tersimpan → skip fase kalibrasi.

    ARSITEKTUR v1.3: single-thread rx_thread (revert producer-consumer v1.2
    yang gagal). Chunk recv diperbesar (RECV_CHUNK) supaya st.recv jarang
    dipanggil → overhead rendah → buffer USB tidak telat dikuras.
    Background subtraction: frozen clutter map (kalibrasi sekali, lalu beku).
    """
    init_params(fs, phase1b=phase1b, text=text)
    import uhd

    tx_gain_sh = mp.Value('d', tx_gain)
    rx_gain_sh = mp.Value('d', rx_gain)
    stop_ev = mp.Event()

    if verbose:
        print(f"\n{'█' * 60}")
        print(f"  RUN @ FS = {fs/1e6:.2f} MS/s  |  TX={tx_gain} dB  RX={rx_gain} dB")
        print(f"  BW eff ≈ {N_ACTIVE * (fs/NSC) / 1e6:.2f} MHz  |  "
              f"δR theoretical ≈ {3e8/(2*N_ACTIVE*(fs/NSC)):.1f} m")
        if phase1b:
            print(f"  PHASE 1B+1C aktif | text='{text}' | "
                  f"comm SC={N_COMM} (QPSK) | sense SC={N_SENSE} (BPSK)")
            print(f"  PHASE 1C v1.4 | max_range={max_range_m:.1f}m  "
                  f"thr={echo_threshold_db:.1f}dB  skip={direct_skip_bins}bin  "
                  f"top_k={max_echoes}")
            if load_clutter:
                print(f"  BG: frozen clutter map dari file (kalibrasi dilewati)")
            else:
                print(f"  BG: frozen | kalibrasi {calib_frames} frame "
                      f"(RUANGAN HARUS DIAM saat kalibrasi)")
            print(f"  ARSITEKTUR: single-thread RX, chunk recv besar (anti-OVF)")
            if n_frames <= calib_frames and not load_clutter:
                print(f"  ⚠ WARNING: --frames ({n_frames}) <= --calib-frames "
                      f"({calib_frames}). Tidak ada frame untuk fase deteksi!")
                print(f"    Naikkan --frames jadi minimal {calib_frames + 200}.")
        print(f"{'█' * 60}")

    print("[INIT] Start TX...")
    tx_proc = mp.Process(target=tx_worker,
                         args=(stop_ev, tx_gain_sh, fs, frame_delay, phase1b, text))
    tx_proc.start()
    time.sleep(3)

    print("[INIT] Start RX...")
    try:
        rx_usrp = init_rx(fs, rx_gain_sh.value)
    except Exception as e:
        print(f"[RX] FATAL: {e}")
        stop_ev.set()
        tx_proc.join(timeout=5)
        return None

    if dc_offset_auto:
        try:
            rx_usrp.set_rx_dc_offset(True, 0)
            rx_usrp.set_rx_iq_balance(True, 0)
            print("[INIT] RX DC + IQ auto-correction: ON")
        except Exception as e:
            print(f"[INIT] DC/IQ not supported: {e}")

    if probe_wait > 0:
        print(f"[INIT] Probe wait {probe_wait}s...")
        time.sleep(probe_wait)

    probe_stats = amplitude_probe(rx_usrp, fs, duration_s=2.0)
    if probe_stats["mean_amp"] < 0.001:
        print("✗ ABORT: signal too low.")
        stop_ev.set()
        tx_proc.join(timeout=5)
        return {
            "fs_mhz": fs/1e6, "tx_gain": tx_gain, "rx_gain": rx_gain,
            "status": "ABORT_LOW_SIGNAL", "n_decoded": 0, "n_target": n_frames,
            "valid_rate": 0.0, "mean_ber": float('nan'), "mean_evm_db": float('nan'),
            "probe_mean_amp": probe_stats["mean_amp"],
            "probe_peak": probe_stats["peak"],
            "ovf_total": 0, "ovf_steady": 0, "ovf_rate_steady": 0.0,
        }
    if probe_only:
        stop_ev.set()
        tx_proc.join(timeout=5)
        return {"fs_mhz": fs/1e6, "status": "PROBE_ONLY",
                "probe_mean_amp": probe_stats["mean_amp"],
                "probe_peak": probe_stats["peak"]}

    state = {"bers": [], "cfos": [], "amps": [], "evms": [],
             "n": 0, "ovf": 0, "ovf_at_warmup_end": None, "running": True,
             # Phase 1B/1C accumulators
             "p1b_records": [],     # per-frame dict
             "p1b_crc_ok_count": 0,
             "p1b_crc_fail_count": 0,
             # Phase 1D plot buffers
             "cir_history": deque(maxlen=cir_history_len),
             # Phase 1C v1.3: frozen clutter map calibration
             "clutter_map": None,        # frozen clutter map (beku setelah kalibrasi)
             "calib_buffer": [],         # CIR terkumpul selama fase kalibrasi
             "calib_done": False,        # True setelah clutter map dibekukan
             # JCAS forward-scatter detector records (KLAIM DETEKSI utama)
             "jcas_records": [],
             }

    # JCAS detector: 1 frame = FRAME_LEN/fs detik → fs_frame = fs/FRAME_LEN.
    # cir_mag SENGAJA tidak dipakai (lihat update()); metrik = avg_amp + H_est,
    # versi yang terbukti akurat (range/echo CIR salah-alat untuk ruangan kecil).
    jcas_detector = ForwardScatterDetector(
        fs_frame=fs / FRAME_LEN,
        ma_len=sense_ma_len, cfar_len=sense_cfar_len,
        threshold_k=sense_threshold_k, min_score=sense_min_score,
    ) if jcas else None

    # Phase 1C v1.3: load clutter map tersimpan (skip kalibrasi)
    if phase1b and load_clutter:
        try:
            cm = np.load(load_clutter)
            if len(cm) == CIR_NFFT:
                state["clutter_map"] = cm
                state["calib_done"] = True
                print(f"[INIT] Clutter map dimuat dari {load_clutter} "
                      f"-> fase kalibrasi DILEWATI")
            else:
                print(f"[WARN] Clutter map {load_clutter} ukuran salah "
                      f"({len(cm)} != {CIR_NFFT}) -> kalibrasi normal")
        except Exception as e:
            print(f"[WARN] Gagal load clutter map: {e} -> kalibrasi normal")

    # Phase 1D: setup matplotlib live plot (kalau aktif & phase1b)
    plot_handles = None
    if plot:
        if not phase1b:
            print("[WARN] --plot needs --phase1b mode. Plot disabled.")
        elif "DISPLAY" not in os.environ and sys.platform == "linux":
            print("[WARN] No DISPLAY env detected (SSH tanpa -X?).")
            print("       Plot membutuhkan X11 display. Disabled.")
            print("       Workaround: pakai --log-csv untuk offline analysis,")
            print("       atau jalankan script dari desktop session langsung.")
        else:
            try:
                print("[INIT] Membuka jendela live plot matplotlib...")
                print("       Tkinter canvas init bisa 3-10 detik pertama kali. MOHON SABAR.")
                t0 = time.time()
                plot_handles = setup_live_plot(fs)
                print(f"[INIT] Plot window ready ✓ ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[WARN] Plot setup gagal: {e}")
                print("       Continuing without plot. Data tetap tersimpan di log CSV.")
                plot_handles = None

    # CSV writer setup (Phase 1B only)
    csv_file = None
    csv_writer = None
    if phase1b and log_csv:
        csv_file = open(log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        active_idx = np.where(np.abs(X_LTF) > 1e-9)[0]      # subcarrier aktif (50)
        csv_writer.writerow([
            "frame_idx", "text", "counter", "crc_ok",
            "comm_evm_db", "sense_evm_db", "sense_ber",
            "cfo_hz", "amp", "ovf",
            "direct_db", "noise_floor_db", "n_echoes", "echoes_m",
            "jcas_status", "jcas_amp_hp_db", "jcas_doppler_hp_hz",
            "jcas_score", "jcas_threshold", "jcas_object_detected",
        ] + [f"habs_{k}" for k in active_idx])

    # ════════════════════════════════════════════════════════════
    # PHASE 1C v1.3 — Single-thread RX (revert producer-consumer)
    #   Producer-consumer v1.2 GAGAL: queue.Queue Python tidak sanggup
    #   throughput chunk-kecil 17k/s → perang GIL → 2 juta drop.
    #   v1.3: kembali single-thread (terbukti 99.6% di run 75k) +
    #   chunk recv DIPERBESAR (RECV_CHUNK) → st.recv ~600x/s bukan 17k.
    #   Itu pangkas overhead recv ~28x → buffer USB tak telat dikuras.
    # ════════════════════════════════════════════════════════════
    RECV_CHUNK = 65536   # sample per st.recv (~1.6 ms @ 40 MS/s)

    def _resolve_bg_cir():
        """Tentukan bg_cir untuk frame sekarang (bg_mode='frozen' saja).
        Returns (bg_cir, phase_label): 'calib' | 'detect' | 'off'."""
        if not phase1b:
            return None, "off"
        if state["calib_done"]:
            return state["clutter_map"], "detect"
        return None, "calib"   # fase kalibrasi: belum subtraksi

    def rx_thread():
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.args = "num_recv_frames=1000"
        st  = rx_usrp.get_rx_stream(st_args)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        st.issue_stream_cmd(cmd)

        chunk = np.zeros(RECV_CHUNK, dtype=np.complex64)
        md_rx = uhd.types.RXMetadata()
        buf   = np.zeros(0, dtype=np.complex64)
        last_g, ctr = rx_gain_sh.value, 0

        while state["running"]:
            if ctr % 50 == 0:
                g = rx_gain_sh.value
                if g != last_g:
                    rx_usrp.set_rx_gain(g, 0)
                    last_g = g

            nsamp = st.recv(chunk, md_rx)
            if md_rx.error_code == uhd.types.RXMetadataErrorCode.overflow:
                # OVF: buang buf untuk resync. clutter_map TIDAK disentuh
                # (sudah beku, OVF tidak merusaknya).
                state["ovf"] += 1
                buf = np.zeros(0, dtype=np.complex64)
                ctr += 1
                continue
            if nsamp <= 0:
                ctr += 1
                continue

            buf = np.concatenate([buf, chunk[:nsamp]])

            while len(buf) >= FRAME_LEN + 256:
                bg_cir, bg_phase = _resolve_bg_cir()

                res = sync_and_demod(buf, sc_threshold=0.7,
                                     echo_threshold_db=echo_threshold_db,
                                     bg_cir=bg_cir,
                                     max_range_m=max_range_m,
                                     direct_skip_bins=direct_skip_bins,
                                     max_echoes=max_echoes)
                if res["bits"] is not None:
                    b = calc_ber(res["bits"])
                    state["bers"].append(b)
                    state["cfos"].append(res["cfo_hz"])
                    state["amps"].append(res["avg_amp"])
                    state["evms"].append(res["evm_db"])
                    state["n"] += 1

                    if state["n"] == warmup_frames:
                        state["ovf_at_warmup_end"] = state["ovf"]

                    # ── JCAS forward-scatter detector (KLAIM DETEKSI) ──
                    # cir_mag=None → detektor pakai avg_amp (metrik andal),
                    # BUKAN max(cir_mag) yg lemah utk forward-scatter.
                    sensing = None
                    if jcas_detector is not None:
                        sensing = jcas_detector.update(
                            avg_amp=res.get("avg_amp"),
                            H_est=res.get("H_est"),
                            cir_mag=None,
                        )
                        state["jcas_records"].append({
                            "frame_idx": state["n"],
                            "jcas_status": sensing["status"],
                            "jcas_amp_hp_db": sensing["amp_hp_db"],
                            "jcas_doppler_hp_hz": sensing["doppler_hp_hz"],
                            "jcas_score": sensing["score"],
                            "jcas_threshold": sensing["threshold"],
                            "jcas_object_detected": sensing["object_detected"],
                        })

                    if phase1b and "p1b_text" in res:
                        # ── Fase kalibrasi: kumpulkan CIR, freeze di akhir ──
                        if not state["calib_done"]:
                            state["calib_buffer"].append(res["p1c_cir_mag"])
                            if len(state["calib_buffer"]) >= calib_frames:
                                cm = np.median(
                                    np.array(state["calib_buffer"]), axis=0)
                                state["clutter_map"] = cm
                                state["calib_done"] = True
                                state["calib_buffer"] = []
                                print(f"\n[CALIB] Clutter map dibekukan dari "
                                      f"{calib_frames} frame -> FASE DETEKSI mulai\n")
                                if save_clutter:
                                    try:
                                        np.save(save_clutter, cm)
                                        print(f"[CALIB] Clutter map disimpan: "
                                              f"{save_clutter}")
                                    except Exception as e:
                                        print(f"[WARN] Gagal simpan clutter: {e}")

                        rec = {
                            "frame_idx": state["n"],
                            "text": res["p1b_text"],
                            "counter": res["p1b_counter"],
                            "crc_ok": res["p1b_crc_ok"],
                            "comm_evm_db": res["p1b_comm_evm_db"],
                            "sense_evm_db": res["p1b_sense_evm_db"],
                            "sense_ber": res["p1b_sense_ber"],
                            "direct_db": res["p1c_direct_db"],
                            "noise_floor_db": res["p1c_noise_floor_db"],
                            "echoes": res["p1c_echoes"],
                            "n_echoes": res["p1c_n_echoes"],
                            "comm_sample_eq": res.get("p1b_comm_sample_eq"),
                            "bg_phase": bg_phase,
                            # JCAS forward-scatter (klaim deteksi)
                            "jcas_status": sensing["status"] if sensing else "OFF",
                            "jcas_amp_hp_db": sensing["amp_hp_db"] if sensing else 0.0,
                            "jcas_doppler_hp_hz": sensing["doppler_hp_hz"] if sensing else 0.0,
                            "jcas_score": sensing["score"] if sensing else 0.0,
                            "jcas_threshold": sensing["threshold"] if sensing else 0.0,
                            "jcas_object_detected": sensing["object_detected"] if sensing else False,
                        }
                        state["p1b_records"].append(rec)
                        if rec["crc_ok"]:
                            state["p1b_crc_ok_count"] += 1
                        else:
                            state["p1b_crc_fail_count"] += 1
                        state["cir_history"].append({
                            "cir_mag": res["p1c_cir_mag"],
                            "bin_to_meter": res["p1c_bin_to_meter"],
                            "frame_idx": state["n"],
                        })
                        if csv_writer is not None:
                            echoes_str = ";".join(
                                f"{r:.2f}@{db:.1f}dB" for r, db in rec["echoes"]
                            )
                            csv_writer.writerow([
                                rec["frame_idx"], rec["text"], rec["counter"], int(rec["crc_ok"]),
                                f"{rec['comm_evm_db']:.2f}", f"{rec['sense_evm_db']:.2f}",
                                f"{rec['sense_ber']:.4f}", f"{res['cfo_hz']:.0f}",
                                f"{res['avg_amp']:.4f}", state["ovf"],
                                f"{rec['direct_db']:.1f}", f"{rec['noise_floor_db']:.1f}",
                                rec["n_echoes"], echoes_str,
                                rec["jcas_status"],
                                f"{rec['jcas_amp_hp_db']:.3f}",
                                f"{rec['jcas_doppler_hp_hz']:.3f}",
                                f"{rec['jcas_score']:.3f}",
                                f"{rec['jcas_threshold']:.3f}",
                                int(rec["jcas_object_detected"]),
                            ] + [f"{v:.5f}" for v in np.abs(res["H_est"][active_idx])])
                            csv_file.flush()

                    if state["n"] <= 2 and res["sample_eq"] is not None and verbose:
                        print(f"\n--- Frame {state['n']} sample ---")
                        for i, v in enumerate(res["sample_eq"][:5]):
                            print(f"  [{i}] {v.real:+.3f} {v.imag:+.3f}j")
                    buf = buf[res["consume"]:]
                else:
                    advance = res["consume"] if res["consume"] > 0 else 1
                    buf = buf[advance:]
                    if advance == 0:
                        break
            ctr += 1

        st.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    threading.Thread(target=rx_thread, daemon=True).start()

    if verbose:
        print(f"\n[RUN] Target {n_frames} frames (warmup={warmup_frames}). Ctrl+C to stop.\n")
        if phase1b:
            print(f"  {'Frm':>4} | {'text':>4} {'ctr':>3} {'CRC':>4} | "
                  f"{'commEVM':>7} {'snsBER':>7} | {'JCAS':>9} {'score':>6} | "
                  f"{'OVF':>4}")
        else:
            print(f"  {'Frm':>4} | {'BER':>8} | {'CFO(Hz)':>8} | {'Amp':>6} | "
                  f"{'EVM(dB)':>7} | {'OVF':>4} | Verdict")

    try:
        last_n = 0
        last_plot_n = 0
        plot_update_interval = 0.15  # ~6.7 Hz
        last_plot_time = 0.0
        while state["n"] < n_frames:
            time.sleep(0.01)
            cur = state["n"]
            if verbose:
                for i in range(last_n, cur):
                    if phase1b and i < len(state["p1b_records"]):
                        rec = state["p1b_records"][i]
                        crc_str = "OK " if rec["crc_ok"] else "FAIL"
                        echoes_str = ", ".join(
                            f"{r:.1f}@{db:.0f}" for r, db in rec["echoes"][:3]
                        ) if rec["echoes"] else "-"
                        ph = rec.get("bg_phase", "")
                        ph_tag = {"calib": " [CAL]", "detect": "",
                                  "warmup": " [w]", "off": ""}.get(ph, "")
                        print(f"  {i+1:>4} | {rec['text']:>4} {rec['counter']:>3} {crc_str:>4} | "
                              f"{rec['comm_evm_db']:>7.2f} {rec['sense_ber']:>7.4f} | "
                              f"{rec.get('jcas_status','-'):>9} {rec.get('jcas_score',0.0):>6.2f} | "
                              f"{state['ovf']:>4}{ph_tag}")
                    else:
                        b = state["bers"][i]; cfo = state["cfos"][i]
                        amp = state["amps"][i]; evm = state["evms"][i]
                        v = "✓ OK" if b < 0.05 else ("⚠ MARG" if b < 0.20 else "✗ BAD")
                        marker = " [warmup]" if (i+1) <= warmup_frames else ""
                        print(f"  {i+1:>4} | {b:>8.4f} | {cfo:>8.0f} | "
                              f"{amp:>6.4f} | {evm:>7.2f} | {state['ovf']:>4} | {v}{marker}")
            last_n = cur

            # Phase 1D: update plot (rate-limited)
            if plot_handles is not None and cur > last_plot_n:
                now = time.time()
                if now - last_plot_time > plot_update_interval:
                    try:
                        last_plot_n = update_live_plot(plot_handles, state, last_plot_n)
                    except Exception as e:
                        print(f"[WARN] Plot update error: {e}")
                        plot_handles = None
                    last_plot_time = now
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C diterima — menghentikan TX/RX...")
    finally:
        state["running"] = False
        stop_ev.set()
        # Shutdown TX bersih: join, lalu paksa kill kalau masih hidup.
        # Mencegah zombie process yang merebut USB di run berikutnya.
        tx_proc.join(timeout=5)
        if tx_proc.is_alive():
            print("[STOP] TX tidak berhenti normal -> terminate paksa")
            tx_proc.terminate()
            tx_proc.join(timeout=3)
        if tx_proc.is_alive():
            print("[STOP] TX masih hidup -> kill")
            tx_proc.kill()
            tx_proc.join(timeout=2)
        time.sleep(0.5)   # beri waktu RX thread lepas stream
        if csv_file is not None:
            csv_file.close()
        if plot_handles is not None:
            try:
                # Final plot update + leave window open until user closes
                update_live_plot(plot_handles, state, 0)
                import matplotlib.pyplot as plt
                plt.ioff()
                print("\n[INFO] Plot window tetap terbuka. Tutup window untuk exit.")
                plt.show(block=True)
            except Exception:
                pass

    # ── Aggregate metrics ─────────────────────────────────────────
    if not state["bers"]:
        return {
            "fs_mhz": fs/1e6, "tx_gain": tx_gain, "rx_gain": rx_gain,
            "status": "NO_FRAMES", "n_decoded": 0, "n_target": n_frames,
            "valid_rate": 0.0, "mean_ber": float('nan'), "mean_evm_db": float('nan'),
            "ovf_total": state["ovf"], "ovf_steady": 0, "ovf_rate_steady": 0.0,
            "probe_mean_amp": probe_stats["mean_amp"],
            "probe_peak": probe_stats["peak"],
        }

    n_decoded = state["n"]
    bers_arr = np.array(state["bers"])
    evms_arr = np.array(state["evms"])
    valid_mask = bers_arr < 0.05
    valid_rate = float(np.mean(valid_mask))

    ovf_total = state["ovf"]
    ovf_at_warmup = state["ovf_at_warmup_end"] if state["ovf_at_warmup_end"] is not None else ovf_total
    ovf_steady = max(0, ovf_total - ovf_at_warmup)
    n_steady = max(1, n_decoded - warmup_frames)
    ovf_rate_steady = ovf_steady / n_steady

    metrics = {
        "fs_mhz": fs / 1e6,
        "tx_gain": tx_gain, "rx_gain": rx_gain,
        "status": "OK",
        "n_decoded": n_decoded, "n_target": n_frames,
        "valid_rate": valid_rate,
        "mean_ber": float(np.mean(bers_arr)),
        "median_ber": float(np.median(bers_arr)),
        "mean_evm_db": float(np.mean(evms_arr)),
        "mean_amp": float(np.mean(state["amps"])),
        "mean_cfo_hz": float(np.mean(np.abs(state["cfos"]))),
        "ovf_total": ovf_total,
        "ovf_warmup": ovf_at_warmup,
        "ovf_steady": ovf_steady,
        "ovf_rate_steady": ovf_rate_steady,
        "probe_mean_amp": probe_stats["mean_amp"],
        "probe_peak": probe_stats["peak"],
        "probe_spur_dr": probe_stats["spur_dr"],
    }

    # ── Phase 1B summary ──────────────────────────────────────────
    if phase1b and state["p1b_records"]:
        recs = state["p1b_records"]
        n_p1b = len(recs)
        crc_ok = state["p1b_crc_ok_count"]
        crc_fail = state["p1b_crc_fail_count"]
        prr = crc_ok / max(n_p1b, 1)
        comm_evms = np.array([r["comm_evm_db"] for r in recs])
        sense_evms = np.array([r["sense_evm_db"] for r in recs])
        sense_bers = np.array([r["sense_ber"] for r in recs])
        n_echoes_arr = np.array([r["n_echoes"] for r in recs])
        # Histogram top echo range
        all_echoes = [e[0] for r in recs for e in r["echoes"]]
        metrics.update({
            "phase1b": True,
            "p1b_n_packets": n_p1b,
            "p1b_crc_ok": crc_ok,
            "p1b_crc_fail": crc_fail,
            "p1b_prr": prr,
            "p1b_mean_comm_evm_db": float(np.mean(comm_evms)),
            "p1b_mean_sense_evm_db": float(np.mean(sense_evms)),
            "p1b_mean_sense_ber": float(np.mean(sense_bers)),
            "p1b_mean_n_echoes": float(np.mean(n_echoes_arr)),
            "p1b_total_echoes": len(all_echoes),
        })

        # Recovered text histogram (which texts decoded)
        text_counts = {}
        for r in recs:
            if r["crc_ok"]:
                text_counts[r["text"]] = text_counts.get(r["text"], 0) + 1
        metrics["p1b_text_counts"] = text_counts

    if verbose:
        print(f"\n{'═' * 60}")
        if phase1b and state["p1b_records"]:
            # Phase 1B: tampilkan metrics yang relevan (PRR, comm/sense EVM)
            print(f"  FS = {fs/1e6:.2f} MS/s  | Decoded: {n_decoded}/{n_frames}  "
                  f"| PRR: {metrics['p1b_prr']*100:.2f}%")
            print(f"  ─ Phase 1B (V2I '{text}') ────────────────────────")
            print(f"  Packets        : {metrics['p1b_n_packets']}")
            print(f"  PRR            : {metrics['p1b_prr']*100:.2f}%  "
                  f"(OK={metrics['p1b_crc_ok']}, FAIL={metrics['p1b_crc_fail']})")
            print(f"  Mean comm EVM  : {metrics['p1b_mean_comm_evm_db']:.2f} dB (QPSK)")
            print(f"  Mean sense EVM : {metrics['p1b_mean_sense_evm_db']:.2f} dB (BPSK)")
            print(f"  Mean sense BER : {metrics['p1b_mean_sense_ber']:.4f}")
            print(f"  ─ Phase 1C (range) ──────────────────────────────")
            # Pisahkan echo fase deteksi vs kalibrasi untuk interpretasi benar
            det_recs = [r for r in recs if r.get("bg_phase") == "detect"]
            det_echoes = [e[0] for r in det_recs for e in r["echoes"]]
            print(f"  Mean #echoes/frame   : {metrics['p1b_mean_n_echoes']:.2f}")
            print(f"  Total echo detections: {metrics['p1b_total_echoes']}  "
                  f"(fase deteksi saja: {len(det_echoes)})")
            print(f"  Frame fase deteksi   : {len(det_recs)}")
            if metrics["p1b_text_counts"]:
                tc = metrics["p1b_text_counts"]
                top = sorted(tc.items(), key=lambda x: -x[1])[:5]
                print(f"  Recovered texts      : {top}")
            # ─ JCAS forward-scatter (KLAIM DETEKSI) ─
            jr = [r for r in state.get("jcas_records", [])
                  if r["frame_idx"] > warmup_frames]   # buang warmup (skor sigma~0)
            if jr:
                ndet = sum(1 for r in jr if r.get("jcas_object_detected"))
                maxsc = max((r.get("jcas_score", 0.0) for r in jr), default=0.0)
                print(f"  ─ JCAS forward-scatter (deteksi objek) ──────────")
                print(f"  Object detections    : {ndet} ({100*ndet/max(len(jr),1):.1f}% frame, post-warmup)")
                print(f"  Max JCAS score       : {maxsc:.2f}  (post-warmup)")
            print(f"  ─ Hardware ──────────────────────────────────────")
            print(f"  OVF total      : {ovf_total}  (warmup: {ovf_at_warmup}, "
                  f"steady: {ovf_steady} → {ovf_rate_steady*100:.1f}%/frame)")
            print(f"  NOTE: legacy BER/EVM (Phase 0 path) tidak applicable di Phase 1B mode.")
        else:
            # Phase 0/1A: legacy reporting
            valid_mask_legacy = bers_arr < 0.05
            print(f"  FS = {fs/1e6:.2f} MS/s  | Decoded: {n_decoded}/{n_frames}  "
                  f"| Valid: {float(np.mean(valid_mask_legacy))*100:.1f}%")
            print(f"  Mean BER       : {metrics['mean_ber']:.4f}")
            print(f"  Mean EVM       : {metrics['mean_evm_db']:.2f} dB")
            print(f"  OVF total      : {ovf_total}  (warmup: {ovf_at_warmup}, "
                  f"steady: {ovf_steady} → {ovf_rate_steady*100:.1f}%/frame)")
        print(f"{'═' * 60}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════
# FS SWEEP — Phase 1A main feature
# ═══════════════════════════════════════════════════════════════════
def evaluate_fs(metrics):
    """Pass criteria sesuai CONTEXT_TRANSFER_v2:
       - OVF rate steady < 5%
       - Valid frame rate >= 90%
       - Mean BER < 0.05
    """
    if metrics is None or metrics.get("status") != "OK":
        return False, "no_data"
    reasons = []
    if metrics["ovf_rate_steady"] >= 0.05:
        reasons.append(f"OVF_high({metrics['ovf_rate_steady']*100:.1f}%)")
    if metrics["valid_rate"] < 0.90:
        reasons.append(f"valid_low({metrics['valid_rate']*100:.1f}%)")
    if metrics["mean_ber"] >= 0.05:
        reasons.append(f"BER_high({metrics['mean_ber']:.3f})")
    if reasons:
        return False, ",".join(reasons)
    return True, "pass"


def run_fs_sweep(fs_candidates, frames_per_fs, tx_gain, rx_gain,
                 probe_wait, output_csv, warmup_frames):
    """Loop over FS candidates, write CSV, identify fs_winner."""
    print(f"\n{'#' * 60}")
    print(f"  FS SWEEP — Phase 1A")
    print(f"  Candidates : {[f/1e6 for f in fs_candidates]} MS/s")
    print(f"  Frames/FS  : {frames_per_fs}  (warmup: {warmup_frames})")
    print(f"  Gain       : TX={tx_gain} dB  RX={rx_gain} dB")
    print(f"  Output CSV : {output_csv}")
    print(f"{'#' * 60}\n")

    results = []
    for i, fs in enumerate(fs_candidates):
        print(f"\n>>> [{i+1}/{len(fs_candidates)}] Testing FS = {fs/1e6:.2f} MS/s ...")
        metrics = run_hardware(
            fs=fs, n_frames=frames_per_fs,
            tx_gain=tx_gain, rx_gain=rx_gain,
            probe_wait=probe_wait, warmup_frames=warmup_frames,
            verbose=False,  # less spam during sweep
        )
        if metrics is None:
            metrics = {"fs_mhz": fs/1e6, "tx_gain": tx_gain, "rx_gain": rx_gain,
                       "status": "INIT_FAIL", "n_decoded": 0, "n_target": frames_per_fs,
                       "valid_rate": 0.0, "mean_ber": float('nan'),
                       "mean_evm_db": float('nan'), "ovf_total": 0,
                       "ovf_steady": 0, "ovf_rate_steady": 0.0,
                       "probe_mean_amp": 0.0, "probe_peak": 0.0}
        passed, reason = evaluate_fs(metrics)
        metrics["pass"] = passed
        metrics["reason"] = reason
        results.append(metrics)

        # Quick summary
        if metrics["status"] == "OK":
            print(f"    Decoded: {metrics['n_decoded']}/{metrics['n_target']} "
                  f"| Valid: {metrics['valid_rate']*100:.1f}% "
                  f"| BER: {metrics['mean_ber']:.4f} "
                  f"| EVM: {metrics['mean_evm_db']:.2f} dB "
                  f"| OVF steady: {metrics['ovf_rate_steady']*100:.1f}%/frame "
                  f"| {'✓ PASS' if passed else '✗ FAIL ('+reason+')'}")
        else:
            print(f"    Status: {metrics['status']}")

        time.sleep(2)  # Cooldown between FS

    # ── Write CSV ─────────────────────────────────────────────────
    fieldnames = ["fs_mhz", "tx_gain", "rx_gain", "status", "pass", "reason",
                  "n_decoded", "n_target", "valid_rate", "mean_ber", "median_ber",
                  "mean_evm_db", "mean_amp", "mean_cfo_hz",
                  "ovf_total", "ovf_warmup", "ovf_steady", "ovf_rate_steady",
                  "probe_mean_amp", "probe_peak", "probe_spur_dr"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  FS SWEEP SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'FS(MHz)':>8} | {'Valid%':>6} | {'BER':>8} | {'EVM(dB)':>7} | "
          f"{'OVF%':>6} | Verdict")
    print(f"  {'-'*60}")
    for r in results:
        if r["status"] == "OK":
            verdict = "✓ PASS" if r["pass"] else f"✗ {r['reason']}"
            print(f"  {r['fs_mhz']:>8.2f} | "
                  f"{r['valid_rate']*100:>6.1f} | "
                  f"{r['mean_ber']:>8.4f} | "
                  f"{r['mean_evm_db']:>7.2f} | "
                  f"{r['ovf_rate_steady']*100:>6.1f} | {verdict}")
        else:
            print(f"  {r['fs_mhz']:>8.2f} | {r['status']:^45}")

    # Identify fs_winner: PASS dengan FS tertinggi
    passed = [r for r in results if r.get("pass")]
    if passed:
        winner = max(passed, key=lambda r: r["fs_mhz"])
        print(f"\n  🏆 FS WINNER: {winner['fs_mhz']:.2f} MS/s "
              f"(EVM {winner['mean_evm_db']:.2f} dB, BER {winner['mean_ber']:.4f})")
        print(f"     → Gunakan ini untuk Phase 1B-D.")
    else:
        print(f"\n  ⚠ NO FS PASSED. Check gain/antenna, atau turunkan kriteria.")

    print(f"\n  CSV: {output_csv}")
    print(f"{'═' * 60}\n")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ofdm_isac_bistatic — Phase 1A FS sweep + single FS run")

    ap.add_argument("--simulate", action="store_true",
                    help="AWGN self-test (no hardware)")
    ap.add_argument("--probe-only", action="store_true",
                    help="HW: capture 2s, print stats, exit")
    ap.add_argument("--fs-sweep", action="store_true",
                    help="Phase 1A: sweep FS candidates")

    ap.add_argument("--fs", type=float, default=20e6,
                    help="Sample rate Hz (single run mode, default 20e6)")
    ap.add_argument("--fs-candidates", type=str,
                    default="5e6,10e6,15e6,20e6,30e6",
                    help="Comma-separated FS list for sweep")
    ap.add_argument("--frames", type=int, default=100,
                    help="Target frames (single run)")
    ap.add_argument("--frames-per-fs", type=int, default=200,
                    help="Frames per FS in sweep mode")
    ap.add_argument("--warmup-frames", type=int, default=50,
                    help="Frames excluded from steady-state OVF rate")

    ap.add_argument("--tx-gain", type=float, default=80.0)
    ap.add_argument("--rx-gain", type=float, default=70.0)
    ap.add_argument("--probe-wait", type=float, default=10.0)
    ap.add_argument("--frame-delay", type=float, default=0.0)
    ap.add_argument("--no-dc-fix", action="store_true")
    ap.add_argument("--output-csv", type=str, default="fs_sweep_results.csv")

    # ── Phase 1B/1C/1D flags ──────────────────────────────────────
    ap.add_argument("--phase1b", action="store_true",
                    help="Enable Phase 1B (V2I 'STEI' comm) + Phase 1C (range est)")
    ap.add_argument("--text", type=str, default="STEI",
                    help="V2I packet text (4 chars max, padded/truncated to 4)")
    ap.add_argument("--log-csv", type=str, default=None,
                    help="Per-frame Phase 1B log CSV path "
                         "(e.g. isac_log_$(date +%%s).csv)")
    ap.add_argument("--plot", action="store_true",
                    help="Live dashboard ISAC 6-panel (range/const/JCAS-score/"
                         "EVM/text-log/amplitude-blip). Requires --phase1b.")
    ap.add_argument("--echo-threshold-db", type=float, default=14.0,
                    help="Echo detection SNR threshold di atas noise floor FISIK "
                         "(default 14 dB). v1.4: dinaikkan dari 8 -> 14 untuk "
                         "tekan echo palsu (8 dB lolos ~10%% noise; 14 dB ~0). "
                         "Turunkan ke 10-12 kalau target lemah perlu terdeteksi "
                         "(risiko: echo palsu naik).")

    # ── Phase 1C v1.1: geometric limit + bg subtraction ──────────
    ap.add_argument("--max-range-m", type=float, default=12.0,
                    help="Hard geometric limit delta path bistatic (default 12 m). "
                         "Set sesuai ukuran lab: lab 4×4 m → 12, lab 10×10 m → 25.")
    ap.add_argument("--direct-skip-bins", type=int, default=3,
                    help="Bin skip setelah direct path untuk mask Hanning sidelobe "
                         "(default 3 @ oversample 4×). Naikkan kalau sidelobe leak.")
    ap.add_argument("--max-echoes", type=int, default=3,
                    help="Top-K echo terkuat per frame yang dilaporkan (default 3). "
                         "Turunkan = lebih konservatif.")

    # ── Phase 1C v1.3: frozen clutter map calibration ────────────
    ap.add_argument("--calib-frames", type=int, default=300,
                    help="Jumlah frame fase kalibrasi clutter map (default 300). "
                         "RUANGAN HARUS DIAM selama fase ini. "
                         "Pastikan --frames > --calib-frames + margin deteksi.")
    ap.add_argument("--save-clutter", type=str, default=None,
                    help="Simpan clutter map hasil kalibrasi ke file .npy.")
    ap.add_argument("--load-clutter", type=str, default=None,
                    help="Muat clutter map .npy tersimpan -> fase kalibrasi "
                         "dilewati (untuk demo cepat / kanal sudah dikenal).")

    # ── JCAS forward-scatter detector (KLAIM DETEKSI utama) ──────
    ap.add_argument("--no-jcas", action="store_true",
                    help="Matikan detektor forward-scatter JCAS.")
    ap.add_argument("--sense-ma-len", type=int, default=30,
                    help="Window moving-median utk high-pass amplitudo/fase LoS.")
    ap.add_argument("--sense-cfar-len", type=int, default=120,
                    help="Window referensi rolling utk threshold CFAR sederhana.")
    ap.add_argument("--sense-threshold-k", type=float, default=4.5,
                    help="CFAR K multiplier; lebih rendah = lebih sensitif.")
    ap.add_argument("--sense-min-score", type=float, default=1.5,
                    help="Lantai threshold deteksi JCAS minimum.")

    args = ap.parse_args()

    if args.simulate:
        init_params(args.fs, phase1b=args.phase1b, text=args.text)
        run_simulation()
        sys.exit(0)

    mp.set_start_method('spawn')

    if args.fs_sweep:
        if args.phase1b:
            print("⚠ --phase1b not supported in --fs-sweep mode. "
                  "Run single FS with --phase1b instead.")
            sys.exit(1)
        fs_list = [float(x) for x in args.fs_candidates.split(",")]
        run_fs_sweep(
            fs_candidates=fs_list,
            frames_per_fs=args.frames_per_fs,
            tx_gain=args.tx_gain, rx_gain=args.rx_gain,
            probe_wait=args.probe_wait,
            output_csv=args.output_csv,
            warmup_frames=args.warmup_frames,
        )
    else:
        # Default log_csv name kalau phase1b aktif tapi log-csv tidak diisi
        log_csv = args.log_csv
        if args.phase1b and log_csv is None:
            log_csv = f"isac_log_{int(time.time())}.csv"
            print(f"[INFO] Phase 1B log CSV: {log_csv}")
        run_hardware(
            fs=args.fs,
            n_frames=args.frames,
            tx_gain=args.tx_gain, rx_gain=args.rx_gain,
            probe_wait=args.probe_wait,
            frame_delay=args.frame_delay,
            dc_offset_auto=not args.no_dc_fix,
            probe_only=args.probe_only,
            warmup_frames=args.warmup_frames,
            phase1b=args.phase1b,
            text=args.text,
            log_csv=log_csv,
            plot=args.plot,
            echo_threshold_db=args.echo_threshold_db,
            max_range_m=args.max_range_m,
            direct_skip_bins=args.direct_skip_bins,
            max_echoes=args.max_echoes,
            calib_frames=args.calib_frames,
            save_clutter=args.save_clutter,
            load_clutter=args.load_clutter,
            jcas=not args.no_jcas,
            sense_ma_len=args.sense_ma_len,
            sense_cfar_len=args.sense_cfar_len,
            sense_threshold_k=args.sense_threshold_k,
            sense_min_score=args.sense_min_score,
        )
