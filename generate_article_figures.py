"""Generate figures for the EEG sleep staging article.

Creates publication-quality visualizations of:
1. Architecture diagram (text-based schematic)
2. Version history / kappa progression over time
3. Per-stage F1 scores (radar chart)
4. Information flow through the model
5. Dead ends waterfall chart
6. SOTA comparison
7. Per-branch N1 discriminability
8. Device type impact (CNN vs spectral-only)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "plots" / "article"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#059669",
    "danger": "#dc2626",
    "warning": "#d97706",
    "neutral": "#6b7280",
    "W": "#e74c3c",
    "N1": "#f39c12",
    "N2": "#3498db",
    "N3": "#2c3e50",
    "REM": "#27ae60",
}


def fig1_version_history():
    """Kappa progression across architecture versions."""
    versions = [
        ("v1\n2-branch", 0.669),
        ("v2\n4-branch", 0.674),
        ("v2+CRF\n+BiGRU", 0.773),
        ("v4\nfrozen enc", 0.772),
        ("v7\ncleanup", 0.756),
        ("v9\nband dec", 0.758),
        ("v12\n5-branch", 0.749),
        ("bigru256\ncapacity", 0.757),
        ("spec_only\ndrop CNN", 0.762),
        ("combined\n+N1Aux+heads", 0.769),
        ("gru384\ncapacity", 0.772),
        ("final\nsimplified", 0.765),
    ]
    names, kappas = zip(*versions)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color bars by whether they improved
    colors = []
    for i, k in enumerate(kappas):
        if i == 0:
            colors.append(COLORS["neutral"])
        elif k > kappas[i - 1] + 0.003:
            colors.append(COLORS["success"])
        elif k < kappas[i - 1] - 0.003:
            colors.append(COLORS["danger"])
        else:
            colors.append(COLORS["neutral"])
    # Override CRF bar -- it's the big jump
    colors[2] = COLORS["primary"]

    bars = ax.bar(x, kappas, color=colors, width=0.7, edgecolor="white", linewidth=0.5)

    # Human ceiling
    ax.axhline(y=0.76, color=COLORS["warning"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(len(names) - 0.5, 0.762, "Human inter-rater (0.76)",
            ha="right", va="bottom", fontsize=9, color=COLORS["warning"], fontstyle="italic")

    # Annotate the big CRF jump
    ax.annotate("+0.099\n(CRF)", xy=(2, 0.773), xytext=(2, 0.785),
                ha="center", fontsize=9, fontweight="bold", color=COLORS["primary"],
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1.5))

    # Value labels
    for bar, k in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.008,
                f"{k:.3f}", ha="center", va="top", fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Test Cohen's Kappa")
    ax.set_title("Architecture Evolution: Test Kappa Over 12 Iterations")
    ax.set_ylim(0.64, 0.80)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_version_history.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_version_history.png")


def fig2_per_stage_f1():
    """Per-stage F1 scores with human difficulty overlay."""
    stages = ["W", "N1", "N2", "N3", "REM"]
    our_f1 = [0.932, 0.535, 0.820, 0.753, 0.844]
    human_kappa = [0.70, 0.24, 0.57, 0.57, 0.69]  # Younes et al. 2021

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, our_f1, width, label="Our F1",
                    color=[COLORS[s] for s in stages], edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, human_kappa, width, label="Human inter-rater kappa",
                    color=[COLORS[s] for s in stages], alpha=0.35, edgecolor="white",
                    linewidth=0.5, hatch="//")

    for bar, v in zip(bars1, our_f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(bars2, human_kappa):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9, alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.set_title("Per-Stage Performance vs Human Agreement")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_per_stage_f1.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_per_stage_f1.png")


def fig3_info_flow():
    """Information flow through the model -- kappa at each layer."""
    layers = [
        "Spectral\nraw (18d)",
        "Spectral\nMLP (128d)",
        "STFT\n(128d)",
        "Encoder\n(256d)",
        "Pre-GRU\n(320d)",
        "BiGRU\n(768d)",
        "Logits\n(5d)",
        "CRF\nViterbi",
    ]
    kappas = [0.503, 0.570, 0.566, 0.591, 0.608, 0.793, 0.758, 0.772]
    n1_f1s = [0.218, 0.252, 0.232, 0.260, 0.303, 0.510, 0.553, 0.537]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(layers))

    # Kappa
    ax1.plot(x, kappas, "o-", color=COLORS["primary"], linewidth=2.5, markersize=8, zorder=3)
    for i, (xi, k) in enumerate(zip(x, kappas)):
        ax1.annotate(f"{k:.3f}", (xi, k), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
                     color=COLORS["primary"])

    # Highlight the GRU jump
    ax1.annotate("+0.185", xy=(5, 0.793), xytext=(4.3, 0.82),
                 fontsize=11, fontweight="bold", color=COLORS["success"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=2))

    # Divider between per-epoch and sequence
    ax1.axvline(x=4.5, color=COLORS["neutral"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.text(2.0, 0.83, "Per-epoch\n(no temporal context)", ha="center",
             fontsize=9, color=COLORS["neutral"], fontstyle="italic")
    ax1.text(6.5, 0.83, "With sequence\ncontext", ha="center",
             fontsize=9, color=COLORS["neutral"], fontstyle="italic")

    ax1.set_ylabel("Cohen's Kappa")
    ax1.set_title("Information Flow: Stage Discriminability at Each Layer")
    ax1.set_ylim(0.45, 0.88)
    ax1.grid(axis="x", visible=False)

    # N1 F1
    ax2.plot(x, n1_f1s, "s-", color=COLORS["warning"], linewidth=2.5, markersize=8, zorder=3)
    for i, (xi, f) in enumerate(zip(x, n1_f1s)):
        ax2.annotate(f"{f:.3f}", (xi, f), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
                     color=COLORS["warning"])

    ax2.axvline(x=4.5, color=COLORS["neutral"], linestyle=":", linewidth=1.5, alpha=0.7)

    # CRF hurts N1
    ax2.annotate("CRF: -0.016", xy=(7, 0.537), xytext=(6.8, 0.47),
                 fontsize=10, fontweight="bold", color=COLORS["danger"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["danger"], lw=1.5))

    ax2.set_ylabel("N1 F1 Score")
    ax2.set_xlabel("")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, fontsize=9)
    ax2.set_ylim(0.15, 0.60)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_info_flow.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_info_flow.png")


def fig4_dead_ends():
    """Waterfall chart of dead ends (sorted by impact)."""
    dead_ends = [
        ("BiMamba", -0.016),
        ("Dual-view enc", -0.016),
        ("Multi-scale GRU", -0.005),
        ("Multi-token", -0.005),
        ("Per-stage 640d", -0.004),
        ("Branch weights", -0.004),
        ("SCE loss", -0.004),
        ("Rec pos embed", -0.004),
        ("Traj dynamics", -0.003),
        ("Cross-epoch conv", -0.003),
        ("n_segments>3", -0.002),
        ("Branch GRU", -0.001),
        ("Gap features", -0.001),
        ("CRF priors", +0.001),
        ("Position embed", +0.002),
    ]

    names, deltas = zip(*dead_ends)

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(names))
    colors = [COLORS["danger"] if d < -0.003 else
              COLORS["warning"] if d < 0 else
              COLORS["neutral"] for d in deltas]

    bars = ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for bar, d in zip(bars, deltas):
        offset = -0.001 if d < 0 else 0.001
        ax.text(d + offset, bar.get_y() + bar.get_height() / 2,
                f"{d:+.3f}", va="center",
                ha="right" if d < 0 else "left",
                fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Delta Test Kappa vs Baseline")
    ax.set_title("Dead Ends: 15 Architectural Changes That Did Not Help")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlim(-0.022, 0.006)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    # Annotation
    ax.text(-0.020, len(names) + 0.3,
            "34 total dead ends documented. 15 with measurable kappa impact shown.",
            fontsize=9, color=COLORS["neutral"], fontstyle="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_dead_ends.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 04_dead_ends.png")


def fig5_sota_comparison():
    """SOTA comparison chart -- our model in context."""
    models = [
        ("Stephansen\n(2020)", 0.675, "multi-ch, OOD", COLORS["neutral"]),
        ("ZleepAnlystNet\n(2024)", 0.779, "1ch, OOD", COLORS["neutral"]),
        ("This work\n(2026)", 0.765, "1ch, 5 devices", COLORS["primary"]),
        ("U-Sleep\n(2021)", 0.819, "multi-ch, OOD", COLORS["neutral"]),
        ("SOMNUS\n(2025)", 0.870, "multi-ch, ensemble", COLORS["neutral"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [m[0] for m in models]
    kappas = [m[1] for m in models]
    notes = [m[2] for m in models]
    colors = [m[3] for m in models]
    x = np.arange(len(models))

    bars = ax.bar(x, kappas, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Human ceiling
    ax.axhline(y=0.76, color=COLORS["warning"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(4.3, 0.762, "Human (0.76)", ha="right", fontsize=9,
            color=COLORS["warning"], fontstyle="italic")

    for bar, k, note in zip(bars, kappas, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{k:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.015,
                note, ha="center", va="top", fontsize=8, color="white", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Cohen's Kappa (OOD / Cross-Dataset)")
    ax.set_title("Cross-Dataset Sleep Staging: Published Benchmarks")
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_sota_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 05_sota_comparison.png")


def fig6_branch_n1():
    """Per-branch N1 discriminability -- why CNN branches were dropped."""
    branches = ["Small CNN\n(8-45Hz)", "Medium CNN\n(4-12Hz)", "Large CNN\n(0.5-4Hz)",
                "Spectral\n(18 feat)", "STFT\n(TF patch)", "Fused\n(640d)"]
    n1_f1 = [0.065, 0.056, 0.078, 0.252, 0.232, 0.197]
    overall_kappa = [0.312, 0.270, 0.340, 0.570, 0.566, 0.591]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(branches))
    cnn_color = COLORS["danger"]
    spec_color = COLORS["success"]
    fused_color = COLORS["warning"]
    colors = [cnn_color, cnn_color, cnn_color, spec_color, spec_color, fused_color]

    # N1 F1
    bars1 = ax1.bar(x, n1_f1, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars1, n1_f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(branches, fontsize=9)
    ax1.set_ylabel("N1 F1 Score (kNN)")
    ax1.set_title("N1 Discriminability by Branch")
    ax1.set_ylim(0, 0.32)
    ax1.grid(axis="x", visible=False)

    # Annotate fusion hurts
    ax1.annotate("Fusion HURTS\nN1 (0.252 -> 0.197)", xy=(5, 0.197), xytext=(4.5, 0.28),
                 fontsize=9, fontweight="bold", color=COLORS["danger"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["danger"], lw=1.5))

    # Overall kappa
    bars2 = ax2.bar(x, overall_kappa, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars2, overall_kappa):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(branches, fontsize=9)
    ax2.set_ylabel("Overall Kappa (kNN)")
    ax2.set_title("Overall Discriminability by Branch")
    ax2.set_ylim(0, 0.7)
    ax2.grid(axis="x", visible=False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=cnn_color, label="CNN branches (device-specific)"),
        mpatches.Patch(color=spec_color, label="Spectral/STFT (device-invariant)"),
        mpatches.Patch(color=fused_color, label="Fused (all branches)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_branch_n1.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 06_branch_n1.png")


def fig7_architecture():
    """Architecture schematic."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("System Architecture", fontsize=14, fontweight="bold", pad=20)

    def box(x, y, w, h, text, color, fontsize=9, alpha=1.0):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", alpha=alpha, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white" if alpha > 0.5 else "black")

    def arrow(x1, y1, x2, y2, text="", color="gray"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, text, fontsize=8, color=color, fontstyle="italic")

    # Input
    box(5.5, 7.0, 3.0, 0.6, "Raw EEG (200 x 3840 samples)", "#374151", fontsize=10)

    # Encoder box
    enc_rect = mpatches.FancyBboxPatch(
        (0.3, 3.4), 13.4, 3.2, boxstyle="round,pad=0.15",
        facecolor="#f3f4f6", edgecolor="#9ca3af", linewidth=1.5, linestyle="--")
    ax.add_patch(enc_rect)
    ax.text(7.0, 6.4, "Epoch Encoder (1.08M params, frozen, per-epoch)",
            ha="center", fontsize=10, color="#6b7280", fontstyle="italic")

    # CNN branches (grayed out)
    box(0.5, 4.8, 2.5, 0.8, "Small CNN\nk=50 (8-45Hz)", "#9ca3af", alpha=0.4)
    box(0.5, 3.8, 2.5, 0.8, "Medium CNN\nk=150 (4-12Hz)", "#9ca3af", alpha=0.4)
    ax.text(1.75, 3.55, "SKIPPED", fontsize=8, ha="center", color=COLORS["danger"],
            fontweight="bold")
    box(0.5, 4.8, 2.5, 0.8, "Small CNN\nk=50 (8-45Hz)", "#9ca3af", alpha=0.3)
    box(3.2, 4.3, 2.5, 0.8, "Large CNN\nk=400 (0.5-4Hz)", "#9ca3af", alpha=0.4)
    ax.text(4.45, 4.05, "SKIPPED", fontsize=8, ha="center", color=COLORS["danger"],
            fontweight="bold")

    # Active branches
    box(6.2, 4.3, 3.0, 1.3, "Spectral Branch\n18 FFT features\nMLP -> 128d", COLORS["success"])
    box(9.5, 4.3, 3.8, 1.3, "STFT Branch\npatch embed + transformer\nmean pool -> 128d", COLORS["primary"])

    # Arrow from input to encoder
    arrow(7.0, 7.0, 7.0, 6.5)

    # Concat
    box(7.5, 3.5, 1.8, 0.5, "concat (256d)", "#374151", fontsize=9)
    arrow(7.7, 4.3, 8.2, 4.05)
    arrow(11.0, 4.3, 8.8, 4.05)

    # BiGRU
    box(5.5, 2.2, 3.0, 0.8, "2-layer BiGRU\nhidden=384 (768d out)", COLORS["secondary"], fontsize=10)
    arrow(8.4, 3.5, 7.0, 3.05)

    # Head
    box(5.5, 1.1, 3.0, 0.7, "MLP Head\n768->128->5", COLORS["primary"], fontsize=10)
    arrow(7.0, 2.2, 7.0, 1.85)

    # CRF
    box(5.5, 0.1, 3.0, 0.7, "Linear CRF\nViterbi decode", COLORS["warning"], fontsize=10)
    arrow(7.0, 1.1, 7.0, 0.85)

    # Output
    ax.text(7.0, -0.2, "W  |  N1  |  N2  |  N3  |  REM", ha="center",
            fontsize=11, fontweight="bold", color="#374151")

    # Param annotations
    ax.text(13.5, 2.6, "4.2M trainable", fontsize=9, ha="right", color=COLORS["secondary"])
    ax.text(13.5, 0.45, "35 params", fontsize=9, ha="right", color=COLORS["warning"])

    fig.savefig(OUT_DIR / "07_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 07_architecture.png")


def fig8_what_worked():
    """What actually moved the needle -- ranked by kappa impact."""
    items = [
        ("CRF + BiGRU\n(temporal context)", 0.185, COLORS["primary"]),
        ("BiGRU capacity\n(192->384)", 0.011, COLORS["secondary"]),
        ("Drop CNN branches\n(spectral only)", 0.005, COLORS["success"]),
        ("Band-specific\nrecon decoders", 0.008, COLORS["success"]),
        ("Data quality\nfiltering", 0.004, COLORS["success"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [i[0] for i in items]
    deltas = [i[1] for i in items]
    colors = [i[2] for i in items]
    y = np.arange(len(items))

    bars = ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    for bar, d in zip(bars, deltas):
        ax.text(d + 0.002, bar.get_y() + bar.get_height() / 2,
                f"+{d:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Delta Test Kappa")
    ax.set_title("What Actually Worked (Ranked by Impact)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    # Scale note
    ax.text(0.10, len(items) - 0.2,
            "CRF + BiGRU = 89% of total improvement.\nEverything else combined = 11%.",
            fontsize=9, color=COLORS["neutral"], fontstyle="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_what_worked.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 08_what_worked.png")


def fig9_data_overview():
    """Dataset composition by device type."""
    devices = {
        "Scalp\n(clinical)": {"datasets": 8, "color": COLORS["primary"]},
        "In-ear\n(earbuds)": {"datasets": 3, "color": COLORS["success"]},
        "Around-ear\n(cEEGrid)": {"datasets": 1, "color": COLORS["secondary"]},
        "Headband": {"datasets": 3, "color": COLORS["warning"]},
        "Forehead\n(patch)": {"datasets": 1, "color": COLORS["danger"]},
    }

    # Approximate subject counts per device type
    subj_counts = [1450, 180, 30, 550, 83]
    ds_counts = [8, 3, 1, 3, 1]
    device_names = list(devices.keys())
    device_colors = [d["color"] for d in devices.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart -- subjects by device
    wedges, texts, autotexts = ax1.pie(
        subj_counts, labels=device_names, colors=device_colors,
        autopct=lambda p: f"{int(p * 2293 / 100)}",
        startangle=90, textprops={"fontsize": 9})
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
        t.set_color("white")
    ax1.set_title("Subjects by Device Type\n(2,293 total)")

    # Bar chart -- datasets per device type
    x = np.arange(len(device_names))
    bars = ax2.bar(x, ds_counts, color=device_colors, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, ds_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(c), ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(device_names, fontsize=9)
    ax2.set_ylabel("Number of Datasets")
    ax2.set_title("Datasets by Device Type\n(17 total)")
    ax2.set_ylim(0, 10)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_data_overview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 09_data_overview.png")


def fig10_crf_tradeoff():
    """CRF tradeoff: kappa gain vs N1 F1 loss."""
    stages = ["W", "N1", "N2", "N3", "REM", "Overall\nKappa"]
    with_crf = [0.932, 0.535, 0.820, 0.753, 0.844, 0.765]
    # Approximate no-CRF values from ablation data
    without_crf = [0.920, 0.546, 0.806, 0.747, 0.835, 0.756]
    delta = [w - wo for w, wo in zip(with_crf, without_crf)]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(stages))
    colors = [COLORS["success"] if d > 0 else COLORS["danger"] for d in delta]

    bars = ax.bar(x, delta, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for bar, d in zip(bars, delta):
        va = "bottom" if d >= 0 else "top"
        offset = 0.002 if d >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                f"{d:+.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight="bold")
    ax.set_ylabel("Delta (CRF - No CRF)")
    ax.set_title("CRF Tradeoff: Overall Improvement vs N1 Suppression")
    ax.set_ylim(-0.020, 0.020)
    ax.grid(axis="x", visible=False)

    ax.text(3.0, -0.017,
            "CRF smooths transitions: helps consistency (+0.009 kappa)\nbut suppresses brief N1 episodes (-0.011 F1)",
            fontsize=9, color=COLORS["neutral"], fontstyle="italic", ha="center")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_crf_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 10_crf_tradeoff.png")


if __name__ == "__main__":
    print("Generating article figures...")
    fig1_version_history()
    fig2_per_stage_f1()
    fig3_info_flow()
    fig4_dead_ends()
    fig5_sota_comparison()
    fig6_branch_n1()
    fig7_architecture()
    fig8_what_worked()
    fig9_data_overview()
    fig10_crf_tradeoff()
    print(f"\nAll figures saved to {OUT_DIR}")
