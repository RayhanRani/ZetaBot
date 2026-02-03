"""Generate report with charts and findings."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    events_df: pd.DataFrame,
    top_movers: pd.DataFrame,
    most_volatile: pd.DataFrame,
    highest_volume: pd.DataFrame,
    category_agg: pd.DataFrame,
    clustering: pd.DataFrame,
    pattern_ranked: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate report.md and save charts."""
    output_dir = Path(output_dir) if output_dir else Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Charts
    _chart_category_dist(events_df, charts_dir)
    _chart_top_movers(top_movers, charts_dir)
    _chart_volume_dist(highest_volume, charts_dir)
    _chart_clusters(clustering, charts_dir)

    # Report
    report_path = output_dir / "report.md"
    findings = _build_findings(
        events_df, top_movers, most_volatile, highest_volume,
        category_agg, clustering, pattern_ranked,
    )
    md = _build_markdown(findings, charts_dir)
    report_path.write_text(md, encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return report_path


def _chart_category_dist(df: pd.DataFrame, out_dir: Path) -> None:
    if "category" not in df.columns or df.empty:
        return
    counts = df["category"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Markets by Category")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "category_dist.png", dpi=100)
    plt.close()


def _chart_top_movers(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "last_price" not in df.columns:
        return
    df = df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df["last_price"].values)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["title"].str[:40] if "title" in df.columns else df["ticker"], fontsize=8)
    ax.set_title("Top Movers (by distance from 50¢)")
    ax.set_xlabel("Last Price (¢)")
    plt.tight_layout()
    plt.savefig(out_dir / "top_movers.png", dpi=100)
    plt.close()


def _chart_volume_dist(df: pd.DataFrame, out_dir: Path) -> None:
    vol_col = "volume_24h" if "volume_24h" in df.columns else "volume"
    if df.empty or vol_col not in df.columns:
        return
    df = df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df[vol_col].values)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["title"].str[:40] if "title" in df.columns else df["ticker"], fontsize=8)
    ax.set_title("Highest Volume Markets")
    ax.set_xlabel(vol_col)
    plt.tight_layout()
    plt.savefig(out_dir / "highest_volume.png", dpi=100)
    plt.close()


def _chart_clusters(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "cluster" not in df.columns:
        return
    counts = df["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Market Clusters (by behavior)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "clusters.png", dpi=100)
    plt.close()


def _build_findings(
    events_df: pd.DataFrame,
    top_movers: pd.DataFrame,
    most_volatile: pd.DataFrame,
    highest_volume: pd.DataFrame,
    category_agg: pd.DataFrame,
    clustering: pd.DataFrame,
    pattern_ranked: pd.DataFrame,
) -> dict:
    findings = []
    if not events_df.empty:
        findings.append(f"- Total markets analyzed: {len(events_df)}")
    if not category_agg.empty:
        top_cat = category_agg.nlargest(1, "count")
        if not top_cat.empty:
            findings.append(f"- Largest category: {top_cat['category'].iloc[0]} ({int(top_cat['count'].iloc[0])} markets)")
    if not top_movers.empty:
        findings.append(f"- Top mover (by distance from 50¢): {top_movers['title'].iloc[0][:60]}...")
    if not highest_volume.empty:
        vol_col = "volume_24h" if "volume_24h" in highest_volume.columns else "volume"
        findings.append(f"- Highest volume market: {highest_volume['title'].iloc[0][:60]}... ({highest_volume[vol_col].iloc[0]:.0f})")
    if not pattern_ranked.empty:
        findings.append(f"- Markets with strongest pattern signals: {len(pattern_ranked)}")
    if not clustering.empty:
        findings.append(f"- Markets clustered into {clustering['cluster'].nunique()} behavior groups")
    return {"findings": findings}


def _build_markdown(findings: dict, charts_dir: Path) -> str:
    lines = [
        "# Kalshi Trends Report",
        "",
        "## Findings Summary",
        "",
    ]
    for f in findings.get("findings", []):
        lines.append(f)
    lines.extend([
        "",
        "## Charts",
        "",
        "![Category Distribution](charts/category_dist.png)",
        "",
        "![Top Movers](charts/top_movers.png)",
        "",
        "![Highest Volume](charts/highest_volume.png)",
        "",
        "![Clusters](charts/clusters.png)",
        "",
    ])
    return "\n".join(lines)
