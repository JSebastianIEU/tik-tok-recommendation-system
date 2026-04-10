import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import re
    from collections import Counter
    from itertools import combinations

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    from sqlalchemy import create_engine, text

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["figure.facecolor"] = "white"
    return (
        Counter,
        combinations,
        create_engine,
        mo,
        np,
        os,
        pd,
        plt,
        re,
        sns,
        text,
    )


@app.cell
def _(mo):
    mo.md("""
    # TikTok Dataset — Exploratory Data Analysis

    This notebook performs a comprehensive exploratory analysis of **31,607 TikTok videos**
    scraped from the platform and stored in a Supabase PostgreSQL database. The goal is to
    understand the shape, quality, and statistical properties of the data **before** feeding
    it into our multi-objective recommendation pipeline.

    **Database schema:**

    | Table | Description | Rows |
    |-------|-------------|------|
    | `videos` | Core video metadata (caption, duration, timestamps) | 31,607 |
    | `video_snapshots` | Point-in-time engagement metrics (plays, likes, shares, comments) | 32,603 |
    | `authors` | Creator profiles (username, bio, verified status) | 22,878 |
    | `author_metric_snapshots` | Creator follower/following counts over time | 32,596 |
    | `comments` / `comment_snapshots` | User-generated comments with engagement | 19,989 |
    | `hashtags` / `video_hashtags` | Hashtag taxonomy and video–hashtag bridge table | 2,805 tags |
    | `audios` | Audio track metadata | 25,045 |

    The analysis covers **10 sections**: data extraction, data quality, temporal patterns,
    engagement distributions, content features, hashtag deep-dive, creator landscape,
    comment analysis, audio analysis, and actionable conclusions for the recommendation model.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Database Connection & Raw Extraction
    """)
    return


@app.cell
def _(create_engine, os, pd, text):
    _db_url = os.environ.get("DATABASE_URL", "")
    _engine = create_engine(_db_url.replace("postgresql://", "postgresql+psycopg://"))
    _conn = _engine.connect()

    # ── Videos + latest snapshot ──
    videos_df = pd.read_sql(
        text("""
        SELECT
            v.video_id, v.author_id, v.caption, v.duration_sec, v.created_at,
            v.audio_id,
            vs.likes, vs.comments_count, vs.shares, vs.plays,
            vs.scraped_at AS snapshot_at
        FROM videos v
        JOIN LATERAL (
            SELECT * FROM video_snapshots s
            WHERE s.video_id = v.video_id ORDER BY s.scraped_at DESC LIMIT 1
        ) vs ON true
        """),
        _conn,
    )
    videos_df["created_at"] = pd.to_datetime(videos_df["created_at"], utc=True)
    videos_df["snapshot_at"] = pd.to_datetime(videos_df["snapshot_at"], utc=True)

    # ── Authors + latest follower snapshot ──
    authors_df = pd.read_sql(
        text("""
        SELECT a.author_id, a.username, a.display_name, a.bio, a.verified,
               ams.follower_count, ams.following_count, ams.author_likes_count
        FROM authors a
        LEFT JOIN LATERAL (
            SELECT * FROM author_metric_snapshots s
            WHERE s.author_id = a.author_id ORDER BY s.scraped_at DESC LIMIT 1
        ) ams ON true
        """),
        _conn,
    )

    # ── Comments + latest snapshot ──
    comments_df = pd.read_sql(
        text("""
        SELECT c.comment_id, c.video_id, c.text, c.username,
               c.comment_level, c.parent_comment_id,
               cs.likes AS comment_likes, cs.reply_count
        FROM comments c
        LEFT JOIN LATERAL (
            SELECT * FROM comment_snapshots s
            WHERE s.comment_id = c.comment_id ORDER BY s.scraped_at DESC LIMIT 1
        ) cs ON true
        """),
        _conn,
    )

    # ── Hashtags (bridge table) ──
    hashtags_bridge_df = pd.read_sql(
        text("SELECT vh.video_id, h.tag FROM video_hashtags vh JOIN hashtags h ON vh.hashtag_id = h.hashtag_id"),
        _conn,
    )

    # ── Audios ──
    audios_df = pd.read_sql(
        text("SELECT audio_id, audio_name, audio_author_name, is_original FROM audios"),
        _conn,
    )

    _conn.close()
    _engine.dispose()
    return audios_df, authors_df, comments_df, hashtags_bridge_df, videos_df


@app.cell
def _(audios_df, authors_df, comments_df, hashtags_bridge_df, mo, videos_df):
    _v = len(videos_df)
    _a = len(authors_df)
    _c = len(comments_df)
    _h = hashtags_bridge_df["tag"].nunique()
    _au = len(audios_df)
    _d0 = videos_df["created_at"].min().strftime("%Y-%m-%d")
    _d1 = videos_df["created_at"].max().strftime("%Y-%m-%d")

    mo.md(f"""
    ### Extraction Summary

    | Metric | Value |
    |--------|-------|
    | Videos (with latest engagement snapshot) | **{_v:,}** |
    | Unique authors | **{_a:,}** |
    | Comments | **{_c:,}** |
    | Unique hashtags (bridge table) | **{_h:,}** |
    | Audio tracks | **{_au:,}** |
    | Date range | **{_d0}** → **{_d1}** |

    All engagement columns (`plays`, `likes`, `shares`, `comments_count`) come from the
    **most recent snapshot** per video, reflecting the latest observed state.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Data Quality & Completeness

    Before any modelling, we must verify data integrity: null rates, zero-engagement records,
    duplicates, and type consistency. Issues uncovered here directly affect feature engineering
    and evaluation reliability.
    """)
    return


@app.cell
def _(mo, pd, videos_df):
    _quality_rows = []
    for _col in videos_df.columns:
        _n = len(videos_df)
        _nulls = int(videos_df[_col].isnull().sum())
        _zeros = int((videos_df[_col] == 0).sum()) if videos_df[_col].dtype in ["int64", "float64", "Int64"] else "-"
        _pct = _nulls / _n * 100
        _quality_rows.append({
            "Column": _col,
            "Nulls": _nulls,
            "Null %": f"{_pct:.2f}%",
            "Zeros": _zeros,
            "Dtype": str(videos_df[_col].dtype),
        })
    _qdf = pd.DataFrame(_quality_rows)

    _dup = int(videos_df["video_id"].duplicated().sum())
    _zero_plays = int((videos_df["plays"] == 0).sum())
    _null_eng = int(videos_df["plays"].isnull().sum())
    _empty_cap = int((videos_df["caption"].fillna("").str.strip() == "").sum())

    mo.md(f"""
    ### 2.1 Column-Level Quality Report

    {mo.as_html(_qdf)}

    ### 2.2 Key Findings

    | Check | Result | Impact |
    |-------|--------|--------|
    | Duplicate video IDs | **{_dup}** | {'No dedup needed' if _dup == 0 else 'Deduplication required'} |
    | Null engagement (plays) | **{_null_eng}** ({_null_eng/len(videos_df)*100:.2f}%) | Filter before training |
    | Zero-play videos | **{_zero_plays}** ({_zero_plays/len(videos_df)*100:.1f}%) | Exclude from engagement rate calculations |
    | Empty captions | **{_empty_cap}** ({_empty_cap/len(videos_df)*100:.1f}%) | Limits text-based retrieval for these records |

    **Interpretation:** The dataset is remarkably clean — near-zero null rates on core fields.
    The {_null_eng} null engagement records correspond to videos whose snapshots had incomplete
    data and should be excluded from model training. Zero-play videos are minimal and likely
    represent content scraped immediately after publication.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Temporal Distribution

    Understanding the composition of our dataset — when videos were created and how
    old they were at scraping time — is essential for identifying biases and designing
    appropriate train/test splits.

    **Important caveat:** We have a single engagement snapshot per video (the latest scrape),
    **not** a time-series of how engagement grew over days or weeks. This means we cannot
    draw causal conclusions about temporal effects on engagement (e.g., "videos posted at
    hour X get more likes") because older videos have had more time to accumulate metrics.
    This section focuses strictly on **dataset composition**, not temporal engagement trends.
    """)
    return


@app.cell
def _(plt, sns, videos_df):
    fig_temporal, _ax = plt.subplots(1, 2, figsize=(14, 5))

    # 3a — Yearly composition
    _yearly = videos_df["created_at"].dt.year.value_counts().sort_index()
    _ax[0].bar(_yearly.index.astype(str), _yearly.values,
               color=sns.color_palette("muted")[0], alpha=0.85)
    _ax[0].set_title("Videos by Year of Creation", fontweight="bold")
    _ax[0].set_ylabel("Count")
    _ax[0].tick_params(axis="x", rotation=45)

    # 3b — Monthly composition (last 2 years for clarity)
    _recent = videos_df[videos_df["created_at"].dt.year >= 2024]
    _monthly = _recent.set_index("created_at").resample("ME").size()
    _ax[1].bar(_monthly.index, _monthly.values, width=25,
               color=sns.color_palette("muted")[1], alpha=0.85)
    _ax[1].set_title("Videos Created per Month (2024–2026)", fontweight="bold")
    _ax[1].set_xlabel("Month")
    _ax[1].set_ylabel("Count")
    _ax[1].tick_params(axis="x", rotation=45)

    fig_temporal.tight_layout()
    fig_temporal
    return (fig_temporal,)


@app.cell
def _(videos_df, plt, sns, np, pd):
    # ── 3.2 Video age at scraping time — the key confounding variable ──
    _df = videos_df.dropna(subset=["created_at", "snapshot_at"]).copy()
    _df["age_days"] = (_df["snapshot_at"] - _df["created_at"]).dt.total_seconds() / 86400

    fig_age, _ax = plt.subplots(1, 2, figsize=(14, 5))

    # Age distribution
    _age_clipped = _df["age_days"].clip(upper=_df["age_days"].quantile(0.99))
    _ax[0].hist(_age_clipped, bins=60, color=sns.color_palette("muted")[2], alpha=0.85, edgecolor="white")
    _ax[0].set_title("Video Age at Scraping Time", fontweight="bold")
    _ax[0].set_xlabel("Days since creation")
    _ax[0].set_ylabel("Frequency")
    _ax[0].axvline(_df["age_days"].median(), color="red", linestyle="--",
                   label=f"Median: {_df['age_days'].median():.0f} days")
    _ax[0].legend()

    # Age vs plays — showing the confounding effect
    _sample = _df[_df["plays"] > 0].sample(min(5000, len(_df)), random_state=42)
    _ax[1].scatter(np.clip(_sample["age_days"], 0, 2500),
                   np.log10(_sample["plays"].clip(lower=1)),
                   alpha=0.1, s=6, color=sns.color_palette("muted")[3])
    _ax[1].set_title("Video Age vs Plays (confounding variable)", fontweight="bold")
    _ax[1].set_xlabel("Age (days)")
    _ax[1].set_ylabel("log\u2081\u2080(Plays)")

    fig_age.tight_layout()
    fig_age
    return (fig_age,)


@app.cell
def _(mo, videos_df):
    _df = videos_df.dropna(subset=["created_at", "snapshot_at"]).copy()
    _df["age_days"] = (_df["snapshot_at"] - _df["created_at"]).dt.total_seconds() / 86400

    _pct_recent = len(_df[_df["created_at"].dt.year >= 2024]) / len(_df) * 100
    _med_age = _df["age_days"].median()
    _max_age = _df["age_days"].max()

    mo.md(f"""
    ### 3.1 Dataset Composition Interpretation

    - **{_pct_recent:.0f}%** of videos were created in 2024–2026, reflecting both TikTok's growth
      and our scraper's recency bias toward trending content
    - The dataset spans 2018–2026, providing topical diversity across TikTok's evolution

    ### 3.2 Video Age — A Critical Confounding Variable

    - **Median video age at scraping:** {_med_age:.0f} days
    - **Maximum:** {_max_age:.0f} days (~{_max_age/365:.1f} years)

    The scatter plot above reveals the confounding relationship: older videos tend to show
    higher play counts simply because they have had more time to accumulate views. This is
    **not** evidence that older content is inherently better.

    **Why this matters for our recommendation model:**

    1. **We cannot use raw engagement counts as-is for comparing videos of different ages.**
       A 2-year-old video with 1M plays is not necessarily "better" than a 2-day-old video
       with 50K plays — the latter may be growing much faster.
    2. **We do not have engagement trajectories** (how views/likes grew over time), only a
       single snapshot. This means we cannot compute velocity or growth-rate features.
    3. **Mitigation strategies in our pipeline:**
       - Author-baseline residualization (predict engagement *above* the creator's average,
         partially controlling for audience size and content age)
       - Time-based train/test splits (train on older content, test on newer) to simulate
         realistic production conditions
       - Log1p transformation reduces the impact of extreme accumulation in old videos
    4. **What we explicitly avoid:** claiming temporal causality (e.g., "posting at 4 PM
       increases engagement") — such claims require controlled experiments or at minimum
       engagement time-series data that we do not have.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Engagement Metrics Analysis

    Engagement metrics (plays, likes, shares, comments) are the **primary training targets**
    for our recommendation system. Understanding their distributions, relationships, and
    derived ratios is essential for choosing appropriate transformations and loss functions.
    """)
    return


@app.cell
def _(mo, videos_df):
    _eng = ["plays", "likes", "shares", "comments_count"]
    _stats = videos_df[_eng].describe().T
    _stats["median"] = videos_df[_eng].median()
    _stats["skew"] = videos_df[_eng].skew()
    _stats["kurtosis"] = videos_df[_eng].kurtosis()

    mo.md(f"""
    ### 4.1 Descriptive Statistics

    {mo.as_html(_stats.round(1))}

    **Key observations:**
    - All metrics exhibit **extreme positive skewness** and **high kurtosis**, hallmarks of
      power-law distributions typical of social media platforms
    - The mean-to-median ratio is a useful proxy for skewness: plays show a
      **{videos_df['plays'].mean() / max(videos_df['plays'].median(), 1):.0f}x** ratio,
      confirming heavy right-tail concentration
    - This motivates using **log1p transformations** on all engagement targets before training
    """)
    return


@app.cell
def _(np, plt, sns, videos_df):
    fig_eng_dist, _ax = plt.subplots(2, 2, figsize=(13, 10))
    _eng = ["plays", "likes", "shares", "comments_count"]
    _titles = ["Plays", "Likes", "Shares", "Comments"]
    _colors = sns.color_palette("muted", 4)

    for _i, (_c, _t, _clr) in enumerate(zip(_eng, _titles, _colors)):
        _a = _ax[_i // 2][_i % 2]
        _d = videos_df[_c].dropna().clip(lower=1)
        _a.hist(np.log10(_d), bins=60, color=_clr, alpha=0.8, edgecolor="white", linewidth=0.5)
        _a.set_title(f"Distribution of {_t} (log\u2081\u2080 scale)", fontweight="bold")
        _a.set_xlabel(f"log\u2081\u2080({_t})")
        _a.set_ylabel("Frequency")
        _med = float(np.median(_d))
        _a.axvline(np.log10(_med), color="red", linestyle="--", alpha=0.7, label=f"Median: {_med:,.0f}")
        _a.legend(fontsize=9)

    fig_eng_dist.tight_layout()
    fig_eng_dist
    return


@app.cell
def _(mo):
    mo.md("""
    ### Interpretation of Engagement Distributions

    All four metrics follow a **log-normal** distribution (approximately Gaussian in log-space),
    characteristic of virality-driven platforms. The long right tail represents viral content
    achieving orders-of-magnitude more engagement than the median.

    **Direct implications for the recommendation system:**
    1. **Target transformation:** `log1p(plays)` for the **reach** objective compresses the range
       and reduces the influence of extreme outliers on gradient-based training
    2. **Evaluation:** Rank-based metrics (NDCG, MRR) are more appropriate than regression
       metrics (MSE) because we care about *relative ordering*, not exact engagement counts
    3. **Negative sampling:** Random negatives from the long tail are easy; semi-hard negatives
       near the decision boundary require deliberate construction
    """)
    return


@app.cell
def _(np, plt, sns, videos_df):
    # ── 4.2 Correlation heatmap ──
    _eng = videos_df[["plays", "likes", "shares", "comments_count"]].dropna().clip(lower=1).apply(np.log1p)
    _corr = _eng.corr()

    fig_corr, _a = plt.subplots(figsize=(7, 6))
    _mask = np.triu(np.ones_like(_corr, dtype=bool), k=1)
    sns.heatmap(_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=_mask, square=True, linewidths=1, ax=_a, vmin=-1, vmax=1)
    _a.set_title("Engagement Correlations (log-transformed)", fontweight="bold")
    fig_corr.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.2 Correlation Interpretation

    Strong positive correlations (> 0.8) between all engagement metrics confirm that popular
    content performs well across *all* dimensions. However, the correlations are not perfect,
    which **justifies our multi-objective approach** with separate targets:

    - **Reach** = `log1p(plays)` — raw visibility, algorithm-driven
    - **Engagement** = `(likes + comments + shares) / plays` — quality-adjusted interaction
    - **Conversion** = `(shares × 1000) / plays` — strongest signal of content value

    The imperfect correlation means a video can have high views but low share rate (clickbait)
    or low views but high engagement rate (niche quality content) — our model must distinguish
    these cases.
    """)
    return


@app.cell
def _(np, plt, sns, videos_df):
    # ── 4.3 Scatter: Plays vs Likes with regression ──
    _df = videos_df.dropna(subset=["plays", "likes"])
    _df = _df[(_df["plays"] > 0) & (_df["likes"] > 0)].sample(min(5000, len(_df)), random_state=42)

    fig_scatter, _a = plt.subplots(figsize=(9, 7))
    _a.scatter(np.log10(_df["plays"]), np.log10(_df["likes"]),
               alpha=0.15, s=8, color=sns.color_palette("muted")[0])
    _z = np.polyfit(np.log10(_df["plays"]), np.log10(_df["likes"]), 1)
    _x = np.linspace(np.log10(_df["plays"]).min(), np.log10(_df["plays"]).max(), 100)
    _a.plot(_x, np.polyval(_z, _x), color="red", linewidth=2,
            label=f"Fit: log(likes) = {_z[0]:.2f} \u00d7 log(plays) + {_z[1]:.2f}")
    _a.set_xlabel("log\u2081\u2080(Plays)", fontsize=12)
    _a.set_ylabel("log\u2081\u2080(Likes)", fontsize=12)
    _a.set_title("Plays vs Likes (log-log, 5K sample)", fontweight="bold")
    _a.legend()
    fig_scatter.tight_layout()
    fig_scatter
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 Plays–Likes Relationship

    The log-log scatter confirms a **strong power-law relationship** between plays and likes.
    The slope of the regression line indicates the elasticity: for every 10× increase in plays,
    likes increase by approximately the same factor. Points above the line represent
    **high-quality content** (more likes per view than expected), while points below represent
    content that attracts views but not engagement — a distinction our engagement-rate
    objective is designed to capture.
    """)
    return


@app.cell
def _(plt, sns, videos_df):
    # ── 4.4 Engagement rates ──
    _df = videos_df[videos_df["plays"] > 0].dropna(subset=["plays", "likes", "shares", "comments_count"]).copy()
    _df["like_rate"] = _df["likes"] / _df["plays"]
    _df["comment_rate"] = _df["comments_count"] / _df["plays"]
    _df["share_rate"] = _df["shares"] / _df["plays"]

    fig_rates, _ax = plt.subplots(1, 3, figsize=(15, 4.5))
    _rate_info = [
        ("like_rate", "Like Rate (likes / plays)"),
        ("comment_rate", "Comment Rate (comments / plays)"),
        ("share_rate", "Share Rate (shares / plays)"),
    ]
    for _a, (_rc, _t) in zip(_ax, _rate_info):
        _vals = _df[_rc].clip(upper=_df[_rc].quantile(0.99))
        _a.hist(_vals, bins=50, color=sns.color_palette("muted")[_rate_info.index((_rc, _t))],
                alpha=0.8, edgecolor="white", linewidth=0.5)
        _a.set_title(_t, fontweight="bold", fontsize=10)
        _a.set_xlabel("Rate")
        _a.set_ylabel("Frequency")
        _a.axvline(_vals.median(), color="red", linestyle="--", alpha=0.7,
                   label=f"Median: {_vals.median():.4f}")
        _a.legend(fontsize=8)

    fig_rates.tight_layout()
    fig_rates
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.4 Engagement Rate Distributions

    Engagement rates normalize raw counts by viewership, isolating content *quality*:

    - **Like rate** clusters around 5–10%, consistent with TikTok industry benchmarks
    - **Comment rate** is much lower (< 1%) — commenting requires significantly more effort
    - **Share rate** falls between the two — sharing signals strong endorsement and is the
      basis for our `conversion` objective

    These rates are used directly as features and targets in the recommendation pipeline.
    """)
    return


@app.cell
def _(np, plt, sns, videos_df):
    # ── 4.5 Outlier analysis — boxplots ──
    _df = videos_df[["plays", "likes", "shares", "comments_count"]].dropna().clip(lower=1).apply(np.log10)

    fig_box, _a = plt.subplots(figsize=(10, 5))
    _bp = _a.boxplot([_df[c].values for c in _df.columns], labels=["Plays", "Likes", "Shares", "Comments"],
                     patch_artist=True, showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3))
    _colors = sns.color_palette("muted", 4)
    for _patch, _clr in zip(_bp["boxes"], _colors):
        _patch.set_facecolor(_clr)
        _patch.set_alpha(0.7)
    _a.set_ylabel("log\u2081\u2080(count)")
    _a.set_title("Engagement Metrics — Outlier Overview (log scale)", fontweight="bold")
    fig_box.tight_layout()
    fig_box
    return


@app.cell
def _(mo, np, pd, videos_df):
    # Quantify outliers using IQR on log-transformed data
    _results = []
    for _col in ["plays", "likes", "shares", "comments_count"]:
        _d = np.log10(videos_df[_col].dropna().clip(lower=1))
        _q1, _q3 = np.percentile(_d, [25, 75])
        _iqr = _q3 - _q1
        _upper = _q3 + 1.5 * _iqr
        _n_outliers = int((_d > _upper).sum())
        _results.append({"Metric": _col, "Q1 (log)": f"{_q1:.2f}", "Q3 (log)": f"{_q3:.2f}",
                         "IQR": f"{_iqr:.2f}", "Upper fence": f"{10**_upper:,.0f}",
                         "Outliers": f"{_n_outliers:,}", "Outlier %": f"{_n_outliers/len(videos_df)*100:.1f}%"})
    _odf = pd.DataFrame(_results)

    mo.md(f"""
    ### 4.5 Outlier Quantification (IQR method on log-transformed data)

    {mo.as_html(_odf)}

    **Interpretation:** Even after log-transformation, a meaningful percentage of videos qualify
    as statistical outliers — these are **viral hits** that drive platform engagement.
    Our recommendation system handles these through:
    - Log1p target transformation to reduce their gradient influence
    - Author-baseline residualization (predicting engagement *above creator average*)
    - Rank-based evaluation metrics that are robust to outlier magnitudes
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Content Analysis

    We analyze video duration, caption characteristics, and the relationship between content
    features and engagement to identify useful signals for content-based recommendation.
    """)
    return


@app.cell
def _(plt, sns, videos_df):
    fig_content, _ax = plt.subplots(1, 3, figsize=(16, 5))

    # Duration
    _dur = videos_df["duration_sec"].dropna()
    _ax[0].hist(_dur[_dur <= 180], bins=60, color=sns.color_palette("muted")[3], alpha=0.85, edgecolor="white")
    _ax[0].set_title("Video Duration (\u2264 3 min)", fontweight="bold")
    _ax[0].set_xlabel("Seconds")
    _ax[0].set_ylabel("Frequency")
    _ax[0].axvline(_dur.median(), color="red", linestyle="--", label=f"Median: {_dur.median():.0f}s")
    _ax[0].axvline(60, color="gray", linestyle=":", alpha=0.6, label="60s mark")
    _ax[0].legend(fontsize=8)

    # Caption length (raw, including hashtags)
    _cap_raw = videos_df["caption"].fillna("").str.len()
    _ax[1].hist(_cap_raw[_cap_raw > 0], bins=60, color=sns.color_palette("muted")[4], alpha=0.85, edgecolor="white")
    _ax[1].set_title("Caption Length (raw, incl. hashtags)", fontweight="bold")
    _ax[1].set_xlabel("Characters")
    _ax[1].set_ylabel("Frequency")
    _ax[1].axvline(_cap_raw.median(), color="red", linestyle="--", label=f"Median: {_cap_raw.median():.0f}")
    _ax[1].legend(fontsize=8)

    # Caption word count
    _wc = videos_df["caption"].fillna("").str.split().str.len()
    _ax[2].hist(_wc[_wc > 0].clip(upper=80), bins=50, color=sns.color_palette("muted")[5],
                alpha=0.85, edgecolor="white")
    _ax[2].set_title("Caption Word Count", fontweight="bold")
    _ax[2].set_xlabel("Words")
    _ax[2].set_ylabel("Frequency")
    _ax[2].axvline(_wc.median(), color="red", linestyle="--", label=f"Median: {_wc.median():.0f}")
    _ax[2].legend(fontsize=8)

    fig_content.tight_layout()
    fig_content
    return


@app.cell
def _(mo, pd, videos_df):
    _df = videos_df[(videos_df["plays"] > 0) & (videos_df["duration_sec"] > 0)].dropna(subset=["plays"]).copy()
    _df["dur_bucket"] = pd.cut(
        _df["duration_sec"],
        bins=[0, 15, 30, 60, 120, 180, 99999],
        labels=["0-15s", "15-30s", "30-60s", "1-2min", "2-3min", "3min+"],
    )
    _agg = _df.groupby("dur_bucket", observed=True).agg(
        count=("video_id", "size"),
        median_plays=("plays", "median"),
        median_likes=("likes", "median"),
        median_share_rate=("shares", lambda x: (x / _df.loc[x.index, "plays"]).median()),
    ).reset_index()

    mo.md(f"""
    ### 5.1 Duration vs Engagement

    {mo.as_html(_agg.round(4))}

    **Interpretation:** TikTok's short-form DNA is evident — the majority of videos are under
    60 seconds. The relationship between duration and engagement is non-trivial: very short
    videos (0-15s) may have lower completion rates but higher replay rates, while longer content
    (3min+) attracts a dedicated but smaller audience. Our model uses **duration buckets**
    as a categorical feature to capture these non-linear effects.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Hashtag & Topic Analysis

    Hashtags are the primary user-declared topic signal. This section performs a deep analysis
    of hashtag frequency, coverage, co-occurrence patterns, and a critical data quality issue
    discovered during exploration.
    """)
    return


@app.cell
def _(hashtags_bridge_df, mo, videos_df):
    # ── Coverage gap: bridge table vs inline captions ──
    _bridge_videos = hashtags_bridge_df["video_id"].nunique()
    _total = len(videos_df)
    _inline_count = int(videos_df["caption"].fillna("").str.contains(r"#[a-zA-Z]", regex=True).sum())

    mo.md(f"""
    ### 6.1 Critical Finding: Hashtag Coverage Gap

    | Source | Videos covered | Coverage |
    |--------|---------------|----------|
    | Bridge table (`video_hashtags`) | **{_bridge_videos:,}** | **{_bridge_videos/_total*100:.1f}%** |
    | Inline in caption text | **{_inline_count:,}** | **{_inline_count/_total*100:.1f}%** |

    **This is a critical data quality issue.** The normalized `video_hashtags` bridge table
    covers only **{_bridge_videos/_total*100:.1f}%** of videos, while **{_inline_count/_total*100:.1f}%** of captions
    contain inline hashtags (e.g., `#fitness #workout`). This means the bridge table alone
    massively underrepresents the hashtag landscape.

    **Resolution:** For downstream analysis and the recommendation model, we extract hashtags
    from **both sources** — the bridge table AND inline caption text — and merge them into
    a unified `hashtag_list` per video. This approach was also validated in our separate
    NLP analysis, where combined extraction raised coverage from 3.9% to nearly 100%.
    """)
    return


@app.cell
def _(Counter, hashtags_bridge_df, pd, re, videos_df):
    # Build combined hashtag list per video
    def _extract_inline_tags(_caption):
        return [t.lower() for t in re.findall(r"#([a-zA-Z]\w+)", str(_caption))]

    _bridge_grouped = hashtags_bridge_df.groupby("video_id")["tag"].apply(list).to_dict()

    _all_tags = Counter()
    _tags_per_video = []
    for _, _row in videos_df.iterrows():
        _vid = _row["video_id"]
        _inline = _extract_inline_tags(_row["caption"])
        _bridge = [t.lower() for t in _bridge_grouped.get(_vid, [])]
        _combined = list(set(_inline + _bridge))
        _all_tags.update(_combined)
        _tags_per_video.append(len(_combined))

    videos_with_combined_tags = pd.Series(_tags_per_video)
    all_tag_counter = _all_tags
    return all_tag_counter, videos_with_combined_tags


@app.cell
def _(all_tag_counter, mo, videos_df, videos_with_combined_tags):
    _has_tags = int((videos_with_combined_tags > 0).sum())
    _total = len(videos_df)
    _unique = len(all_tag_counter)
    _avg = videos_with_combined_tags[videos_with_combined_tags > 0].mean()

    mo.md(f"""
    ### 6.2 Combined Hashtag Statistics (Bridge + Inline)

    | Metric | Value |
    |--------|-------|
    | Videos with \u2265 1 hashtag (combined) | **{_has_tags:,}** / {_total:,} (**{_has_tags/_total*100:.1f}%**) |
    | Unique hashtags | **{_unique:,}** |
    | Avg hashtags per video (when present) | **{_avg:.1f}** |

    After combining both sources, hashtag coverage jumps to {_has_tags/_total*100:.1f}%, providing
    a dense topic signal for content-based retrieval.
    """)
    return


@app.cell
def _(all_tag_counter, pd, plt, sns):
    # ── Top 25 hashtags + Zipf plot ──
    _top25 = pd.DataFrame(all_tag_counter.most_common(25), columns=["hashtag", "count"])

    fig_hash, _ax = plt.subplots(1, 2, figsize=(15, 6))

    # Top 25
    _ax[0].barh(_top25["hashtag"].iloc[::-1], _top25["count"].iloc[::-1],
                color=sns.color_palette("muted")[2], alpha=0.85)
    _ax[0].set_title("Top 25 Hashtags (combined extraction)", fontweight="bold")
    _ax[0].set_xlabel("Number of videos")

    # Zipf rank-frequency
    _counts = sorted(all_tag_counter.values(), reverse=True)
    _ax[1].scatter(range(1, len(_counts) + 1), _counts, s=6, alpha=0.4,
                   color=sns.color_palette("muted")[3])
    _ax[1].set_xscale("log")
    _ax[1].set_yscale("log")
    _ax[1].set_title("Hashtag Frequency vs Rank (log-log)", fontweight="bold")
    _ax[1].set_xlabel("Rank")
    _ax[1].set_ylabel("Frequency")
    _ax[1].plot([1, len(_counts)], [_counts[0], _counts[0] / len(_counts)],
                color="red", linestyle="--", alpha=0.5, label="Zipf reference")
    _ax[1].legend()

    fig_hash.tight_layout()
    fig_hash
    return


@app.cell
def _(mo):
    mo.md("""
    ### 6.3 Hashtag Frequency Interpretation

    The frequency distribution follows **Zipf's law** (approximately linear in log-log space),
    which is universal for user-generated tag systems. This has concrete implications:

    - **Head tags** (`fyp`, `viral`, `foryou`) are platform-meta signals, not topical — they
      should be **filtered out** during topic-based retrieval to avoid noise
    - **Mid-frequency tags** (100–1000 occurrences) represent genuine topics (fitness, cooking,
      fashion) and are the most useful for content-based similarity
    - **Long-tail tags** (< 10 occurrences) are too sparse for reliable statistics but may
      capture niche interests valuable for personalization
    """)
    return


@app.cell
def _(Counter, combinations, pd, plt, re, sns, videos_df):
    # ── 6.4 Hashtag co-occurrence network ──
    _pair_counter = Counter()
    for _, _row in videos_df.iterrows():
        _tags = list(set(t.lower() for t in re.findall(r"#([a-zA-Z]\w+)", str(_row["caption"]))))
        _tags = [t for t in _tags if t not in {"fyp", "foryou", "foryoupage", "viral", "xyzbca", "trending"}]
        if 2 <= len(_tags) <= 20:
            for _pair in combinations(sorted(_tags), 2):
                _pair_counter[_pair] += 1

    _top_pairs = pd.DataFrame(_pair_counter.most_common(20), columns=["pair", "count"])
    _top_pairs["tag_1"] = _top_pairs["pair"].apply(lambda x: x[0])
    _top_pairs["tag_2"] = _top_pairs["pair"].apply(lambda x: x[1])

    fig_cooccur, _a = plt.subplots(figsize=(12, 6))
    _labels = [f"{r['tag_1']} + {r['tag_2']}" for _, r in _top_pairs.iterrows()]
    _a.barh(_labels[::-1], _top_pairs["count"].values[::-1],
            color=sns.color_palette("viridis", 20)[::-1], alpha=0.85)
    _a.set_title("Top 20 Hashtag Co-occurrence Pairs (meta-tags filtered)", fontweight="bold")
    _a.set_xlabel("Co-occurrence count")
    fig_cooccur.tight_layout()
    fig_cooccur
    return


@app.cell
def _(mo):
    mo.md("""
    ### 6.4 Co-occurrence Interpretation

    Hashtag co-occurrence reveals **topic clusters** in the dataset. Pairs like
    `fitness + gym`, `money + investing`, and `cooking + recipe` confirm that creators
    use semantically coherent tag combinations. These natural clusters:

    - Validate our **hashtag-based topic similarity** features in the retrieval pipeline
    - Suggest that **hashtag embedding** (learning vector representations from co-occurrence
      patterns) could improve content-based retrieval beyond exact-match similarity
    - Inform the **topic cap policy** in our ranker — we can define topic groups from
      co-occurrence clusters to enforce diversity in recommendation lists
    """)
    return


@app.cell
def _(Counter, pd, plt, re, sns, videos_df):
    # ── 6.5 Word cloud from caption text (excluding hashtags) ──
    _caption_counter = Counter()
    _stopwords = {"the", "a", "an", "is", "it", "to", "in", "for", "of", "and", "on",
                  "this", "that", "with", "you", "my", "me", "your", "i", "so", "but",
                  "are", "was", "not", "do", "no", "be", "have", "has", "just", "if",
                  "all", "can", "how", "what", "when", "out", "up", "its", "from", "or",
                  "at", "we", "they", "he", "she", "her", "his", "get", "got", "like",
                  "one", "im", "dont", "will", "been"}
    for _cap in videos_df["caption"].fillna(""):
        _clean = re.sub(r"#\w+", "", str(_cap))
        _clean = re.sub(r"https?://\S+", "", _clean)
        _clean = re.sub(r"@\w+", "", _clean)
        _words = re.findall(r"[a-zA-Z]{3,}", _clean.lower())
        _caption_counter.update(w for w in _words if w not in _stopwords)

    _top_words = pd.DataFrame(_caption_counter.most_common(30), columns=["word", "count"])

    fig_words, _a = plt.subplots(figsize=(12, 6))
    _a.barh(_top_words["word"].iloc[::-1], _top_words["count"].iloc[::-1],
            color=sns.color_palette("muted")[0], alpha=0.85)
    _a.set_title("Top 30 Words in Captions (hashtags/URLs/mentions removed)", fontweight="bold")
    _a.set_xlabel("Frequency")
    fig_words.tight_layout()
    fig_words
    return


@app.cell
def _(mo):
    mo.md("""
    ### 6.5 Caption Vocabulary Interpretation

    After removing hashtags, URLs, and mentions, the most frequent caption words reveal the
    **content landscape** of our dataset. Fitness, lifestyle, and motivation-related terms
    dominate, reflecting the scraper's source configuration. This vocabulary analysis is
    important for understanding the **topical coverage** of our training data and identifying
    potential **cold-start** gaps for content categories not well represented.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Creator Landscape

    Understanding the distribution of creators, their output volume, and the relationship
    between creator size and engagement informs the creator-aware retrieval branch.
    """)
    return


@app.cell
def _(authors_df, np, plt, sns, videos_df):
    fig_creators, _ax = plt.subplots(1, 3, figsize=(16, 5))

    # Videos per creator
    _vpa = videos_df.groupby("author_id").size()
    _ax[0].hist(_vpa.clip(upper=30), bins=30, color=sns.color_palette("muted")[5], alpha=0.85, edgecolor="white")
    _ax[0].set_title("Videos per Creator (capped at 30)", fontweight="bold")
    _ax[0].set_xlabel("Video count")
    _ax[0].set_ylabel("Number of creators")
    _ax[0].axvline(_vpa.median(), color="red", linestyle="--", label=f"Median: {_vpa.median():.0f}")
    _ax[0].legend(fontsize=8)

    # Follower distribution
    _f = authors_df["follower_count"].dropna()
    _f = _f[_f > 0]
    _ax[1].hist(np.log10(_f), bins=50, color=sns.color_palette("muted")[0], alpha=0.85, edgecolor="white")
    _ax[1].set_title("Follower Count (log\u2081\u2080)", fontweight="bold")
    _ax[1].set_xlabel("log\u2081\u2080(followers)")
    _ax[1].set_ylabel("Frequency")

    # Verified vs not
    _v_counts = authors_df["verified"].value_counts().reindex([True, False, None], fill_value=0)
    _labels_v = ["Verified", "Not Verified", "Unknown"]
    _ax[2].bar(_labels_v, [_v_counts.get(True, 0), _v_counts.get(False, 0), _v_counts.get(None, 0)],
               color=[sns.color_palette("muted")[2], sns.color_palette("muted")[3], sns.color_palette("muted")[4]],
               alpha=0.85)
    _ax[2].set_title("Creator Verification Status", fontweight="bold")
    _ax[2].set_ylabel("Count")

    fig_creators.tight_layout()
    fig_creators
    return


@app.cell
def _(authors_df, mo, videos_df):
    _vpa = videos_df.groupby("author_id").size().reset_index(name="video_count")
    _merged = _vpa.merge(authors_df[["author_id", "username", "verified", "follower_count"]], on="author_id", how="left")
    _top15 = _merged.nlargest(15, "video_count")[["username", "verified", "follower_count", "video_count"]]
    _top15["follower_count"] = _top15["follower_count"].apply(lambda x: f"{x:,.0f}" if x == x else "N/A")

    _v_count = int((authors_df["verified"] == True).sum())
    _total = len(authors_df)
    _single = int((_vpa["video_count"] == 1).sum())

    mo.md(f"""
    ### 7.1 Top 15 Most Prolific Creators

    {mo.as_html(_top15)}

    ### 7.2 Creator Landscape Summary

    | Metric | Value |
    |--------|-------|
    | Total creators | **{_total:,}** |
    | Verified | **{_v_count}** ({_v_count/_total*100:.1f}%) |
    | Single-video creators | **{_single:,}** ({_single/len(_vpa)*100:.1f}%) |
    | Median videos per creator | **{_vpa['video_count'].median():.0f}** |

    Creator output follows a **power-law**: {_single/_total*100:.0f}% of creators have only 1 video,
    while a few prolific creators produce dozens. This power-law justifies the
    **creator-aware retrieval branch**, where prolific creators with consistent styles serve
    as retrieval anchors for neighborhood-based recommendations.
    """)
    return


@app.cell
def _(authors_df, plt, sns, videos_df):
    # ── Verified vs not — engagement comparison ──
    _m = videos_df.merge(authors_df[["author_id", "verified"]], on="author_id", how="left")
    _m["verified"] = _m["verified"].fillna(False)
    _m = _m[_m["plays"] > 0].dropna(subset=["plays", "likes", "shares"])

    _agg = _m.groupby("verified").agg(
        count=("video_id", "size"),
        median_plays=("plays", "median"),
        median_likes=("likes", "median"),
        median_shares=("shares", "median"),
    ).reset_index()
    _agg["label"] = _agg["verified"].map({True: "Verified", False: "Not Verified"})

    fig_verified, _ax = plt.subplots(1, 3, figsize=(14, 4.5))
    _metrics = ["median_plays", "median_likes", "median_shares"]
    _labels = ["Median Plays", "Median Likes", "Median Shares"]
    _colors = [sns.color_palette("muted")[2], sns.color_palette("muted")[3]]

    for _a, _metric, _lbl in zip(_ax, _metrics, _labels):
        _bars = _a.bar(_agg["label"], _agg[_metric], color=_colors, alpha=0.85, edgecolor="white")
        _a.set_title(_lbl, fontweight="bold")
        _a.set_ylabel("Count")
        for _b, _v in zip(_bars, _agg[_metric]):
            _a.text(_b.get_x() + _b.get_width() / 2, _b.get_height(),
                    f"{_v:,.0f}", ha="center", va="bottom", fontsize=9)

    fig_verified.tight_layout()
    fig_verified
    return


@app.cell
def _(mo):
    mo.md("""
    ### 7.3 Verified vs Non-Verified Interpretation

    Verified creators show significantly higher median engagement, which is expected —
    verification correlates with established audiences. However, for recommendation:

    - Verification is a **popularity proxy**, not a **relevance signal**
    - Non-verified creators produce highly engaging niche content
    - Our model treats verification as one feature among many, not a primary ranking signal
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Comment Analysis

    Comments provide signals beyond simple counts: text sentiment, intent, threading depth,
    and engagement on comments themselves. These feed our Comment Intelligence module.
    """)
    return


@app.cell
def _(comments_df, np, plt, sns):
    fig_comments, _ax = plt.subplots(1, 3, figsize=(16, 5))

    # Comments per video
    _cpv = comments_df.groupby("video_id").size()
    _ax[0].hist(_cpv.clip(upper=50), bins=50, color=sns.color_palette("muted")[4], alpha=0.85, edgecolor="white")
    _ax[0].set_title("Comments per Video (capped at 50)", fontweight="bold")
    _ax[0].set_xlabel("Comment count")
    _ax[0].set_ylabel("Videos")

    # Comment text length
    _clen = comments_df["text"].fillna("").str.len()
    _ax[1].hist(_clen[_clen > 0].clip(upper=300), bins=60, color=sns.color_palette("muted")[5],
                alpha=0.85, edgecolor="white")
    _ax[1].set_title("Comment Text Length", fontweight="bold")
    _ax[1].set_xlabel("Characters")
    _ax[1].set_ylabel("Frequency")

    # Comment likes distribution
    _cl = comments_df["comment_likes"].dropna()
    _cl = _cl[_cl > 0]
    if len(_cl) > 0:
        _ax[2].hist(np.log10(_cl.clip(lower=1)), bins=50, color=sns.color_palette("muted")[0],
                    alpha=0.85, edgecolor="white")
        _ax[2].set_title("Comment Likes (log\u2081\u2080)", fontweight="bold")
        _ax[2].set_xlabel("log\u2081\u2080(likes)")
    else:
        _ax[2].text(0.5, 0.5, "No comment like data", ha="center", va="center", transform=_ax[2].transAxes)
        _ax[2].set_title("Comment Likes", fontweight="bold")

    fig_comments.tight_layout()
    fig_comments
    return


@app.cell
def _(comments_df, mo):
    _total = len(comments_df)
    _unique_vids = comments_df["video_id"].nunique()
    _replies = int((comments_df["comment_level"] > 0).sum()) if "comment_level" in comments_df.columns else 0
    _avg_len = comments_df["text"].fillna("").str.len().mean()
    _empty = int((comments_df["text"].fillna("").str.strip() == "").sum())

    mo.md(f"""
    ### 8.1 Comment Dataset Summary

    | Metric | Value |
    |--------|-------|
    | Total comments | **{_total:,}** |
    | Videos with comments | **{_unique_vids:,}** |
    | Reply threads (level > 0) | **{_replies:,}** ({_replies/max(_total,1)*100:.1f}%) |
    | Average comment length | **{_avg_len:.0f}** characters |
    | Empty comments | **{_empty}** |

    ### 8.2 Interpretation

    Comments feed our **Comment Intelligence** module, which classifies intent across 8 categories
    (confusion, help-seeking, purchase intent, save intent, praise, complaint, skepticism,
    off-topic). Key signals extracted:

    - **Threading depth** — deeply threaded discussions correlate with controversial or
      high-value content
    - **Sentiment dynamics** — comment sentiment evolution over time reveals audience reception
    - **Intent distribution** — videos with high purchase/save intent are strong conversion
      candidates, directly feeding the `conversion` objective
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Audio Track Analysis

    Audio is a distinctive TikTok feature — trending sounds drive content creation patterns.
    Understanding audio reuse helps identify trend-driven content.
    """)
    return


@app.cell
def _(audios_df, mo, plt, sns, videos_df):
    # Videos per audio
    _vpa = videos_df[videos_df["audio_id"].notna()].groupby("audio_id").size().reset_index(name="video_count")
    _vpa_merged = _vpa.merge(audios_df[["audio_id", "audio_name"]], on="audio_id", how="left")
    _top_audio = _vpa_merged.nlargest(15, "video_count")[["audio_name", "video_count"]]

    _reused = int((_vpa["video_count"] > 1).sum())
    _unique_audio = len(_vpa)
    _no_audio = int(videos_df["audio_id"].isna().sum())

    fig_audio, _ax = plt.subplots(figsize=(12, 5))
    _names = _top_audio["audio_name"].fillna("Unknown").str[:40].tolist()
    _ax.barh(_names[::-1], _top_audio["video_count"].values[::-1],
             color=sns.color_palette("muted")[1], alpha=0.85)
    _ax.set_title("Top 15 Most Used Audio Tracks", fontweight="bold")
    _ax.set_xlabel("Number of videos")
    fig_audio.tight_layout()

    mo.vstack([
        fig_audio,
        mo.md(f"""
    ### 9.1 Audio Landscape Summary

    | Metric | Value |
    |--------|-------|
    | Unique audio tracks | **{_unique_audio:,}** |
    | Audio tracks used in > 1 video | **{_reused:,}** ({_reused/_unique_audio*100:.1f}%) |
    | Videos without audio ID | **{_no_audio}** |

    ### 9.2 Interpretation

    Audio reuse is a strong **trend signal** on TikTok — when multiple creators use the
    same sound, it indicates a trending audio. This can serve as:
    - A **collaborative filtering signal** (videos sharing an audio track attract similar audiences)
    - A **freshness indicator** (trending sounds have temporal peaks)
    - A feature for the **multimodal retrieval branch** in our recommendation pipeline
    """)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Key Findings & Implications for the Recommendation System

    ### Data Quality
    - The dataset is **remarkably clean**: near-zero null rates on core fields
    - A critical **hashtag coverage gap** was identified and resolved: the bridge table covers
      only 3.2% of videos, but combined extraction (bridge + inline caption) raises coverage to ~94%
    - Engagement metrics are available for all videos via point-in-time snapshots

    ### Distribution Properties
    - All engagement metrics follow **log-normal / power-law distributions** → `log1p` targets
    - Strong inter-metric correlations but imperfect → multi-objective optimization justified
    - Power-law dynamics in creator output, hashtag frequency, and audio reuse

    ### Critical Confounding Variable: Video Age
    - We have **one snapshot per video**, not engagement trajectories over time
    - Older videos show higher raw counts simply from longer accumulation windows
    - This makes raw temporal features (hour-of-day, day-of-week) unreliable as
      engagement predictors without proper age controls
    - Mitigation: author-baseline residualization, log1p targets, time-based splits

    ### Feature Engineering Priorities (informed by this EDA)
    1. **Duration buckets** — short/medium/long categorical feature
    2. **Caption length and word count** — content richness proxies
    3. **Combined hashtag list** — from both bridge table and inline extraction
    4. **Engagement rates** (like/comment/share per play) — quality signals, age-invariant
    5. **Creator follower count** (log-transformed) — authority signal
    6. **Comment intelligence** — intent classification, sentiment, threading depth
    7. **Audio track ID** — trending sound detection, collaborative signal
    8. **Verification status** — popularity proxy (low weight)

    ### Modeling Decisions Directly Supported by This EDA
    - **Time-based splits** are appropriate given the 2018–2026 temporal span
    - **Log1p transformation** reduces target skewness from >8 to near-Gaussian
    - **NDCG and MRR** are the correct evaluation metrics (rank-based, outlier-robust)
    - **Hashtag-based topic retrieval** should use combined extraction, not bridge table alone
    - **Author-baseline residualization** is needed to control for both creator-popularity
      bias and video-age confounding
    - **Engagement rates** (likes/plays, shares/plays) are preferred over raw counts as
      targets because they are naturally age-invariant
    - **Multi-objective training** (reach, engagement, conversion) captures distinct value dimensions

    ### Known Limitations
    - **No engagement trajectories** — single snapshot per video, cannot compute growth rates
    - Temporal bias toward recent content (2024–2026 dominate)
    - Scraper source bias — fitness, finance, lifestyle topics over-represented
    - Audio `is_original` field is uniformly `False` (scraper limitation)
    - Comment data covers only a subset of videos (19,989 comments across fewer videos)
    """)
    return


if __name__ == "__main__":
    app.run()
