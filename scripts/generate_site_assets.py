import json
import os
import sqlite3
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "assets" / "figures"
METRICS_PATH = ROOT / "docs" / "assets" / "metrics.json"
DB_PATH = ROOT / "nba.sqlite"
RANDOM_STATE = 42
TRAIN_CUTOFF = 2017


def savefig(name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTDIR / name, bbox_inches="tight", dpi=140)
    plt.close()


def load_data():
    conn = sqlite3.connect(DB_PATH)

    print("Loading games table...", flush=True)
    games = pd.read_sql("SELECT * FROM game", conn)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["season_year"] = games["game_date"].dt.year.where(
        games["game_date"].dt.month >= 10,
        games["game_date"].dt.year - 1,
    )
    for col in ["pts_home", "pts_away"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games.dropna(subset=["pts_home", "pts_away"]).copy()

    print("Loading player info...", flush=True)
    players = pd.read_sql("SELECT * FROM common_player_info", conn)

    print("Aggregating play-by-play into player-game stats...", flush=True)
    scoring = pd.read_sql(
        """
        SELECT
            game_id,
            player1_id,
            player1_name,
            player1_team_abbreviation,
            SUM(pts) AS pts
        FROM play_by_play
        JOIN (
            SELECT rowid AS rid,
                   CASE
                     WHEN eventmsgtype = 1
                       THEN CASE
                         WHEN (COALESCE(homedescription, '') || COALESCE(visitordescription, '')) LIKE '%3PT%'
                           THEN 3 ELSE 2 END
                     WHEN eventmsgtype = 3 AND score IS NOT NULL
                       THEN 1
                     ELSE 0
                   END AS pts
            FROM play_by_play
            WHERE eventmsgtype IN (1, 3)
        ) scored
          ON play_by_play.rowid = scored.rid
        WHERE pts > 0
        GROUP BY game_id, player1_id, player1_name, player1_team_abbreviation
        """,
        conn,
    )
    rebounds = pd.read_sql(
        """
        SELECT game_id, player1_id, player1_name, COUNT(*) AS reb
        FROM play_by_play
        WHERE eventmsgtype = 4
        GROUP BY game_id, player1_id, player1_name
        """,
        conn,
    )
    turnovers = pd.read_sql(
        """
        SELECT game_id, player1_id, player1_name, COUNT(*) AS tov
        FROM play_by_play
        WHERE eventmsgtype = 5
        GROUP BY game_id, player1_id, player1_name
        """,
        conn,
    )
    conn.close()

    print("Merging player-game box stats...", flush=True)
    box = scoring.merge(rebounds, on=["game_id", "player1_id", "player1_name"], how="outer")
    box = box.merge(turnovers, on=["game_id", "player1_id", "player1_name"], how="outer")
    box = box.rename(
        columns={
            "player1_name": "player_name",
            "player1_id": "player_id",
            "player1_team_abbreviation": "team",
        }
    )
    box[["pts", "reb", "tov"]] = box[["pts", "reb", "tov"]].fillna(0)
    box = box[box["player_name"].notna() & (box["player_name"] != "")].copy()

    games["total_pts"] = games["pts_home"] + games["pts_away"]
    games["home_win"] = (games["pts_home"] > games["pts_away"]).astype(int)
    games["point_diff"] = games["pts_home"] - games["pts_away"]

    print("Building player-season and season-level summaries...", flush=True)
    box_with_season = box.merge(games[["game_id", "season_year"]], on="game_id", how="left")
    player_season = (
        box_with_season.groupby(["player_name", "player_id", "season_year"])
        .agg(
            games_played=("pts", "count"),
            pts_pg=("pts", "mean"),
            reb_pg=("reb", "mean"),
            tov_pg=("tov", "mean"),
        )
        .reset_index()
    )
    player_season = player_season[player_season["games_played"] >= 20].copy()

    season_scoring = (
        games.groupby("season_year")
        .agg(
            avg_total_pts=("total_pts", "mean"),
            avg_home_pts=("pts_home", "mean"),
            avg_away_pts=("pts_away", "mean"),
            home_win_rate=("home_win", "mean"),
            n_games=("game_id", "count"),
        )
        .reset_index()
    )
    season_scoring = season_scoring[
        (season_scoring["season_year"] >= 1950) & (season_scoring["season_year"] <= 2022)
    ].copy()

    return games, players, box, player_season, season_scoring


def build_rolling_team_features(df, window=10):
    home = df[["game_id", "game_date", "team_id_home", "pts_home", "pts_away", "home_win"]].copy()
    home.columns = ["game_id", "game_date", "team_id", "pts_for", "pts_against", "win"]
    home["is_home"] = 1

    away = df[["game_id", "game_date", "team_id_away", "pts_away", "pts_home", "home_win"]].copy()
    away.columns = ["game_id", "game_date", "team_id", "pts_for", "pts_against", "win"]
    away["win"] = 1 - away["win"]
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date"])
    team_games["recent_pts"] = team_games.groupby("team_id")["pts_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    team_games["recent_wins"] = team_games.groupby("team_id")["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    return team_games.reset_index(drop=True)


def classification_pipeline(games):
    clf_games = games[games["season_year"] >= 1980].copy().sort_values("game_date").reset_index(drop=True)
    clf_games["is_playoff"] = clf_games["game_id"].astype(str).str.startswith("004").astype(int)

    team_rolling = build_rolling_team_features(clf_games, window=10)
    home_rolling = team_rolling[team_rolling["is_home"] == 1][
        ["game_id", "team_id", "recent_pts", "recent_wins"]
    ].rename(
        columns={
            "team_id": "team_id_home",
            "recent_pts": "home_team_recent_pts",
            "recent_wins": "home_team_recent_wins",
        }
    )
    away_rolling = team_rolling[team_rolling["is_home"] == 0][
        ["game_id", "team_id", "recent_pts", "recent_wins"]
    ].rename(
        columns={
            "team_id": "team_id_away",
            "recent_pts": "away_team_recent_pts",
            "recent_wins": "away_team_recent_wins",
        }
    )

    clf_games = clf_games.merge(home_rolling, on=["game_id", "team_id_home"], how="left")
    clf_games = clf_games.merge(away_rolling, on=["game_id", "team_id_away"], how="left")

    clf_features = [
        "season_year",
        "is_playoff",
        "home_team_recent_pts",
        "away_team_recent_pts",
        "home_team_recent_wins",
        "away_team_recent_wins",
    ]
    clf_data = clf_games.dropna(subset=clf_features).copy()

    train_mask = clf_data["season_year"] < TRAIN_CUTOFF
    X_train = clf_data.loc[train_mask, clf_features].values
    y_train = clf_data.loc[train_mask, "home_win"].values
    X_test = clf_data.loc[~train_mask, clf_features].values
    y_test = clf_data.loc[~train_mask, "home_win"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores_logreg = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    y_pred_logreg = logreg.predict(X_test_scaled)
    y_proba_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    readable_names = {
        "season_year": "Season Year (era)",
        "is_playoff": "Is Playoff Game",
        "home_team_recent_pts": "Home: avg pts (last 10)",
        "away_team_recent_pts": "Away: avg pts (last 10)",
        "home_team_recent_wins": "Home: win rate (last 10)",
        "away_team_recent_wins": "Away: win rate (last 10)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax = axes[0, 0]
    for name, y_proba, color in [
        ("Logistic Regression", y_proba_logreg, "#1f77b4"),
        ("Random Forest", y_proba_rf, "#d62728"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Home Win Prediction (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred: Away Win", "Pred: Home Win"],
        yticklabels=["Actual: Away Win", "Actual: Home Win"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title("Random Forest — Confusion Matrix (Test Set)")

    ax = axes[1, 0]
    fi = pd.DataFrame({"feature": clf_features, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=True
    )
    fi["readable"] = fi["feature"].map(readable_names)
    ax.barh(fi["readable"], fi["importance"], color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("Which Features Drive Home-Win Predictions?")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    coef_df = pd.DataFrame({"feature": clf_features, "coef": logreg.coef_[0]}).sort_values("coef")
    coef_df["readable"] = coef_df["feature"].map(readable_names)
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_df["coef"]]
    ax.barh(coef_df["readable"], coef_df["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Logistic Regression Coefficient (standardized)")
    ax.set_title("Direction of Effects on Home Win Probability")
    ax.grid(axis="x", alpha=0.3)
    ax.text(
        0.02,
        0.98,
        "Green: increases home win prob.\nRed: decreases home win prob.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.8},
    )

    fig.suptitle("Classification Results — Predicting NBA Home-Team Wins", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig("classification_results.png")

    metrics = {
        "classification_dataset_games": int(len(clf_data)),
        "classification_train_games": int(len(X_train)),
        "classification_test_games": int(len(X_test)),
        "classification_cv_auc_logreg": float(cv_scores_logreg.mean()),
        "classification_cv_auc_rf": float(cv_scores_rf.mean()),
        "classification_test_auc_logreg": float(roc_auc_score(y_test, y_proba_logreg)),
        "classification_test_auc_rf": float(roc_auc_score(y_test, y_proba_rf)),
        "classification_test_acc_logreg": float(accuracy_score(y_test, y_pred_logreg)),
        "classification_test_acc_rf": float(accuracy_score(y_test, y_pred_rf)),
        "classification_baseline_acc": float(y_test.mean()),
        "clf_feature_importances": fi.sort_values("importance", ascending=False)
        .set_index("feature")["importance"]
        .to_dict(),
        "clf_logreg_coefs": coef_df.set_index("feature")["coef"].to_dict(),
    }
    return clf_data, clf_features, readable_names, logreg, rf, scaler, y_test, y_pred_rf, y_proba_logreg, y_proba_rf, metrics


def eda_figures(games, player_season, season_scoring):
    home_pts = games["pts_home"].dropna()
    away_pts = games["pts_away"].dropna()
    t_stat, p_val = ttest_ind(home_pts, away_pts, alternative="greater")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    sns.kdeplot(home_pts, ax=ax, fill=True, color="#1f77b4", alpha=0.55, label="Home")
    sns.kdeplot(away_pts, ax=ax, fill=True, color="#ff7f0e", alpha=0.55, label="Away")
    ax.axvline(home_pts.mean(), color="#1f77b4", lw=2.5, ls="--", label=f"Home mean = {home_pts.mean():.1f}")
    ax.axvline(away_pts.mean(), color="#ff7f0e", lw=2.5, ls="--", label=f"Away mean = {away_pts.mean():.1f}")
    ax.set_xlabel("Points Scored per Game")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Home vs. Away Scoring")
    ax.legend()
    ax.text(
        0.97,
        0.92,
        f"p = {p_val:.2e}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    ax2 = axes[1]
    era_bins = [1946, 1970, 1990, 2010, 2023]
    era_labels = ["1947–69", "1970–89", "1990–09", "2010–22"]
    games["era"] = pd.cut(games["season_year"], bins=era_bins, labels=era_labels)
    era_win = games.groupby("era", observed=True)["home_win"].mean().reset_index()
    palette = sns.color_palette("Blues_d", len(era_labels))
    bars = ax2.bar(era_win["era"].astype(str), era_win["home_win"] * 100, color=palette, edgecolor="white")
    ax2.axhline(50, color="grey", ls="--", label="50% (no advantage)")
    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_ylabel("Home Win Rate (%)")
    ax2.set_xlabel("Era")
    ax2.set_ylim(40, 70)
    ax2.set_title("Home-Court Win Rate by Era")
    ax2.legend()
    fig.suptitle("Conclusion 1 — Home-Court Advantage", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_home_court.png")

    modern = season_scoring[season_scoring["season_year"] >= 1980].copy()
    r, p = pearsonr(modern["season_year"], modern["avg_total_pts"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(
        season_scoring["season_year"],
        season_scoring["avg_total_pts"],
        color="#2196F3",
        lw=2,
        marker="o",
        markersize=3,
        label="Avg pts/game",
    )
    m, b = np.polyfit(modern["season_year"], modern["avg_total_pts"], 1)
    ax.plot(
        modern["season_year"],
        m * modern["season_year"] + b,
        color="#F44336",
        lw=2,
        ls="--",
        label=f"Trend line (r={r:.2f})",
    )
    ax.axvline(1980, color="grey", lw=1.2, ls=":", alpha=0.8)
    ax.text(1981, season_scoring["avg_total_pts"].max() * 0.97, "3PT rule\nadopted", fontsize=8, color="grey")
    ax.axvline(2015, color="#4CAF50", lw=1.5, ls=":", label="Pace-and-space era (~2015)")
    ax.set_xlabel("Season Start Year")
    ax.set_ylabel("Avg Total Points / Game")
    ax.set_title("NBA Scoring Trends (1950–2022)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.fill_between(
        season_scoring["season_year"],
        season_scoring["avg_home_pts"],
        season_scoring["avg_away_pts"],
        color="orchid",
        alpha=0.25,
        label="Home–Away gap",
    )
    ax2.plot(season_scoring["season_year"], season_scoring["avg_home_pts"], color="#2E86C1", lw=2, label="Home avg pts")
    ax2.plot(season_scoring["season_year"], season_scoring["avg_away_pts"], color="#FF6F00", lw=2, label="Away avg pts")
    ax2.set_xlabel("Season Start Year")
    ax2.set_ylabel("Avg Points / Game")
    ax2.set_title("Home vs. Away Scoring by Season")
    ax2.legend(fontsize=9)
    fig.suptitle("Conclusion 2 — The Scoring Revolution", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_scoring_revolution.png")

    ps = player_season.dropna(subset=["pts_pg", "reb_pg", "tov_pg"]).copy()
    pts_q75 = ps["pts_pg"].quantile(0.75)
    reb_q75 = ps["reb_pg"].quantile(0.75)
    ps["high_scorer"] = (ps["pts_pg"] >= pts_q75).astype(int)
    ps["high_rebounder"] = (ps["reb_pg"] >= reb_q75).astype(int)
    ct = pd.crosstab(ps["high_scorer"], ps["high_rebounder"])
    chi2, p_chi, dof, expected = chi2_contingency(ct)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    corr_cols = ["pts_pg", "reb_pg", "tov_pg"]
    corr_labels = ["Pts/G", "Reb/G", "TOV/G"]
    corr_mat = ps[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat,
        mask=mask,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=corr_labels,
        yticklabels=corr_labels,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Pairwise Correlation — Player Per-Game Stats")
    axes[0].tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    sample = ps.sample(min(4000, len(ps)), random_state=42)
    sc = ax2.scatter(
        sample["reb_pg"],
        sample["pts_pg"],
        c=sample["games_played"],
        cmap="plasma",
        alpha=0.45,
        s=18,
        edgecolors="none",
    )
    z = np.polyfit(sample["reb_pg"], sample["pts_pg"], 1)
    x_line = np.linspace(sample["reb_pg"].min(), sample["reb_pg"].max(), 200)
    ax2.plot(x_line, z[0] * x_line + z[1], color="crimson", lw=2.2, label="Linear fit")
    ax2.text(
        0.05,
        0.95,
        f"r = {corr_mat.loc['pts_pg', 'reb_pg']:.2f}  (p < 0.001)",
        transform=ax2.transAxes,
        va="top",
        color="crimson",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Rebounds per Game")
    ax2.set_ylabel("Points per Game")
    ax2.set_title("Points vs. Rebounds per Game\n(colour = games played)")
    ax2.legend()
    plt.colorbar(sc, ax=ax2, label="Games Played in Season")
    fig.suptitle("Conclusion 3 — Player Statistical Archetypes & Feature Correlations", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_player_archetypes.png")

    top10 = ps.nlargest(10, "pts_pg")[["player_name", "season_year", "pts_pg", "reb_pg", "tov_pg"]]
    ps["pts_z"] = stats.zscore(ps["pts_pg"].fillna(0))
    outliers = ps[ps["pts_z"] > 3][["player_name", "season_year", "pts_pg", "pts_z"]].sort_values(
        "pts_z", ascending=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    top_plot = top10.sort_values("pts_pg")
    ax.barh(top_plot["player_name"] + " (" + top_plot["season_year"].astype(int).astype(str) + ")", top_plot["pts_pg"], color="#c0392b")
    ax.set_title("Highest Single-Season Scoring Averages")
    ax.set_xlabel("Points per Game")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    sns.histplot(ps["pts_pg"], bins=35, color="#1f77b4", alpha=0.8, ax=ax)
    ax.axvline(ps["pts_pg"].mean(), color="black", lw=1.5, ls="--", label=f"Mean = {ps['pts_pg'].mean():.1f}")
    for _, row in outliers.head(6).iterrows():
        ax.axvline(row["pts_pg"], color="#d35400", alpha=0.35)
    ax.set_title("Distribution of Player-Season Scoring")
    ax.set_xlabel("Points per Game")
    ax.legend()
    fig.suptitle("Elite Scoring Seasons Are Genuine Outliers", fontsize=17, fontweight="bold")
    plt.tight_layout()
    savefig("eda_scoring_outliers.png")

    metrics = {
        "games_count": int(len(games)),
        "players_unique": int(ps["player_name"].nunique()),
        "box_rows": int(player_season.shape[0]),
        "season_start": int(games["season_year"].min()),
        "season_end": int(games["season_year"].max()),
        "home_mean": float(home_pts.mean()),
        "away_mean": float(away_pts.mean()),
        "home_win_rate": float(games["home_win"].mean()),
        "home_t_stat": float(t_stat),
        "home_p_value": float(p_val),
        "scoring_r": float(r),
        "scoring_p": float(p),
        "chi2": float(chi2),
        "chi2_p": float(p_chi),
        "era_home_win": {str(k): float(v) for k, v in zip(era_win["era"], era_win["home_win"])},
        "top_scorers": top10.to_dict(orient="records"),
    }
    return ps, metrics


def clustering_pipeline(player_season):
    ps_clust = player_season.dropna(subset=["pts_pg", "reb_pg", "tov_pg"]).copy()
    clust_features = ["pts_pg", "reb_pg", "tov_pg"]
    scaler_c = StandardScaler()
    X_clust = scaler_c.fit_transform(ps_clust[clust_features])

    k_range = range(2, 9)
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_clust)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_clust, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(k_range), inertias, "o-", color="#1f77b4", lw=2, markersize=8)
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster SSE)")
    axes[0].set_title("Elbow Method")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), sil_scores, "o-", color="#d62728", lw=2, markersize=8)
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].grid(alpha=0.3)
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    axes[1].axvline(best_k, color="green", ls="--", alpha=0.7, label=f"Best k = {best_k}")
    axes[1].legend()
    fig.suptitle("Choosing the Number of Player Archetypes", fontweight="bold")
    plt.tight_layout()
    savefig("clustering_diagnostics.png")

    final_km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    ps_clust["cluster"] = final_km.fit_predict(X_clust)
    centroids_scaled = final_km.cluster_centers_
    centroids_orig = scaler_c.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_orig, columns=clust_features)
    centroid_df["size"] = ps_clust.groupby("cluster").size().values
    centroid_df.index.name = "cluster"

    order = centroid_df.sort_values("pts_pg", ascending=False).index.tolist()
    labels_by_rank = {
        order[0]: "Stars (high pts, mid reb)",
        order[1]: "Starters / Wings",
        order[2]: "Role players",
        order[3]: "Bench / low usage",
    }
    ps_clust["archetype"] = ps_clust["cluster"].map(labels_by_rank)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_clust)
    ps_clust["pc1"] = X_pca[:, 0]
    ps_clust["pc2"] = X_pca[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    archetype_colors = {
        "Stars (high pts, mid reb)": "#d62728",
        "Starters / Wings": "#2ca02c",
        "Role players": "#1f77b4",
        "Bench / low usage": "#9467bd",
    }
    for arch, color in archetype_colors.items():
        sub = ps_clust[ps_clust["archetype"] == arch]
        ax.scatter(sub["pc1"], sub["pc2"], c=color, label=f"{arch} (n={len(sub):,})", alpha=0.45, s=15, edgecolors="none")
    centroids_pca = pca.transform(centroids_scaled)
    for cx, cy in centroids_pca:
        ax.scatter(cx, cy, marker="X", s=300, c="black", edgecolors="white", linewidths=2, zorder=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Player-Season Archetypes (K-Means, k=4)\nPCA Projection")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    profile = centroid_df.copy()
    profile["archetype"] = profile.index.map(labels_by_rank)
    profile = profile.sort_values("pts_pg", ascending=True)
    x = np.arange(len(profile))
    width = 0.27
    ax.barh(x - width, profile["pts_pg"], height=width, label="Pts/G", color="#d62728")
    ax.barh(x, profile["reb_pg"], height=width, label="Reb/G", color="#2ca02c")
    ax.barh(x + width, profile["tov_pg"], height=width, label="TOV/G", color="#ff7f0e")
    ax.set_yticks(x)
    ax.set_yticklabels(profile["archetype"])
    ax.set_xlabel("Per-Game Statistic")
    ax.set_title("Archetype Statistical Profiles")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Player Archetypes Discovered by K-Means Clustering", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig("clustering_results.png")

    metrics = {
        "best_k": int(best_k),
        "silhouette_scores": {str(k): float(s) for k, s in zip(k_range, sil_scores)},
        "pca_total_variance": float(pca.explained_variance_ratio_.sum()),
        "archetype_sizes": {k: int(v) for k, v in ps_clust["archetype"].value_counts().to_dict().items()},
        "cluster_centroids": centroid_df.assign(archetype=centroid_df.index.map(labels_by_rank)).to_dict(orient="records"),
    }
    return ps_clust, pca, archetype_colors, metrics


def regression_pipeline(clf_data, readable_names):
    reg_features = [
        "season_year",
        "is_playoff",
        "home_team_recent_pts",
        "away_team_recent_pts",
        "home_team_recent_wins",
        "away_team_recent_wins",
    ]
    reg_target = "total_pts"
    reg_data = clf_data.dropna(subset=reg_features + [reg_target]).copy()

    train_mask = reg_data["season_year"] < TRAIN_CUTOFF
    X_train_r = reg_data.loc[train_mask, reg_features].values
    y_train_r = reg_data.loc[train_mask, reg_target].values
    X_test_r = reg_data.loc[~train_mask, reg_features].values
    y_test_r = reg_data.loc[~train_mask, reg_target].values

    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    lr = LinearRegression()
    lr.fit(X_train_r_scaled, y_train_r)
    y_pred_lr = lr.predict(X_test_r_scaled)

    rfr = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    rfr.fit(X_train_r, y_train_r)
    y_pred_rfr = rfr.predict(X_test_r)

    cv_r2_lr = cross_val_score(lr, X_train_r_scaled, y_train_r, cv=5, scoring="r2").mean()
    cv_r2_rfr = cross_val_score(rfr, X_train_r, y_train_r, cv=5, scoring="r2").mean()

    residuals = y_test_r - y_pred_rfr
    test_df = reg_data.loc[~train_mask].copy()
    test_df["pred_rfr"] = y_pred_rfr
    season_summary = (
        test_df.groupby("season_year")
        .agg(actual_mean=("total_pts", "mean"), predicted_mean=("pred_rfr", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    ax.scatter(y_test_r, y_pred_rfr, alpha=0.25, s=12, color="#1f77b4", edgecolors="none")
    lims = [min(y_test_r.min(), y_pred_rfr.min()) - 5, max(y_test_r.max(), y_pred_rfr.max()) + 5]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Total Points")
    ax.set_ylabel("Predicted Total Points")
    ax.set_title(f"Random Forest — Predicted vs. Actual\nTest R² = {r2_score(y_test_r, y_pred_rfr):.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_pred_rfr, residuals, alpha=0.25, s=12, color="#9467bd", edgecolors="none")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Predicted Total Points")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot — Random Forest")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    fi_r = pd.DataFrame({"feature": reg_features, "importance": rfr.feature_importances_}).sort_values(
        "importance", ascending=True
    )
    fi_r["readable"] = fi_r["feature"].map(readable_names)
    ax.barh(fi_r["readable"], fi_r["importance"], color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("What Drives Game Total Points?")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(season_summary["season_year"], season_summary["actual_mean"], "o-", color="#1f77b4", lw=2, markersize=9, label="Actual avg total pts")
    ax.plot(season_summary["season_year"], season_summary["predicted_mean"], "s--", color="#d62728", lw=2, markersize=9, label="Predicted avg total pts")
    ax.set_xlabel("Season Year")
    ax.set_ylabel("Average Total Points / Game")
    ax.set_title("Test-Set Predictions vs. Actuals by Season")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Regression Diagnostics — Predicting Game Total Points", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig("regression_results.png")

    metrics = {
        "regression_cv_r2_lr": float(cv_r2_lr),
        "regression_cv_r2_rf": float(cv_r2_rfr),
        "regression_test_rmse_lr": float(np.sqrt(mean_squared_error(y_test_r, y_pred_lr))),
        "regression_test_mae_lr": float(mean_absolute_error(y_test_r, y_pred_lr)),
        "regression_test_r2_lr": float(r2_score(y_test_r, y_pred_lr)),
        "regression_test_rmse_rf": float(np.sqrt(mean_squared_error(y_test_r, y_pred_rfr))),
        "regression_test_mae_rf": float(mean_absolute_error(y_test_r, y_pred_rfr)),
        "regression_test_r2_rf": float(r2_score(y_test_r, y_pred_rfr)),
        "regression_feature_importances": fi_r.sort_values("importance", ascending=False)
        .set_index("feature")["importance"]
        .to_dict(),
    }
    return season_summary, metrics


def summary_dashboard(games, season_scoring, y_test, y_proba_logreg, y_proba_rf, ps_clust, pca, archetype_colors, season_summary):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    axA = fig.add_subplot(gs[0, :2])
    axA.plot(
        season_scoring["season_year"],
        season_scoring["avg_total_pts"],
        color="#1f77b4",
        lw=2.5,
        marker="o",
        markersize=4,
        label="Avg total pts/game",
    )
    modern2 = season_scoring[season_scoring["season_year"] >= 1980]
    m_, b_ = np.polyfit(modern2["season_year"], modern2["avg_total_pts"], 1)
    axA.plot(modern2["season_year"], m_ * modern2["season_year"] + b_, color="#d62728", lw=2, ls="--", label="Modern-era trend")
    axA.axvline(1980, color="grey", lw=1, ls=":", alpha=0.7)
    axA.axvline(2015, color="#2ca02c", lw=1, ls=":", alpha=0.7)
    axA.text(1981, axA.get_ylim()[1] * 0.97, "3PT line", fontsize=8, color="grey")
    axA.text(2015.5, axA.get_ylim()[1] * 0.97, "Pace-and-space era", fontsize=8, color="#2ca02c")
    axA.set_xlabel("Season Year")
    axA.set_ylabel("Avg Total Points / Game")
    axA.set_title("A. NBA Scoring Across Seven Decades", fontweight="bold", fontsize=12)
    axA.legend(loc="lower right", fontsize=9)
    axA.grid(alpha=0.3)

    axB = fig.add_subplot(gs[0, 2])
    era_win2 = games.groupby("era", observed=True)["home_win"].mean().reset_index()
    bars = axB.bar(
        era_win2["era"].astype(str),
        era_win2["home_win"] * 100,
        color=sns.color_palette("Blues_d", len(era_win2)),
        edgecolor="white",
    )
    axB.axhline(50, color="grey", ls="--", lw=1)
    for bar in bars:
        axB.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    axB.set_ylim(40, 69)
    axB.set_xlabel("Era")
    axB.set_ylabel("Home Win Rate (%)")
    axB.set_title("B. Declining Home-Court Advantage", fontweight="bold", fontsize=12)
    axB.tick_params(axis="x", rotation=20)

    axC = fig.add_subplot(gs[1, 0])
    for name, y_proba, color in [("Logistic Reg.", y_proba_logreg, "#1f77b4"), ("Random Forest", y_proba_rf, "#d62728")]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axC.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    axC.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    axC.set_xlabel("False Positive Rate")
    axC.set_ylabel("True Positive Rate")
    axC.set_title("C. Home-Win Classifier ROC", fontweight="bold", fontsize=12)
    axC.legend(loc="lower right", fontsize=9)
    axC.grid(alpha=0.3)

    axD = fig.add_subplot(gs[1, 1])
    for arch, color in archetype_colors.items():
        sub = ps_clust[ps_clust["archetype"] == arch]
        axD.scatter(sub["pc1"], sub["pc2"], c=color, label=arch.split(" ")[0], alpha=0.45, s=10, edgecolors="none")
    axD.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% var)")
    axD.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% var)")
    axD.set_title("D. Player Archetypes (K-Means + PCA)", fontweight="bold", fontsize=12)
    axD.legend(fontsize=8, loc="best")
    axD.grid(alpha=0.3)

    axE = fig.add_subplot(gs[1, 2])
    axE.plot(season_summary["season_year"], season_summary["actual_mean"], "o-", color="#1f77b4", lw=2, markersize=8, label="Actual")
    axE.plot(season_summary["season_year"], season_summary["predicted_mean"], "s--", color="#d62728", lw=2, markersize=8, label="Predicted (RF)")
    axE.set_xlabel("Test Season")
    axE.set_ylabel("Avg Total Pts / Game")
    axE.set_title("E. Total-Points Predictions", fontweight="bold", fontsize=12)
    axE.legend(fontsize=9)
    axE.grid(alpha=0.3)

    fig.suptitle("NBA Data Science Pipeline — Summary Dashboard", fontsize=16, fontweight="bold", y=0.995)
    savefig("summary_dashboard.png")


def main():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 14, "axes.labelsize": 12, "font.family": "DejaVu Sans"})

    games, players, box, player_season, season_scoring = load_data()
    print("Generating EDA figures...", flush=True)
    ps, eda_metrics = eda_figures(games, player_season, season_scoring)

    print("Running classification pipeline...", flush=True)
    clf_data, clf_features, readable_names, logreg, rf, scaler, y_test, y_pred_rf, y_proba_logreg, y_proba_rf, clf_metrics = classification_pipeline(games)
    print("Running clustering pipeline...", flush=True)
    ps_clust, pca, archetype_colors, cluster_metrics = clustering_pipeline(player_season)
    print("Running regression pipeline...", flush=True)
    season_summary, reg_metrics = regression_pipeline(clf_data, readable_names)
    print("Building summary dashboard...", flush=True)
    summary_dashboard(games, season_scoring, y_test, y_proba_logreg, y_proba_rf, ps_clust, pca, archetype_colors, season_summary)

    all_metrics = {
        **eda_metrics,
        **clf_metrics,
        **cluster_metrics,
        **reg_metrics,
    }
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))
    print(json.dumps({k: all_metrics[k] for k in [
        "games_count",
        "players_unique",
        "home_win_rate",
        "classification_test_auc_rf",
        "classification_test_acc_rf",
        "best_k",
        "regression_test_mae_rf",
        "regression_test_r2_rf",
    ]}, indent=2))


if __name__ == "__main__":
    main()
