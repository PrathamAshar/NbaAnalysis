import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
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
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "nba.sqlite"
OUTDIR = ROOT / "docs" / "assets" / "figures"
METRICS_PATH = ROOT / "docs" / "assets" / "metrics.json"
TRAIN_CUTOFF = 2017
RANDOM_STATE = 42


def savefig(name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTDIR / name, bbox_inches="tight", dpi=150)
    plt.close()


def load_games():
    print("Loading game table...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql("SELECT * FROM game", conn)
    conn.close()

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["season_year"] = games["game_date"].dt.year.where(
        games["game_date"].dt.month >= 10,
        games["game_date"].dt.year - 1,
    )
    for col in ["pts_home", "pts_away"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games.dropna(subset=["pts_home", "pts_away"]).copy()

    games["total_pts"] = games["pts_home"] + games["pts_away"]
    games["home_win"] = (games["pts_home"] > games["pts_away"]).astype(int)
    games["point_diff"] = games["pts_home"] - games["pts_away"]
    games["is_playoff"] = games["game_id"].astype(str).str.startswith("004").astype(int)

    era_bins = [1946, 1970, 1990, 2010, 2023]
    era_labels = ["1947–69", "1970–89", "1990–09", "2010–22"]
    games["era"] = pd.cut(games["season_year"], bins=era_bins, labels=era_labels)
    return games


def build_rolling_team_features(df, window=10):
    home = df[["game_id", "game_date", "team_id_home", "pts_home", "pts_away", "home_win"]].copy()
    home.columns = ["game_id", "game_date", "team_id", "pts_for", "pts_against", "win"]
    home["is_home"] = 1

    away = df[["game_id", "game_date", "team_id_away", "pts_away", "pts_home", "home_win"]].copy()
    away.columns = ["game_id", "game_date", "team_id", "pts_for", "pts_against", "win"]
    away["win"] = 1 - away["win"]
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date"]).reset_index(drop=True)
    team_games["recent_pts"] = team_games.groupby("team_id")["pts_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    team_games["recent_wins"] = team_games.groupby("team_id")["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    return team_games


def make_eda_figures(games):
    print("Generating game-level EDA figures...", flush=True)
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
    season_scoring = season_scoring[(season_scoring["season_year"] >= 1950) & (season_scoring["season_year"] <= 2022)].copy()

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
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    ax2 = axes[1]
    era_win = games.groupby("era", observed=True)["home_win"].mean().reset_index()
    bars = ax2.bar(era_win["era"].astype(str), era_win["home_win"] * 100, color=sns.color_palette("Blues_d", len(era_win)), edgecolor="white")
    ax2.axhline(50, color="grey", ls="--", label="50% (no advantage)")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Home Win Rate (%)")
    ax2.set_xlabel("Era")
    ax2.set_ylim(40, 70)
    ax2.set_title("Home-Court Win Rate by Era")
    ax2.legend()
    fig.suptitle("Home-Court Advantage in Historical Context", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_home_court.png")

    modern = season_scoring[season_scoring["season_year"] >= 1980].copy()
    r, p = pearsonr(modern["season_year"], modern["avg_total_pts"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(season_scoring["season_year"], season_scoring["avg_total_pts"], color="#2196F3", lw=2, marker="o", markersize=3, label="Avg pts/game")
    m, b = np.polyfit(modern["season_year"], modern["avg_total_pts"], 1)
    ax.plot(modern["season_year"], m * modern["season_year"] + b, color="#F44336", lw=2, ls="--", label=f"Trend line (r={r:.2f})")
    ax.axvline(1980, color="grey", lw=1.2, ls=":", alpha=0.8)
    ax.text(1981, season_scoring["avg_total_pts"].max() * 0.97, "3PT rule\nadopted", fontsize=8, color="grey")
    ax.axvline(2015, color="#4CAF50", lw=1.5, ls=":", label="Pace-and-space era (~2015)")
    ax.set_xlabel("Season Start Year")
    ax.set_ylabel("Avg Total Points / Game")
    ax.set_title("NBA Scoring Trends (1950–2022)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.fill_between(season_scoring["season_year"], season_scoring["avg_home_pts"], season_scoring["avg_away_pts"], color="orchid", alpha=0.25, label="Home–Away gap")
    ax2.plot(season_scoring["season_year"], season_scoring["avg_home_pts"], color="#2E86C1", lw=2, label="Home avg pts")
    ax2.plot(season_scoring["season_year"], season_scoring["avg_away_pts"], color="#FF6F00", lw=2, label="Away avg pts")
    ax2.set_xlabel("Season Start Year")
    ax2.set_ylabel("Avg Points / Game")
    ax2.set_title("Home vs. Away Scoring by Season")
    ax2.legend(fontsize=9)
    fig.suptitle("Scoring Has Changed by Era, Not by One Simple Trend", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_scoring_revolution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    season_win = games.groupby("season_year")["home_win"].mean().reset_index()
    ax.plot(season_win["season_year"], season_win["home_win"] * 100, color="#17324d", lw=2.2)
    rolling = season_win["home_win"].rolling(7, center=True, min_periods=1).mean()
    ax.plot(season_win["season_year"], rolling * 100, color="#d96c3a", lw=3, label="7-season rolling mean")
    ax.axhline(50, color="grey", ls="--", lw=1)
    ax.set_title("Season-by-Season Home Win Rate")
    ax.set_xlabel("Season Year")
    ax.set_ylabel("Home Win Rate (%)")
    ax.legend()

    ax = axes[1]
    sns.histplot(games["point_diff"], bins=45, color="#1f7a8c", ax=ax)
    ax.axvline(0, color="black", lw=1.2)
    ax.axvline(games["point_diff"].mean(), color="#d96c3a", lw=2, ls="--", label=f"Mean = {games['point_diff'].mean():.1f}")
    ax.set_title("Distribution of Point Differentials")
    ax.set_xlabel("Home Team Margin")
    ax.legend()
    fig.suptitle("Game Margins Reinforce the Home-Court Story", fontsize=18, fontweight="bold")
    plt.tight_layout()
    savefig("eda_margin_story.png")

    metrics = {
        "games_count": int(len(games)),
        "season_start": int(games["season_year"].min()),
        "season_end": int(games["season_year"].max()),
        "home_mean": float(home_pts.mean()),
        "away_mean": float(away_pts.mean()),
        "home_win_rate": float(games["home_win"].mean()),
        "home_t_stat": float(t_stat),
        "home_p_value": float(p_val),
        "scoring_r": float(r),
        "scoring_p": float(p),
        "era_home_win": {str(k): float(v) for k, v in zip(era_win["era"], era_win["home_win"])},
    }
    return season_scoring, metrics


def classification_and_regression(games):
    print("Running classification and regression...", flush=True)
    clf_games = games[games["season_year"] >= 1980].copy().sort_values("game_date").reset_index(drop=True)
    team_rolling = build_rolling_team_features(clf_games, window=10)

    home_rolling = team_rolling[team_rolling["is_home"] == 1][["game_id", "team_id", "recent_pts", "recent_wins"]].rename(
        columns={"team_id": "team_id_home", "recent_pts": "home_team_recent_pts", "recent_wins": "home_team_recent_wins"}
    )
    away_rolling = team_rolling[team_rolling["is_home"] == 0][["game_id", "team_id", "recent_pts", "recent_wins"]].rename(
        columns={"team_id": "team_id_away", "recent_pts": "away_team_recent_pts", "recent_wins": "away_team_recent_wins"}
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
    readable_names = {
        "season_year": "Season Year (era)",
        "is_playoff": "Is Playoff Game",
        "home_team_recent_pts": "Home: avg pts (last 10)",
        "away_team_recent_pts": "Away: avg pts (last 10)",
        "home_team_recent_wins": "Home: win rate (last 10)",
        "away_team_recent_wins": "Away: win rate (last 10)",
    }
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
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores_logreg = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    y_pred_logreg = logreg.predict(X_test_scaled)
    y_proba_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax = axes[0, 0]
    for name, y_proba, color in [("Logistic Regression", y_proba_logreg, "#1f77b4"), ("Random Forest", y_proba_rf, "#d62728")]:
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
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, xticklabels=["Pred: Away Win", "Pred: Home Win"], yticklabels=["Actual: Away Win", "Actual: Home Win"], cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Random Forest — Confusion Matrix")

    ax = axes[1, 0]
    fi = pd.DataFrame({"feature": clf_features, "importance": rf.feature_importances_}).sort_values("importance", ascending=True)
    fi["readable"] = fi["feature"].map(readable_names)
    ax.barh(fi["readable"], fi["importance"], color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("What Drives Home-Win Predictions?")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    coef_df = pd.DataFrame({"feature": clf_features, "coef": logreg.coef_[0]}).sort_values("coef")
    coef_df["readable"] = coef_df["feature"].map(readable_names)
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_df["coef"]]
    ax.barh(coef_df["readable"], coef_df["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Logistic Regression Coefficient (standardized)")
    ax.set_title("Direction of Effects")
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Predicting NBA Home-Team Wins", fontsize=17, fontweight="bold")
    plt.tight_layout()
    savefig("classification_results.png")

    reg_features = clf_features
    reg_data = clf_data.dropna(subset=reg_features + ["total_pts"]).copy()
    train_mask_r = reg_data["season_year"] < TRAIN_CUTOFF
    X_train_r = reg_data.loc[train_mask_r, reg_features].values
    y_train_r = reg_data.loc[train_mask_r, "total_pts"].values
    X_test_r = reg_data.loc[~train_mask_r, reg_features].values
    y_test_r = reg_data.loc[~train_mask_r, "total_pts"].values

    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    lr = LinearRegression()
    lr.fit(X_train_r_scaled, y_train_r)
    y_pred_lr = lr.predict(X_test_r_scaled)

    rfr = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1)
    rfr.fit(X_train_r, y_train_r)
    y_pred_rfr = rfr.predict(X_test_r)

    cv_r2_lr = cross_val_score(lr, X_train_r_scaled, y_train_r, cv=5, scoring="r2").mean()
    cv_r2_rfr = cross_val_score(rfr, X_train_r, y_train_r, cv=5, scoring="r2").mean()

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
    residuals = y_test_r - y_pred_rfr
    ax.scatter(y_pred_rfr, residuals, alpha=0.25, s=12, color="#9467bd", edgecolors="none")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Predicted Total Points")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot — Random Forest")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    fi_r = pd.DataFrame({"feature": reg_features, "importance": rfr.feature_importances_}).sort_values("importance", ascending=True)
    fi_r["readable"] = fi_r["feature"].map(readable_names)
    ax.barh(fi_r["readable"], fi_r["importance"], color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("What Drives Game Total Points?")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    test_df = reg_data.loc[~train_mask_r].copy()
    test_df["pred_rfr"] = y_pred_rfr
    season_summary = test_df.groupby("season_year").agg(actual_mean=("total_pts", "mean"), predicted_mean=("pred_rfr", "mean")).reset_index()
    ax.plot(season_summary["season_year"], season_summary["actual_mean"], "o-", color="#1f77b4", lw=2, markersize=8, label="Actual")
    ax.plot(season_summary["season_year"], season_summary["predicted_mean"], "s--", color="#d62728", lw=2, markersize=8, label="Predicted")
    ax.set_xlabel("Season Year")
    ax.set_ylabel("Average Total Points / Game")
    ax.set_title("Test-Set Predictions vs. Actuals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Predicting Total Points Scored", fontsize=17, fontweight="bold")
    plt.tight_layout()
    savefig("regression_results.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    auc_df = pd.DataFrame(
        {
            "model": ["Always home", "Logistic reg.", "Random forest"],
            "accuracy": [y_test.mean(), accuracy_score(y_test, y_pred_logreg), accuracy_score(y_test, y_pred_rf)],
            "auc": [0.5, roc_auc_score(y_test, y_proba_logreg), roc_auc_score(y_test, y_proba_rf)],
        }
    )
    x = np.arange(len(auc_df))
    width = 0.34
    ax.bar(x - width / 2, auc_df["accuracy"], width=width, color="#1f7a8c", label="Accuracy")
    ax.bar(x + width / 2, auc_df["auc"], width=width, color="#d96c3a", label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(auc_df["model"])
    ax.set_ylim(0.45, 0.75)
    ax.set_title("Home-Win Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    reg_df = pd.DataFrame(
        {
            "model": ["Linear reg.", "Random forest"],
            "mae": [mean_absolute_error(y_test_r, y_pred_lr), mean_absolute_error(y_test_r, y_pred_rfr)],
            "r2": [r2_score(y_test_r, y_pred_lr), r2_score(y_test_r, y_pred_rfr)],
        }
    )
    ax.bar(reg_df["model"], reg_df["mae"], color=["#8e44ad", "#2e86ab"])
    ax.set_title("Total-Points Error (MAE)")
    ax.set_ylabel("Mean Absolute Error")
    for i, row in reg_df.iterrows():
        ax.text(i, row["mae"] + 0.2, f"R²={row['r2']:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Model Scoreboard", fontsize=17, fontweight="bold")
    plt.tight_layout()
    savefig("model_scoreboard.png")

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
        "regression_cv_r2_lr": float(cv_r2_lr),
        "regression_cv_r2_rf": float(cv_r2_rfr),
        "regression_test_rmse_lr": float(np.sqrt(mean_squared_error(y_test_r, y_pred_lr))),
        "regression_test_mae_lr": float(mean_absolute_error(y_test_r, y_pred_lr)),
        "regression_test_r2_lr": float(r2_score(y_test_r, y_pred_lr)),
        "regression_test_rmse_rf": float(np.sqrt(mean_squared_error(y_test_r, y_pred_rfr))),
        "regression_test_mae_rf": float(mean_absolute_error(y_test_r, y_pred_rfr)),
        "regression_test_r2_rf": float(r2_score(y_test_r, y_pred_rfr)),
    }
    return metrics


def main():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 14, "axes.labelsize": 12, "font.family": "DejaVu Sans"})
    games = load_games()
    season_scoring, eda_metrics = make_eda_figures(games)
    model_metrics = classification_and_regression(games)
    metrics = {**eda_metrics, **model_metrics}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
