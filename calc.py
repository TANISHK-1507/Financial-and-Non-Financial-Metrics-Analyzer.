import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-v0_8")


data1 = pd.read_csv("company1_data.csv")
data2 = pd.read_csv("company2_data.csv")

data1["company"] = "Company 1"
data2["company"] = "Company 2"

df = pd.concat([data1, data2], ignore_index=True)

df["profit"] = df["revenue"] - df["cost_of_goods_sold"] - df["operating_expenses"]

df["profit_margin"] = df["profit"] / df["revenue"]

df["operating_margin_denominator"] = df["revenue"]
df["operating_profit_margin"] = (
    df["revenue"] - df["cost_of_goods_sold"] - df["operating_expenses"]
) / df["revenue"] 

df["roa"] = df["profit"] / df["total_assets"]
df["roe"] = df["profit"] / df["shareholder_equity"]



def basic_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    for company in df["company"].unique():
        sub = df[df["company"] == company].sort_values("year")
        ax.plot(sub["year"], sub["revenue"], marker="o", label=company)
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue Over Time")
    ax.legend()

    ax = axes[0, 1]
    for company in df["company"].unique():
        sub = df[df["company"] == company].sort_values("year")
        ax.plot(sub["year"], sub["profit_margin"], marker="o", label=company)
    ax.set_xlabel("Year")
    ax.set_ylabel("Profit Margin")
    ax.set_title("Profit Margin Over Time")
    ax.legend()
    
    ax = axes[1, 0]
    for company in df["company"].unique():
        sub = df[df["company"] == company].sort_values("year")
        ax.plot(sub["year"], sub["roa"], marker="o", label=company)
    ax.set_xlabel("Year")
    ax.set_ylabel("ROA")
    ax.set_title("ROA Over Time")
    ax.legend()
    
    ax = axes[1, 1]
    for company in df["company"].unique():
        sub = df[df["company"] == company].sort_values("year")
        ax.plot(sub["year"], sub["roe"], marker="o", label=company)
    ax.set_xlabel("Year")
    ax.set_ylabel("ROE")
    ax.set_title("ROE Over Time")
    ax.legend()

    plt.tight_layout()
    plt.show()

def build_regression_model(df):
    df_reg = df.sort_values(["company", "year"]).copy()
    df_reg["revenue_next_year"] = df_reg.groupby("company")["revenue"].shift(-1)
    df_reg = df_reg.dropna(subset=["revenue_next_year"])

    features = [
        "year",
        "revenue",
        "profit",
        "profit_margin",
        "operating_profit_margin",
        "roa",
        "roe",
        "customer_satisfaction",
        "employee_turnover",
    ]

    X = df_reg[features]
    y = df_reg["revenue_next_year"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n====== Regression (Linear): Predict Next Year Revenue ======")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score:            {r2:.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Next Year Revenue")
    plt.ylabel("Predicted Next Year Revenue")
    plt.title("Actual vs Predicted Revenue (Next Year)")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.tight_layout()
    plt.show()

    return model


def build_classification_model(df):
    df_cls = df.copy()

    med_margin = df_cls["profit_margin"].median()
    med_roe = df_cls["roe"].median()
    med_roa = df_cls["roa"].median()

    df_cls["good_performance"] = (
        (df_cls["profit_margin"] > med_margin)
        & (df_cls["roe"] > med_roe)
        & (df_cls["roa"] > med_roa)
    ).astype(int)

    features = [
        "revenue",
        "profit",
        "profit_margin",
        "operating_profit_margin",
        "roa",
        "roe",
    ]

    X = df_cls[features]
    y = df_cls["good_performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n====== Classification: Good vs Weak Performance ======")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(features)), importances[indices])
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.title("Feature Importance (Classification)")
    plt.tight_layout()
    plt.show()

    return clf, df_cls


def run_clustering(df):
    df_cluster = df.copy()

    cluster_features = [
        "profit_margin",
        "operating_profit_margin",
        "roa",
        "roe",
    ]

    X = df_cluster[cluster_features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

    print("\n====== Clustering: Segment Summary ======")
    print(
        df_cluster.groupby(["company", "cluster"])[cluster_features].mean().round(3),
        "\n",
    )

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_cluster,
        x="roa",
        y="roe",
        hue="cluster",
        style="company",
        s=90,
    )
    plt.title("Clusters by ROA & ROE")
    plt.tight_layout()
    plt.show()

    return df_cluster


def generate_text_report(df):
    print("\n====== TEXT REPORT: COMPANY COMPARISON ======\n")

    company_group = df.groupby("company")

    summary = company_group[[
        "revenue",
        "profit",
        "profit_margin",
        "operating_profit_margin",
        "roa",
        "roe",
    ]].mean().round(3)

    print(summary)
    print("\nInsights:")
    print(f"- Higher average profit: {summary['profit'].idxmax()}")
    print(f"- Higher average ROE:    {summary['roe'].idxmax()}")
    print(f"- Higher average ROA:    {summary['roa'].idxmax()}\n")

def main():
    print("Data loaded. Shape:", df.shape)
    print(df.head(), "\n")

    basic_plots(df)

    build_regression_model(df)

    build_classification_model(df)

    clustered_df = run_clustering(df)

    generate_text_report(df)

    output_df = df.merge(
        clustered_df[["year", "company", "cluster"]],
        on=["year", "company"],
        how="left",
        suffixes=("", "_clustered"),
    )
    output_df.to_csv("processed_company_data_with_ratios.csv", index=False)
    print("Saved: processed_company_data_with_ratios.csv")


if __name__ == "__main__":
    main()
