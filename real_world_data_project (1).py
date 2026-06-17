"""
Project 4: Real-world Data Project (Retail Domain)
-------------------------------------------------------
Goal   : Work on a domain-specific dataset (retail sales) for applied
         learning - end-to-end data analysis and prediction.
Tools  : Pandas, Matplotlib, Seaborn, Scikit-learn
Output : Trend visualizations, category breakdown, a sales forecasting
         model, and a final findings/conclusions summary.

NOTE: A synthetic daily retail sales dataset (1 year, 3 stores, 4 product
categories) is generated below so the script runs anywhere without an
external file. To use your own data, replace `generate_dataset()` with
`pd.read_csv("your_file.csv")`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(21)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dataset():
    dates = pd.date_range("2025-06-01", "2026-05-31", freq="D")
    stores = ["Store_A", "Store_B", "Store_C"]
    categories = ["Electronics", "Groceries", "Apparel", "Home & Living"]

    rows = []
    for date in dates:
        for store in stores:
            for category in categories:
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_boost = 1.25 if date.weekday() >= 5 else 1.0
                base = {"Electronics": 4000, "Groceries": 2500,
                        "Apparel": 1800, "Home & Living": 1500}[category]
                noise = np.random.normal(0, 250)
                sales = max(0, base * seasonal * weekend_boost + noise)
                rows.append([date, store, category, round(sales, 2)])

    df = pd.DataFrame(rows, columns=["Date", "Store", "Category", "Sales"])
    return df


def explore_data(df):
    print("\n--- Dataset Overview ---")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nTotal sales by category:")
    print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))


def build_trend_visuals(df):
    monthly = df.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_sales = monthly.groupby("Month")["Sales"].sum().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Retail Sales: End-to-End Analysis Dashboard", fontsize=16, fontweight="bold")

    sns.lineplot(data=monthly_sales, x="Month", y="Sales", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Monthly Sales Trend")
    axes[0, 0].tick_params(axis="x", rotation=45)

    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=cat_sales.values, y=cat_sales.index, hue=cat_sales.index,
                ax=axes[0, 1], palette="viridis", legend=False)
    axes[0, 1].set_title("Total Sales by Category")
    axes[0, 1].set_xlabel("Sales")

    store_sales = df.groupby("Store")["Sales"].sum()
    axes[1, 0].pie(store_sales, labels=store_sales.index, autopct="%1.1f%%",
                    colors=sns.color_palette("pastel"))
    axes[1, 0].set_title("Sales Share by Store")

    df["DayOfWeek"] = df["Date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_sales = df.groupby("DayOfWeek")["Sales"].mean().reindex(dow_order)
    sns.barplot(x=dow_sales.index, y=dow_sales.values, hue=dow_sales.index,
                ax=axes[1, 1], palette="crest", legend=False)
    axes[1, 1].set_title("Average Sales by Day of Week")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "retail_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nDashboard saved to: {path}")


def build_forecast_model(df):
    model_df = df.copy()
    model_df["Month"] = model_df["Date"].dt.month
    model_df["DayOfWeekNum"] = model_df["Date"].dt.weekday
    model_df["IsWeekend"] = (model_df["DayOfWeekNum"] >= 5).astype(int)
    model_df = pd.get_dummies(model_df, columns=["Store", "Category"], drop_first=True)

    feature_cols = [c for c in model_df.columns if c not in ["Date", "Sales", "DayOfWeek"]]
    X = model_df[feature_cols]
    y = model_df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    model = RandomForestRegressor(n_estimators=200, random_state=21)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Sales Forecasting Model (Random Forest Regressor) ---")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {r2:.3f}")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color="darkorange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to: {path}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 5 features influencing sales:")
    print(importances.sort_values(ascending=False).head(5))

    return mae, r2


def main():
    df = generate_dataset()
    explore_data(df)
    build_trend_visuals(df)
    mae, r2 = build_forecast_model(df)

    print(f"""
========== FINDINGS & CONCLUSIONS ==========
- Electronics consistently generates the highest revenue among all categories.
- Weekends show a noticeable boost in average daily sales across stores.
- Seasonal demand follows a cyclical pattern across the year.
- The Random Forest forecasting model explains about {r2*100:.1f}% of the
  variance in daily sales (R^2 = {r2:.3f}), with an average prediction
  error of {mae:.2f} units, making it a reliable baseline for planning
  inventory and staffing.
================================================
""")


if __name__ == "__main__":
    main()
