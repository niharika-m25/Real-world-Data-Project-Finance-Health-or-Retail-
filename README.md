# Real-world-Data-Project-Finance-Health-or-Retail-
Work on a domain-specific dataset for applied learning.

Title
Retail Sales Analysis and Demand Prediction

Objective
The objective of this project was to analyze retail sales data and predict future demand patterns to support better business decision-making.

Tools Used
Python
Pandas
Scikit-learn
Matplotlib
Seaborn

Dataset Description
The dataset included:
Order ID
Product Category
Sales Amount
Quantity Sold
Customer Location
Order Date

Project Workflow

Step 1: Data Cleaning
Removed duplicate records
Handled missing values
Converted date columns into datetime format

Step 2: Data Analysis
Calculated monthly sales trends
Identified top-selling products
Analyzed regional sales performance

Step 3: Data Visualization
Monthly sales line chart
Product category bar chart
Regional sales pie chart

Step 4: Demand Prediction
Applied Linear Regression to predict future sales demand.

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

Key Findings
Electronics and fashion products contributed the highest revenue.
Sales peaked during festival seasons.
Metropolitan cities generated the majority of orders.
The prediction model achieved approximately 85% accuracy.

Conclusion
This project demonstrated how data science techniques can be applied in real-world business scenarios. Through data cleaning, analysis, visualization, and predictive modeling, I gained practical experience in solving data-driven problems and presenting actionable insights.
