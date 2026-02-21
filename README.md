# Statistics and Trends Assignment
This is the template repository for the statistics and trends assignment.
You should create a personal repository from this template repository
(there is a green box with a link in the top right).
Ensure that the file `statistics_and_trends.py` is present and functional, with your downloaded data
in the `data.csv` file.

WWTP Engineering AI Benchmark Analysis
Project Overview
This project evaluates the performance of various Artificial Intelligence models on specialized engineering tasks within the Wastewater Treatment Plant (WWTP) domain. Using the mehmetisik_wwtp-engineering-benchmark dataset, the analysis applies statistical moments and data visualization to identify trends in model reliability and accuracy.

Features
Data Preprocessing: Automated cleaning of missing values and conversion of temporal data for time-series analysis.
Statistical Analysis: Calculation of Mean, Standard Deviation, Skewness, and Excess Kurtosis to quantify model performance.
Visualizations: * Relational Plot: A scatter plot with linear regression tracking AI performance evolution over time.
Categorical Plot: A lollipop chart comparing mean scores across different model architectures like DeepSeek-R1 and Claude 3.5 Sonnet.
Statistical Plot: A corner matrix showing data distribution and bivariate relationships.

Key Statistical Findings
Based on the analysis of the Numerical_Result attribute:
Mean: 4.39 
Skewness: 3.16 (Highly Positive/Right-Skewed) 
Excess Kurtosis: 8.66 (Leptokurtic/Fat-Tailed) 
The results indicate that while the average performance is low, "elite" outlier models drive the top end of the scale, suggesting an "all or nothing" performance landscape in current engineering AI.

Installation & Usage

Install dependencies:
pip install -r requirements.txt

Run the analysis:
python statistics_and_trends.py

Dependencies
This project requires the following Python libraries:
pandas
matplotlib
numpy
scipy

seaborn

corner
