"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Generates a Scatter Plot with a Trend Line showing performance over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepping date for regression (needs to be numeric)
    df_plot = df.sort_values('Evaluation_Date')
    x_days = (df_plot['Evaluation_Date'] - df_plot['Evaluation_Date'].min()).dt.days
    
    # Scatter points
    ax.scatter(df_plot['Evaluation_Date'], df_plot['Numerical_Result'], 
               alpha=0.5, color='#34495e', label='Individual Evaluations')
    
    # Regression Line (Trend)
    m, b = np.polyfit(x_days, df_plot['Numerical_Result'], 1)
    ax.plot(df_plot['Evaluation_Date'], m*x_days + b, color='#e67e22', linewidth=3, label='Performance Trend')
    
    ax.set_title('AI Performance Trajectory in WWTP Benchmarking', fontsize=12, fontweight='bold')
    ax.set_xlabel('Evaluation Date')
    ax.set_ylabel('Numerical Score')
    ax.legend()
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    """Generates a Lollipop Chart for model performance comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_scores = df.groupby('Model')['Numerical_Result'].mean().sort_values()
    
    ax.hlines(y=avg_scores.index, xmin=0, xmax=avg_scores.values, color='lightgrey', linewidth=2)
    ax.plot(avg_scores.values, avg_scores.index, "o", color='#e74c3c', markersize=8)
    
    ax.set_title('Model Efficiency Comparison (Lollipop Analysis)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Numerical Result')
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """Generates a Corner Plot to visualize distributions and correlations."""
    cols_for_corner = ['Task_Version', 'Numerical_Result']
    data_for_corner = df[cols_for_corner].values
    
    fig = corner(data_for_corner, labels=cols_for_corner, color='#3498db', 
                 show_titles=True, title_kwargs={"fontsize": 12})
    
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    """Calculates the 4 main statistical moments."""
    data = df[col]
    mean = data.mean()
    stddev = data.std()
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Cleans data and formats dates for regression analysis."""
    df = df.dropna(subset=['Numerical_Result']).copy()
    df['Evaluation_Date'] = pd.to_datetime(df['Evaluation_Date'], dayfirst=True)
    # Print diagnostics as required by template
    print("Head:\n", df.head())
    print("\nCorr:\n", df.select_dtypes(include=[np.number]).corr())
    return df


def writing(moments, col):
    """Prints the analysis findings for the report."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    skew_str = "right" if moments[2] > 0.5 else "left" if moments[2] < -0.5 else "not"
    kurt_str = "leptokurtic" if moments[3] > 0.5 else "platykurtic" if moments[3] < -0.5 else "mesokurtic"
    print(f'The data was {skew_str} skewed and {kurt_str}.')
    return


def main():
    """Main execution entry point."""
    filename = 'data.csv'
    df = pd.read_csv(filename)
    df = preprocessing(df)
    
    target_col = 'Numerical_Result'
    moments = statistical_analysis(df, target_col)
    writing(moments, target_col)
    
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)


if __name__ == "__main__":
    main()