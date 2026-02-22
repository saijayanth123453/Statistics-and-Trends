from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Generates a Scatter Plot with a Regression line showing growth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Sort and prepare dates for numeric regression
    df_plot = df.sort_values('Evaluation_Date')
    x_days = (df_plot['Evaluation_Date'] - df_plot['Evaluation_Date'].min()).dt.days
    ax.scatter(df_plot['Evaluation_Date'], df_plot['Numerical_Result'], 
               alpha=0.5, color='#34495e', label='Evaluations')
    # Linear Regression
    m, b = np.polyfit(x_days, df_plot['Numerical_Result'], 1)
    ax.plot(df_plot['Evaluation_Date'], m*x_days + b, color='#e67e22', linewidth=3)
    ax.set_title("AI Engineering Performance Trajectory")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Benchmark Score")
    plt.savefig("relational_plot.png")
    plt.close()


def plot_categorical_plot(df):
    """Creates a Lollipop Chart ranking mean performance per model."""
    avg_scores = df.groupby('Model')['Numerical_Result'].mean().sort_values()
    plt.figure(figsize=(10, 10))
    plt.hlines(y=avg_scores.index, xmin=0, xmax=avg_scores.values, color='skyblue')
    plt.plot(avg_scores.values, avg_scores.index, "o", color='#c0392b')
    plt.title("Model Performance Leaderboard")
    plt.xlabel("Average Numerical Result")
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close()


def plot_statistical_plot(df):
    """Generates a Corner Plot to visualize distribution and covariance."""
    data_to_plot = df[['Numerical_Result', 'Task_Version']]
    _ = corner(data_to_plot, labels=['Numerical Result', 'Task Version'], 
                 color='darkslateblue')
    plt.savefig("statistical_plot.png")
    plt.close()


def statistical_analysis(df, col):
    """Calculates the four statistical moments for the specified column."""
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
    
    # CodeGrade requires these exact prints to verify your data cleaning
    print("Head:\n", df.head())
    print("\nCorr:\n", df.select_dtypes(include=[np.number]).corr())
    return df


def writing(moments, col):
    """Prints the statistical findings and interpretations."""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew_str = "right" if moments[2] > 0.5 else "left" if moments[2] < -0.5 else "not"
    kurt_str = "leptokurtic" if moments[3] > 0.5 else "platykurtic" if moments[3] < -0.5 else "mesokurtic"
    print(f'The data was {skew_str} skewed and {kurt_str}.')


if __name__ == "__main__":
    # Ensure this filename matches the dataset in your repository
    filename = 'data.csv'
    try:
        raw_df = pd.read_csv(filename)
        processed_df = preprocessing(raw_df)
        # Attribute for analysis
        target_column = 'Numerical_Result'
        results = statistical_analysis(processed_df, target_column)
        writing(results, target_column)
        # Generate and save the three plots
        plot_relational_plot(processed_df)
        plot_categorical_plot(processed_df)
        plot_statistical_plot(processed_df)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")

