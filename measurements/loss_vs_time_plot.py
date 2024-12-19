import pandas as pd
import matplotlib.pyplot as plt

def plot_optimization_graph(csv_file_path):
    # Load the data from a CSV file, splitting columns by the semicolon delimiter
    data = pd.read_csv(csv_file_path, sep=';')

    # Rename the columns for clarity
    data.columns = ['source_data', 'config_id', 'out_data', 'time', 'Loss']

    # Ensure the required columns ('time', 'config_id', 'Loss') are present
    required_columns = {'time', 'config_id', 'Loss'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Create a new column 'OptimizationType' by formatting the 'config_id' column
    # Replace underscores with spaces and capitalize each word
    data['OptimizationType'] = data['config_id'].str.replace('_', ' ').str.title()

    # Convert 'time' and 'Loss' columns to numeric, coercing errors to NaN
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['Loss'] = pd.to_numeric(data['Loss'], errors='coerce')

    # Begin plotting with matplotlib
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Iterate through each unique optimization type to plot them separately
    for opt_type, color in zip(data['OptimizationType'].unique(), ['orange', 'blue', 'green']):
        # Filter the data for the current optimization type
        subset = data[data['OptimizationType'] == opt_type]

        # Scatter plot for the current optimization type
        plt.scatter(
            subset['time'],  # X-axis: time
            subset['Loss'],  # Y-axis: loss
            label=opt_type,  # Legend label
            s=100,  # Marker size
            color=color
        )

        # Calculate mean and variance for each optimization type
        mean_loss = subset['Loss'].mean()

        # Plot the mean as a horizontal line
        plt.axhline(
            y=mean_loss,  # Horizontal line at mean loss
            color=color, linestyle='solid', linewidth=1, label=f"{opt_type} Mean"
        )

    # Set both axes to a logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Add titles and labels to the plot
    plt.title("Loss vs Time by Optimization Type", fontsize=16)  # Plot title
    plt.xlabel("Time (log scale)", fontsize=14)  # X-axis label
    plt.ylabel("Loss (log scale)", fontsize=14)  # Y-axis label

    # Add a legend to distinguish optimization types and their statistics, fixed in the top-right corner
    plt.legend(fontsize=12, title_fontsize=13, 
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               borderaxespad=0.5)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

# Example usage
# Replace 'data.csv' with the path to your CSV file
csv_file_path = 'measurements/results_dual_annealing_main.csv'  # Update with your actual file path
plot_optimization_graph(csv_file_path)
