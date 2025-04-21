import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import files
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class NsightAnalyzer:
    def __init__(self, sqlite_file):
        """Initialize the analyzer with the SQLite file."""
        self.conn = sqlite3.connect(sqlite_file)
        self.cursor = self.conn.cursor()
        self.report_name = os.path.basename(sqlite_file).split('.')[0]
        
    def close(self):
        """Close the database connection."""
        self.conn.close()
        
    def execute_query(self, query):
        """Execute a SQL query and return results as a DataFrame."""
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            return pd.DataFrame()
    
    def get_cuda_memcpy_summary_by_size(self):
        """Get summary of CUDA memory operations by size."""
        query = """
        SELECT 
            CASE
                WHEN copyKind = 1 THEN 'Host to Device'
                WHEN copyKind = 2 THEN 'Device to Host'
                WHEN copyKind = 8 THEN 'Device to Device'
                ELSE 'Other'
            END as Direction,
            COUNT(*) as Count,
            SUM(bytes) as TotalBytes,
            AVG(bytes) as AvgBytes,
            MIN(bytes) as MinBytes,
            MAX(bytes) as MaxBytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
        ORDER BY TotalBytes DESC
        """
        return self.execute_query(query)
    
    def get_cuda_memcpy_summary_by_time(self):
        """Get summary of CUDA memory operations by time."""
        query = """
        SELECT 
            CASE
                WHEN copyKind = 1 THEN 'Host to Device'
                WHEN copyKind = 2 THEN 'Device to Host'
                WHEN copyKind = 8 THEN 'Device to Device'
                ELSE 'Other'
            END as Direction,
            COUNT(*) as Count,
            SUM(end - start) as TotalTime_ns,
            AVG(end - start) as AvgTime_ns,
            MIN(end - start) as MinTime_ns,
            MAX(end - start) as MaxTime_ns,
            SUM(bytes) as TotalBytes,
            CASE WHEN SUM(end - start) > 0 
                THEN (SUM(bytes) / (SUM(end - start) / 1e9)) / 1e9
                ELSE 0 
            END as Bandwidth_GBps
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
        ORDER BY TotalTime_ns DESC
        """
        return self.execute_query(query)
    
    def get_cuda_kernel_summary(self):
        """Get summary of CUDA kernel executions."""
        query = """
        SELECT 
            s.value as KernelName,
            COUNT(*) as Invocations,
            SUM(end - start) as TotalTime_ns,
            AVG(end - start) as AvgTime_ns,
            MIN(end - start) as MinTime_ns,
            MAX(end - start) as MaxTime_ns,
            AVG(registersPerThread) as AvgRegistersPerThread,
            MAX(gridX) as GridX, 
            MAX(gridY) as GridY, 
            MAX(gridZ) as GridZ,
            MAX(blockX) as BlockX, 
            MAX(blockY) as BlockY, 
            MAX(blockZ) as BlockZ,
            MAX(gridX) * MAX(blockX) * MAX(gridY) * MAX(blockY) * MAX(gridZ) * MAX(blockZ) as TotalThreads
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        GROUP BY s.value
        ORDER BY TotalTime_ns DESC
        """
        return self.execute_query(query)
    
    def get_kernel_execution_times(self):
        """Get execution times for each kernel invocation."""
        query = """
        SELECT 
            s.value as KernelName,
            (end - start) / 1000000.0 as ExecutionTime_ms,
            gridX, gridY, gridZ, blockX, blockY, blockZ,
            gridX * blockX * gridY * blockY * gridZ * blockZ as TotalThreads
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY start
        """
        return self.execute_query(query)
    
    def get_cuda_api_summary(self):
        """Get summary of CUDA API calls."""
        query = """
        SELECT 
            s.value as API_Name,
            COUNT(*) as Count,
            SUM(end - start) as TotalTime_ns,
            AVG(end - start) as AvgTime_ns,
            MIN(end - start) as MinTime_ns,
            MAX(end - start) as MaxTime_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY s.value
        ORDER BY TotalTime_ns DESC
        """
        return self.execute_query(query)
    
    def get_total_runtime(self):
        """Get total application runtime."""
        query = """
        SELECT 
            MIN(start) as StartTime,
            MAX(end) as EndTime,
            MAX(end) - MIN(start) as TotalRuntime_ns
        FROM (
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME
            UNION ALL
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
            UNION ALL
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
        )
        """
        return self.execute_query(query)
    
    def generate_report(self, output_path=None):
        """Generate a comprehensive report and save to Excel."""
        if output_path is None:
            output_path = f"{self.report_name}_analysis.xlsx"
            
        # Create a Pandas Excel writer using openpyxl
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Get all report data
        memcpy_size = self.get_cuda_memcpy_summary_by_size()
        memcpy_time = self.get_cuda_memcpy_summary_by_time()
        kernel_summary = self.get_cuda_kernel_summary()
        kernel_times = self.get_kernel_execution_times()
        api_summary = self.get_cuda_api_summary()
        runtime = self.get_total_runtime()
        
        # Clean up and enhance data
        if not memcpy_size.empty:
            memcpy_size['TotalMB'] = memcpy_size['TotalBytes'] / (1024 * 1024)
            
        # Convert nanoseconds to milliseconds for better readability
        for df in [memcpy_time, kernel_summary, api_summary]:
            if not df.empty:
                time_columns = [col for col in df.columns if '_ns' in col]
                for col in time_columns:
                    new_col = col.replace('_ns', '_ms')
                    df[new_col] = df[col] / 1000000.0
                    df.drop(col, axis=1, inplace=True)
        
        # Calculate total runtime in ms
        total_runtime_ms = 0
        if not runtime.empty:
            total_runtime_ms = runtime.iloc[0]['TotalRuntime_ns'] / 1000000.0
        
        # Extract kernel grid and block info
        grid_dims = "N/A"
        block_dims = "N/A"
        total_threads = 0
        
        if not kernel_summary.empty:
            # Get the dimensions from the kernel that took the most time
            main_kernel = kernel_summary.iloc[0]
            grid_dims = f"{main_kernel['GridX']} x {main_kernel['GridY']} x {main_kernel['GridZ']}"
            block_dims = f"{main_kernel['BlockX']} x {main_kernel['BlockY']} x {main_kernel['BlockZ']}"
            total_threads = main_kernel['TotalThreads']
            
        # Create summary sheet
        summary_data = {
            'Metric': [
                'Total Runtime (ms)',
                'Kernel Invocations',
                'Total Kernel Time (ms)',
                'Kernel Time Percentage',
                'Total Memory Transfer (MB)',
                'Host to Device Transfer (MB)',
                'Device to Host Transfer (MB)',
                'Memory Transfer Time (ms)',
                'Memory Transfer Percentage',
                'Grid Dimensions', 
                'Thread Block Dimensions',
                'Total Threads'
            ],
            'Value': [
                f"{total_runtime_ms:.2f}",
                kernel_summary['Invocations'].sum() if not kernel_summary.empty else 0,
                kernel_summary['TotalTime_ms'].sum() if not kernel_summary.empty else 0,
                f"{(kernel_summary['TotalTime_ms'].sum() / total_runtime_ms * 100):.2f}%" if total_runtime_ms > 0 and not kernel_summary.empty else "0%",
                f"{memcpy_size['TotalMB'].sum():.2f}" if not memcpy_size.empty else 0,
                f"{memcpy_size[memcpy_size['Direction'] == 'Host to Device']['TotalMB'].sum():.2f}" if not memcpy_size.empty and 'Host to Device' in memcpy_size['Direction'].values else 0,
                f"{memcpy_size[memcpy_size['Direction'] == 'Device to Host']['TotalMB'].sum():.2f}" if not memcpy_size.empty and 'Device to Host' in memcpy_size['Direction'].values else 0,
                f"{memcpy_time['TotalTime_ms'].sum():.2f}" if not memcpy_time.empty else 0,
                f"{(memcpy_time['TotalTime_ms'].sum() / total_runtime_ms * 100):.2f}%" if total_runtime_ms > 0 and not memcpy_time.empty else "0%",
                grid_dims,
                block_dims,
                total_threads
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save all dataframes to Excel sheets
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        if not memcpy_size.empty:
            memcpy_size.to_excel(writer, sheet_name='MemOps_BySize', index=False)
        
        if not memcpy_time.empty:
            memcpy_time.to_excel(writer, sheet_name='MemOps_ByTime', index=False)
        
        if not kernel_summary.empty:
            kernel_summary.to_excel(writer, sheet_name='Kernel_Summary', index=False)
        
        if not kernel_times.empty:
            kernel_times.to_excel(writer, sheet_name='Kernel_Executions', index=False)
        
        if not api_summary.empty:
            api_summary.to_excel(writer, sheet_name='API_Summary', index=False)
        
        # Save the Excel file
        writer.close()
        
        # Generate plots
        fig_directory = f"{self.report_name}_figures"
        self.generate_visualizations(fig_directory, memcpy_size, memcpy_time, kernel_summary, kernel_times)
        
        return {
            'summary': summary_df,
            'report_file': output_path,
            'fig_directory': fig_directory
        }
    
    def generate_visualizations(self, fig_directory, memcpy_size, memcpy_time, kernel_summary, kernel_times):
        """Generate visualizations based on the collected data."""
        os.makedirs(fig_directory, exist_ok=True)
        
        # Set style for all plots
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Plot 1: Memory operations by size
        if not memcpy_size.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Direction', y='TotalMB', data=memcpy_size)
            plt.title('CUDA Memory Operations by Size')
            plt.ylabel('Total Size (MB)')
            plt.xlabel('Transfer Direction')
            
            # Add values on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f} MB", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/memops_size.png", dpi=300)
            plt.close()
        
        # Plot 2: Memory operations by time
        if not memcpy_time.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Direction', y='TotalTime_ms', data=memcpy_time)
            plt.title('CUDA Memory Operations by Time')
            plt.ylabel('Total Time (ms)')
            plt.xlabel('Transfer Direction')
            
            # Add values on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f} ms", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/memops_time.png", dpi=300)
            plt.close()
        
        # Plot 3: Memory bandwidth
        if not memcpy_time.empty and 'Bandwidth_GBps' in memcpy_time.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Direction', y='Bandwidth_GBps', data=memcpy_time)
            plt.title('CUDA Memory Bandwidth')
            plt.ylabel('Bandwidth (GB/s)')
            plt.xlabel('Transfer Direction')
            
            # Add values on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f} GB/s", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/memops_bandwidth.png", dpi=300)
            plt.close()
        
        # Plot 4: Kernel execution times
        if not kernel_summary.empty:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='KernelName', y='TotalTime_ms', data=kernel_summary)
            plt.title('CUDA Kernel Execution Time')
            plt.ylabel('Total Time (ms)')
            plt.xlabel('Kernel Name')
            plt.xticks(rotation=45, ha='right')
            
            # Add values on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f} ms", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/kernel_time.png", dpi=300)
            plt.close()
        
        # Plot 5: Kernel invocations
        if not kernel_summary.empty:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='KernelName', y='Invocations', data=kernel_summary)
            plt.title('CUDA Kernel Invocations')
            plt.ylabel('Number of Invocations')
            plt.xlabel('Kernel Name')
            plt.xticks(rotation=45, ha='right')
            
            # Add values on top of bars
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height())}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/kernel_invocations.png", dpi=300)
            plt.close()
            
        # Plot 6: Time distribution for multiple kernel invocations
        if not kernel_times.empty and len(kernel_times) > 1:
            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(x='KernelName', y='ExecutionTime_ms', data=kernel_times)
            plt.title('Distribution of Kernel Execution Times')
            plt.ylabel('Execution Time (ms)')
            plt.xlabel('Kernel Name')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/kernel_time_distribution.png", dpi=300)
            plt.close()
            
        # Plot 7: Execution timeline
        if not kernel_times.empty:
            kernel_times = kernel_times.sort_values('ExecutionTime_ms', ascending=False)
            plt.figure(figsize=(12, 6))
            ax = sns.stripplot(x='KernelName', y='ExecutionTime_ms', data=kernel_times, 
                              jitter=True, size=8, palette='viridis')
            plt.title('Kernel Execution Times (Each point is one invocation)')
            plt.ylabel('Execution Time (ms)')
            plt.xlabel('Kernel Name')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{fig_directory}/kernel_execution_points.png", dpi=300)
            plt.close()

def compare_profiling_results(sequential_file, parallel_file, output_prefix="comparison"):
    """Compare sequential and parallel implementations profiling results."""
    # Load data from both files
    seq_analyzer = NsightAnalyzer(sequential_file)
    par_analyzer = NsightAnalyzer(parallel_file)
    
    # Get summary data
    seq_runtime = seq_analyzer.get_total_runtime()
    par_runtime = par_analyzer.get_total_runtime()
    
    seq_total_time = seq_runtime.iloc[0]['TotalRuntime_ns'] / 1000000.0 if not seq_runtime.empty else 0
    par_total_time = par_runtime.iloc[0]['TotalRuntime_ns'] / 1000000.0 if not par_runtime.empty else 0
    
    # Calculate speedup
    speedup = seq_total_time / par_total_time if par_total_time > 0 else 0
    
    # Get kernel data
    seq_kernel = seq_analyzer.get_cuda_kernel_summary()
    par_kernel = par_analyzer.get_cuda_kernel_summary()
    
    seq_kernel_time = seq_kernel['TotalTime_ms'].sum() if not seq_kernel.empty else 0
    par_kernel_time = par_kernel['TotalTime_ms'].sum() if not par_kernel.empty else 0
    
    # Get memory operations data
    seq_memcpy = seq_analyzer.get_cuda_memcpy_summary_by_time()
    par_memcpy = par_analyzer.get_cuda_memcpy_summary_by_time()
    
    seq_memcpy_time = seq_memcpy['TotalTime_ms'].sum() if not seq_memcpy.empty else 0
    par_memcpy_time = par_memcpy['TotalTime_ms'].sum() if not par_memcpy.empty else 0
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': [
            'Total Runtime (ms)',
            'Kernel Execution Time (ms)',
            'Memory Transfer Time (ms)',
            'Speedup Factor'
        ],
        'Sequential': [
            f"{seq_total_time:.2f}",
            f"{seq_kernel_time:.2f}",
            f"{seq_memcpy_time:.2f}",
            'N/A'
        ],
        'Parallel': [
            f"{par_total_time:.2f}",
            f"{par_kernel_time:.2f}",
            f"{par_memcpy_time:.2f}",
            f"{speedup:.2f}x"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to Excel
    comparison_df.to_excel(f"{output_prefix}_comparison.xlsx", index=False)
    
    # Generate comparison visualizations
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    metrics = ['Total Runtime', 'Kernel Time', 'Memory Time']
    seq_values = [seq_total_time, seq_kernel_time, seq_memcpy_time]
    par_values = [par_total_time, par_kernel_time, par_memcpy_time]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, seq_values, width, label='Sequential')
    rects2 = ax.bar(x + width/2, par_values, width, label='Parallel')
    
    # Add labels and titles
    ax.set_ylabel('Time (ms)')
    ax.set_title('Sequential vs Parallel Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Add speedup text
    plt.text(0.5, 0.95, f"Speedup: {speedup:.2f}x", 
             horizontalalignment='center',
             verticalalignment='center', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=300)
    plt.close()
    
    # Close connections
    seq_analyzer.close()
    par_analyzer.close()
    
    return comparison_df

def analyze_scalability(data_sizes, sequential_times, parallel_times, thread_counts=None, output_prefix="scalability"):
    """Analyze scalability with different input sizes and thread configurations."""
    # Create DataFrame
    scalability_data = {
        'Data Size': data_sizes,
        'Sequential Time (ms)': sequential_times,
        'Parallel Time (ms)': parallel_times,
        'Speedup': [s/p if p > 0 else 0 for s, p in zip(sequential_times, parallel_times)]
    }
    
    if thread_counts is not None:
        scalability_data['Thread Count'] = thread_counts
    
    scalability_df = pd.DataFrame(scalability_data)
    
    # Save to Excel
    scalability_df.to_excel(f"{output_prefix}_scalability.xlsx", index=False)
    
    # Plot speedup vs data size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Data Size', y='Speedup', data=scalability_df, marker='o', linewidth=2)
    plt.title('Speedup vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Speedup Factor (Sequential / Parallel)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_speedup_vs_size.png", dpi=300)
    plt.close()
    
    # Plot execution time vs data size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Data Size', y='Sequential Time (ms)', data=scalability_df, marker='o', label='Sequential')
    sns.lineplot(x='Data Size', y='Parallel Time (ms)', data=scalability_df, marker='s', label='Parallel')
    plt.title('Execution Time vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_time_vs_size.png", dpi=300)
    plt.close()
    
    # If thread counts are provided, plot speedup vs thread count
    if thread_counts is not None:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Thread Count', y='Speedup', data=scalability_df, marker='o', linewidth=2)
        plt.title('Speedup vs Thread Count')
        plt.xlabel('Thread Count')
        plt.ylabel('Speedup Factor')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_speedup_vs_threads.png", dpi=300)
        plt.close()
    
    return scalability_df

def upload_sqlite_file():
    """Upload SQLite file through Google Colab."""
    print("Please upload your SQLite file generated by Nsight Systems...")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file was uploaded.")
        return None
    
    # Get the first uploaded file
    filename = list(uploaded.keys())[0]
    
    if not filename.endswith('.sqlite'):
        print(f"Warning: {filename} might not be a SQLite file. Attempting to process anyway.")
    
    return filename

def main():
    """Main function for the Nsight analyzer in Google Colab."""
    print("NVIDIA Nsight Systems SQLite Analyzer")
    print("=====================================")
    print("This tool analyzes SQLite files generated by Nsight Systems profiler.")
    
    mode = input("Choose analysis mode:\n1. Analyze a single SQLite file\n2. Compare sequential and parallel implementations\n3. Analyze scalability with different input sizes\nEnter choice (1-3): ")
    
    if mode == "1":
        # Single file analysis
        sqlite_file = upload_sqlite_file()
        if sqlite_file:
            analyzer = NsightAnalyzer(sqlite_file)
            report = analyzer.generate_report()
            
            print(f"\nAnalysis completed successfully!")
            print(f"Report saved to: {report['report_file']}")
            print(f"Visualizations saved to: {report['fig_directory']}")
            
            # Display summary table
            print("\nSummary Results:")
            print("----------------")
            print(report['summary'].to_string(index=False))
            
            # Option to download the Excel report
            print("\nWould you like to download the Excel report? (y/n)")
            if input().lower() in ['y', 'yes']:
                files.download(report['report_file'])
            
            analyzer.close()
    
    elif mode == "2":
        # Compare sequential and parallel implementations
        print("\nPlease upload the SQLite file for the sequential implementation:")
        seq_file = upload_sqlite_file()
        
        if not seq_file:
            print("No sequential file uploaded. Exiting.")
            return
            
        print("\nPlease upload the SQLite file for the parallel implementation:")
        par_file = upload_sqlite_file()
        
        if not par_file:
            print("No parallel file uploaded. Exiting.")
            return
        
        output_prefix = input("\nEnter a prefix for the output files (default: comparison): ") or "comparison"
        
        comparison = compare_profiling_results(seq_file, par_file, output_prefix)
        
        print("\nComparison Results:")
        print("------------------")
        print(comparison.to_string(index=False))
        
        print(f"\nComparison saved to {output_prefix}_comparison.xlsx")
        print(f"Visualization saved to {output_prefix}_comparison.png")
        
        # Option to download the Excel report
        print("\nWould you like to download the Excel report? (y/n)")
        if input().lower() in ['y', 'yes']:
            files.download(f"{output_prefix}_comparison.xlsx")
    
    elif mode == "3":
        # Analyze scalability
        print("\nPlease provide data for scalability analysis:")
        
        # Get data sizes
        data_sizes_str = input("Enter data sizes (comma-separated): ")
        data_sizes = [int(x.strip()) for x in data_sizes_str.split(",")]
        
        # Get sequential times
        seq_times_str = input("Enter sequential times in ms (comma-separated): ")
        seq_times = [float(x.strip()) for x in seq_times_str.split(",")]
        
        # Get parallel times
        par_times_str = input("Enter parallel times in ms (comma-separated): ")
        par_times = [float(x.strip()) for x in par_times_str.split(",")]
        
        # Check if thread counts should be included
        include_threads = input("Include thread counts in analysis? (y/n): ").lower() in ['y', 'yes']
        thread_counts = None
        
        if include_threads:
            thread_counts_str = input("Enter thread counts (comma-separated): ")
            thread_counts = [int(x.strip()) for x in thread_counts_str.split(",")]
        
        output_prefix = input("Enter a prefix for the output files (default: scalability): ") or "scalability"
        
        # Validate input lengths
        if len(data_sizes) != len(seq_times) or len(data_sizes) != len(par_times) or (include_threads and len(data_sizes) != len(thread_counts)):
            print("Error: All input lists must have the same length.")
            return
        
        scalability = analyze_scalability(data_sizes, seq_times, par_times, thread_counts, output_prefix)
        
        print("\nScalability Analysis Results:")
        print("----------------------------")
        print(scalability.to_string(index=False))
        
        print(f"\nScalability data saved to {output_prefix}_scalability.xlsx")
        print(f"Visualizations saved as {output_prefix}_*.png")
        
        # Option to download the Excel report
        print("\nWould you like to download the Excel report? (y/n)")
        if input().lower() in ['y', 'yes']:
            files.download(f"{output_prefix}_scalability.xlsx")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()