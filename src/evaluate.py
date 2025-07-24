# File: src/evaluate.py

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Local imports
from infer import MRIInferenceEngine
from utils import (
    load_config, set_seed, setup_logging, get_device, create_directory
)


class MRIEvaluator:
    """
    Comprehensive evaluation suite for MRI reconstruction models.
    
    Provides detailed analysis including statistical tests, visualizations,
    and comparison metrics for different reconstruction methods.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_paths: List[str],
        model_names: List[str],
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            model_paths: List of paths to model checkpoints
            model_names: List of model names for comparison
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
        """
        self.config = config
        self.model_paths = model_paths
        self.model_names = model_names
        self.device = device
        self.output_dir = Path(output_dir)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup directories
        self._setup_directories()
        
        # Initialize inference engines
        self._setup_inference_engines()
        
        self.logger.info(f"Evaluator initialized with {len(model_paths)} models")
    
    def _setup_directories(self):
        """Setup evaluation output directories."""
        self.results_dir = self.output_dir / "evaluation_results"
        self.plots_dir = self.output_dir / "plots"
        self.comparisons_dir = self.output_dir / "comparisons"
        self.statistics_dir = self.output_dir / "statistics"
        
        for directory in [self.results_dir, self.plots_dir, self.comparisons_dir, self.statistics_dir]:
            create_directory(directory)
    
    def _setup_inference_engines(self):
        """Setup inference engines for each model."""
        self.inference_engines = []
        
        for model_path, model_name in zip(self.model_paths, self.model_names):
            engine_output_dir = self.results_dir / model_name
            
            engine = MRIInferenceEngine(
                config=self.config,
                model_path=model_path,
                device=self.device,
                output_dir=str(engine_output_dir)
            )
            
            self.inference_engines.append(engine)
    
    def run_comprehensive_evaluation(
        self,
        max_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Run comprehensive evaluation on all models.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of all model metrics
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        all_model_metrics = {}
        
        # Run inference for each model
        for engine, model_name in zip(self.inference_engines, self.model_names):
            self.logger.info(f"Evaluating model: {model_name}")
            
            metrics = engine.run_inference(
                max_samples=max_samples,
                save_all=True
            )
            
            all_model_metrics[model_name] = metrics
        
        # Save raw metrics
        self._save_all_metrics(all_model_metrics)
        
        # Generate comparative analysis
        self._generate_comparative_analysis(all_model_metrics)
        
        # Generate statistical analysis
        self._generate_statistical_analysis(all_model_metrics)
        
        # Generate visualizations
        self._generate_visualizations(all_model_metrics)
        
        # Generate performance report
        self._generate_performance_report(all_model_metrics)
        
        return all_model_metrics
    
    def _save_all_metrics(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Save all metrics to files."""
        # Save as JSON
        json_file = self.results_dir / "all_metrics.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for model_name, metrics in all_metrics.items():
            json_metrics[model_name] = {
                metric_name: [float(x) for x in values]
                for metric_name, values in metrics.items()
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        # Save as numpy
        npz_file = self.results_dir / "all_metrics.npz"
        flat_metrics = {}
        for model_name, metrics in all_metrics.items():
            for metric_name, values in metrics.items():
                flat_metrics[f"{model_name}_{metric_name}"] = np.array(values)
        
        np.savez_compressed(npz_file, **flat_metrics)
        
        self.logger.info(f"All metrics saved to {json_file} and {npz_file}")
    
    def _generate_comparative_analysis(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ):
        """Generate comparative analysis between models."""
        self.logger.info("Generating comparative analysis...")
        
        comparison_file = self.comparisons_dir / "model_comparison.txt"
        
        with open(comparison_file, 'w') as f:
            f.write("Model Comparison Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics for each model
            for model_name, metrics in all_metrics.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                for metric_name, values in metrics.items():
                    if values:  # Check if list is not empty
                        f.write(f"{metric_name.upper()}:\n")
                        f.write(f"  Mean: {np.mean(values):.6f}\n")
                        f.write(f"  Std:  {np.std(values):.6f}\n")
                        f.write(f"  Median: {np.median(values):.6f}\n")
                        f.write(f"  Min: {np.min(values):.6f}\n")
                        f.write(f"  Max: {np.max(values):.6f}\n\n")
                
                f.write("\n")
            
            # Pairwise comparisons
            if len(all_metrics) > 1:
                f.write("Pairwise Comparisons\n")
                f.write("=" * 30 + "\n\n")
                
                model_names = list(all_metrics.keys())
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names[i+1:], i+1):
                        f.write(f"{model1} vs {model2}:\n")
                        f.write("-" * 20 + "\n")
                        
                        for metric_name in ['psnr', 'ssim', 'nmse']:
                            if (metric_name in all_metrics[model1] and 
                                metric_name in all_metrics[model2]):
                                
                                values1 = all_metrics[model1][metric_name]
                                values2 = all_metrics[model2][metric_name]
                                
                                if values1 and values2:
                                    # Perform t-test
                                    t_stat, p_value = stats.ttest_rel(values1, values2)
                                    
                                    mean_diff = np.mean(values1) - np.mean(values2)
                                    
                                    f.write(f"{metric_name.upper()}:\n")
                                    f.write(f"  Mean difference: {mean_diff:.6f}\n")
                                    f.write(f"  T-statistic: {t_stat:.6f}\n")
                                    f.write(f"  P-value: {p_value:.6f}\n")
                                    f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")
                        
                        f.write("\n")
    
    def _generate_statistical_analysis(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ):
        """Generate detailed statistical analysis."""
        self.logger.info("Generating statistical analysis...")
        
        stats_file = self.statistics_dir / "statistical_analysis.txt"
        
        with open(stats_file, 'w') as f:
            f.write("Statistical Analysis\n")
            f.write("=" * 30 + "\n\n")
            
            for model_name, metrics in all_metrics.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 20 + "\n")
                
                for metric_name, values in metrics.items():
                    if values and len(values) > 1:
                        # Basic statistics
                        f.write(f"{metric_name.upper()} Distribution:\n")
                        f.write(f"  Count: {len(values)}\n")
                        f.write(f"  Mean: {np.mean(values):.6f}\n")
                        f.write(f"  Std: {np.std(values):.6f}\n")
                        f.write(f"  Variance: {np.var(values):.6f}\n")
                        f.write(f"  Skewness: {stats.skew(values):.6f}\n")
                        f.write(f"  Kurtosis: {stats.kurtosis(values):.6f}\n")
                        
                        # Percentiles
                        percentiles = [5, 25, 50, 75, 95]
                        f.write("  Percentiles:\n")
                        for p in percentiles:
                            f.write(f"    {p}th: {np.percentile(values, p):.6f}\n")
                        
                        # Normality test
                        if len(values) >= 8:  # Minimum for Shapiro-Wilk test
                            shapiro_stat, shapiro_p = stats.shapiro(values)
                            f.write(f"  Normality (Shapiro-Wilk):\n")
                            f.write(f"    Statistic: {shapiro_stat:.6f}\n")
                            f.write(f"    P-value: {shapiro_p:.6f}\n")
                            f.write(f"    Normal: {'Yes' if shapiro_p > 0.05 else 'No'}\n")
                        
                        f.write("\n")
                
                f.write("\n")
    
    def _generate_visualizations(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Box plots for each metric
        self._create_box_plots(all_metrics)
        
        # Distribution plots
        self._create_distribution_plots(all_metrics)
        
        # Correlation analysis
        self._create_correlation_plots(all_metrics)
        
        # Performance radar chart
        self._create_radar_chart(all_metrics)
        
        # Scatter plots for metric relationships
        self._create_scatter_plots(all_metrics)
    
    def _create_box_plots(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Create box plots for metric comparison."""
        metrics_to_plot = ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
            
            data_to_plot = []
            labels = []
            
            for model_name, metrics in all_metrics.items():
                if metric_name in metrics and metrics[metric_name]:
                    data_to_plot.append(metrics[metric_name])
                    labels.append(model_name)
            
            if data_to_plot:
                axes[i].boxplot(data_to_plot, labels=labels)
                axes[i].set_title(f'{metric_name.upper()} Distribution')
                axes[i].grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                if len(labels) > 3:
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metric_boxplots.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plots(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Create distribution plots for metrics."""
        metrics_to_plot = ['psnr', 'ssim', 'nmse']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        for i, metric_name in enumerate(metrics_to_plot):
            for model_name, metrics in all_metrics.items():
                if metric_name in metrics and metrics[metric_name]:
                    axes[i].hist(
                        metrics[metric_name],
                        alpha=0.7,
                        label=model_name,
                        bins=20,
                        density=True
                    )
            
            axes[i].set_title(f'{metric_name.upper()} Distribution')
            axes[i].set_xlabel(metric_name.upper())
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metric_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_plots(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Create correlation plots between metrics."""
        # Combine all models' data for correlation analysis
        combined_data = {}
        metrics_of_interest = ['psnr', 'ssim', 'nmse', 'mae', 'nrmse']
        
        for model_name, metrics in all_metrics.items():
            for metric_name in metrics_of_interest:
                if metric_name in metrics and metrics[metric_name]:
                    if metric_name not in combined_data:
                        combined_data[metric_name] = []
                    combined_data[metric_name].extend(metrics[metric_name])
        
        # Create correlation matrix
        if len(combined_data) > 1:
            # Ensure all metrics have the same length
            min_length = min(len(values) for values in combined_data.values())
            correlation_data = {
                metric: values[:min_length] 
                for metric, values in combined_data.items()
            }
            
            import pandas as pd
            df = pd.DataFrame(correlation_data)
            correlation_matrix = df.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.3f'
            )
            plt.title('Metric Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "metric_correlations.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def _create_radar_chart(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Create radar chart for model comparison."""
        try:
            metrics_for_radar = ['psnr', 'ssim']  # Metrics where higher is better
            inverse_metrics = ['nmse', 'mae']  # Metrics where lower is better
            
            # Prepare data
            model_scores = {}
            
            for model_name, metrics in all_metrics.items():
                scores = {}
                
                # Higher is better metrics (normalize to 0-1)
                for metric in metrics_for_radar:
                    if metric in metrics and metrics[metric]:
                        # Use relative scoring across models
                        all_values = []
                        for m in all_metrics.values():
                            if metric in m and m[metric]:
                                all_values.extend(m[metric])
                        
                        if all_values:
                            min_val, max_val = min(all_values), max(all_values)
                            if max_val > min_val:
                                scores[metric] = (np.mean(metrics[metric]) - min_val) / (max_val - min_val)
                            else:
                                scores[metric] = 1.0
                
                # Lower is better metrics (invert and normalize)
                for metric in inverse_metrics:
                    if metric in metrics and metrics[metric]:
                        all_values = []
                        for m in all_metrics.values():
                            if metric in m and m[metric]:
                                all_values.extend(m[metric])
                        
                        if all_values:
                            min_val, max_val = min(all_values), max(all_values)
                            if max_val > min_val:
                                # Invert: lower values get higher scores
                                scores[metric] = 1 - (np.mean(metrics[metric]) - min_val) / (max_val - min_val)
                            else:
                                scores[metric] = 1.0
                
                if scores:
                    model_scores[model_name] = scores
            
            if model_scores:
                self._plot_radar_chart(model_scores)
                
        except Exception as e:
            self.logger.warning(f"Could not create radar chart: {e}")
    
    def _plot_radar_chart(self, model_scores: Dict[str, Dict[str, float]]):
        """Plot radar chart."""
        metrics = list(next(iter(model_scores.values())).keys())
        num_metrics = len(metrics)
        
        if num_metrics < 3:
            return  # Need at least 3 metrics for a meaningful radar chart
        
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for model_name, scores in model_scores.items():
            values = [scores[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_radar.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_scatter_plots(self, all_metrics: Dict[str, Dict[str, List[float]]]):
        """Create scatter plots showing relationships between metrics."""
        metric_pairs = [('psnr', 'ssim'), ('nmse', 'mae'), ('psnr', 'nmse')]
        
        fig, axes = plt.subplots(1, len(metric_pairs), figsize=(15, 5))
        
        for i, (metric_x, metric_y) in enumerate(metric_pairs):
            for model_name, metrics in all_metrics.items():
                if (metric_x in metrics and metric_y in metrics and 
                    metrics[metric_x] and metrics[metric_y]):
                    
                    # Ensure same length
                    min_len = min(len(metrics[metric_x]), len(metrics[metric_y]))
                    x_values = metrics[metric_x][:min_len]
                    y_values = metrics[metric_y][:min_len]
                    
                    axes[i].scatter(x_values, y_values, alpha=0.6, label=model_name)
            
            axes[i].set_xlabel(metric_x.upper())
            axes[i].set_ylabel(metric_y.upper())
            axes[i].set_title(f'{metric_x.upper()} vs {metric_y.upper()}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metric_scatter_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_report(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ):
        """Generate comprehensive performance report."""
        self.logger.info("Generating performance report...")
        
        report_file = self.output_dir / "performance_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MRI Reconstruction Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric-table {{ border-collapse: collapse; width: 100%; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .best-score {{ background-color: #d4edda; font-weight: bold; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .image-item {{ text-align: center; }}
                .image-item img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MRI Reconstruction Evaluation Report</h1>
                <p>Generated on: {self._get_timestamp()}</p>
                <p>Models evaluated: {len(all_metrics)}</p>
                <p>Acceleration factor: {self.config['data']['acceleration']}</p>
                <p>Center fraction: {self.config['data']['center_fraction']}</p>
            </div>
        """
        
        # Summary table
        html_content += self._generate_summary_table_html(all_metrics)
        
        # Individual model details
        html_content += self._generate_model_details_html(all_metrics)
        
        # Visualizations
        html_content += self._generate_visualizations_html()
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Performance report saved to {report_file}")
    
    def _generate_summary_table_html(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ) -> str:
        """Generate HTML summary table."""
        html = """
        <div class="section">
            <h2>Summary Statistics</h2>
            <table class="metric-table">
                <tr>
                    <th>Model</th>
                    <th>PSNR (dB)</th>
                    <th>SSIM</th>
                    <th>NMSE</th>
                    <th>MAE</th>
                    <th>Inference Time (s)</th>
                </tr>
        """
        
        # Find best scores for highlighting
        best_scores = {}
        metrics_to_compare = ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']
        
        for metric in metrics_to_compare:
            model_means = {}
            for model_name, metrics in all_metrics.items():
                if metric in metrics and metrics[metric]:
                    model_means[model_name] = np.mean(metrics[metric])
            
            if model_means:
                if metric in ['psnr', 'ssim']:  # Higher is better
                    best_scores[metric] = max(model_means, key=model_means.get)
                else:  # Lower is better
                    best_scores[metric] = min(model_means, key=model_means.get)
        
        # Generate table rows
        for model_name, metrics in all_metrics.items():
            html += f"<tr><td>{model_name}</td>"
            
            for metric in metrics_to_compare:
                if metric in metrics and metrics[metric]:
                    mean_val = np.mean(metrics[metric])
                    std_val = np.std(metrics[metric])
                    
                    # Format based on metric type
                    if metric == 'psnr':
                        formatted = f"{mean_val:.2f} ± {std_val:.2f}"
                    elif metric in ['ssim', 'nmse', 'mae']:
                        formatted = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:  # inference_time
                        formatted = f"{mean_val:.3f} ± {std_val:.3f}"
                    
                    # Highlight best score
                    css_class = "best-score" if best_scores.get(metric) == model_name else ""
                    html += f'<td class="{css_class}">{formatted}</td>'
                else:
                    html += "<td>N/A</td>"
            
            html += "</tr>"
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _generate_model_details_html(
        self,
        all_metrics: Dict[str, Dict[str, List[float]]]
    ) -> str:
        """Generate detailed HTML for each model."""
        html = """
        <div class="section">
            <h2>Detailed Model Statistics</h2>
        """
        
        for model_name, metrics in all_metrics.items():
            html += f"""
            <h3>{model_name}</h3>
            <table class="metric-table">
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Median</th>
                    <th>Count</th>
                </tr>
            """
            
            for metric_name, values in metrics.items():
                if values:
                    html += f"""
                    <tr>
                        <td>{metric_name.upper()}</td>
                        <td>{np.mean(values):.6f}</td>
                        <td>{np.std(values):.6f}</td>
                        <td>{np.min(values):.6f}</td>
                        <td>{np.max(values):.6f}</td>
                        <td>{np.median(values):.6f}</td>
                        <td>{len(values)}</td>
                    </tr>
                    """
            
            html += "</table>"
        
        html += "</div>"
        
        return html
    
    def _generate_visualizations_html(self) -> str:
        """Generate HTML section for visualizations."""
        html = """
        <div class="section">
            <h2>Visualizations</h2>
            <div class="image-grid">
        """
        
        # List of visualization files to include
        viz_files = [
            ("metric_boxplots.png", "Metric Box Plots"),
            ("metric_distributions.png", "Metric Distributions"),
            ("metric_correlations.png", "Metric Correlations"),
            ("performance_radar.png", "Performance Radar Chart"),
            ("metric_scatter_plots.png", "Metric Scatter Plots")
        ]
        
        for filename, title in viz_files:
            file_path = self.plots_dir / filename
            if file_path.exists():
                # Use relative path for HTML
                rel_path = f"plots/{filename}"
                html += f"""
                <div class="image-item">
                    <h4>{title}</h4>
                    <img src="{rel_path}" alt="{title}">
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def compare_with_baseline_methods(
        self,
        baseline_results: Dict[str, Dict[str, List[float]]]
    ):
        """
        Compare neural operator results with baseline methods.
        
        Args:
            baseline_results: Dictionary containing baseline method results
                             Format: {method_name: {metric_name: [values]}}
        """
        self.logger.info("Comparing with baseline methods...")
        
        # Combine neural operator and baseline results
        all_results = {}
        
        # Add neural operator results
        for i, (engine, model_name) in enumerate(zip(self.inference_engines, self.model_names)):
            # Assuming results are already computed and stored
            results_file = engine.output_dir / "all_metrics.npz"
            if results_file.exists():
                data = np.load(results_file)
                all_results[model_name] = {
                    key: data[key].tolist() for key in data.keys()
                }
        
        # Add baseline results
        all_results.update(baseline_results)
        
        # Generate comparison analysis
        self._generate_comparative_analysis(all_results)
        self._generate_visualizations(all_results)
        
        # Save comparison report
        comparison_file = self.comparisons_dir / "baseline_comparison.txt"
        
        with open(comparison_file, 'w') as f:
            f.write("Neural Operator vs Baseline Methods Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance improvement analysis
            for model_name in self.model_names:
                if model_name in all_results:
                    f.write(f"Neural Operator Model: {model_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    for baseline_name, baseline_metrics in baseline_results.items():
                        f.write(f"\nComparison with {baseline_name}:\n")
                        
                        for metric in ['psnr', 'ssim', 'nmse']:
                            if (metric in all_results[model_name] and 
                                metric in baseline_metrics):
                                
                                no_values = all_results[model_name][metric]
                                baseline_values = baseline_metrics[metric]
                                
                                if no_values and baseline_values:
                                    no_mean = np.mean(no_values)
                                    baseline_mean = np.mean(baseline_values)
                                    
                                    if metric in ['psnr', 'ssim']:  # Higher is better
                                        improvement = ((no_mean - baseline_mean) / baseline_mean) * 100
                                    else:  # Lower is better (nmse)
                                        improvement = ((baseline_mean - no_mean) / baseline_mean) * 100
                                    
                                    f.write(f"  {metric.upper()}: {improvement:+.2f}% improvement\n")
                    
                    f.write("\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive MRI Reconstruction Evaluation")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for the models')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--baseline-results', type=str, default=None,
                       help='Path to baseline results file (JSON format)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.models) != len(args.names):
        raise ValueError("Number of models and names must match")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device and seed
    device = get_device(args.device)
    set_seed(config['system']['seed'])
    
    # Setup logging
    create_directory(args.output)
    setup_logging(
        os.path.join(args.output, 'logs'),
        config['system']['log_level']
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comprehensive evaluation on {device}")
    logger.info(f"Models: {args.names}")
    logger.info(f"Output directory: {args.output}")
    
    # Create evaluator
    evaluator = MRIEvaluator(
        config=config,
        model_paths=args.models,
        model_names=args.names,
        device=device,
        output_dir=args.output
    )
    
    # Run evaluation
    try:
        all_metrics = evaluator.run_comprehensive_evaluation(
            max_samples=args.max_samples
        )
        
        # Compare with baseline methods if provided
        if args.baseline_results:
            with open(args.baseline_results, 'r') as f:
                baseline_results = json.load(f)
            
            evaluator.compare_with_baseline_methods(baseline_results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()