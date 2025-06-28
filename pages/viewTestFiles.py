import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
# import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = "./results"
CACHE_DURATION = 3600  # 1 hour cache

class DataLoader:
    """Efficient data loading with caching and error handling"""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self._file_cache = {}
        self._data_cache = {}
    
    @st.cache_data(ttl=CACHE_DURATION)
    def load_file_structure(_self) -> Dict[str, List[str]]:
        """Load and cache file structure efficiently"""
        files = {}
        if not _self.results_dir.exists():
            logger.error(f"Results directory {_self.results_dir} does not exist")
            return files
        
        try:
            csv_files = list(_self.results_dir.glob("*.csv"))
            for csv_file in csv_files:
                filename = csv_file.name
                if "_image_2_" in filename:
                    sequence = filename[:2]
                    feature_detector = filename.split("_image_2_")[1].replace(".csv", "")
                    
                    if sequence not in files:
                        files[sequence] = []
                    files[sequence].append(feature_detector)
            
            # Sort for consistent ordering
            for sequence in files:
                files[sequence].sort()
                
        except Exception as e:
            logger.error(f"Error loading file structure: {e}")
            st.error(f"Error loading files: {e}")
        return files
    
    @st.cache_data(ttl=CACHE_DURATION)
    def load_csv_data(_self, sequence: str, feature_detector: str) -> Optional[pd.DataFrame]:
        """Load and cache CSV data with error handling"""
        filename = f"{sequence}_image_2_{feature_detector}.csv"
        filepath = _self.results_dir / filename
        
        try:
            if filepath.exists():
                data = pd.read_csv(filepath)
                logger.info(f"Loaded {len(data)} rows from {filename}")
                return data
            else:
                logger.warning(f"File not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            st.error(f"Error loading {filename}: {e}")
            return None

class AdvancedAnalysis:
    """Enhanced analysis with comprehensive metrics and comparisons"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.files = data_loader.load_file_structure()
    
    def get_single_analysis(self, sequence: str, feature_detector: str) -> Dict:
        """Get comprehensive analysis for a single sequence-detector combination"""
        data = self.data_loader.load_csv_data(sequence, feature_detector)
        timeTaken = None
        try:
            df = pd.read_csv("results/all_sequences_mse_results.csv")
            df['Sequence'] = df['Sequence'].apply(lambda x: f"{int(x):02d}")
            # st.dataframe(df, use_container_width=True)
            condition = (df['Sequence'] == sequence) & (df['Feature_Detector'] == feature_detector)
            identifiedRow = df[condition]
            timeTaken = identifiedRow["Time Taken"].iloc[0]
        except Exception as e:
            print(f"Error reading time taken from CSV: {e}")
            timeTaken = 0
        if data is None or data.empty:
            return {"error": "No data available"}
        
        try:
            metrics = {
                "sequence": sequence,
                "feature_detector": feature_detector,
                "total_frames": len(data),
                "mean_error_x": data['Error_X'].mean(),
                "mean_error_y": data['Error_Y'].mean(),
                "mean_error_z": data['Error_Z'].mean(),
                "std_error_x": data['Error_X'].std(),
                "std_error_y": data['Error_Y'].std(),
                "std_error_z": data['Error_Z'].std(),
                "max_error_x": data['Error_X'].max(),
                "max_error_y": data['Error_Y'].max(),
                "max_error_z": data['Error_Z'].max(),
                "min_error_x": data['Error_X'].min(),
                "min_error_y": data['Error_Y'].min(),
                "min_error_z": data['Error_Z'].min(),
                "median_error_x": data['Error_X'].median(),
                "median_error_y": data['Error_Y'].median(),
                "median_error_z": data['Error_Z'].median(),
                "total_error": float(f"{(data['Error_X'].mean() + data['Error_Y'].mean() + data['Error_Z'].mean())/3:.3f}"),
                "rmse": np.sqrt((data['Error_X']**2 + data["Error_Y"]**2 + data['Error_Z']**2).mean()),
                "Total Time Taken": timeTaken/len(df),
                "data": data
            }
            
            # Calculate trajectory length
            if all(col in data.columns for col in ['Predicted_X', 'Predicted_Z']):
                pred_x, pred_z = data['Predicted_X'].values, data['Predicted_Z'].values
                trajectory_length = np.sum(np.sqrt(np.diff(pred_x)**2 + np.diff(pred_z)**2))
                metrics["trajectory_length"] = trajectory_length
                metrics["error_per_meter"] = metrics["total_error"] / trajectory_length if trajectory_length > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in analysis for {sequence}-{feature_detector}: {e}")
            return {"error": str(e)}
    
    def get_comparative_analysis(self) -> pd.DataFrame:
        """Get comparative analysis across all sequences and feature detectors"""
        results = []
        
        for sequence in self.files:
            for feature_detector in self.files[sequence]:
                analysis = self.get_single_analysis(sequence, feature_detector)
                if "error" not in analysis:
                    results.append({
                        "Sequence": sequence,
                        "Feature_Detector": feature_detector,
                        "Total_Frames": analysis["total_frames"],
                        "Mean_Error_X": analysis["mean_error_x"],
                        "Mean_Error_Y": analysis["mean_error_y"],
                        "Mean_Error_Z": analysis["mean_error_z"],
                        "Std_Error_X": analysis["std_error_x"],
                        "Std_Error_Y": analysis["std_error_y"],
                        "Std_Error_Z": analysis["std_error_z"],
                        "Total_Error": analysis["total_error"],
                        "RMSE": analysis["rmse"],
                        "Total Time Taken": analysis["Total Time Taken"],
                        "Max_Error_X": analysis["max_error_x"],
                        "Max_Error_Y": analysis["max_error_y"],
                        "Max_Error_Z": analysis["max_error_z"],
                        "Trajectory_Length": analysis.get("trajectory_length", 0),
                        "Error_Per_Meter": analysis.get("error_per_meter", 0)
                    })
        
        return pd.DataFrame(results)
    
    def get_best_feature_detectors(self, comparative_df: pd.DataFrame, top_n: int = 3) -> Dict:
        """Identify best feature detectors based on multiple criteria"""
        if comparative_df.empty:
            return {}
        
        rankings = {}
        
        # Rank by different metrics
        metrics = ["Total_Error", "RMSE", "Error_Per_Meter", "Mean_Error_X", "Mean_Error_Y", "Mean_Error_Z", "Total Time Taken"]
        
        for metric in metrics:
            if metric in comparative_df.columns:
                # Filter out zero values for Error_Per_Meter
                df_filtered = comparative_df[comparative_df[metric] > 0] if metric == "Error_Per_Meter" else comparative_df
                
                if not df_filtered.empty:
                    ranked = df_filtered.nsmallest(top_n, metric)
                    rankings[metric] = ranked[["Sequence", "Feature_Detector", metric]].to_dict('records')
        
        # Overall ranking using composite score
        if not comparative_df.empty:
            # Normalize metrics to 0-1 scale
            normalized_df = comparative_df.copy()
            score_metrics = ["Total_Error", "RMSE", "Total Time Taken"]
            
            for metric in score_metrics:
                if metric in normalized_df.columns:
                    max_val = normalized_df[metric].max()
                    min_val = normalized_df[metric].min()
                    if max_val != min_val:
                        normalized_df[f"{metric}_norm"] = (normalized_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[f"{metric}_norm"] = 0
            
            # Calculate composite score (lower is better)
            normalized_df["Composite_Score"] = (
                normalized_df.get("Total_Error_norm", 0) + 
                normalized_df.get("RMSE_norm", 0) +
                normalized_df.get("Total Time Taken_norm", 0)
            ) / len(score_metrics)
            
            overall_best = normalized_df.nsmallest(top_n, "Composite_Score")
            rankings["Overall"] = overall_best[["Sequence", "Feature_Detector", "Composite_Score", "Total_Error", "RMSE", "Total Time Taken"]].to_dict('records')
        
        return rankings

class VisualizationEngine:
    """Advanced visualization with interactive plots"""
    
    @staticmethod
    def create_trajectory_comparison(data: pd.DataFrame) -> go.Figure:
        """Create interactive trajectory comparison plot"""
        fig = go.Figure()
        
        # Add predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=data['Predicted_X'], 
            y=data['Predicted_Z'],
            z=data['Predicted_Y'],
            mode='lines+markers',
            name='Predicted Trajectory',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add ground truth trajectory
        fig.add_trace(go.Scatter3d(
            x=data['Ground_Truth_X'], 
            y=data['Ground_Truth_Z'],
            z=data['Ground_Truth_Y'],
            mode='lines+markers',
            name='Ground Truth',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Z (m)',  # Z as floor axis
                zaxis_title='Y (m)',  # Y as height
                xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                zaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1, y=-5, z=2)
                ),
                bgcolor='rgba(0, 0, 0, 0)'
            ),
            title=dict(
                text='Ground Truth Trajectory (3D)',
                font=dict(size=22, color='white')
            ),
            legend=dict(
                x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    @staticmethod
    def create_error_distribution(data: pd.DataFrame) -> go.Figure:
        """Create interactive error distribution plot"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Error in X Direction', 'Error in Y Direction', 'Error in Z Direction'),
            vertical_spacing=0.1
        )
        
        # X direction errors
        fig.add_trace(go.Scatter(
            x=data['Frame'], 
            y=data['Error_X'],
            mode='lines',
            name='Error X',
            line=dict(color='red')
        ), row=1, col=1)
        
        # Z direction errors
        fig.add_trace(go.Scatter(
            x=data['Frame'], 
            y=data['Error_Y'],
            mode='lines',
            name='Error Y',
            line=dict(color='red')
        ), row=2, col=1)

        # Z direction errors
        fig.add_trace(go.Scatter(
            x=data['Frame'], 
            y=data['Error_Z'],
            mode='lines',
            name='Error Z',
            line=dict(color='red')
        ), row=3, col=1)
        
        fig.update_layout(
            title="Error Distribution Across Frames",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Frame", row=2, col=1)
        fig.update_yaxes(title_text="Error (m)", row=1, col=1)
        fig.update_yaxes(title_text="Error (m)", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_comparative_heatmap(comparative_df: pd.DataFrame, values: str) -> go.Figure:
        """Create heatmap comparing feature detectors across sequences"""
        if comparative_df.empty:
            return go.Figure()
        
        # Pivot data for heatmap
        pivot_data = comparative_df.pivot(index='Feature_Detector', columns='Sequence', values=values)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            text=np.round(pivot_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title=values)
        ))
        
        fig.update_layout(
            title=f"Feature Detector Performance Across Sequences - {values}",
            xaxis_title="Sequence",
            yaxis_title="Feature Detector"
        )
        
        return fig

def main():
    """Main application"""
    st.set_page_config(page_title="Feature Descriptor Analysis", layout="wide")
    st.title("🎯 Advanced Feature Descriptor Analysis Dashboard")
    
    # Initialize components
    data_loader = DataLoader()
    analyzer = AdvancedAnalysis(data_loader)
    viz_engine = VisualizationEngine()
    
    # Check if data is available
    if not analyzer.files:
        st.error("No CSV files found in the results directory. Please ensure the results are available.")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Single Analysis", "Comparative Analysis", "Best Feature Detectors"]
        )
    
    if analysis_mode == "Single Analysis":
        st.header("📊 Single Sequence Analysis")
        
        # Selection controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Configuration")
            sequence = st.selectbox("Sequence", list(analyzer.files.keys()))
            feature_detector = st.selectbox("Feature Detector", analyzer.files[sequence])
        
        # Perform analysis
        analysis_result = analyzer.get_single_analysis(sequence, feature_detector)
        
        if "error" in analysis_result:
            st.error(f"Analysis failed: {analysis_result['error']}")
            return
        
        with col2:
            st.subheader("Performance Metrics")
            
            # Display metrics in columns
            metrics_cols = st.columns(6)
            metrics_cols[0].metric("Total Frames", f"{analysis_result['total_frames']}")
            metrics_cols[1].metric("Mean Error X", f"{analysis_result['mean_error_x']:.3f}m")
            metrics_cols[2].metric("Mean Error Z", f"{analysis_result['mean_error_z']:.3f}m")
            metrics_cols[3].metric("Total Error", f"{analysis_result['total_error']:.3f}m")
            metrics_cols[4].metric("RMSE", f"{analysis_result['rmse']:.3f}m")
            metrics_cols[5].metric("Traj. Length", f"{analysis_result.get('trajectory_length', 0):.1f}m")
        
        # Visualizations
        st.subheader("📈 Detailed Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Trajectory comparison
            trajectory_fig = viz_engine.create_trajectory_comparison(analysis_result["data"])
            st.plotly_chart(trajectory_fig, use_container_width=True)
        
        with viz_col2:
            # Error distribution
            error_fig = viz_engine.create_error_distribution(analysis_result["data"])
            st.plotly_chart(error_fig, use_container_width=True)
        
        # Detailed statistics
        with st.expander("📋 Detailed Statistics"):
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Median"],
                "Error X": [
                    analysis_result['mean_error_x'],
                    analysis_result['std_error_x'],
                    analysis_result['min_error_x'],
                    analysis_result['max_error_x'],
                    analysis_result['median_error_x']
                ],
                "Error Y": [
                    analysis_result['mean_error_y'],
                    analysis_result['std_error_y'],
                    analysis_result['min_error_y'],
                    analysis_result['max_error_y'],
                    analysis_result['median_error_y']
                ],
                "Error Z": [
                    analysis_result['mean_error_z'],
                    analysis_result['std_error_z'],
                    analysis_result['min_error_z'],
                    analysis_result['max_error_z'],
                    analysis_result['median_error_z']
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    elif analysis_mode == "Comparative Analysis":
        st.header("🔍 Comparative Analysis")
        
        # Get comparative data
        comparative_df = analyzer.get_comparative_analysis()
        
        if comparative_df.empty:
            st.error("No comparative data available")
            return
        
        # Display summary
        st.subheader("📊 Performance Summary")
        st.dataframe(comparative_df, use_container_width=True)
        
        # Heatmap visualization
        st.subheader("🌡️ Performance Heatmap")
        col1, col2 = st.columns(2)
        heatmap_fig_TotalError = viz_engine.create_comparative_heatmap(comparative_df, 'Total_Error')
        col1.plotly_chart(heatmap_fig_TotalError, use_container_width=True)
        heatmap_figTimeTaken = viz_engine.create_comparative_heatmap(comparative_df, 'Total Time Taken')
        col2.plotly_chart(heatmap_figTimeTaken, use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Best Performers (Lowest Total Error):**")
            best_performers = comparative_df.nsmallest(5, 'Total_Error')[['Sequence', 'Feature_Detector', 'Total_Error']]
            st.dataframe(best_performers, use_container_width=True)
        with col2:
            st.write("**Most Consistent (Lowest RMSE):**")
            most_consistent = comparative_df.nsmallest(5, 'RMSE')[['Sequence', 'Feature_Detector', 'RMSE']]
            st.dataframe(most_consistent, use_container_width=True)
        with col3:
            st.write("**Fastest Detectors:**")
            most_consistent = comparative_df.nsmallest(5, 'Total Time Taken')[['Sequence', 'Feature_Detector', 'Total Time Taken']]
            st.dataframe(most_consistent, use_container_width=True)
        
    
    elif analysis_mode == "Best Feature Detectors":
        st.header("🏆 Best Feature Detectors Identification")
        
        # Get comparative data
        comparative_df = analyzer.get_comparative_analysis()
        
        if comparative_df.empty:
            st.error("No comparative data available")
            return
        
        # Get rankings
        rankings = analyzer.get_best_feature_detectors(comparative_df)
        
        if not rankings:
            st.error("Unable to generate rankings")
            return
        
        # Display rankings
        st.subheader("🎯 Overall Best Feature Detectors")
        
        if "Overall" in rankings:
            overall_df = pd.DataFrame(rankings["Overall"])
            overall_df.index = overall_df.index + 1
            st.dataframe(overall_df, use_container_width=True)
            
            # Highlight the best
            if not overall_df.empty:
                best_overall = overall_df.iloc[0]
                st.success(f"🥇 **Best Overall:** {best_overall['Feature_Detector']} on Sequence {best_overall['Sequence']}")
        
        # Category-wise rankings
        st.subheader("📊 Category-wise Rankings")
        
        ranking_cols = st.columns(min(3, len(rankings)))
        
        for i, (category, ranking_data) in enumerate(rankings.items()):
            if category != "Overall" and ranking_data:
                with ranking_cols[i % 3]:
                    st.write(f"**Best by {category}:**")
                    ranking_df = pd.DataFrame(ranking_data)
                    ranking_df.index = ranking_df.index + 1
                    st.dataframe(ranking_df, use_container_width=True)
        
        # Summary insights
        st.subheader("💡 Key Insights")
        
        # Feature detector frequency in top rankings
        feature_counts = {}
        for category, ranking_data in rankings.items():
            for entry in ranking_data:
                detector = entry['Feature_Detector']
                if detector not in feature_counts:
                    feature_counts[detector] = 0
                else:
                    feature_counts[detector] = feature_counts.get(detector, 0) + 1
        
        if feature_counts:
            best_detector = max(feature_counts, key=feature_counts.get)
            st.info(f"🔍 **Most Consistently Good:** {best_detector} appears in {feature_counts[best_detector]} top rankings")
        
        # Sequence-specific insights
        seq_performance = comparative_df.groupby('Sequence')['Total_Error'].min()
        easiest_seq = seq_performance.idxmin()
        hardest_seq = seq_performance.idxmax()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🟢 **Easiest Sequence:** {easiest_seq} (Min Error: {seq_performance[easiest_seq]:.3f})")
        with col2:
            st.info(f"🔴 **Hardest Sequence:** {hardest_seq} (Min Error: {seq_performance[hardest_seq]:.3f})")

if __name__ == "__main__":
    main()