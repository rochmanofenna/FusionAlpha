#!/usr/bin/env python3
"""
Model Explainability Dashboard

Interactive dashboard for understanding and interpreting the Fusion Alpha trading model:
1. Feature importance analysis
2. SHAP value explanations
3. Contradiction detection visualization
4. Decision pathway tracing
5. Performance attribution
6. Risk factor analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json

# Add project paths
sys.path.append('/home/ryan/trading/mismatch-trading')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit not available. Install with: pip install streamlit")
    STREAMLIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplainabilityConfig:
    """Configuration for explainability analysis"""
    model_path: str = "/home/ryan/trading/mismatch-trading/training/checkpoints/best_model.pt"
    data_path: str = "/home/ryan/trading/mismatch-trading/training_data/synthetic_training_data.parquet"
    output_dir: str = "/home/ryan/trading/mismatch-trading/explainability/outputs"
    
    # Analysis settings
    n_samples_shap: int = 100  # Number of samples for SHAP analysis
    top_features: int = 20     # Top N features to analyze
    
    # Visualization settings
    figure_width: int = 12
    figure_height: int = 8
    save_plots: bool = True

class ModelExplainer:
    """
    Model explainability analyzer for the Fusion Alpha system
    """
    
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and model
        self.data = self._load_data()
        self.model = self._load_model()
        
        # Feature names
        self.feature_names = self._get_feature_names()
        
        # SHAP explainer
        self.shap_explainer = None
        if SHAP_AVAILABLE and self.model is not None:
            self._setup_shap_explainer()
        
        logger.info("Model explainer initialized")
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load training/validation data for analysis"""
        try:
            if Path(self.config.data_path).exists():
                df = pd.read_parquet(self.config.data_path)
                logger.info(f"Loaded {len(df)} samples for explainability analysis")
                return df
            else:
                logger.warning(f"Data file not found: {self.config.data_path}")
                # Generate synthetic data for demo
                return self._generate_demo_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._generate_demo_data()
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """Generate synthetic data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        data = {
            'rsi': np.random.uniform(0, 100, n_samples),
            'macd': np.random.normal(0, 0.1, n_samples),
            'bb_upper': np.random.uniform(100, 150, n_samples),
            'bb_lower': np.random.uniform(50, 100, n_samples),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
            'volatility': np.random.uniform(0.01, 0.05, n_samples),
            'returns': np.random.normal(0.001, 0.02, n_samples),
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'sentiment_volume': np.random.poisson(10, n_samples),
            'contradiction_type': np.random.choice(['none', 'overhype', 'underhype', 'paradox'], n_samples),
            'hour': np.random.randint(9, 16, n_samples),
            'day_of_week': np.random.randint(0, 5, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Encode categorical variable
        df['contradiction_encoded'] = df['contradiction_type'].map({
            'none': 0, 'overhype': 1, 'underhype': 2, 'paradox': 3
        })
        
        # Generate target (future returns)
        df['future_return_5m'] = (
            0.001 * (df['sentiment_score'] > 0.5) - 
            0.001 * (df['sentiment_score'] < -0.5) +
            0.0005 * (df['rsi'] < 30) -
            0.0005 * (df['rsi'] > 70) +
            np.random.normal(0, 0.01, n_samples)
        )
        
        # Generate model predictions
        df['prediction'] = df['future_return_5m'] + np.random.normal(0, 0.005, n_samples)
        df['confidence'] = np.random.uniform(0.5, 1.0, n_samples)
        
        logger.info("Generated synthetic data for demo")
        return df
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if Path(self.config.model_path).exists():
                import torch
                checkpoint = torch.load(self.config.model_path, map_location='cpu')
                logger.info("Model loaded successfully")
                return checkpoint
            else:
                logger.warning(f"Model file not found: {self.config.model_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for analysis"""
        if self.data is not None:
            # Exclude target and prediction columns
            exclude_cols = ['future_return_5m', 'prediction', 'confidence', 'symbol', 'timestamp', 'contradiction_type']
            return [col for col in self.data.columns if col not in exclude_cols]
        else:
            return ['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio', 'volatility', 'returns', 'sentiment_score']
    
    def _setup_shap_explainer(self):
        """Setup SHAP explainer for the model"""
        if not SHAP_AVAILABLE or self.data is None:
            return
        
        try:
            # Get feature data
            X = self.data[self.feature_names].fillna(0)
            
            # For demo purposes, use a simple linear explainer
            # In production, this would be integrated with the actual model
            self.shap_explainer = shap.LinearExplainer(
                np.ones(len(self.feature_names)), X.iloc[:100]
            )
            logger.info("SHAP explainer setup completed")
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {e}")
            self.shap_explainer = None
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze overall feature importance"""
        logger.info("Analyzing feature importance...")
        
        if self.data is None:
            return {}
        
        # Calculate correlation with target
        target_corr = self.data[self.feature_names].corrwith(self.data['future_return_5m']).abs()
        
        # Calculate prediction correlation (model feature importance proxy)
        pred_corr = self.data[self.feature_names].corrwith(self.data['prediction']).abs()
        
        # Combine importance metrics
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'target_correlation': target_corr.values,
            'prediction_correlation': pred_corr.values,
            'combined_importance': (target_corr.abs() + pred_corr.abs()).values / 2
        }).sort_values('combined_importance', ascending=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_width, self.config.figure_height))
        
        # Top features bar chart
        top_features = importance_df.head(self.config.top_features)
        ax1.barh(range(len(top_features)), top_features['combined_importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top Feature Importance')
        ax1.grid(True, alpha=0.3)
        
        # Correlation comparison
        ax2.scatter(importance_df['target_correlation'], importance_df['prediction_correlation'], alpha=0.7)
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('Target Correlation')
        ax2.set_ylabel('Prediction Correlation')
        ax2.set_title('Feature Correlation Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Add feature labels for top features
        for _, row in top_features.head(5).iterrows():
            ax2.annotate(row['feature'], 
                        (row['target_correlation'], row['prediction_correlation']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return {
            'importance_scores': importance_df.to_dict('records'),
            'top_features': top_features['feature'].tolist()
        }
    
    def analyze_shap_values(self) -> Optional[Dict[str, Any]]:
        """Analyze SHAP values for model interpretability"""
        if not SHAP_AVAILABLE or self.shap_explainer is None or self.data is None:
            logger.warning("SHAP analysis not available")
            return None
        
        logger.info("Calculating SHAP values...")
        
        try:
            # Get sample data for SHAP analysis
            X_sample = self.data[self.feature_names].fillna(0).iloc[:self.config.n_samples_shap]
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Create SHAP summary plot
            plt.figure(figsize=(self.config.figure_width, self.config.figure_height))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            
            if self.config.save_plots:
                plt.savefig(self.output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate feature importance from SHAP values
            shap_importance = np.abs(shap_values).mean(0)
            shap_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            return {
                'shap_values': shap_values.tolist(),
                'feature_importance': shap_df.to_dict('records'),
                'base_value': float(self.shap_explainer.expected_value)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def analyze_contradiction_patterns(self) -> Dict[str, Any]:
        """Analyze contradiction detection patterns"""
        logger.info("Analyzing contradiction patterns...")
        
        if self.data is None or 'contradiction_type' not in self.data.columns:
            return {}
        
        # Contradiction distribution
        contradiction_counts = self.data['contradiction_type'].value_counts()
        
        # Performance by contradiction type
        performance_by_type = self.data.groupby('contradiction_type').agg({
            'future_return_5m': ['mean', 'std', 'count'],
            'prediction': 'mean',
            'confidence': 'mean'
        }).round(4)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_width, self.config.figure_height))
        
        # Contradiction distribution
        ax1 = axes[0, 0]
        contradiction_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Contradiction Type Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Returns by contradiction type
        ax2 = axes[0, 1]
        returns_by_type = self.data.groupby('contradiction_type')['future_return_5m'].mean()
        colors = ['green' if x > 0 else 'red' for x in returns_by_type.values]
        returns_by_type.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Average Returns by Contradiction Type')
        ax2.set_ylabel('Average Return')
        ax2.tick_params(axis='x', rotation=45)
        
        # Prediction accuracy by type
        ax3 = axes[1, 0]
        for ctype in self.data['contradiction_type'].unique():
            subset = self.data[self.data['contradiction_type'] == ctype]
            ax3.scatter(subset['prediction'], subset['future_return_5m'], 
                       label=ctype, alpha=0.6, s=20)
        
        ax3.plot([-0.05, 0.05], [-0.05, 0.05], 'k--', alpha=0.5)
        ax3.set_xlabel('Prediction')
        ax3.set_ylabel('Actual Return')
        ax3.set_title('Prediction vs Actual by Contradiction Type')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Confidence distribution
        ax4 = axes[1, 1]
        for ctype in self.data['contradiction_type'].unique():
            subset = self.data[self.data['contradiction_type'] == ctype]
            ax4.hist(subset['confidence'], alpha=0.7, label=ctype, bins=20)
        
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution by Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(self.output_dir / 'contradiction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'contradiction_counts': contradiction_counts.to_dict(),
            'performance_by_type': performance_by_type.to_dict(),
            'returns_by_type': returns_by_type.to_dict()
        }
    
    def analyze_decision_pathways(self, sample_indices: List[int] = None) -> Dict[str, Any]:
        """Analyze decision pathways for specific samples"""
        logger.info("Analyzing decision pathways...")
        
        if self.data is None:
            return {}
        
        if sample_indices is None:
            # Select diverse samples for analysis
            sample_indices = [
                self.data['prediction'].idxmax(),  # Highest prediction
                self.data['prediction'].idxmin(),  # Lowest prediction
                self.data['confidence'].idxmax(),  # Highest confidence
                self.data.sample(1).index[0]       # Random sample
            ]
        
        pathways = []
        
        for idx in sample_indices:
            if idx >= len(self.data):
                continue
            
            sample = self.data.iloc[idx]
            
            # Create decision pathway
            pathway = {
                'sample_id': int(idx),
                'prediction': float(sample['prediction']),
                'actual_return': float(sample['future_return_5m']),
                'confidence': float(sample['confidence']),
                'contradiction_type': sample.get('contradiction_type', 'unknown'),
                
                # Feature contributions (simplified)
                'feature_values': {
                    feature: float(sample[feature]) for feature in self.feature_names 
                    if feature in sample.index
                },
                
                # Decision logic (mock - would be from actual model)
                'decision_steps': [
                    {
                        'step': 'Technical Analysis',
                        'signal': float(sample.get('rsi', 50) - 50) / 50,  # Normalized RSI
                        'weight': 0.4
                    },
                    {
                        'step': 'Sentiment Analysis', 
                        'signal': float(sample.get('sentiment_score', 0)),
                        'weight': 0.3
                    },
                    {
                        'step': 'Contradiction Detection',
                        'signal': 1.0 if sample.get('contradiction_type') != 'none' else 0.0,
                        'weight': 0.3
                    }
                ]
            }
            
            pathways.append(pathway)
        
        return {'decision_pathways': pathways}
    
    def create_interactive_dashboard(self) -> Optional[str]:
        """Create interactive Plotly dashboard"""
        if not self.data is not None:
            return None
        
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Feature Importance', 'Prediction vs Actual', 
                          'Contradiction Analysis', 'Performance Over Time'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Feature importance
        importance = self.data[self.feature_names].corrwith(self.data['prediction']).abs().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=importance.head(10).values, y=importance.head(10).index, orientation='h'),
            row=1, col=1
        )
        
        # Prediction vs Actual
        fig.add_trace(
            go.Scatter(
                x=self.data['prediction'],
                y=self.data['future_return_5m'],
                mode='markers',
                marker=dict(color=self.data['confidence'], colorscale='Viridis', showscale=True),
                text=self.data.index,
                name='Predictions'
            ),
            row=1, col=2
        )
        
        # Contradiction distribution
        if 'contradiction_type' in self.data.columns:
            contradiction_counts = self.data['contradiction_type'].value_counts()
            fig.add_trace(
                go.Pie(labels=contradiction_counts.index, values=contradiction_counts.values),
                row=2, col=1
            )
        
        # Performance over time (if timestamp available)
        if 'timestamp' in self.data.columns:
            daily_returns = self.data.groupby(self.data['timestamp'].dt.date)['future_return_5m'].mean()
            fig.add_trace(
                go.Scatter(x=daily_returns.index, y=daily_returns.values, mode='lines'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Fusion Alpha Model Explainability Dashboard",
            showlegend=False
        )
        
        # Save as HTML
        html_path = self.output_dir / 'interactive_dashboard.html'
        fig.write_html(str(html_path))
        
        logger.info(f"Interactive dashboard saved to {html_path}")
        return str(html_path)
    
    def generate_explainability_report(self) -> Dict[str, Any]:
        """Generate comprehensive explainability report"""
        logger.info("Generating comprehensive explainability report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'data_summary': {
                'n_samples': len(self.data) if self.data is not None else 0,
                'n_features': len(self.feature_names),
                'date_range': f"{self.data.index.min()} to {self.data.index.max()}" if self.data is not None and hasattr(self.data.index, 'min') else "N/A"
            }
        }
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        report['feature_importance'] = feature_importance
        
        # SHAP analysis
        shap_analysis = self.analyze_shap_values()
        if shap_analysis:
            report['shap_analysis'] = shap_analysis
        
        # Contradiction analysis
        contradiction_analysis = self.analyze_contradiction_patterns()
        report['contradiction_analysis'] = contradiction_analysis
        
        # Decision pathways
        decision_pathways = self.analyze_decision_pathways()
        report['decision_pathways'] = decision_pathways
        
        # Model performance summary
        if self.data is not None:
            predictions = self.data['prediction']
            actuals = self.data['future_return_5m']
            
            report['performance_summary'] = {
                'correlation': float(predictions.corr(actuals)),
                'mae': float(np.mean(np.abs(predictions - actuals))),
                'rmse': float(np.sqrt(np.mean((predictions - actuals) ** 2))),
                'directional_accuracy': float(np.mean(np.sign(predictions) == np.sign(actuals)))
            }
        
        # Save report
        report_path = self.output_dir / 'explainability_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Explainability report saved to {report_path}")
        return report
    
    def print_summary(self):
        """Print explainability summary"""
        print("\n" + "="*60)
        print("FUSION ALPHA MODEL EXPLAINABILITY SUMMARY")
        print("="*60)
        
        if self.data is not None:
            print(f"Data Samples: {len(self.data):,}")
            print(f"Features: {len(self.feature_names)}")
            
            # Top features
            importance = self.data[self.feature_names].corrwith(self.data['prediction']).abs().sort_values(ascending=False)
            print(f"\nTop 5 Most Important Features:")
            for i, (feature, score) in enumerate(importance.head(5).items(), 1):
                print(f"  {i}. {feature}: {score:.3f}")
            
            # Contradiction analysis
            if 'contradiction_type' in self.data.columns:
                contradiction_dist = self.data['contradiction_type'].value_counts(normalize=True)
                print(f"\nContradiction Distribution:")
                for ctype, pct in contradiction_dist.items():
                    print(f"  {ctype}: {pct:.1%}")
            
            # Performance metrics
            if 'prediction' in self.data.columns and 'future_return_5m' in self.data.columns:
                corr = self.data['prediction'].corr(self.data['future_return_5m'])
                mae = np.mean(np.abs(self.data['prediction'] - self.data['future_return_5m']))
                print(f"\nModel Performance:")
                print(f"  Prediction Correlation: {corr:.3f}")
                print(f"  Mean Absolute Error: {mae:.4f}")
        
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"SHAP Available: {SHAP_AVAILABLE}")
        print("="*60)

def create_streamlit_app():
    """Create Streamlit web app for interactive exploration"""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit not available for web app")
        return
    
    # This would be in a separate file for actual deployment
    st.title("Fusion Alpha Model Explainability Dashboard")
    
    st.sidebar.header("Configuration")
    
    # File uploads
    data_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'parquet'])
    model_file = st.sidebar.file_uploader("Upload Model File", type=['pt', 'pkl'])
    
    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Feature Importance", "SHAP Analysis", "Contradiction Patterns", "Decision Pathways"]
    )
    
    if st.sidebar.button("Run Analysis"):
        config = ExplainabilityConfig()
        explainer = ModelExplainer(config)
        
        if analysis_type == "Feature Importance":
            st.header("Feature Importance Analysis")
            results = explainer.analyze_feature_importance()
            
            if results:
                importance_df = pd.DataFrame(results['importance_scores'])
                st.dataframe(importance_df)
                
                # Plot
                fig = px.bar(
                    importance_df.head(10), 
                    x='combined_importance', 
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig)
        
        elif analysis_type == "Contradiction Patterns":
            st.header("Contradiction Detection Analysis")
            results = explainer.analyze_contradiction_patterns()
            
            if results:
                st.subheader("Contradiction Distribution")
                fig = px.pie(
                    values=list(results['contradiction_counts'].values()),
                    names=list(results['contradiction_counts'].keys()),
                    title="Distribution of Contradiction Types"
                )
                st.plotly_chart(fig)
        
        # Add more analysis types...

def main():
    """Test the explainability framework"""
    print("Testing Model Explainability Dashboard")
    print("="*50)
    
    # Create configuration
    config = ExplainabilityConfig()
    
    # Create explainer
    explainer = ModelExplainer(config)
    
    # Generate comprehensive report
    try:
        report = explainer.generate_explainability_report()
        explainer.print_summary()
        
        # Create interactive dashboard
        dashboard_path = explainer.create_interactive_dashboard()
        
        print(f"\nExplainability analysis completed!")
        print(f"Analyzed {len(explainer.feature_names)} features")
        print(f"Report generated with {len(report)} sections")
        if dashboard_path:
            print(f"Interactive dashboard: {dashboard_path}")
        print(f"💾 Outputs saved to: {explainer.output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()