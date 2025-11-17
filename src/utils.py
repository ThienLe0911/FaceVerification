"""
Utility functions for face verification project.

This module provides helper functions for path management, logging,
data handling, and other common operations used throughout the project.
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


class ProjectPaths:
    """
    Centralized path management for the project.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize project paths.
        
        Args:
            project_root (Optional[str]): Root directory of the project.
                                        If None, tries to detect automatically.
        """
        self.project_root = Path(project_root) if project_root else self._detect_project_root()
        
        # Define all project paths
        self.data_dir = self.project_root / "data"
        self.data_raw = self.data_dir / "raw"
        self.data_processed = self.data_dir / "processed"
        self.data_pairs = self.data_dir / "pairs"
        
        self.notebooks_dir = self.project_root / "notebooks"
        self.src_dir = self.project_root / "src"
        self.experiments_dir = self.project_root / "experiments"
        
        # Ensure all directories exist
        self._create_directories()
    
    def _detect_project_root(self) -> Path:
        """
        Try to automatically detect the project root directory.
        
        Returns:
            Path: Project root directory
        """
        current_dir = Path.cwd()
        
        # Look for project indicators
        indicators = ['requirements.txt', 'README.md', 'src', 'data', 'notebooks']
        
        # Check current directory and parents
        for path in [current_dir] + list(current_dir.parents):
            if any((path / indicator).exists() for indicator in indicators):
                return path
        
        # Default to current directory
        return current_dir
    
    def _create_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir, self.data_raw, self.data_processed, self.data_pairs,
            self.notebooks_dir, self.src_dir, self.experiments_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_relative_path(self, path: Union[str, Path]) -> str:
        """
        Get path relative to project root.
        
        Args:
            path (Union[str, Path]): Absolute or relative path
            
        Returns:
            str: Path relative to project root
        """
        path = Path(path)
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


class Logger:
    """
    Enhanced logging utility for the project.
    """
    
    def __init__(
        self,
        name: str = "face_verification",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name (str): Logger name
            level (int): Logging level
            log_file (Optional[str]): Path to log file
            console_output (bool): Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


class ConfigManager:
    """
    Configuration management for the project.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_path (Optional[str]): Path to config file
        """
        self.config_path = config_path or "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "model": {
                "verification_threshold": 0.8,
                "detection_threshold": 0.6,
                "image_size": 160,
                "device": "auto"
            },
            "preprocessing": {
                "normalize_method": "facenet",
                "resize_method": "cubic"
            },
            "paths": {
                "models_dir": "models",
                "logs_dir": "logs"
            },
            "experiment": {
                "random_seed": 42,
                "batch_size": 32
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key (str): Configuration key (supports dot notation)
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key (supports dot notation)
            value (Any): Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class DatasetUtils:
    """
    Utilities for dataset management and manipulation.
    """
    
    @staticmethod
    def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all image files in a directory.
        
        Args:
            directory (str): Directory to search
            extensions (List[str]): Valid extensions
            
        Returns:
            List[str]: List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    @staticmethod
    def create_image_pairs_file(
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
        output_file: str
    ):
        """
        Create pairs file for face verification evaluation.
        
        Args:
            positive_pairs (List[Tuple[str, str]]): List of same-person pairs
            negative_pairs (List[Tuple[str, str]]): List of different-person pairs
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            # Write positive pairs
            for img1, img2 in positive_pairs:
                f.write(f"{img1}\t{img2}\t1\n")
            
            # Write negative pairs
            for img1, img2 in negative_pairs:
                f.write(f"{img1}\t{img2}\t0\n")
    
    @staticmethod
    def load_pairs_file(pairs_file: str) -> List[Tuple[str, str, int]]:
        """
        Load pairs from file.
        
        Args:
            pairs_file (str): Pairs file path
            
        Returns:
            List[Tuple[str, str, int]]: List of (img1, img2, label) tuples
        """
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    img1, img2, label = parts
                    pairs.append((img1, img2, int(label)))
        
        return pairs


class VisualizationUtils:
    """
    Utilities for visualization and plotting.
    """
    
    @staticmethod
    def plot_verification_results(
        similarities: List[float],
        labels: List[int],
        threshold: float = 0.8,
        title: str = "Face Verification Results"
    ):
        """
        Plot verification results distribution.
        
        Args:
            similarities (List[float]): Similarity scores
            labels (List[int]): True labels (1 for same person, 0 for different)
            threshold (float): Verification threshold
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Plot distributions
        plt.subplot(1, 2, 1)
        same_person_scores = similarities[labels == 1]
        diff_person_scores = similarities[labels == 0]
        
        plt.hist(same_person_scores, alpha=0.7, label='Same Person', bins=30, color='green')
        plt.hist(diff_person_scores, alpha=0.7, label='Different Person', bins=30, color='red')
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Similarity Score Distribution')
        plt.legend()
        
        # ROC-like plot
        plt.subplot(1, 2, 2)
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for thresh in thresholds:
            predictions = similarities > thresh
            
            # Calculate TPR and FPR
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        plt.plot(fpr_list, tpr_list, 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_image_pair(
        img1: Union[str, np.ndarray],
        img2: Union[str, np.ndarray],
        similarity: float,
        is_match: bool,
        threshold: float = 0.8
    ):
        """
        Display image pair with verification result.
        
        Args:
            img1 (Union[str, np.ndarray]): First image
            img2 (Union[str, np.ndarray]): Second image
            similarity (float): Similarity score
            is_match (bool): Whether faces match
            threshold (float): Verification threshold
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Load images if paths are provided
        for i, img in enumerate([img1, img2]):
            if isinstance(img, str):
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = img
            
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f'Image {i+1}')
        
        # Set overall title with results
        result_text = "MATCH" if is_match else "NO MATCH"
        color = "green" if is_match else "red"
        
        plt.suptitle(
            f'{result_text}: Similarity = {similarity:.3f} (Threshold = {threshold})',
            color=color,
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()


class ExperimentTracker:
    """
    Simple experiment tracking utility.
    """
    
    def __init__(self, log_file: str = "experiments/experiment_log.json"):
        """
        Initialize experiment tracker.
        
        Args:
            log_file (str): Path to experiment log file
        """
        self.log_file = log_file
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> List[Dict[str, Any]]:
        """Load existing experiments."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def log_experiment(
        self,
        name: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        notes: str = ""
    ):
        """
        Log an experiment.
        
        Args:
            name (str): Experiment name
            parameters (Dict[str, Any]): Experiment parameters
            results (Dict[str, Any]): Experiment results
            notes (str): Additional notes
        """
        experiment = {
            "name": name,
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": parameters,
            "results": results,
            "notes": notes
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
    
    def _save_experiments(self):
        """Save experiments to file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_best_experiment(self, metric: str, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get best experiment based on a metric.
        
        Args:
            metric (str): Metric name
            maximize (bool): Whether to maximize the metric
            
        Returns:
            Optional[Dict[str, Any]]: Best experiment or None
        """
        if not self.experiments:
            return None
        
        valid_experiments = [
            exp for exp in self.experiments
            if metric in exp.get('results', {})
        ]
        
        if not valid_experiments:
            return None
        
        return max(
            valid_experiments,
            key=lambda x: x['results'][metric] if maximize else -x['results'][metric]
        )


# Convenience functions
def setup_project_environment(project_root: Optional[str] = None) -> Tuple[ProjectPaths, Logger, ConfigManager]:
    """
    Setup complete project environment.
    
    Args:
        project_root (Optional[str]): Project root directory
        
    Returns:
        Tuple[ProjectPaths, Logger, ConfigManager]: Project utilities
    """
    paths = ProjectPaths(project_root)
    logger = Logger(log_file=str(paths.experiments_dir / "project.log"))
    config = ConfigManager(str(paths.project_root / "config.json"))
    
    logger.info("Project environment setup complete")
    return paths, logger, config


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example usage
if __name__ == "__main__":
    print("Face Verification Utilities Module")
    print("=================================")
    
    # Setup project environment
    paths, logger, config = setup_project_environment()
    
    print(f"Project root: {paths.project_root}")
    print(f"Data directory: {paths.data_dir}")
    print(f"Source directory: {paths.src_dir}")
    
    # Test configuration
    print(f"Verification threshold: {config.get('model.verification_threshold', 0.8)}")
    
    # Set random seed
    set_random_seed(42)
    
    logger.info("Utilities module loaded successfully")
    print("\nUtilities module ready for use!")