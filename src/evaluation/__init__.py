"""Evaluation modules for OceanEmbed."""

from .metrics import OceanMetrics, compute_depth_wise_metrics
from .argo_validator import ArgoValidator

__all__ = ['OceanMetrics', 'compute_depth_wise_metrics', 'ArgoValidator']
