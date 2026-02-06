"""Inference module for the multimodal place recognition system.

This module provides pipelines, data structures, and utilities for running
inference including place recognition, registration, and localization.

This code is based on the OpenPlaceRecognition library (Apache 2.0 License).
Source: https://github.com/OPR-Project/OpenPlaceRecognition
"""

from mmpr.inference.data import (
    LocalizationResult,
    LocalizedCandidate,
    PlaceRecognitionResult,
    RegistrationResult,
    SequencePRDebug,
)
from mmpr.inference.index import (
    FaissFlatIndex,
    Index,
    IndexMetric,
    IndexSchema,
)
from mmpr.inference.io import PointCloudStore
from mmpr.inference.pipelines import (
    LocalizationPipeline,
    PlaceRecognitionPipeline,
    RansacPointCloudRegistrationPipeline,
    SequencePlaceRecognitionPipeline,
)

__all__ = [
    # Data classes
    "PlaceRecognitionResult",
    "SequencePRDebug",
    "RegistrationResult",
    "LocalizedCandidate",
    "LocalizationResult",
    # Index
    "Index",
    "IndexMetric",
    "IndexSchema",
    "FaissFlatIndex",
    # IO
    "PointCloudStore",
    # Pipelines
    "PlaceRecognitionPipeline",
    "SequencePlaceRecognitionPipeline",
    "RansacPointCloudRegistrationPipeline",
    "LocalizationPipeline",
]
