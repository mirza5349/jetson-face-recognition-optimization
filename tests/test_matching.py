# tests/test_matching.py
import os
import sys
import numpy as np

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.matching.matcher import FaceMatcher

def test_euclidean_distance_computation():
    """
    Verifies Euclidean distance calculations on single embeddings.
    """
    matcher = FaceMatcher(metric="euclidean", threshold=0.6)
    
    # Test identical vectors (distance should be 0)
    emb1 = np.array([0.3, 0.4, 0.0] + [0.0] * 509, dtype=np.float32)
    assert np.allclose(matcher.compute_distance(emb1, emb1), 0.0), "Identical embeddings must have 0.0 Euclidean distance"
    
    # Test known distance
    emb2 = np.array([0.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32) # origin
    emb3 = np.array([3.0, 4.0, 0.0] + [0.0] * 509, dtype=np.float32) # point (3,4)
    # L2 distance should be sqrt(3^2 + 4^2) = 5.0
    assert np.allclose(matcher.compute_distance(emb2, emb3), 5.0), "Euclidean distance calculation failed standard L2 test"

def test_cosine_distance_computation():
    """
    Verifies Cosine distance calculations. Because embeddings are L2-normalized,
    Cosine distance = 1.0 - dot_product(n_emb1, n_emb2).
    """
    matcher = FaceMatcher(metric="cosine", threshold=0.4)
    
    # Identical vectors (cosine similarity = 1.0 -> distance = 0.0)
    emb1 = np.random.normal(0, 1, 512).astype(np.float32)
    assert np.allclose(matcher.compute_distance(emb1, emb1), 0.0, atol=1e-5)
    
    # Orthogonal vectors (cosine similarity = 0.0 -> distance = 1.0)
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[0] = 1.0
    emb3 = np.zeros(512, dtype=np.float32)
    emb3[1] = 1.0
    
    assert np.allclose(matcher.compute_distance(emb2, emb3), 1.0, atol=1e-5), "Orthogonal vectors must have Cosine distance of 1.0"

def test_match_verification_thresholds():
    """
    Asserts that verification threshold flags match correctness boundaries.
    """
    # Euclidean Matcher (threshold 1.0)
    matcher_euc = FaceMatcher(metric="euclidean", threshold=1.0)
    emb1 = np.zeros(512, dtype=np.float32)
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[0] = 0.5
    
    is_same, dist = matcher_euc.verify(emb1, emb2)
    assert is_same, f"Distance {dist} is under 1.0, should verify as True"
    
    emb3 = np.zeros(512, dtype=np.float32)
    emb3[0] = 1.5
    is_same, dist = matcher_euc.verify(emb1, emb3)
    assert not is_same, f"Distance {dist} is over 1.0, should verify as False"

def test_vectorized_database_sweep():
    """
    Asserts that vectorized search sweeps correctly scan databases, identify
    closest indices, and execute in sub-millisecond times.
    """
    matcher = FaceMatcher(metric="cosine", threshold=0.4)
    
    # Build database of 100 random identities (L2 normalized)
    db = np.random.normal(0, 1, (100, 512)).astype(np.float32)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    
    # Query is exact match of identity at index 42
    query = db[42].copy()
    
    distances, best_idx = matcher.search_database(query, db)
    
    assert best_idx == 42, f"Vectorized sweep failed to find exact match. Expected 42, got {best_idx}"
    assert np.allclose(distances[42], 0.0, atol=1e-5), f"Distance to exact match should be 0.0, got {distances[42]}"
    assert len(distances) == 100, "Should return distances for all database records"
