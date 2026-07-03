# src/metrics/evaluator.py
import numpy as np

class AccuracyEvaluator:
    """
    Accuracy evaluation engine for face verification and identification.
    
    Metrics Computed:
    - Verification: Accuracy, Precision, Recall, F1, FAR, FRR, TAR
    - Identification: Closed-set accuracy, Open-set Unknown-rejection rate
    - Pose Analysis: Segmented accuracy by angular deviation (0-15, 15-30, 30-40, 40-50 deg)
    - Lighting Analysis: Segmented accuracy by illumination (normal, low, side, shadow)
    - Calibration: ROC threshold sweep curves
    """
    def __init__(self):
        pass

    @staticmethod
    def calculate_verification_metrics(labels, distances, threshold):
        """
        Computes standard binary verification metrics at a given threshold.
        Args:
            labels: np.ndarray [M] (1 for same identity, 0 for different)
            distances: np.ndarray [M] (pairwise face distances)
            threshold: float, matching cutoff
        Returns:
            dict of metrics
        """
        labels = np.array(labels, dtype=bool)
        distances = np.array(distances, dtype=float)
        
        # Predictions (distance < threshold means they are verified as the same person)
        predictions = distances < threshold
        
        TP = np.sum(predictions & labels)
        TN = np.sum(~predictions & ~labels)
        FP = np.sum(predictions & ~labels)
        FN = np.sum(~predictions & labels)
        
        total = len(labels)
        if total == 0:
            return {}
            
        accuracy = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # FAR = FP / (FP + TN) (False Acceptance Rate: fraction of negative pairs incorrectly accepted)
        far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        # FRR = FN / (FN + TP) (False Rejection Rate: fraction of positive pairs incorrectly rejected)
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        # TAR = TP / (TP + FN) (True Acceptance Rate, same as Recall)
        tar = recall
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_acceptance_rate": float(far),
            "false_rejection_rate": float(frr),
            "true_acceptance_rate": float(tar),
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN)
        }

    @staticmethod
    def calculate_identification_metrics(true_ids, pred_ids, distances, threshold, unknown_id=-1):
        """
        Computes closed-set and open-set identification accuracy.
        Args:
            true_ids: np.ndarray [M] (actual person IDs)
            pred_ids: np.ndarray [M] (predicted best-match database IDs)
            distances: np.ndarray [M] (distance to the predicted match)
            threshold: float, matching cutoff (if distance >= threshold, predicted as unknown_id)
            unknown_id: ID used for unknown identities
        """
        true_ids = np.array(true_ids)
        pred_ids = np.array(pred_ids)
        distances = np.array(distances)
        
        # Apply threshold filtering: if closest match is too far, classify as unknown
        final_predictions = np.where(distances < threshold, pred_ids, unknown_id)
        
        # Segment into known queries vs unknown queries
        is_known = true_ids != unknown_id
        is_unknown = true_ids == unknown_id
        
        closed_set_correct = 0
        closed_set_total = np.sum(is_known)
        
        open_set_correct = 0
        open_set_total = len(true_ids)
        
        # Unknown rejection tracking
        unknown_rejected = 0
        unknown_total = np.sum(is_unknown)
        
        if closed_set_total > 0:
            # For known queries, match must be correct AND under the threshold
            closed_set_correct = np.sum((final_predictions == true_ids) & is_known)
            closed_set_accuracy = closed_set_correct / closed_set_total
        else:
            closed_set_accuracy = 1.0
            
        if unknown_total > 0:
            # For unknown queries, match must be rejected (classified as unknown)
            unknown_rejected = np.sum((final_predictions == unknown_id) & is_unknown)
            unknown_rejection_rate = unknown_rejected / unknown_total
        else:
            unknown_rejection_rate = 1.0
            
        if open_set_total > 0:
            open_set_correct = np.sum(final_predictions == true_ids)
            open_set_accuracy = open_set_correct / open_set_total
        else:
            open_set_accuracy = 1.0
            
        return {
            "identification_accuracy": float(open_set_accuracy),
            "closed_set_accuracy": float(closed_set_accuracy),
            "unknown_rejection_rate": float(unknown_rejection_rate),
            "total_queries": int(open_set_total),
            "known_queries": int(closed_set_total),
            "unknown_queries": int(unknown_total)
        }

    @staticmethod
    def sweep_thresholds(labels, distances, steps=100, is_cosine=False):
        """
        Sweeps through threshold ranges to calculate ROC and calibration curves.
        """
        max_thresh = 1.0 if is_cosine else 2.5
        thresholds = np.linspace(0.0, max_thresh, steps)
        
        far_list = []
        tar_list = []
        acc_list = []
        f1_list = []
        
        best_f1 = 0.0
        best_threshold = 0.0
        
        for t in thresholds:
            m = AccuracyEvaluator.calculate_verification_metrics(labels, distances, t)
            if not m:
                continue
            far_list.append(m["false_acceptance_rate"])
            tar_list.append(m["true_acceptance_rate"])
            acc_list.append(m["accuracy"])
            f1_list.append(m["f1_score"])
            
            if m["f1_score"] > best_f1:
                best_f1 = m["f1_score"]
                best_threshold = t
                
        return {
            "thresholds": thresholds.tolist(),
            "FAR": far_list,
            "TAR": tar_list,
            "accuracies": acc_list,
            "f1_scores": f1_list,
            "optimal_threshold": float(best_threshold),
            "max_f1": float(best_f1)
        }
