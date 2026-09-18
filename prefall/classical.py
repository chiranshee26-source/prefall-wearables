"""Step 2a: feature-based models from slide 11 (Random Forest, SVM, Decision Tree)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_models(seed=0):
    """Hyperparameters exactly as on slide 11. class_weight='balanced' because pre-fall
    windows are ~2% of the data."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8,
                                                class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
    }


def forest_kb(rf, bytes_per_node=12):
    """Rough MCU flash estimate: each node stores feature id, float threshold, child links."""
    nodes = sum(t.tree_.node_count for t in rf.estimators_)
    return nodes, nodes * bytes_per_node / 1024
