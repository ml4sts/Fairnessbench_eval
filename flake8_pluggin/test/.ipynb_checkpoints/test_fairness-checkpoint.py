import ast
import textwrap
import pytest
from flake8_pluggin_eval import Fairnessevaluator

def run_checker(code: str):
    tree = ast.parse(textwrap.dedent(code))
    checker = Fairnessevaluator(tree)
    issues = list(checker.run())
    return issues, checker.score

def test_missing_fairness_library():
    code = "import pandas as pd"
    issues, score = run_checker(code)

    # should flag missing bias mitigation (FNA104)
    assert any(msg.startswith("FNA104:") for _, _, msg, _ in issues)

    # only data collection rewarded (+15), no other +points
    assert score == 15

def test_disparate_impact_ratio_detected():
    code = '''
        from aif360.sklearn.metrics import disparate_impact_ratio
        def evaluate():
            return disparate_impact_ratio(y_true, y_pred, prot_attr, priv_group=1)
    '''
    issues, score = run_checker(code)

    # should not flag missing fairness metrics (FNA105)
    assert not any(msg.startswith("FNA105: No fairness metrics") for _, _, msg, _ in issues)

    # should report that we found disparate_impact_ratio under FNA105
    assert any("Found disparate_impact_ratio" in msg for _, _, msg, _ in issues)

def test_detects_categorical_encoding():
    code = '''
        import pandas as pd

        def preprocess(data):
            return pd.get_dummies(data)
    '''
    issues, score = run_checker(code)

    # should not flag missing encoding (FNA103)
    assert not any(msg.startswith("FNA103: No categorical encoding") for _, _, msg, _ in issues)

    # should report that we found get_dummies under FNA103
    assert any("Found get_dummies" in msg for _, _, msg, _ in issues)

def test_flags_missing_categorical_encoding():
    code = '''
        import pandas as pd

        def preprocess(data):
            return data.copy()
    '''
    issues, score = run_checker(code)

    # Should flag missing encoding
    assert any(msg.startswith("FNA103: No categorical encoding") for _, _, msg, _ in issues)

def test_detects_fairness_model_training():
    code = '''
        def adversarial_debiasing():
            pass
    '''
    issues, score = run_checker(code)

    # should not flag missing training techniques (FNA106)
    assert not any(msg.startswith("FNA106: No fairness-aware training") for _, _, msg, _ in issues)

    # should report that we found adversarial under FNA106
    assert any("Found adversarial" in msg for _, _, msg, _ in issues)

def test_flags_missing_fairness_training():
    code = '''
        def train_model():
            pass
    '''
    issues, score = run_checker(code)

    # should flag missing training techniques
    assert any(msg.startswith("FNA106: No fairness-aware training") for _, _, msg, _ in issues)
