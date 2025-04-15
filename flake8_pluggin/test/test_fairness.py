import ast 
import textwrap
from flake8_pluggin_eval import Fairnessevaluator

def run_checker(code):
    tree = ast.parse(textwrap.dedent(code))
    checker = Fairnessevaluator(tree)
    return list(checker.run()), checker.score

def test_missing_fairness_library():
    code = "import pandas as pd"
    issues, score = run_checker(code)
    assert any("FNA104" in msg for _, _, msg, _ in issues)
    assert score < 100

def test_disparate_impact_ratio_detected():
    code = '''
         from aif360.sklearn.metrics import disparate_impact_ratio
         def evaluate():
             return disparate_impact_ratio(y_true, y_pred, prot_attr, priv_group=1)
    '''
    issues, score = run_checker(code)
    assert not any("FNA105" in msg for _, _, msg, _ in issues)

def test_detects_categorical_encoding():
    code = '''
         import pandas as pd

         def preprocess(data):
             return pd.get_dummies(data)
    '''
    issues, score = run_checker(code)
    assert not any("FNA103" in msg for _, _, msg, _ in issues), "Expected categorical encoding to be detected"

def test_flags_missing_categorical_encoding():
    code = '''
         import pandas as pd

         def preprocess(data):
             return data.copy()
    '''
    issues, score = run_checker(code)
    assert any("FNA103" in msg for _, _, msg, _ in issues), "Expected missing categorical encoding to be flagged"

def test_detects_fairness_model_training():
    code = '''
         def adversarial_debiasing():
             pass
    '''
    issues, score = run_checker(code)
    assert not any("FNA106" in msg for _, _, msg, _ in issues), "Expected fairness-aware training technique to be detected"

def test_flags_missing_fairness_training():
    code = '''
         def train_model():
             pass
    '''
    issues, score = run_checker(code)
    assert any("FNA106" in msg for _, _, msg, _ in issues), "Expected missing fairness-aware training to be flagged"
