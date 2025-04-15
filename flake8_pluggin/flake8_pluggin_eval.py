import ast
import sys 
if sys.version_info < (3, 8):
    import importlib_metadata
else: 
    import importlib.metadata as importlib_metadata
class Fairnessevaluator:
    name = __name__
    version = importlib_metadata.version(__name__)

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        # Comment to do
        self.score = 80
        self.issues = []
    def run(self): 
        self.check_data_collection()
        # self.check_missing_value_handling
        self.check_categorical_encoding()
        self.check_bias_mitigation()
        self.check_fairness_metrics()
        self.check_model_training()
        self.check_evaluation()

        for line, col, msg in self.issues:
            yield line, col, msg, type(self)
        print(f"Fairness Score: {self.score}/80")
    # format on how the error message should look like, it takes as input the line, column and the message   
    def add_issue(self, node, message, deduction):
        lineno = getattr(node, 'lineno', 1)
        col_offset = getattr(node, 'col_offset', 0)
        self.issues.append((lineno, col_offset, message))
        self.score -= deduction    

    def check_data_collection(self):
        """checking for dataset representativeness and privacy trade-offs."""
        # You make sure that it should return exactly which one its missing, which library we want to check
        data_libs = {"pandas", "numpy", "sklearn", "datasets"}
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import) and any(n.name in data_libs for n in node.names):
                found = True
                break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA101: No dataset processing library found (e.g., pandas, numpy, sklearn, datasets).", 15)
            
    def check_missing_value_handling(self):
        """checking if code addresses missing values."""
        missing_funcs = {"dropna", "fillna"}
        found = False
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Attribute) and node.attr in missing_funcs) or \
               (isinstance(node, ast.Name) and node.id in missing_funcs):
                found = True
                break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA102: No handling of missing values detected (e.g., dropna, fillna).", 10)
    def check_categorical_encoding(self):
        """checking for encoding of categorical variables."""
        encoding_funcs = {"get_dummies", "OneHotEncoder", "LabelEncoder"}
        found = False
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Name) and node.id in encoding_funcs) or \
               (isinstance(node, ast.Attribute) and node.attr in encoding_funcs):
                found = True
                break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA103: No categorical encoding found (e.g., get_dummies, OneHotEncoder, LabelEncoder).", 10)
    def check_bias_mitigation(self):
        """check for bias mitigation techniques in the code."""
        fairness_libs = {"aif360", "fairlearn", "equitas", "fairness_indicator"}
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in fairness_libs:
                        found = True
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in fairness_libs:
                    found = True
                    break
            if found:
                break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA104: No bias mitigation techniques found (e.g., aif360, fairlearn, equitas, fairness_indicator).", 15)
    # add more metrics https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html
    # Check disagreegated fairness metrics eg groupby.apply.......
    def check_fairness_metrics(self):
        """Ensure appropriate fairness metrics are used."""
        metric_functions = {"equalized_odds", "demographic_parity", "statistical_parity", "disparate_impact_ratio"}
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in metric_functions:
                found = True
                break
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in metric_functions:
                    found = True
                    break
                elif isinstance(node.func, ast.Attribute) and node.func.attr in metric_functions:
                    found = True
                    break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA105: No fairness metrics function found (e.g., equalized_odds, demographic_parity, statistical_parity, disparate_impact_ratio).", 10)           
    def check_model_training(self):
        """Ensure fairness mitigation techniques are considered during training."""
        # TODO: add more https://aif360.readthedocs.io/en/latest/modules/algorithms.html
        fairness_terms = {"adversarial", "reweighting"}
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and any(term in node.name for term in fairness_terms):
                found = True
                break
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in fairness_terms:
                    found = True
                    break
                elif isinstance(node.func, ast.Attribute) and node.func.attr in fairness_terms:
                    found = True
                    break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA106: No fairness-aware training techniques found (e.g., adversarial, reweighting.)", 10)
    def check_evaluation(self):
        """Ensure fairness evaluation and bias auditing is performed."""
        eval_functions = {"audit_bias", "disparate_impact_ratio"}
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in eval_functions:
                found = True
                break
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in eval_functions:
                    found = True
                    break
                elif isinstance(node.func, ast.Attribute) and node.func.attr in eval_functions:
                    found = True
                    break
        if not found:
            self.add_issue(next(ast.walk(self.tree)), "FNA107: No fairness evaluation or auditing function found (e.g.,audit_bias, disparate_impact_ratio.", 10)

    