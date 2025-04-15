import ast
import astpretty

with open("flake8_pluggin_eval.py") as f:  
    source = f.read()

tree = ast.parse(source)

astpretty.pprint(tree)