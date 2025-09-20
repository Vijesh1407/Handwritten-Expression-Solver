# Placeholder parser for evaluating recognized expressions.
# Will implement BODMAS rules later.

def evaluate_expression(symbols):
    """
    Input: list of recognized symbols as strings, e.g. ['2', '+', '3']
    Output: Evaluation result
    """
    expression = "".join(symbols)
    try:
        return eval(expression)  # TEMP (Python eval, replace later with safe parser)
    except Exception:
        return "Invalid Expression"
