import re
from typing import Tuple, Union

def parse_and_evaluate(symbols):
    """
    Enhanced parser with better error handling and equation solving
    Takes a list of predicted symbol labels and returns expression and result
    No decimal point support - removed 'dec' handling
    """
    if not symbols:
        return "", "No symbols detected"
    
    # Mapping for operators (removed 'dec')
    mapping = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "eq": "="
    }
    
    # Build expression string
    expr = ""
    for sym in symbols:
        expr += mapping.get(sym, sym)
    
    # Clean up expression
    expr = expr.strip()
    
    if not expr:
        return "", "Empty expression"
    
    # Handle equations (contains '=')
    if '=' in expr:
        parts = expr.split('=')
        
        if len(parts) != 2:
            return expr, "Invalid equation format (multiple '=' signs)"
        
        left, right = parts[0].strip(), parts[1].strip()
        
        if not left or not right:
            return expr, "Incomplete equation"
        
        try:
            # Evaluate both sides
            left_result = safe_eval(left)
            right_result = safe_eval(right)
            
            # Check if equation is valid
            if abs(left_result - right_result) < 1e-9:
                return expr, f"✓ Equation is correct ({left_result} = {right_result})"
            else:
                return expr, f"✗ Equation is incorrect ({left_result} ≠ {right_result})"
                
        except Exception as e:
            return expr, f"Cannot evaluate equation: {str(e)}"
    
    # Handle regular expressions
    else:
        try:
            result = safe_eval(expr)
            return expr, result
        except ZeroDivisionError:
            return expr, "Error: Division by zero"
        except Exception as e:
            return expr, f"Invalid expression: {str(e)}"


def safe_eval(expression: str) -> float:
    """
    Safely evaluate mathematical expressions
    Only allows numbers and basic operators
    """
    # Remove any whitespace
    expression = expression.replace(" ", "")
    
    # Check for invalid characters (removed decimal point from allowed)
    allowed_chars = set("0123456789+-*/()")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Expression contains invalid characters")
    
    # Check for empty expression
    if not expression or expression == "":
        raise ValueError("Empty expression")
    
    # Prevent dangerous patterns
    if "__" in expression or "import" in expression:
        raise ValueError("Invalid expression")
    
    try:
        # Use eval with restricted namespace for safety
        result = eval(expression, {"__builtins__": {}}, {})
        
        # Ensure result is a number
        if not isinstance(result, (int, float)):
            raise ValueError("Result is not a number")
        
        return float(result)
        
    except SyntaxError:
        raise ValueError("Syntax error in expression")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero")
    except Exception as e:
        raise ValueError(f"Evaluation error: {str(e)}")


def format_expression(symbols):
    """
    Format symbols into a readable mathematical expression
    Removed decimal point support
    """
    mapping = {
        "add": " + ",
        "sub": " - ",
        "mul": " × ",
        "div": " ÷ ",
        "eq": " = "
    }
    
    expr = ""
    for i, sym in enumerate(symbols):
        if sym in mapping:
            expr += mapping[sym]
        else:
            expr += sym
    
    return expr.strip()