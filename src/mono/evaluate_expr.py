
empty_env = lambda: {}

def evaluate_expression(expr, term_env):
    constructor = expr.constructor
    if constructor == 'var':
        name = expr.name
        return term_env.lookup('name')
    elif constructor == 'lam':
        name = expr.name
        body = expr.body
        return # closure(name=name, body=body)
    elif constructor == 'app':
        fun = expr.fun
        arg = expr.arg
        clos = evaluate_expression(fun, term_env)
        arg_val = evaluate_expression(arg, term_env)
        new_env = copy_env(term_env)
        new_env.assign(clos.name, arg_val)
        return evaluate_expression(clos.body, new_env)
    elif constructor == 'ebool':
        return expr.val
    elif constructor == 'booleq':
        arg1 = evaluate_expression(expr.arg1, term_env)
        arg2 = evaluate_expression(expr.arg2, term_env)
        return arg1 == arg2
    elif constructor == 'ifthenelse':
        prop = evaluate_expression(expr.prop, term_env)
        if prop:
            return evaluate_expression(expr.true_branch, term_env)
        else:
            return evaluate_expression(expr.false_branch, term_env)
    elif ...