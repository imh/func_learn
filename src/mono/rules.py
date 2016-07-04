from type_sys import FunType, BoolType, IntType, Name, Expression
import numpy as np

class Closure(object):
    def __init__(self, name, body, env):
        self.name = name
        self.body = body
        self.env = env

def mk_new_name_of_type(params, context, scope, type_def):
    ttd = type(type_def)
    if ttd == FunType:
        pref = 'f'
    elif ttd == BoolType:
        pref = 'b'
    elif ttd == IntType:
        pref = 'i'
    else:
        pref = 'x'
    i = 0
    while True:
        name = '{}{}'.format(pref, i)
        if any([var.name == name for var in scope]):
            i += 1
        else:
            return Name(name, type_def), context

def mk_new_int(params, context):
    return np.random.randint(1000), context

def mk_new_bool(params, context):
    return (np.random.rand() <= 0.5), context

class LanguageRule(object):
    def __init__(self, rule_name, make_expr, can_make, eval_expr):
        self.rule_name = rule_name
        self.make_expr = make_expr
        self.can_make = can_make
        self.eval_expr = eval_expr

def choose_var_of_type(spec, context, scope, type_def):
    compatible_scope = [var for var in scope if var.type_def.can_be(type_def)]
    return np.random.choice(compatible_scope), context

def spec_lookup(spec, expr):
    return next(x for x in spec['rules'] if x.rule_name == expr.constructor)

def make_var_of_type(spec, context, scope, type_def):
    name, context = choose_var_of_type(spec, context, scope, type_def)
    return Expression(constructor='var', type_def=type_def, name=name), context, True
def eval_var(spec, expr, term_env):
    name = expr.name
    return term_env[name]
def var_can_make(scope, type_def):
    return any([var.type_def.can_be(type_def) for var in scope])
var = LanguageRule('var', make_var_of_type, var_can_make, eval_var)

def make_lam_of_type(spec, context, scope, type_def):
    #typedef must be function(a, b)
    in_type = type_def.in_type
    out_type = type_def.out_type
    name, context = mk_new_name_of_type(spec, context, scope, in_type)
    new_scope = set(scope)
    new_scope.add(name)
    context['depth'] += 1
    body, context, ok = mk_expression_of_type(spec, context, new_scope, out_type)
    context['depth'] -= 1
    return Expression(constructor='lam', type_def=type_def, name=name, body=body), context, ok
def eval_lam(spec, expr, term_env):
    name = expr.name
    body = expr.body
    return Closure(name, body, term_env)
def lam_can_make(scope, type_def):
    return type_def.is_function()
lam = LanguageRule('lam', make_lam_of_type, lam_can_make, eval_lam)


def make_app_of_type(spec, context, scope, type_def):
    arg_type = ForallType('a')
    fun_type = FunType(arg_type, type_def)
    context['depth'] += 1
    fun, context, ok1 = mk_expression_of_type(spec, context, scope, fun_type)
    arg_type = fun.type_def.in_type
    arg, context, ok2 = mk_expression_of_type(spec, context, scope, arg_type)
    context['depth'] -= 1
    return Expression(constructor='app', type_def=type_def, fun=fun, arg=arg), context, ok1 and ok2
def eval_app(spec, expr, term_env):
    fun = expr.fun
    arg = expr.arg
    clos = spec_lookup(spec, fun).eval_expr(spec, fun, term_env)
    arg_val = spec_lookup(spec, arg).eval_expr(spec, arg, term_env)
    new_env = dict(clos.env, **term_env)
    new_env[clos.name] = arg_val
    return spec_lookup(spec, clos.body).eval_expr(spec, clos.body, new_env)
def app_can_make(scope, type_def):
    return True
app = LanguageRule('app', make_app_of_type, app_can_make, eval_app)


def make_booleq(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, BoolType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, BoolType())
    context['depth'] -= 1
    return Expression(constructor='booleq', type_def=BoolType(), arg1=arg1, arg2=arg2,), context, ok1 and ok2
def eval_booleq(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val == arg2val
def booleq_can_make(scope, type_def):
    return type_def.is_bool()
booleq = LanguageRule('booleq', make_booleq, booleq_can_make, eval_booleq)

def make_bool(spec, context, scope, type_def):
    val, context = mk_new_bool(spec, context)
    return Expression(constructor='bool', type_def=BoolType(), val=val), context, True
def eval_bool(spec, expr, term_env):
    return expr.val
def bool_can_make(scope, type_def):
    return type_def.is_bool()
bool_expr = LanguageRule('bool', make_bool, bool_can_make, eval_bool)

def make_and(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, BoolType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, BoolType())
    context['depth'] -= 1
    return Expression(constructor='and', type_def=BoolType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_and(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val and arg2val
def and_can_make(scope, type_def):
    return type_def.is_bool()
and_expr = LanguageRule('and', make_and, and_can_make, eval_and)

def make_or(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, BoolType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, BoolType())
    context['depth'] -= 1
    return Expression(constructor='or', type_def=BoolType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_or(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val or arg2val
def or_can_make(scope, type_def):
    return type_def.is_bool()
or_expr = LanguageRule('or', make_or, or_can_make, eval_or)

def make_not(spec, context, scope, type_def):
    context['depth'] += 1
    arg, context, ok = mk_expression_of_type(spec, context, scope, BoolType())
    context['depth'] -= 1
    return Expression(constructor='not', type_def=BoolType(), arg=arg), context, ok
def eval_not(spec, expr, term_env):
    arg = expr.arg
    return spec_lookup(spec, arg).eval_expr(spec, arg, term_env)
def not_can_make(scope, type_def):
    return type_def.is_bool()
not_expr = LanguageRule('not', make_not, not_can_make, eval_not)

def make_ifthenelse_of_type(spec, context, scope, type_def):
    context['depth'] += 1
    prop, context, ok0 = mk_expression_of_type(spec, context, scope, BoolType())
    true_branch, context, ok1 = mk_expression_of_type(spec, context, scope, type_def)
    false_branch, context, ok2 = mk_expression_of_type(spec, context, scope, type_def)
    context['depth'] -= 1
    return Expression(constructor='ifthenelse', type_def=type_def, prop=prop, true_branch=true_branch,
                      false_branch=false_branch), context, ok0 and ok1 and ok2
def eval_ifthenelse(spec, expr, term_env):
    prop = expr.prop
    true_branch = expr.true_branch
    false_branch = expr.false_branch
    prop_val = spec_lookup(spec, prop).eval_expr(spec, prop, term_env)
    if prop:
        return spec_lookup(spec, true_branch).eval_expr(spec, true_branch, term_env)
    else:
        return spec_lookup(spec, false_branch).eval_expr(spec, false_branch, term_env)
def ifthenelse_can_make(scope, type_def):
    return True
ifthenelse = LanguageRule('ifthenelse', make_ifthenelse_of_type,
    ifthenelse_can_make, eval_ifthenelse)

def make_int(spec, context, scope, type_def):
    val, context = mk_new_int(spec, context)
    return Expression(constructor='int', type_def=IntType(), val=val), context, True
def eval_int(spec, expr, term_env):
    return expr.val
def int_can_make(scope, type_def):
    return type_def.is_int()
int_expr = LanguageRule('int', make_int, int_can_make, eval_int)

def make_plus(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='plus', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_plus(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val + arg2val
def plus_can_make(scope, type_def):
    return type_def.is_int()
plus_expr = LanguageRule('plus', make_plus, plus_can_make, eval_plus)

def make_times(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='times', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_times(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val * arg2val
def times_can_make(scope, type_def):
    return type_def.is_int()
times_expr = LanguageRule('times', make_times, times_can_make, eval_times)

def make_div(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='div', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_div(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val / arg2val
def div_can_make(scope, type_def):
    return type_def.is_int()
div_expr = LanguageRule('div', make_div, div_can_make, eval_div)

def make_int_negate(spec, context, scope, type_def):
    context['depth'] += 1
    arg, context, ok = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='int_negate', type_def=IntType(), arg=arg), context, ok
def eval_int_negate(spec, expr, term_env):
    arg = expr.arg
    return -spec_lookup(spec, arg).eval_expr(spec, arg, term_env)
def int_negate_can_make(scope, type_def):
    return type_def.is_int()
int_negate = LanguageRule('int_negate', make_int_negate, int_negate_can_make,
    eval_int_negate)

def make_gthan(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='gthan', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_gthan(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val > arg2val
def gthan_can_make(scope, type_def):
    return type_def.is_bool()
gthan = LanguageRule('gthan', make_gthan, gthan_can_make, eval_gthan)

def make_lthan(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='lthan', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_lthan(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val < arg2val
def lthan_can_make(scope, type_def):
    return type_def.is_bool()
lthan = LanguageRule('lthan', make_lthan, lthan_can_make, eval_lthan)

def make_inteq(spec, context, scope, type_def):
    context['depth'] += 1
    arg1, context, ok1 = mk_expression_of_type(spec, context, scope, IntType())
    arg2, context, ok2 = mk_expression_of_type(spec, context, scope, IntType())
    context['depth'] -= 1
    return Expression(constructor='inteq', type_def=IntType(), arg1=arg1, arg2=arg2), context, ok1 and ok2
def eval_or(spec, expr, term_env):
    arg1 = expr.arg1
    arg2 = expr.arg2
    arg1val = spec_lookup(spec, arg1).eval_expr(spec, arg1, term_env)
    arg2val = spec_lookup(spec, arg2).eval_expr(spec, arg2, term_env)
    return arg1val == arg2val
def inteq_can_make(scope, type_def):
    return type_def.is_bool()
inteq = LanguageRule('inteq', make_inteq, inteq_can_make, eval_or)

# def make_nil_of_type(spec, context, scope, type_def):
#     return Expression(constructor='nil', type_def=type_def), context, True
# def nil_can_make(scope, type_def):
#     return type_def.is_list()
# nil = LanguageRule('nil', make_nil_of_type, nil_can_make)

# def make_cons_of_type(spec, context, scope, type_def):
#     context['depth'] += 1
#     hd, context, ok1 = mk_expression_of_type(spec, context, scope, type_def.elem_type)
#     tl, context, ok2 = mk_expression_of_type(spec, context, scope, type_def)
#     context['depth'] -= 1
#     return Expression(constructor='cons', type_def=type_def, hd=hd, tl=tl), context, ok1 and ok2
# def cons_can_make(scope, type_def):
#     return type_def.is_list()
# cons = LanguageRule('cons', make_cons_of_type, cons_can_make)

# def make_head_of_type(spec, context, scope, type_def):
#     context['depth'] += 1
#     lst, context, ok = mk_expression_of_type(spec, context, scope, ListType(type_def))
#     context['depth'] -= 1
#     return Expression(constructor='head', type_def=type_def, lst=lst), context, ok
# def head_can_make(spec, context, scope, type_def):
#     return True
# head = LanguageRule('head', make_head_of_type, head_can_make)

# def make_tail_of_type(spec, context, scope, type_def):
#     context['depth'] += 1
#     lst, context, ok = mk_expression_of_type(spec, context, scope, type_def)
#     context['depth'] -= 1
#     return Expression(constructor='tail', type_def=type_def, lst=lst), context, ok
# def tail_can_make(spec, context, scope, type_def):
#     return True
# tail = LanguageRule('tail', make_head_of_type, head_can_make)

# def make_tuple_of_type(spec, context, scope, type_def):
#     context['depth'] += 1
#     fst, context, ok1 = mk_expression_of_type(spec, context, scope, type_def.fst_type)
#     snd, context, ok2 = mk_expression_of_type(spec, context, scope, type_def.snd_type)
#     context['depth'] -= 1
#     return Expression(constructor='tuple', type_def=type_def, fst=fst, snd=snd), context, ok
# def tuple_can_make(spec, context, scope, type_def):
#     return True
# head = LanguageRule('tuple', make_tuple_of_type, tuple_can_make)

# def make_fst_of_type(spec, context, scope, type_def):
#     context['depth'] += 1
#     tup, context, ok1 = mk_expression_of_type(spec, context, scope, type_def.tup_type)
#     context['depth'] -= 1
#     return Expression(constructor='tuple', type_def=type_def, fst=fst, snd=snd), context, ok
# def tuple_can_make(spec, context, scope, type_def):
#     return True
# head = LanguageRule('tuple', make_tuple_of_type, tuple_can_make)

def prob_from_rule(rule, context, spec):
    rule_key = spec['choice_keys'][rule]
    rule_choice_vec = spec['choice'][rule_key]
    logodds = rule_choice_vec.dot(context['choice'])
    return 1/(1+np.exp(-logodds))

def mk_expression_of_type(spec, context, scope, type_def):
    if context['depth'] > context['max_depth']:
        return Expression('depth_exceeded', type_def=type_def), context, False
    candidates = [rule for rule in spec['rules'] if rule.can_make(scope, type_def)]
    expr_probs = np.array([prob_from_rule(rule, context, spec) for rule in candidates])
    expr_probs /= np.sum(expr_probs)
    i = np.random.choice(range(len(expr_probs)), p=expr_probs)
    rule = candidates[i]
    p = expr_probs[i]
    return rule.make_expr(spec, context, scope, type_def)
