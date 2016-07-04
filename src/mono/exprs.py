from rules import var, lam, app, booleq, and_expr, or_expr, not_expr, \
  ifthenelse, gthan, lthan, plus_expr, times_expr, div_expr, int_negate, \
  bool_expr, int_expr, mk_expression_of_type
from type_sys import FunType, BoolType, IntType, Name, Expression
import numpy as np

empty_env = lambda: dict()
spec = dict()
spec['rules'] = [var,
                 lam,
                 # app,
                 booleq,
                 and_expr,
                 or_expr,
                 not_expr,
                 ifthenelse,
                 gthan,
                 lthan,
                 plus_expr,
                 times_expr,
                 div_expr,
                 int_negate
#                  bool_expr,
#                  int_expr
                ]
spec['choice'] = {rule: np.array(0.5) for rule in spec['rules']}
spec['choice_keys'] = {rule: rule for rule in spec['rules']}

eval_spec = dict(spec)
eval_spec['rules'] = eval_spec['rules'] + [app, bool_expr, int_expr]

ok = False
tries = 0
context = dict()
context['choice'] = np.array(0.5)
context['depth'] = 0
context['max_depth'] = 8
for i in xrange(200):
    for tries in xrange(10000):
        e, context, ok = mk_expression_of_type(spec, context, set(), FunType(IntType(), FunType(IntType(), IntType())))
        # e, context, ok = mk_expression_of_type(spec, context, set(), FunType(IntType(), IntType()))
        if ok:
            break
    if ok:
        print '{:>10}'.format(tries), e.pstr()
        for j in xrange(5):
            x, y = np.random.randint(1,11, 2)
            xe = Expression(constructor='int', type_def=IntType(), val=x)
            ye = Expression(constructor='int', type_def=IntType(), val=y)
            outer_app = Expression(constructor='app',
                type_def=FunType(IntType(), IntType()),
                fun=e, arg=xe)
            inner_app = Expression(constructor='app',
                type_def=IntType(),
                fun=outer_app, arg=ye)
            val = app.eval_expr(eval_spec, inner_app, empty_env())
            print '\t\t\t{} `op` {} --> {}'.format(x, y, val)
    else:
        print '{:>10}'.format(tries), e.pstr()