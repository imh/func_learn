from rules import var, lam, app, booleq, and_expr, or_expr, not_expr, \
  ifthenelse, gthan, lthan, plus_expr, times_expr, div_expr, int_negate, \
  bool_expr, int_expr, mk_expression_of_type
from type_sys import FunType, BoolType, IntType, Name, Expression
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

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
# spec['choice'] = {rule: np.array(0.5) for rule in spec['rules']}
# spec['choice_keys'] = {rule: rule for rule in spec['rules']}

eval_spec = dict(spec)
eval_spec['rules'] = eval_spec['rules'] + [app, bool_expr, int_expr]


class ProgModel(Chain):
    def __init__(self, n_progs, dim):
        self.normalize = F.LogSumExp()
        super(ProgModel, self).__init__(
            rule_embeddings=L.EmbedID(len(spec['rules']), dim),
            init_state=L.EmbedID(n_progs, dim),
            state2state=L.Linear(dim, dim),
            choice2state=L.Linear(dim, dim)
            )
model = ProgModel(1, 10)
optimizer = optimizers.SGD(lr=0.000001)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))

x1s = np.random.randint(-100, 100, size=100)
x2s = np.random.randint(-100, 100, size=100)
ys = x1s

class LossComputer(object):
    def __init__(self, ok, lp, errs):
        self.ok = ok
        self.lp = lp
        self.errs = np.array(errs)

for i in xrange(2000):
    print 'starting epoch {}'.format(i)
    tries = 0
    context = dict()
    context['model'] = model
    # context['choice'] = np.array(0.5)
    context['max_depth'] = 2

    lcs = []
    optimizer.zero_grads()
    n_ok = 0
    tries = 0
    while not (n_ok >= 10 and tries >= 100):
        context['depth'] = 0
        context['state'] = model.init_state(Variable(np.array([0], dtype=np.int32)))
        context['lp'] = 0
        e, context, ok = mk_expression_of_type(spec, context, set(), FunType(IntType(), FunType(IntType(), IntType())))
        errs = []
        if ok:
            print '{:>10}'.format(tries), e.pstr()
            for x1, x2, y in zip(x1s, x2s, ys):
                x, y = np.random.randint(1,11, 2)
                x1e = Expression(constructor='int', type_def=IntType(), val=x1)
                x2e = Expression(constructor='int', type_def=IntType(), val=x2)
                outer_app = Expression(constructor='app',
                    type_def=FunType(IntType(), IntType()),
                    fun=e, arg=x1e)
                inner_app = Expression(constructor='app',
                    type_def=IntType(),
                    fun=outer_app, arg=x2e)
                yhat = app.eval_expr(eval_spec, inner_app, empty_env())
                errs.append((yhat - y)**2)
            n_ok += 1
        tries += 1
        lcs.append(LossComputer(ok, context['lp'], errs))
    ok_errs = [lc.errs for lc in lcs if lc.ok]
    if len(ok_errs) > 0:
        print 'ok, update'
        mean_ok_errs = np.mean(ok_errs)
        max_ok_errs = np.max(ok_errs)
        if max_ok_errs > 5e3:
            scal = 1.0 #5e3 / max_ok_errs
        else:
            scal = 1.0
        print 'computed errs'
        loss = Variable(np.array([0], dtype=np.float32))
        print 'adding losses'
        n_ok = len([lc for lc in lcs if lc.ok])
        for lc in lcs:
            err = np.mean(lc.errs) if lc.ok else max_ok_errs+100
            loss += float(err - mean_ok_errs) * lc.lp
        loss /= len(lcs)

        print 'added losses'
        print 'LOSS_OK (mean={}, max={}):'.format(mean_ok_errs, max_ok_errs), '{} / {} ok'.format(len([lc for lc in lcs if lc.ok]), len(lcs)),
        loss.backward()
        optimizer.update()