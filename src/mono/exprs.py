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

# np.random.seed(3) #good for dim=10
# np.random.seed(1) #good for dim=5
# np.random.seed(3)
np.random.seed(2)

class ProgModel(Chain):
    def __init__(self, n_progs, dim):
        self.normalize = F.LogSumExp()
        self.dim = dim
        self.n_progs = n_progs
        super(ProgModel, self).__init__(
            rule_embeddings=L.EmbedID(len(spec['rules']), dim),
            init_state=L.EmbedID(n_progs, dim),
            state2state=L.Linear(dim, dim),
            choice2state=L.Linear(dim, dim),
            state2var=L.Linear(dim, dim),
            state2var_choice=L.Linear(dim, dim)
            )
model = ProgModel(n_progs=1, dim=20)
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(20.0))
optimizer.add_hook(chainer.optimizer.WeightDecay(0.999))
model.to_gpu()

xmin = 1
xmax = 100
xy_n = 100
x1s = np.random.randint(xmin, xmax, size=xy_n)
x2s = np.random.randint(xmin, xmax, size=xy_n)
ys = np.empty((model.n_progs, xy_n), dtype=np.int32)
# ys[0] = x1s
# ys[1] = x2s
ys[0] = x1s + x2s

class LossComputer(object):
    def __init__(self, ok, lp, errs):
        self.ok = ok
        self.lp = lp
        self.errs = np.array(errs)

i = 0
while True:
    i += 1
    print 'epoch {},'.format(i),
    tries = 0
    context = dict()
    context['model'] = model
    # context['choice'] = np.array(0.5)
    context['max_depth'] = 3

    lcs = []
    optimizer.zero_grads()
    # Program 0
    n_ok = 0
    tries = 0
    n_ok_prog = [0]* model.n_progs
    while (not (n_ok >= 100 and 
                tries >= 100 and 
                all([nop > 100 for nop in n_ok_prog]))
           ) and tries < 1e3:
        for prog in range(model.n_progs):
            context['depth'] = 0
            context['state'] = model.init_state(Variable(cuda.to_gpu(np.array([prog], dtype=np.int32))))
            context['lp'] = 0
            e, context, ok = mk_expression_of_type(spec, context, set(), FunType(IntType(), FunType(IntType(), IntType())))
            errs = []
            if ok:
                for x1, x2, y in zip(x1s, x2s, ys[prog]):
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
                # print '{:>10}, mean(errs)={:.3E}'.format(tries, np.mean(errs)), e.pstr()
                n_ok += 1
                n_ok_prog[prog] += 1
            tries += 1
            lcs.append(LossComputer(ok, context['lp'], errs))
    # Program
    ok_errs = [lc.errs for lc in lcs if lc.ok]
    ok_err_means = [np.mean(err) for err in ok_errs]
    
    if len(ok_errs) > 0:
        # print 'ok, update'
        mean_ok_errs = np.mean(ok_err_means)
        max_ok_errs = np.max(ok_err_means)
        min_ok_errs = np.min(ok_err_means)
        not_ok_err = max_ok_errs + (max_ok_errs - min_ok_errs) + 500
        mean_errs = np.mean([np.mean(lc.errs) if lc.ok else not_ok_err for lc in lcs])
        if max_ok_errs > 5e3:
            scal = 1.0 #5e3 / max_ok_errs
        else:
            scal = 1.0
        # print 'computed errs'
        loss = Variable(cuda.to_gpu(np.array([0], dtype=np.float32)))
        loss2 = Variable(cuda.to_gpu(np.array([0], dtype=np.float32)))
        avg_err = 0
        # print 'adding losses'
        n_ok = len([lc for lc in lcs if lc.ok])
        for lc in lcs:
            err = np.mean(lc.errs) if lc.ok else not_ok_err
            loss += float(err - mean_errs) * lc.lp
            loss2 += float(err) * lc.lp
            avg_err += float(err)
        loss /= len(lcs)
        loss2 /= len(lcs)
        avg_err /= len(lcs)

        # print 'added losses'
        print 'LOSS_OK (min={:.4E}, mean={:.4E}, max={:.4E}),'.format(min_ok_errs, mean_ok_errs, max_ok_errs), '{} / {} ok.'.format(len([lc for lc in lcs if lc.ok]), len(lcs)),
        loss.backward()
        print '|grad|_2: {:.4E},'.format(optimizer.compute_grads_norm()),
        print 'LOSS={:.4E}, LOSS2={:.4E}, avg_err={:.4E}'.format(loss.data.get()[0], loss2.data.get()[0], avg_err)
        optimizer.update()
