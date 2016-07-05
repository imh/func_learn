class Expression(object):
    def __init__(self, constructor, type_def, **kwargs):
        self.constructor = constructor
        self.type_def = type_def
        for k, v in kwargs.iteritems():
            self.__dict__[k] = v
    def pstr(self, indent=0):
        con = self.constructor
        if con == 'var':
            return self.name.name
        if con == 'lam':
            return '\{} -> ({})'.format(self.name.name, self.body.pstr(indent))
        if con == 'app':
#             fun = self.fun
#             print fun.constructor
#             bind = fun.name.name
#             body = fun.body.pstr(indent+1)
#             arg = self.arg.pstr(indent+1)
            indents = '  '*indent
#             return '{indents}let {bind} = {arg}\n{indents} in {body}'.format(indents=indents, bind=bind, arg=arg, body=body)
            return 'app ({}) ({})'.format(self.fun.pstr(indent), self.arg.pstr(indent))
        if con == 'bool':
            return str(self.val)
        if con == 'booleq':
            return '({}) == ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'and':
            return '({}) and ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'or':
            return '({}) or ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'not':
            return 'not ({})'.format(self.arg.pstr(indent))
        if con == 'ifthenelse':
            return 'if ({}) then ({}) else ({})'.format(self.prop.pstr(indent), self.true_branch.pstr(indent), self.false_branch.pstr(indent))
        if con == 'int':
            return str(self.val)
        if con == 'gthan':
            return '({}) > ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'lthan':
            return '({}) < ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'inteq':
            return '({}) == ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'plus':
            return '({}) + ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'times':
            return '({}) * ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'div':
            return '({}) / ({})'.format(self.arg1.pstr(indent), self.arg2.pstr(indent))
        if con == 'int_negate':
            return '-({})'.format(self.arg.pstr(indent))
        if con == 'depth_exceeded':
            return 'DEPTH_EXCEEDED :: {}'.format(self.type_def)
        raise Exception('{con} not implemented'.format(con=con))

class Type(object):
    def can_be(self, type_def):
        if type(self) == FunType and type(type_def) == FunType:
            return self.in_type.can_be(type_def.in_type) and self.out_type.can_be(type_def.out_type)
        if type(self) == ListType and type(type_def) == ListType:
            return self.elem_type.can_be(type_def.elem_type)
        if type(self) == TupleType and type(type_def) == TupleType:
            return self.fst_type.can_be(type_def.fst_type) and self.snd_type.can_be(type_def.snd_type)
        # if type(self) == ForallType:
        #     return True
        if type(self) == type(type_def):
            return True
        return False
    def is_function(self):
        return type(self) == FunType
    def is_bool(self):
        return type(self) == BoolType
    def is_int(self):
        return type(self) == IntType
    def is_list(self):
        return type(self) == ListType
    def is_tuple(self):
        return type(self) == TupleType

class Name(object):
    def __init__(self, name, type_def, vec):
        self.name = name
        self.type_def = type_def
        self.vec = vec

class FunType(Type):
    def __init__(self, in_type, out_type):
        self.in_type = in_type
        self.out_type = out_type
    def __str__(self):
        return '{} -> {}'.format(self.in_type, self.out_type)

# class ForallType(Type):
#     def __init__(self, type_var):
#         self.type_var = type_var
#     def __str__(self):
#         return self.type_var
        
class BoolType(Type):
    def __init__(self):
        pass
    def __str__(self):
        return 'Bool'

class IntType(Type):
    def __init__(self):
        pass
    def __str__(self):
        return 'Int'
    
class ListType(Type):
    def __init__(self, elem_type):
        self.elem_type = elem_type
    def __str__(self):
        return '[{}]'.format(self.elem_type)

class TupleType(Type):
    def __init__(self, fst_type):
        self.fst_type = fst_type
        self.snd_type = snd_type
    def __str__(self):
        return '({},{})'.format(self.fst_type, self.snd_type)