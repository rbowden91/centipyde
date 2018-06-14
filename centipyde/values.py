class Function(object):
    pass

# TODO: what about when they are opaque? decls=None?
class Struct(object):
    __slots__ = ['name', 'decls']
    def __init__(self, name, decls):
        self.name = name
        self.decls = decls

class Union(object):
    __slots__ = ['name', 'decls']
    def __init__(self, name, decls):
        self.name = name
        self.decls = decls

class Union(object):
    __slots__ = ['name', 'decls']
    def __init__(self, name, decls):
        self.name = name
        self.decls = decls

class Address(object):
    __slots__ = ['base', 'offset']
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def __str__(self):
        return 'Address(' + str(self.base) + ', ' + str(self.offset) + ')'
        return str(self)

    # TODO: is __str__ necessary at this point? will str() automatically call repr if __str__ doesn't exist??
    def __repr__(self):
        return str(self)

class Flow(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

    def __str__(self):
        return 'Flow(' + str(self.type) + ', ' + str(self.value) + ')'

    def __repr__(self):
        return str(self)

class Memory(object):
    __slots__ = ['type', 'expanded_type', 'name', 'len', 'array', 'segment']
    def __init__(self, type_, expanded_type, name, len_, array, segment):
        self.type = type_
        self.expanded_type = expanded_type
        self.name = name
        self.len = len_
        self.array = array
        self.segment = segment


class Val(object):
    __slots__ = ['type', 'expanded_type', 'value']
    def __init__(self, type_, expanded_type, value):
        self.type = type_
        self.value = value
        self.expanded_type = expanded_type

    def __str__(self):
        return 'Val(' + str(self.type) + ', ' + str(self.value) + ')'

    def __repr__(self):
        return str(self)
