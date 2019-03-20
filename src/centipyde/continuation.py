import sys
import re
import inspect
from contextlib import contextmanager

from inspect import signature, getmembers

# TODO:figure out how to include column offset.
# http://xion.io/post/code/python-get-lambda-code.html
def print_lambda(func):
    print('lambda: ' + func.__qualname__.split('.')[1] + ':' + str(func.__code__.co_firstlineno))
    #print({x[0]:x[1] for x in getmembers(func)}.keys())


# This class handles all the "backwardness" of Continuations, because it was making things
# way too difficult to write / read
# TODO: something about tracking "info" as we go, so we always can print that info at each step in the continuation
class Continuation(object):
    __slots__ = ['continuations', 'passthroughs', 'if_id', 'completed_if_id', 'loop_passthroughs']
    def __init__(self):

        self.continuations = []
        self.passthroughs = []
        self.loop_passthroughs = [[]]
        self.if_id = 0
        self.completed_if_id = None

    def apply(self, func):
        # does literally nothing right now
        assert len(signature(func).parameters) == 0
        self.continuations.append((self.handle_apply, [func]))
        return self

    def handle_apply(self, func):
        func()

    def expect(self, func):
        assert len(signature(func).parameters) == 1
        self.continuations.append((self.handle_expect, [func]))
        return self

    # passthrough is so that we don't process a passthrough from before this node was created
    def handle_expect(self, func):
        #print_lambda(func)
        val = self.get_passthrough()
        func(val)

    def info(self, info):
        self.continuations.append((self.handle_info, [info]))
        return self

    def handle_info(self, info):
        return info

    def kassert(self, func, str_=''):
        assert len(signature(func).parameters) == 0
        self.continuations.append((self.handle_kassert, [func, str_]))
        return self

    def handle_kassert(self, func, str_):
        assert func(), str_

    def kprint(self, p):
        self.continuations.append((self.handle_kprint, [p]))
        return self

    def handle_kprint(self, p):
        print(p())

    def loop(self, func, list_=False, shortcircuit=False):
        assert len(signature(func).parameters) == 1
        self.continuations.append((self.handle_loop, [func, list_, shortcircuit, True]))
        return self

    # TODO: figure out how context_num works here??? Is it needed?
    def handle_loop(self, func, list_, shortcircuit, first):
        loop_vars = self.loop_passthroughs[-2][0]
        do_shortcircuit = False
        if not first:
            # TODO: these *need* to be pushed immediately before, so should be at the end the loop...
            val = self.get_passthrough(-1)
            loop_vars[1].append(val)
            if shortcircuit:
                do_shortcircuit = self.get_passthrough(-1)

        if len(loop_vars[0]) == 0 or do_shortcircuit:
            self.loop_passthroughs.pop()
            self.loop_passthroughs.pop()
            # TODO TODO: do we have to worry about a passthrough we're using being pulled out,
            # such that this should go in front? that's impossible, right?
            self.put_passthrough(lambda: loop_vars[1])
            #if len(self.passthroughs[-1]) == 0 or self.passthroughs[-1][0][1] < context_num:
            #    self.passthroughs[-1] = [(loop_vars[1], context_num)] + self.passthroughs[-1]
            #else:
            #    self.passthroughs[-1].append((loop_vars[1], context_num))
            return

        var = loop_vars[0].pop(0)

        # This should add a continuation that we can now apply to get the element
        # TODO: context can become huge, right??
        # we want to revisit the loop eventually
        tmp = func(var)
        # handle list_
        if list_:
            tmp(loop_vars[1])
        self.continuations.append((self.handle_loop, [func, list_, shortcircuit, False]))

    def loop_var(self, var):
        assert var is None or isinstance(var, list)
        self.continuations.append((self.handle_loop_var, [var]))
        return self

    def handle_loop_var(self, var):
        #assert(self.continuations[0][0] == self.handle_loop)
        self.loop_passthroughs.append([])
        if var is None:
            # default return value is empty list
            self.loop_passthroughs[-1].append([[],[]])
        else:
            # we don't want to use the original var, since then we'll pop things off
            # the node
            self.loop_passthroughs[-1].append([var.copy(),[]])
        self.loop_passthroughs.append([])

    def passthrough(self, val):
        assert len(signature(val).parameters) == 0
        self.continuations.append((self.handle_passthrough, [val]))
        return self

    def get_passthrough(self, idx=0):
        val, func = self.passthroughs.pop(idx)
        #print('Removing {} from idx {}, remaining {}, value generated at: '.format(
        #    val, idx, self.passthroughs), end="")
        #print_lambda(func)
        return val

    def put_passthrough(self, func):
        #print_lambda(func)
        val = func()
        #print('adding', val, self.passthroughs, '\n')
        # keep around the func so we know where a value was generated
        self.passthroughs.append((val, func))

    def handle_passthrough(self, func):
        # TODO: the pushing context is only needed if we're passing through something that can add to
        # the continuation. Generally no?
        # TODO: will this only *ever* go at the front or the back?
        self.put_passthrough(func)
        # This is currently true for Return
        #assert ret is not None

    # TODO: this NEEDS to take a lambda, since free variables could get modified
    # that are in the condition (such as self.context)
    def if_(self, cond, func):
        assert len(signature(func).parameters) == 0
        self.if_id += 1
        self.continuations.append((self.handle_if, [cond, func, self.if_id]))
        return self

    def handle_if(self, cond, func, if_id):
        if cond:
            func()
            # we only want to do this after the other continuations
            # from func() are executed. Not immediately!!
            self.apply(lambda: setattr(self, 'completed_if_id', if_id))

    def elseif(self, cond, func):
        assert len(signature(func).parameters) == 0
        self.continuations.append((self.handle_elseif, [cond, func, self.if_id]))
        return self

    def handle_elseif(self, cond, func, if_id):
        # TODO: is the >= correct??????????????????????
        # scoping with the continuations is weird...
        if self.completed_if_id != if_id and cond:
            func()
            self.apply(lambda: setattr(self, 'completed_if_id', if_id))


    def else_(self, func):
        self.continuations.append((self.handle_else, [func, self.if_id]))
        return self

    def handle_else(self, func, if_id):
        # no need to set the completed_id, since we're the else anyway
        if self.completed_if_id != if_id:
            func()

    #def pop_context(self):
    #    c = self.continuations.pop()
    #    p = self.passthroughs.pop()
    #    self.continuations[-1] = c + self.continuations[-1]
    #    # a context can only have one top-level passthrough, right?
    #    # does anything else necessarily make sense?
    #    # TODO: not necessarily true?
    #    assert len(p) <= 1
    #    self.passthroughs[-1] =  p + self.passthroughs[-1]


    def step(self):
        ret = None
        while ret is None:
            #@if len(self.continuations[-1]) > 0:
            #    assert False
            #    return None
            if len(self.continuations) == 0:
                return None
            k = self.continuations.pop(0)
            old_continuations = self.continuations
            self.continuations = []
            ret = k[0](*k[1])
            self.continuations = self.continuations + old_continuations
        return ret
