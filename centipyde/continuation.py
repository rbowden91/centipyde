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
        self.reverse_k = []
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
        print(p)

    def loop(self, func, list_=False, shortcircuit=False):
        assert len(signature(func).parameters) == 1
        self.continuations.append((self.handle_loop, [func, list_, shortcircuit, True]))
        return self

    def handle_loop(self, is_forward, func, shortcircuit, items, old_items, results):
        if is_forward:
            do_shortcircuit = False

            if len(old_items) != 0:
                # this is not the first iteration of the loop

                # don't use get_passthrough and put_passthrough, because those will do things that aren't quite what we
                # want. They'd probably work, but information would be duplicated
                assert len(self.passthroughs) == 1
                val = self.passthroughs.pop(0)
                results.append(val)
                if shortcircuit:
                    do_shortcircuit = val[0][0]

            if len(items) == 0 or do_shortcircuit:
                vals = [(val[0][1] if shortcircuit else val[0]) for val in results]
                self.passthroughs.append(vals)
                return

            # this technically isn't a passthrough. maybe it should be...
            item = items.pop(0)
            old_items.append(item)

            # This should add a continuation that we can then apply at the next step to complete one iteration
            # of the loop
            tmp = func(item)

            # we want to revisit the loop eventually
            self.continuations.append((self.handle_loop, [func, shortcircuit, items, old_items, results]))
        else:
            if len(items) == 0 or (shortcircuit and len(results) > 0 and results[-1][0][0] == True):
                # this was the last iteration of the loop. Remove the return value
                self.passthroughs.pop()

            if len(results) != 0:
                self.passthroughs.insert(0, results.pop())

            if len(old_items) != 0:
                items.insert(0, old_items.pop())


    def passthrough(self, val):
        assert len(signature(val).parameters) == 0
        self.continuations.append((self.handle_passthrough, [val]))
        return self

    # only ever appends
    def handle_passthrough(self, is_forward, func):
        self.put_passthrough(is_forward, func)
        # This is currently true for Return
        #assert ret is not None

    # TODO: this NEEDS to take a lambda, since free variables could get modified
    # that are in the condition (such as self.context)
    def if_(self, is_forward, cond, func):
        assert len(signature(func).parameters) == 0
        self.if_id += 1
        self.continuations.append((self.handle_if, [cond, func, self.if_id]))
        return self

    def handle_if(self, cond, func, if_id):
        if not is_forward: return
        if cond:
            func()
            # we only want to do this after the other continuations
            # from func() are executed. Not immediately!!
            self.apply(lambda: setattr(self, 'completed_if_id', if_id))

    def elseif(self, is_forward, cond, func):
        assert len(signature(func).parameters) == 0
        self.continuations.append((self.handle_elseif, [cond, func, self.if_id]))
        return self

    def handle_elseif(self, is_forward, cond, func, if_id):
        if not is_forward: return
        if self.completed_if_id != if_id and cond:
            func()
            self.apply(lambda: setattr(self, 'completed_if_id', if_id))


    def else_(self, func):
        self.continuations.append((self.handle_else, [func, self.if_id]))
        return self

    def handle_else(self, is_forward, func, if_id):
        if not is_forward: return
        # no need to set the completed_id, since we're the else anyway
        if self.completed_if_id != if_id:
            func()


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
