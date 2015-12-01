# -*- coding: utf-8 -*-
from functools import wraps
import json
from collections import defaultdict
import random
import copy
import math
import loremipsum

from parsecom import *

def retrying(f, max_tries=100):
    @wraps(f)
    def wrapper(*args, **kwargs):
        for _ in xrange(max_tries):
            try:
                return f(*args, **kwargs)
            except AssertionError:
                continue
        assert False
    return wrapper

def caching(f):
    cache = {}
    @wraps(f)
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            res = f(*args)
            cache[args] = res
            return res
    return wrapper

def capital(word):
    return word[0].upper() + word[1:]

def distinct(*args):
    return len(args) == len(set(args))


# In[11]:

class Generator(object):
    def __add__(self, other):
        return ConcatGen(self, other)

    def pretty(self, wrap=80):
        r = repr(self).strip()
        if len(r) < wrap:
            return [r]
        return ["***%s***" % self.__class__.__name__]

class ConcatGen(Generator):
    def __init__(self, *gs):
        self.gs = [g for g in gs]
    
    def __call__(self, director, **kwargs):
        return ''.join(g(director, **kwargs) for g in self.gs)

    def __repr__(self):
        return ''.join(repr(g) for g in self.gs)
   
    def pretty(self, wrap=80):
        lines = []
        line = ""
        remaining = wrap
        for g in self.gs:
            gls = g.pretty(remaining)
            if line and line[-1] not in "~^ " and gls[0] and gls[0][0] not in ",. ":
                line = line + " "
                remaining -= 1
                gls = g.pretty(remaining)
            line = line + gls[0]
            for gl in gls[1:]:
                assert len(gl) <= remaining
                lines.append(line)
                line = " " * (wrap - remaining) + gl
            remaining = wrap - len(line)
            if remaining < 20:
                lines.append(line)
                line = ""
                remaining = wrap
        if line:
            assert len(line) <= wrap
            lines.append(line)
        return lines

def concatgen(*gs):
    gens = []
    for g in gs:
        if isemptygen(g):
            continue
        if gens and isinstance(gens[-1], LiteralGen) \
                and isinstance(g, LiteralGen):
            newg = LiteralGen(gens[-1].value + g.value)
            gens[-1] = newg

    if len(gs) == 1:
        return gs[0]
    return ConcatGen(*gs)

class LiteralGen(Generator):
    def __init__(self, value):
        self.value = value
    
    def __call__(self, director, **kwargs):
        director.literals.add(self.value.strip())
        return self.value

    def __repr__(self):
        return self.value
   
def isemptygen(g):
    return isinstance(g, LiteralGen) and not g.value

class MaybeGen(Generator):
    def __init__(self, gen):
        self.gen = gen
    
    def __call__(self, director, **kwargs):
        if random.random() < 0.5 ** 0.5:
            return self.gen(director, **kwargs)
        return ''

    def __repr__(self):
        return '[%s]' % repr(self.gen)
    
    
class ChoiceGen(Generator):
    def __init__(self, *gs):
        self.gs = gs
    
    def __call__(self, director, **kwargs):
        gs = director.shuffle(self.gs)
        for g in gs:
            try:
                ret = g(director, **kwargs)
            except AssertionError:
                continue
            if not ret.strip():
                continue
            director.mark_used(g)
            for g2 in gs:
                if g != g2:
                    director.mark_unused(g2)
            return ret    
        assert False
    
    def __repr__(self):
        return '<' + '|'.join(repr(g) for g in self.gs) + '>'

    def pretty(self, wrap=80):
        r = repr(self)
        if len(r) <= wrap:
            return [r]
        char = "<"
        lines = []
        for g in self.gs:
            glines = g.pretty(wrap=wrap-1)
            for gl in glines:
                assert len(gl) < wrap, gl
                lines.append(char + gl)
                char = " "
            char = "|"
        if len(lines[-1]) < wrap:
            lines[-1] = lines[-1] + ">"
        else:
            lines.append(">")
        return lines

class LookupGen(Generator):
    def __init__(self, name):
        self.name = name
    
    def __call__(self, director, **kwargs):
        if self.name in director.kwargs:
            value = director.kwargs[self.name]
            if isinstance(value, bool):
                value = "true" if value else ""
            return value
        value = director.lexicon[self.name](director, **kwargs)
        director.counts[self.name] += 1
        return value

    def __repr__(self):
        return '@' + self.name

class FunctionGen(Generator):
    def __init__(self, name, *args):
        self.name = name
        self.args = args
    
    def __call__(self, director, **kwargs):
        current, director.current = director.current, self
        ret = director.functions[self.name](*self.args, **kwargs)
        director.current = current
        return ret

    def __repr__(self):
        return r"\%s{%s}" % (self.name, "|".join(repr(a) for a in self.args))

# In[12]:
# In[23]:

def combine(*gens):
    g = gens[0]
    gens = gens[1:]
    if len(gens) == 0:
        return g
    while len(gens) > 1:
        g, gens = g + gens[0], gens[1:]
    g = g + gens[0]
    return g

def cleanup(txt):
    txt = re.sub('\s+', ' ', txt) # remove multiple spaces
    txt = re.sub(r' ([\.,;:!\?])', r'\1', txt) # remove space before punctuation
    txt = re.sub(ur'a\(n\)\s*(\W+)([aeiouáéíóúäëïöü])', r'an \1\2', txt,
            flags=re.UNICODE | re.IGNORECASE)
    txt = re.sub(r'a\(n\)', r'a', txt)
    def cap(m):
        pre = m.group(1) or ""
        return pre + m.group(2).upper()
    txt = re.sub(r'\^(\W*)(.)', cap, txt,
            flags=re.UNICODE) # capitalize after ^
    txt = txt.strip()
    txt = re.sub('\s+', ' ', txt) # remove multiple spaces again
    return txt

class Director(object):
    stdlib = []
    def __init__(self, *args):
        lexicon = defaultdict(list)
        for k, v in args:
            lexicon[k].append(v)
        self.lexicon = {}
        for k, v in lexicon.iteritems():
            if len(v) > 1:
                print "Got %d definitions for %s" % (len(v), k)
                self.lexicon[k] = ChoiceGen(*v)
            else:
                self.lexicon[k] = v[0]
        self.functions = {}
        for f, args in self.stdlib:
            self.function(*args)(f)
        self.current = None
        self.ages = defaultdict(float)
        self.counts = defaultdict(int)
        self.literals = set()

    @retrying
    def __call__(self, name, **kwargs):
        self.data = defaultdict(lambda: None)
        self.kwargs = kwargs
        txt = self.lexicon[name](self)
        del self.kwargs
        return "\n".join(cleanup(para) for para in txt.split("~"))

    def function(self, name, *types):
        def wrapper(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                def thunk(x):
                    return lambda: x(self, **kwargs)
                args = list(args)
                the_types = list(types)
                new_args = []
                while args:
                    try:
                        t = the_types.pop(0)
                    except:
                        t = None
                    a = args.pop(0)
                    if t is None:
                        na = thunk(a)
                    else:
                        na = t(a(self, **kwargs))
                    new_args.append(na)
                ret = f(self, *new_args, **kwargs)
                return ret
            self.functions[name] = wrapped

            return wrapped
        
        return wrapper
    
    def shuffle(self, items):
        items = list(items)
        items = sorted(items, key=lambda x: self.ages[x] + random.random())
        return items
    
    def mark_used(self, item):
        self.ages[item] += 0.5

    def mark_unused(self, item):
        self.ages[item] *= 0.5
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value


def stdlib(*args):
    def wrapper(f):
        Director.stdlib.append((f, args))
    return wrapper
   
def wrapped(txt, width=80):
    lines = [""]
    words = txt.split()
    while words:
        word = words.pop(0)
        if len(lines[-1]) + len(word) + 1 > width:
            lines.append("")
        lines[-1] += (" " if lines[-1] else "") + word
    return "\n".join(lines)

template = Forward()

text = Regex(r"[^@#{}[\]\\|<>_\n]+") >> LiteralGen
insert = (lit("@") & word) >> LookupGen
literal = (lit("_") & Regex('.')) >> LiteralGen
function = (lit("\\") & word & Maybe(lit("{") & Maybe(template & Many(lit("|") & template)) & lit("}"))) >> FunctionGen
choice = (lit("<") & template & Many(lit("|") & template) & lit(">")) >> ChoiceGen
option = (lit("[") & template & lit("]")) >> MaybeGen
comment = (lit("#") & Regex(r"[^\n]*")) > LiteralGen(" ")
linebreak = Regex(r"\n +") > LiteralGen(" ")
template.p = Many(text | insert | literal | function | choice | option | comment | linebreak, allow_none=False) >> concatgen
defn = (word + lit("=") + template) >> tuplize
parser = Many((defn & lit("\n")) | lit("\n") | ~comment) >> Director

specials = r"""
@ - insert template
# - comment
^ - capitalize
{} - grouping
[] - optional
\ - function call
| - separator
_ - literal
~ - paragraph break
<> - choice

`
"""

@stdlib("retry")
def retry(director, gen):
    tries = 0
    while True:
        old_data = copy.deepcopy(director.data)
        try:
            return gen()
        except AssertionError:
            tries += 1
            if tries > 3:
                raise
            director.data = old_data
        
@stdlib("ignore")
def ignore(director, gen):
    return ""

@stdlib("error")
def error(director):
    raise Exception

@stdlib("if")
def if_(director, query, yes):
    q = query()
    if q.strip():
        return yes()
    else:
        return ""

@stdlib("some")
def some_(director, *gens):
    genlist = list(gens)
    random.shuffle(genlist)
    n = random.randrange(2,5)
    return ' '.join(g() for g in genlist[:n])

@stdlib("fix")
def fix(director, gen):
    if director["fix"] is None:
        director["fix"] = {}
    df = director["fix"]
    if gen not in df:
        df[gen] = gen()
    return df[gen]

@stdlib("once")
def once(director, gen=None):
    if director["once"] is None:
        director["once"] = set()
    assert director.current not in director["once"]
    director["once"].add(director.current)
    if gen: 
         return gen()
    return ""

@stdlib("choose", float)
def choose(director, value, *gens):
    n = len(gens) - 1
    p = 1 / (1 + math.exp(-value/2))
    k = 0
    for _ in xrange(n):
        k += random.random() < p
    return gens[k]()

@stdlib("repeat", int)
def repeat(director, n, gen):
    return ' '.join(gen() for _ in xrange(n))

@stdlib("lorem")
def lorem(director):
    return loremipsum.get_sentence()

