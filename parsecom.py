import re

class ParseFailure(Exception):
    pass

class Parser(object):
    def __add__(self, other):
        return ThenWS(self, other)

    def __and__(self, other):
        return Then(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __rshift__(self, other):
        return Apply(other, self)
    
    def __gt__(self, other):
        return Apply(lambda *x: other, self)

    def __invert__(self):
        return Null(self)

    def __call__(self, string):
        matches, rest = self.parse(string)
        assert not rest, rest
        assert len(matches) == 1
        return matches[0]

class Or(Parser):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def parse(self, string):
        try:
            return self.p1.parse(string)
        except ParseFailure:
            return self.p2.parse(string)

class Then(Parser):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def parse(self, string):
        first, rest = self.p1.parse(string)
        second, rest = self.p2.parse(rest)
        return first + second, rest

class ThenWS(Then):
    def parse(self, string):
        first, rest = self.p1.parse(string)
        _, rest = whitespace.parse(rest)
        second, rest = self.p2.parse(rest)
        return first + second, rest

    def __repr__(self):
        return "%r + %r" % (self.p1, self.p2)

class Many(Parser):
    def __init__(self, base, allow_none=True):
        self.base = base
        self.allow_none = allow_none
        
    def parse(self, string):
        matches = []
        while True:
            try:
                match, string = self.base.parse(string)
                #_, string = whitespace.parse(string)
                matches.extend(match)
            except ParseFailure:
                if matches or self.allow_none:
                    break
                else:
                    raise
        return matches, string

class Maybe(Parser):
    def __init__(self, parser):
        self.parser = parser

    def parse(self, string):
        try:
            return self.parser.parse(string)
        except ParseFailure:
            return [], string

class Apply(Parser):
    def __init__(self, func, parser):
        self.func = func
        self.parser = parser

    def parse(self, string):
        matches, rest = self.parser.parse(string)
        return [self.func(*matches)], rest

    def __repr__(self):
        return "%r >> %r" % (self.parser, self.func)

class Regex(Parser):
    def __init__(self, regex):
        self.regex = re.compile(regex)

    def parse(self, string):
        match = self.regex.match(string)
        if match is None:
            raise ParseFailure
        return [match.group()], string[match.end():]

    def __repr__(self):
        return "Regex(%s)" % self.regex.pattern

class Null(Parser):
    def __init__(self, parser):
        self.parser = parser

    def parse(self, string):
        match, rest = self.parser.parse(string)
        return [], rest

    def __repr__(self):
        return "~%r" % self.parser

class End(Parser):
    def parse(self, string):
        if string:
            raise ParseFailure
        return [], ''

    def __repr__(self):
        return "eof"

class Forward(Parser):
    def parse(self, string):
        return self.p.parse(string)

    def __repr__(self):
        return "Forward()"

class Literal(Parser):
    def __init__(self, text):
        self.text = text

    def parse(self, string):
        n = len(self.text)
        if string[:n] == self.text:
            return [self.text], string[n:]
        else:
            raise ParseFailure

lit = lambda x: ~Literal(x)
sep = lambda x, s: x + Many(~s + x)
eof = End()
numeric = Regex('\d+')
whitespace = Regex('\s*')
integer = Regex('-?\d+') >> int
word = Regex('\w+')

comment = lambda char: lit(char) + ~Regex(r"[^\n]*\n")
doublequoted = Regex(r'"([^"\\]|\\.)*"')
singlequoted = Regex(r"'([^'\\]|\\.)*'")
quoted = doublequoted | singlequoted
coords = (integer + lit(',') + integer) >> (lambda x, y: (x, y))

def tuplize(*args): return tuple(args)
def listize(*args): return list(args)
def dictize(*args): return dict(args)



