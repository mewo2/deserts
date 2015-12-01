import click
import describe
import re
import readline
import sys
import traceback

from collections import Counter

@click.group()
def cli():
    pass

def print_defn(key, value=None, width=80):
    if value is None:
        value = describe.desc.lexicon[key]
    length = len(key) + 3
    pretty = value.pretty(wrap=width - length)
    prefix = "%s = " % key
    for line in pretty:
        print prefix + line
        prefix = " " * length


@cli.command()
def grammar():
    for key, value in sorted(describe.desc.lexicon.items()):
        print_defn(key, value)

def get_graph():
    lex = describe.desc.lexicon
    nodes = sorted(lex.keys())
    edges = []
    for parent, value in lex.items():
        value = repr(value)
        for child in nodes:
            if re.search(r"\@%s\b" % child, value):
                edges.append((parent, child))
    return nodes, sorted(edges)

@cli.command()
def orphans():
    nodes, edges = get_graph()
    parent_counts = Counter([x[1] for x in edges])
    for node in nodes:
        if parent_counts[node] <= 1:
            print_defn(node)

@cli.command()
def graph():
    nodes, edges = get_graph()
    parent_counts = Counter([x[1] for x in edges])    
    
    print """
digraph G {
    graph [rankdir="LR"];
    """
    for node in nodes:
        print 'node_%s [label="%s", color="%s"];' % (node, node, "red" if
                parent_counts[node] == 1 else "green")
    for parent, child in edges:
        print "node_%s -> node_%s;" % (parent, child)
    print """
}
    """

@cli.command()
def test():
    readline.parse_and_bind("tab: complete")
    desc = describe.desc
    kwargs = {
            "city": "CITY",
            "lastcity": "LASTCITY",
            "direction": "DIRECTION",
            "islandjourney": False,
            "isseajourney": True
            }
    while True:
        @readline.set_completer
        def completer(txt, state):
            labels = sorted(desc.lexicon.keys())
            labels = [x for x in labels if txt in x]
            if state < len(labels):
                return labels[state]
        
        cmd = raw_input("> ").split()
        name = cmd[0]
        repeats = 1
        if len(cmd) > 1:
            repeats = int(cmd[1])
        try:
            desc = describe.parser(open("description.grammar").read())
        except:
            print "Parse error"
            continue
        desc.function("word")(lambda *args: "WORD")
        desc.function("name")(lambda *args: "NAME")
        try:
            for _ in xrange(repeats):
                print desc(name, **kwargs)
        except:
            t, v, tb = sys.exc_info()
            print "Runtime error", t, v
            traceback.print_tb(tb)
            del tb
            continue
if __name__ == '__main__':
    cli()
