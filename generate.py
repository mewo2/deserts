import click
import describe
import glob
import re

template = r"""
\begin{figure}[t]
\includegraphics[width=0.9\textwidth]{%02d/map.png}
\end{figure}
\input{%02d/map}
"""

@click.command()
@click.argument("directory")
@click.option("-n", default=100)
def main(directory, n):
    describe.do_novel(directory, n)
    with open(directory + "/contents.tex", "w") as f:
        for i in xrange(n):
            f.write(template % (i,i))

if __name__ == '__main__':
    main()
