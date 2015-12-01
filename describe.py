import re
import os
import numpy as np

import terrain
from grammar import parser

grammar = open("description.grammar").read()
desc = parser(grammar)

@desc.function("word", str)
def word(director, key):
    m = director.kwargs["mapgrid"]
    return "*" + m.lang.word(key) + "*"

@desc.function("name", str)
def name(director, key):
    m = director.kwargs["mapgrid"]
    return m.lang.name(key)


def direction(m, p1, p2):
    x, y = m.vxs[p2,:] - m.vxs[p1,:]
    angle = int(4 * np.arctan2(y, x) / np.pi + 4.5) % 8
    return ["west", "south-west", "south", "south-east",
            "east", "north-east", "north", "north-west"][angle]

def describe(m, city, last_city=None):
    paras = []
    direc = ""
    far = False
    near = False
    land = True
    sea = False
    if last_city is not None:
        direc = direction(m, city, last_city)
        path,_ = m.shortest_path(last_city, city)
        dist = terrain.distance(m.vxs[city,:], m.vxs[last_city,:])
        far = bool(dist > 0.3)
        near = bool(dist < 0.15)
        land = bool(np.mean(m.elevation[path] > 0) > 0.4)
        sea = bool(np.mean(m.elevation[path] <= 0) > 0.4)
    kwargs = {"lastcity": m.city_names.get(last_city, ""),
              "city": m.city_names[city],
              "region": m.region_names[city],
              "direction": direc,
              "far": far,
              "near": near,
              "islandjourney": land,
              "isseajourney": sea,
              "mapgrid": m}
    paras.extend(desc("description", **kwargs).split("\n"))
    return paras

def mdize(name):
    name = re.sub('`', u'\u02bb', name)
    return name

def get_map(directory, mode):
    try:
        m = terrain.load(directory + "/map.pickle")
    except:
        m = terrain.MapGrid(mode=mode)
        m.save(directory + "/map.pickle")
    return m

def draw_map(directory, m):
    m.plot(directory + "/map.png", dpi=200)

def write_description(directory, m):
    with open(directory + "/map.md", "w") as f:
        last_city = None
        cities = m.ordered_cities()
        reg1 = m.region_names[cities[0]]
        reg2 = m.region_names[cities[-1]]
        title = "%s to %s" % (reg1, reg2)
        f.write("# %s\n\n" % mdize(title).encode("utf8"))
        for city in cities:
            name = mdize(m.city_names[city])
            f.write("## %s\n" % name.encode("utf8"))
            for para in describe(m, city, last_city=last_city):
                f.write("%s\n\n" % mdize(para).encode("utf8"))
            last_city = city

def process_directory(directory, mode="shore"):
    print "PROCESSING", directory, mode
    m = get_map(directory, mode)
    draw_map(directory, m)
    write_description(directory, m)

def choose(lst, p):
    n = len(lst) - 1
    return lst[sum(np.random.random() < p for _ in xrange(n))]

def do_novel(directory='tests/full', n=100):
    modes = ["shore", "island", "mountain", "desert"]
    last_mode = "shore"
    for i in xrange(n):
        mode = choose(modes, i/(n-1.))
        if mode == last_mode:
            mapmode = mode
        else:
            mapmode = last_mode + "/" + mode
        direc = "%s/%02d" % (directory, i)
        try:
            os.makedirs(direc)
        except:
            pass
        process_directory(direc, mode=mapmode)
        last_mode = mode

