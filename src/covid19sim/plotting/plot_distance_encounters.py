import sys
import os
import pickle
import math
import matplotlib.pyplot as plt

def proportion(l):
    x = [v/sum(l) for v in l]
    return x

def mean_variance(l):
    if len(l) == 0:
        return 0, 0, 0, 0
    else:
        mean = sum(l) / len(l)
        var = math.sqrt(sum([(v-mean)*(v-mean) for v in l]) / len(l))
        return mean, var, max(l), min(l)

def plot_distance(file_name, outdir):
    clip = [0, 0]
    location2clip = dict()
    out2distance = {'packing term':[], 'encounter term':[], 'social distancing term':[], 'distance':[]}
    in2distance = {'packing term':[], 'encounter term':[], 'social distancing term':[], 'distance':[]}

    with open(file_name, 'rb') as fi:
        data = pickle.load(fi)['encounter_distances']

    for line in data:
        items = line.strip().split('\t')
        if items[0] == 'A':
            c = int(items[1])
            l = items[2].split(':')[0]
            p = float(items[3])
            e = float(items[4])
            s = float(items[5])
            d = float(items[6])

            clip[c] += 1
            if l not in location2clip:
                location2clip[l] = [0, 0]
            location2clip[l][c] += 1
            out2distance['packing term'] += [p]
            out2distance['encounter term'] += [e]
            out2distance['social distancing term'] += [s]
            out2distance['distance'] += [d]
        if items[0] == 'B':
            p = float(items[1])
            e = float(items[2])
            s = float(items[3])
            d = float(items[4])

            in2distance['packing term'] += [p]
            in2distance['encounter term'] += [e]
            in2distance['social distancing term'] += [s]
            in2distance['distance'] += [d]

    fig, ax = plt.subplots(figsize=(10,5))
    ally = proportion(clip)
    allx = ['Overall']
    colors = ['b', 'g']
    for l, c in location2clip.items():
        ally += proportion(c)
        allx += [l]
        colors += ['b', 'g']
    index = list(range(1, len(ally), 2))
    ax.bar(index, [ally[i] for i in index], color=[colors[i] for i in index], label="No Clip")
    index = list(range(0, len(ally), 2))
    ax.bar(index, [ally[i] for i in index], color=[colors[i] for i in index], label="Clip")
    ax.set_xticks([i+0.5 for i in index])
    ax.set_xticklabels(allx)
    ax.legend()
    ax.set(title='Statistics of Clip')
    plt.savefig(os.path.join(outdir, 'clip.png'), dpi=1000)

    fig, ax = plt.subplots(figsize=(10,5))
    allx = []
    ally = []
    allvar = []
    for o, d in out2distance.items():
        result = mean_variance(d)
        allx.append(o)
        ally.append(result[0])
        allvar.append(result[1])
    index = list(range(len(ally)))
    ax.bar(index, ally, color='b', label="Outside", yerr=allvar)
    ally = []
    allvar = []
    for o, d in in2distance.items():
        result = mean_variance(d)
        allx.append(o)
        ally.append(result[0])
        allvar.append(result[1])
    index = list(range(len(ally), 2*len(ally)))
    ax.bar(index, ally, color='g', label="Inside", yerr=allvar)
    ax.set_xticks(list(range(len(allx))))
    ax.set_xticklabels(allx)
    ax.legend()
    ax.set(title='Statistics of Distance')
    plt.savefig(os.path.join(outdir, 'distance.png'), dpi=1000)

if __name__ == '__main__':
    plot_distance('data_sd.txt')
