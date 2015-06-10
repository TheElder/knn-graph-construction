import networkx as nx
from random import choice
from random import sample
from random import seed
import datetime
import matplotlib.pyplot as plt
from traits.tests.check_timing import new_style_value

amounts = 0

def dist(obj1, obj2):
    global amounts
    amounts += 1
    d = 0
    for i in range(0,len(obj1['pars'])):
        d += pow(float(obj1['pars'][i]) - float(obj2['pars'][i]), 2)
    return d


def greedySearch (query, venter_point, g):
    vcurr = venter_point
    dmin = dist(g.node[query], g.node[vcurr])
    vnext = None
    neighborhood = list(set(g.predecessors(vcurr) + g.successors(vcurr)))
    if query in neighborhood:
        neighborhood.remove(query)
    for vneib in neighborhood:
        if g.node[vneib]['lastchecked'][0] == query:
            d = g.node[vneib]['lastchecked'][1]
        else:
            d = dist(g.node[query], g.node[vneib])
            g.node[vneib]['lastchecked'] = (query, d)
        if d < dmin:
            dmin = d
            vnext = vneib
    if vnext is None:
        return vcurr
    else:
        return greedySearch(query, vnext, g)


def multiSearch (query, attempts, g):
    results = []
    if len(g.nodes()) < 2:
        return results
    for i in range(0, attempts):
        verts = list(g.nodes())
        verts.remove(query)
        seed()
        venter = choice(verts)
        local_min = greedySearch(query, venter, g)
        if local_min not in results:
            results.append(local_min)
    return results


def NNAdd (new_index, l, init_attempts, g):
    neighborhood = []
    locmins = multiSearch(new_index, init_attempts, g)
    neighborhood.extend(locmins)
    for locmin in locmins:
        neighborhood.extend(g.predecessors(locmin) + g.successors(locmin))
    neighborhood = list(set(neighborhood))
    if new_index in neighborhood:
        neighborhood.remove(new_index)
    distances = {}
    for i in neighborhood:
        if g.node[i]['lastchecked'][0] == new_index:
            distances[i] = g.node[i]['lastchecked'][1]
        else:
            distances[i] = dist(g.node[new_index], g.node[i])
    sort_by_dist = sorted(distances, key=distances.get)
    if g.node[new_index]['neibs']:
        for i in sort_by_dist[0:l]:
            if len(g.node[new_index]['neibs']) < l:
                if i not in g.node[new_index]['neibs']:
                    g.add_edge(new_index, i)
                    g.node[new_index]['neibs'][i] = distances[i]
                    g.node[i]['rev_neibs'].append(new_index)
            else:
                maxD = getMaxDist(g, new_index)
                if i not in g.node[new_index]['neibs'] and maxD[1] > distances[i]:
                    g.add_edge(new_index, i)
                    g.node[new_index]['neibs'][i] = distances[i]
                    g.node[i]['rev_neibs'].append(new_index)
                    g.node[maxD[0]]['rev_neibs'].remove(new_index)
                    g.node[new_index]['neibs'].pop(maxD[0])
    else:
        for i in sort_by_dist[0:l]:
            g.add_edge(new_index, i)
            g.node[new_index]['neibs'][i] = distances[i]
            g.node[i]['rev_neibs'].append(new_index)


def getMaxDist(dg, n):
    if not dg.node[n]['neibs']:
        return None
    k = max(dg.node[n]['neibs'], key=dg.node[n]['neibs'].get)
    d = dg.node[n]['neibs'][k]
    return (k,d)


def NNDescentBasicReworked (dg, k):
    for i in dg.nodes():
        other_nodes = dg.nodes()[:(i-1)] + dg.nodes()[i:]
        s = sample(other_nodes, k)
        for j in s:
            dg.node[j]['rev_neibs'].append(i)
            dg.node[i]['neibs'][j] = dist(dg.node[j], dg.node[i])
    while True:
        updates = 0
        for i in dg.nodes():
            neighborhood = []
            neighborhood.extend(dg.node[i]['rev_neibs'] + dg.node[i]['neibs'].keys())
            neighborhood = list(set(neighborhood))
            for j in neighborhood:
                for l in dg.node[j]['neibs'].keys() + dg.node[j]['rev_neibs']:
                    updates += updateNeighborsReworked(dg, i, l)
        if updates == 0:
            for n1 in dg.nodes():
                for n2 in dg.node[n1]['neibs']:
                    dg.add_edge(n1, n2)
            return


def NNDescentFullReworked (dg, k, p, stoprate):
    for i in dg.nodes():
        seed()
        other_nodes = dg.nodes()[:(i-1)] + dg.nodes()[i:]
        s = sample(other_nodes, k)
        for j in s:
            dg.node[j]['rev_neibs'].append(i)
            dg.node[i]['neibs'][j] = dist(dg.node[j], dg.node[i])
    while True:
        updates = 0
        for i in dg.nodes():
            seed()
            old = [n for n in dg.node[i]['neibs'] if dg.node[n]['isnew'] is False]
            new = [n for n in dg.node[i]['neibs'] if dg.node[n]['isnew'] is True]
            new = sample(new, int(len(new)*p))
            for n in new:
                dg.node[n]['isnew'] = False
            old_rev = []
            new_rev = []
            for n in old:
                old_rev.extend(dg.node[n]['rev_neibs'])
            for n in new:
                new_rev.extend(dg.node[n]['rev_neibs'])
            old_rev = list(set(old_rev))
            new_rev = list(set(new_rev))
            old.extend(sample(old_rev, int(len(old_rev)*p)))
            new.extend(sample(new_rev, int(len(new_rev)*p)))
            for j in new:
                for l in new:
                    if j < l:
                        updates += updateNeighborsReworked(dg, j, l)
                        updates += updateNeighborsReworked(dg, l, j)
                for l in old:
                    if l != j:
                        updates += updateNeighborsReworked(dg, j, l)
                        updates += updateNeighborsReworked(dg, l, j)
        if updates < stoprate*k*len(dg.nodes()):
            for n1 in dg.nodes():
                for n2 in dg.node[n1]['neibs']:
                    dg.add_edge(n1, n2)
            return


def updateNeighborsReworked (dg, head, tail):
    if tail in dg.node[head]['neibs']:
        return 0
    maxD = getMaxDist(dg, head)
    d = dist(dg.node[head], dg.node[tail])
    if maxD[1] > d:
        del dg.node[head]['neibs'][maxD[0]]
        dg.node[maxD[0]]['rev_neibs'].remove(head)
        dg.node[head]['neibs'][tail] = d
        dg.node[tail]['rev_neibs'].append(head)
        return 1
    else:
        return 0

def evaluateRecall (dg, K):
    rights = 0.0
    wrongs = 0.0
    for i in dg.nodes():
        distances = {}
        for j in dg.nodes():
            if i != j:
                distances[j] = dist(dg.node[i], dg.node[j])
        sort_by_dist = sorted(distances, key=distances.get)
        for j in sort_by_dist[0:K]:
            if j in dg.node[i]['neibs'] or distances[j] == getMaxDist(dg,i)[1]:
                rights += 1
            else:
                wrongs += 1
    return rights/(rights+wrongs)

K = 10

#Cities Test

for j in [1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000]:
    G = nx.DiGraph()
    amounts = 0
    pos = {}
    labels = {}
    f = open("worldcitiespop.txt",'r')
    i = 1
    for line in f:
        temp = line.rstrip('\n').split(',')
        if len(temp) == 7 and temp[0] == "it":
            pos[i] = (float(temp[5]), float(temp[6]))
            labels[i] = i
            G.add_node(i, pars=pos[i], type=labels[i], isnew=True, neibs={}, rev_neibs=[], lastchecked=(0, 0))
            #NNAdd(i, K+3, 2, G)
            i += 1
        if i > j:
            break

    f.close()

    print(len(pos))

    #a = datetime.datetime.now()

    #for i in reversed(range(1, len(pos)+1)):
        #NNAdd(i, K, 1, G)

    # b = datetime.datetime.now()

    # print("creating kNN graph for NNAdd algorithm")
    # print(b-a)

    #a = datetime.datetime.now()

    NNDescentFullReworked(G, K, 0.5, 0.001)
    #NNDescentBasicReworked(G, K)
    #b = datetime.datetime.now()

    #print("creating kNN graph for NNDescent algorithm")
    #print(b-a)


    # print ("Distance has been calculated:")
    print(amounts)
    # print("times")
    amounts = 0
    #print ("Recall")
    print(evaluateRecall(G, K))

	

# Iris test

# G = nx.DiGraph()
#
# a = datetime.datetime.now()
#
# colors = []
# labels = {}
#
# f = open("iris.data",'r')
# i = 1
# for line in f:
#     temp = line.rstrip('\n').split(',')
#     if len(temp) == 5:
#         G.add_node(i, pars=temp[0:4], type=temp[4], isnew=True, neibs={}, rev_neibs=[], lastchecked=(0, 0))
#         if G.node[i]['type'] == "Iris-setosa":
#             colors.append('r')
#             labels[i] = "setosa"
#         elif G.node[i]['type'] == "Iris-versicolor":
#             colors.append('b')
#             labels[i] = "versicolor"
#         else:
#             colors.append('g')
#             labels[i] = "virginica"
#         NNAdd(i, K, 10, G)
#     i += 1
#
# f.close()
#
# for i in reversed(range(1, len(G.nodes())+1)):
#     NNAdd(i, K, 3, G)
#     #print i
#     # if i%20 == 0:
#     #     nx.draw(G, node_size=300, width=0.5, font_size=10, node_color=colors, with_labels=labels)
#     #     plt.show()
#
# b = datetime.datetime.now()
#
# print("Reading and creating kNN graph for NNAdd algorithm")
# print(b-a)
#
# print ("Distance has been calculated:")
# print(amounts)
#
# # r = evaluateQuality(G)
# # print ("Correct/incorrect nodes:")
# # print(r)
#
# print ("Recall")
# print(evaluateRecall(G, K))
#
#
# print("times")
# amounts = 0
# #
# colors = []
# labels = {}
# for i in G.nodes():
#     if G.node[i]['type'] == "Iris-setosa":
#         colors.append('r')
#         labels[i] = "setosa"
#     elif G.node[i]['type'] == "Iris-versicolor":
#         colors.append('b')
#         labels[i] = "versicolor"
#     else:
#         colors.append('g')
#         labels[i] = "virginica"
#
# pos = nx.spring_layout(G, iterations=20)
# nx.draw(G, with_labels=True, node_size=300, width=0.5, font_size=10, node_color=colors, labels=labels)
# plt.show()
# #
# # G.remove_edges_from(G.edges())
# # G.remove_nodes_from(G.nodes())
# # a = datetime.datetime.now()
# #
# # f = open("iris.data",'r')
# # i = 1
# # for line in f:
# #     temp = line.rstrip('\n').split(',')
# #     if len(temp) == 5:
# #         G.add_node(i, pars=temp[0:4], type=temp[4], isnew=True, neibs={}, rev_neibs=[])
# #     i += 1
# #
# # f.close()
# #
# # NNDescentFullReworked(G, K)
# #
# # b = datetime.datetime.now()
# #
# # nx.draw(G, with_labels=True, node_size=300, width=0.5, font_size=10, node_color=colors, labels=labels)
# # plt.show()
# #
# # print("Reading and creating kNN graph for NNDescent algorithm")
# # print(b-a)
# #
# # r = evaluateQuality(G)
# #
# # print ("Correct/incorrect nodes:")
# # print(r)
# #
# # print ("Recall")
# # print(evaluateRecall(G, K))
# #
# # print ("Distance has been calculated:")
# # print(amounts)
# # print("times")
# #amounts = 0
#
