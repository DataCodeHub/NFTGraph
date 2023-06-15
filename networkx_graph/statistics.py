#!/usr/bin/env python
# coding: utf-8


import networkx as nx
import numpy as np
import pandas as pd
import collections
from scipy.special import perm, comb
import matplotlib.pyplot as plt
# %matplotlib inline

chain_name = 'NFTGraph_All'


df = pd.read_csv('./edges.csv')
df['transactionFee'] = (df['transactionFee']).astype(np.float64)
df['transferedAmount'] = (df['transferedAmount']).astype(np.float64)
df['value'] = (df['value']).astype(np.float64)
print(df.dtypes)
print(df.head(10))

AvgTransFee = df.transactionFee.mean()
print("AvgTransFee=",AvgTransFee)

AvgAmt = df.transferedAmount.mean()
print("AvgAmt=",AvgAmt)

AvgVal = df.value.mean()
print("AvgVal=",AvgVal)

#MultiDiGraph:directed,Self-loops allowed,Parallel edges allowed
mG = nx.from_pandas_edgelist(df, 'source', 'target',create_using = nx.MultiDiGraph())
#simple, undirected graph:undirected,Self-loops allowed, NO Parallel edges
uG = nx.from_pandas_edgelist(df, 'source', 'target',create_using = nx.Graph())
#simple, directed graph:directed,Self-loops allowed, NO Parallel edges
diG = nx.from_pandas_edgelist(df, 'source', 'target', create_using = nx.DiGraph())


#1.node count
print("diG.number_of_nodes()=",diG.number_of_nodes())

#2. edge count
print("mG.number_of_edges()=",mG.number_of_edges())
print("diG.number_of_edges()=",diG.number_of_edges())
print("uG.number_of_edges()=",uG.number_of_edges())

#3. self-loops
print("nx.number_of_selfloops(diG)=",nx.number_of_selfloops(diG))

#4. density
print("nx.density(diG)=",nx.density(uG))

#5. Vertex Degree Distribution: power-law
degree_sequence = sorted([d for n,d in diG.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
frac = [n/diG.number_of_nodes() for n in cnt]
fig, ax = plt.subplots()

#6. Correlation of Indegree and Outdegree
degree_in = sorted([d for d in diG.in_degree()], reverse=True)  # degree_in sequence
degree_out = sorted([d for d in diG.out_degree()], reverse=True)  # degree_out sequence
degree_ratio = []
for i in range(len(degree_in)):
    try:
        degree_ratio.append(degree_out[i][1]/degree_in[i][1])
    except:
        # degree_ratio.append(1000)
        pass

#6. Correlation of Indegree and Outdegree
degree_ratio_sequence = sorted([d for d in degree_ratio], reverse=False)
degreeRatioCount = collections.Counter(degree_ratio_sequence)
ratio_degree, ratio_cnt = zip(*degreeRatioCount.items())

ratio_deg = np.array(ratio_degree)
ratio_cnt = np.array(ratio_cnt)
ratio_frac = np.cumsum(ratio_cnt)
ratio_cum = [n/diG.number_of_nodes() for n in ratio_frac]

fig, ax = plt.subplots()
plt.plot(ratio_deg, ratio_cum,'.-')

t = np.linspace(1,100,1000)
# ax.set_yscale('log')
ax.set_xscale('log')
plt.ylabel("CDF")
plt.xlabel("Outdegree to indegree ratio")
# plt.show()
plt.savefig("./logs/{}/fig2:{}-inOutDegree.pdf".format(chain_name,chain_name),dpi=600)

#中间文件
tmp = pd.concat([pd.DataFrame(ratio_deg),pd.DataFrame(ratio_cum)],axis=1)
tmp.columns = ['ratio_deg','ratio_cum']
tmp.to_csv('./inter-data/inOutDegree-{}.csv'.format(chain_name), index=False)


#7.centrality
degree_centrality = nx.degree_centrality(diG)
# print("degree_centrality=",degree_centrality)

# # expensive cost
# betweenness_centrality = nx.betweenness_centrality(uG)
# print("betweenness_centrality=",betweenness_centrality)

pagerank_centrality = nx.pagerank(diG)
# print("pagerank_centrality=",pagerank_centrality)



degree_centrality_sequence = sorted(list(degree_centrality.values()),reverse=True)
degree_Count = collections.Counter(degree_centrality_sequence)
degree, degree_cnt = zip(*degree_Count.items())
degree_cnt_cum = np.cumsum(np.array(degree_cnt))

# wcc_betweenness_centrality_sequence = sorted(list(wcc_betweenness_centrality.values()),reverse=True)
# wcc_betweenness_Count = collections.Counter(wcc_betweenness_centrality_sequence)
# wcc_betweenness, wcc_betweenness_cnt = zip(*wcc_betweenness_Count.items())
# wcc_betweenness_cnt_cum = np.cumsum(np.array(wcc_betweenness_cnt))

pagerank_centrality_sequence = sorted(list(pagerank_centrality.values()),reverse=True)
pagerank_Count = collections.Counter(pagerank_centrality_sequence)
pagerank, pagerank_cnt = zip(*pagerank_Count.items())
pagerank_cnt_cum = np.cumsum(np.array(pagerank_cnt))

fig, ax = plt.subplots()


# plt.plot(wcc_clossness_cnt_cum,wcc_clossness,'b.-')
plt.plot(degree_cnt_cum,degree,'r--')
# plt.plot(wcc_betweenness_cnt_cum,wcc_betweenness,'y-')
plt.plot(pagerank_cnt_cum,pagerank,'b.-')

ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(['degree','pagerank'])
plt.ylabel("Centrality value")
plt.xlabel("Vertex")
plt.savefig("./logs/{}/fig1:{}-centrality.pdf".format(chain_name,chain_name),dpi=600)


#中间文件
tmp1 = pd.concat([pd.DataFrame(degree_cnt_cum),pd.DataFrame(degree),pd.DataFrame(pagerank_cnt_cum),pd.DataFrame(pagerank)],axis=1)
tmp1.columns = ['degree_cnt_cum','degree','pagerank_cnt_cum','pagerank']
tmp1.to_csv('./inter-data/centrality-{}.csv'.format(chain_name), index=False)


#8. Reciprocity
print("nx.reciprocity(diG)=",nx.reciprocity(diG))

#9.Assortativity
print("nx.degree_assortativity_coefficient(diG)=",nx.degree_assortativity_coefficient(diG))


#10.Strong and Weakly Connected Components:simple, directed version
#SCC
print("nx.number_strongly_connected_components(diG)=",nx.number_strongly_connected_components(diG))

scc_size_list = [len(c) for c in sorted(nx.strongly_connected_components(diG), key=len, reverse=True)]
largest_scc_vertex_set = max(nx.strongly_connected_components(diG), key=len)
largest_scc_size = len(largest_scc_vertex_set)

print("#nodes of the largest scc of diG=",diG.subgraph(largest_scc_vertex_set).number_of_nodes())
print("#edges of the largest scc of diG=",diG.subgraph(largest_scc_vertex_set).number_of_edges())

#WCC
print("nx.number_weakly_connected_components(diG)=",nx.number_weakly_connected_components(diG))

wcc_size_list = [len(c) for c in sorted(nx.weakly_connected_components(diG), key=len, reverse=True)]
largest_wcc_vertex_set = max(nx.weakly_connected_components(diG), key=len)
largest_wcc_size = len(largest_wcc_vertex_set)

print("#nodes of the largest wcc of diG=",diG.subgraph(largest_wcc_vertex_set).number_of_nodes())
print("#edges of the largest wcc of diG=",diG.subgraph(largest_wcc_vertex_set).number_of_edges())

# wcc - draw
wcc_size_sequence = sorted([d for d in wcc_size_list], reverse=True)  # degree sequence
wccSizeCount = collections.Counter(wcc_size_sequence)
wcc_size, wcc_cnt = zip(*wccSizeCount.items())

# scc - draw
scc_size_sequence = sorted([d for d in scc_size_list], reverse=True)  # degree sequence
sccSizeCount = collections.Counter(scc_size_sequence)
scc_size, scc_cnt = zip(*sccSizeCount.items())


fig, ax = plt.subplots()
plt.plot(wcc_cnt,wcc_size,'+')
plt.plot(scc_cnt,scc_size,'x')
t = np.linspace(1,100,1000)
ax.set_yscale('log')
ax.set_xscale('log')
plt.ylabel("#vertices")
plt.xlabel("#components")
plt.legend(['weakly','strongly'])
# plt.show()
plt.savefig("./logs/{}/fig3:{}-wccScc.pdf".format(chain_name,chain_name),dpi=600)


# the largest wcc subgraph without self-loops
frozen_graph = nx.freeze(diG.subgraph(largest_wcc_vertex_set))
unfrozen_graph = nx.Graph(frozen_graph)
unfrozen_graph.remove_edges_from(nx.selfloop_edges(unfrozen_graph))


#11. k-core
print("nx.k_core(unfrozen_graph):",nx.k_core(unfrozen_graph))
print("#nodes of nx.k_core(unfrozen_graph)=",nx.k_core(unfrozen_graph).number_of_nodes())
print("#edges of nx.k_core(unfrozen_graph)=",nx.k_core(unfrozen_graph).number_of_edges())
k_core = nx.core_number(nx.k_core(unfrozen_graph))
print("max(k-core.values())=",max(k_core.values()))


#12. Triangles
# number of triangles in the largest wcc
print("#triangles of uG",np.sum(list(nx.triangles(uG).values())))


#13. Transitivity(传递性)
print("transitivity of uG",nx.transitivity(uG))



#14. Clustering Coeff
print("clustering of uG",nx.average_clustering(uG))

