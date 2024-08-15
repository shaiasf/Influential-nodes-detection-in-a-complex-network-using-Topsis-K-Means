import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep


def show_graph(g):
  plt.figure(figsize=(30,20))
  sp=nx.spring_layout(g)
  
  nx.draw_networkx(G=g,pos=sp,with_labels=False,node_size=35)
  plt.title('figure 1')
  
  plt.axis('off')
  
  plt.show()


class Topsis():
    evaluation_matrix = np.array([])  # Matrix
    weighted_normalized = np.array([])  # Weight matrix
    normalized_decision = np.array([])  # Normalisation matrix
    M = 0  # Number of rows
    N = 0  # Number of columns

    '''
	Create an evaluation matrix consisting of m alternatives and n criteria,
	with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij},
	we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
	'''

    def __init__(self, evaluation_matrix, weight_matrix=[0.25,0.25,0.25,0.25], criteria=np.array([True,True,True,True])):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix/np.sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    '''
	# Step 2
	The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
	'''

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j]**2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,
                                         j] = self.evaluation_matrix[i, j]/(sqrd_sum[j]**0.5)

    '''
	# Step 3
	Calculate the weighted normalised decision matrix
	'''

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    '''
	# Step 4
	Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
	'''

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    '''
	# Step 5
	Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
	{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
	and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
	{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
	where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances 
	from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
	'''

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.worst_alternatives[j])**2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.best_alternatives[j])**2
                
                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i]**0.5
            self.best_distance[i] = self.best_distance[i]**0.5

    '''
	# Step 6
	Calculate the similarity
	'''

    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / \
                (self.worst_distance[i]+self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / \
                (self.worst_distance[i]+self.best_distance[i])
    
    def ranking(self, data):
        return [i+1 for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        print("Step 1\n", self.evaluation_matrix, end="\n\n")
        self.step_2()
        print("Step 2\n", self.normalized_decision, end="\n\n")
        self.step_3()
        print("Step 3\n", self.weighted_normalized, end="\n\n")
        self.step_4()
        print("Step 4\n", self.worst_alternatives,
              self.best_alternatives, end="\n\n")
        self.step_5()
        print("Step 5\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        print("Step 6\n", self.worst_similarity,
              self.best_similarity, end="\n\n")

def creat_evaluation_matrix(g, dc,bc,cc,ec):
  dclist=[]
  bclist=[]
  cclist=[]
  eclist=[]
  nodes=[]
  
  
  for i in sorted(dc):
    dclist.append(dc[i])
  
  for i in sorted(bc):
    bclist.append(bc[i])
  
  for i in sorted(cc):
    cclist.append(cc[i])
  
  for i in sorted(ec):
    eclist.append(ec[i])
  
  for node in g:
    nodes.append(node)
  
  evolution_matrix = pd.DataFrame({
      'node':nodes,
      'dc':dclist,
      'bc':bclist,
      'cc':cclist,
      'ec':eclist
     })
  return evolution_matrix

def evolution_matrix_tonp(evaluation_matrix):
    ev_matrix_np = evaluation_matrix.to_numpy()
    list_ar=[]
    
    for i in range(ev_matrix_np.shape[0]):
      list_ar.append(ev_matrix_np[i][1:5])
    ev_matrix_np=np.array(list_ar)
  
    return ev_matrix_np

def entropy(g, ev_mat):
  m = nx.number_of_nodes(g)  
  sum = np.zeros(4)
  
  for i in range(m):
    for j in range(4):
      sum[j] += ev_mat[i][j]
  
  p = np.copy(ev_mat)
  
  for i in range(m):
    for j in range(4):
      p[i][j] = ev_mat[i][j]/sum[j]

  k = 1/log(m)
  E = []
  plnp = []
 
  
  for j in range(4):
    temp_sum = 0
    for i in range(m):
      try:
         pij = p[i][j]
         temp_sum += pij*log(pij)
      except:
        pij = 1
        temp_sum += pij*log(pij)
    E.append( -k*temp_sum)

  D = []
  for i in range(4):
     D.append(1-E[i])

  sm = np.sum(D)
  return [D[i]/sm for i in range(4)]

def top_meth(evolution_matrix,t:Topsis):
    best_dist = np.array(t.best_distance)
    worst_dist = np.array(t.worst_distance)
    closenness = []
    closenness = worst_dist / (worst_dist + best_dist)
    
    C_df=pd.DataFrame(closenness,columns=['C'])
    c=sorted(range(len(closenness)),key=closenness.__getitem__)
    c= c[::-1]
    
    
    
    nodes_list=[]
    
    for i in range(len(c)):
      nodes_list.append(c[i])
    
    closeness_topsis = pd.DataFrame(
    {
        'S+': best_dist,
        'S-': worst_dist,
        'C': closenness,
        'node2':nodes_list
    }
    )

    centralities_closeness = pd.concat([evolution_matrix, C_df], axis=1)
    centralities_closeness = centralities_closeness.sort_values(by="C",ascending=False)
    centralities_closeness = centralities_closeness.drop(centralities_closeness.columns[0], axis=1)
    centralities_closeness.reset_index(inplace=True)
    centralities_closeness=centralities_closeness.rename(columns={'index':'node'})
    centralities_closeness.to_csv('kinit.csv')
    result = pd.concat([closeness_topsis, centralities_closeness], axis=1)


    degree=pd.DataFrame(result[['node','dc']].sort_values(by='dc',ascending=False))
    degree.rename(columns={'node':'dcn'},inplace=True)
    
    closeness=pd.DataFrame(result[['node','cc']].sort_values(by='cc',ascending=False))
    closeness.rename(columns={'node':'ccn'},inplace=True)
    
    betweenness=result[['node','bc']].sort_values(by='bc',ascending=False)
    betweenness.rename(columns={'node':'bcn'},inplace=True)
    
    eigenvector=result[['node','ec']].sort_values(by='ec',ascending=False)
    eigenvector.rename(columns={'node':'ecn'},inplace=True)
    
    
    degree_list=degree['dcn'].values.tolist()
    closeness_list=closeness['ccn'].values.tolist()
    betweenness_list=betweenness['bcn'].values.tolist()
    eigenvector_list=eigenvector['ecn'].values.tolist()
    
    
    mes_nodes=pd.DataFrame({
        
    
        'dcn':degree_list,
        'ccn':closeness_list,
        'bcn':betweenness_list,
        'ecn':eigenvector_list
    
    })

    final_res = pd.concat([mes_nodes,closeness_topsis], axis=1)

    nomber_des_nodes = int(input('entrer le nomber des noeuds influents voulu: '))
    return (final_res.head(nomber_des_nodes), final_res, )


def SI(g,node):
    n = nx.number_of_nodes(g)
    model = ep.SIModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', 1)
    cfg.add_model_initial_configuration('Infected', node)
    model.set_initial_status(cfg)
    res = pd.DataFrame(columns=['iteration', 'nb_Susceptible', 'Nb_infected'])
    for i in range(n):
        iteration = model.iteration()
        res.loc[len(res.index)] = [iteration['iteration'], iteration['node_count'][0], iteration['node_count'][1] ]
        if iteration['node_count'][1] == n:
            break 
    return res

def si_evaluate(g,nodes:pd.DataFrame):
  Rank_DC = SI(g,set(nodes['dcn']))
  Rank_BC = SI(g,set(nodes['bcn']))
  Rank_CC = SI(g,set(nodes['ccn']))
  Rank_EC = SI(g,set(nodes['ecn']))
  Rank_Topsis = SI(g,set(nodes['node2']))


  fig, axs = plt.subplots(2, 2, figsize=(10, 7))
  axs[0, 0].plot(Rank_DC['iteration'], Rank_DC['Nb_infected'], label="DC")
  axs[0, 0].plot(Rank_Topsis['iteration'], Rank_Topsis['Nb_infected'], label="Topsis")
  axs[0, 0].set_title("DC vs W-Topsis")
  axs[0, 0].legend()
  
  axs[0, 1].plot(Rank_BC['iteration'], Rank_BC['Nb_infected'], label="BC")
  axs[0, 1].plot(Rank_Topsis['iteration'], Rank_Topsis['Nb_infected'], label="Topsis")
  axs[0, 1].set_title("BC vs W-Topsis")
  axs[0, 1].legend()
  
  axs[1, 0].plot(Rank_CC['iteration'], Rank_CC['Nb_infected'], label="CC")
  axs[1, 0].plot(Rank_Topsis['iteration'], Rank_Topsis['Nb_infected'], label="Topsis")
  axs[1, 0].set_title("CC vs W-Topsis")
  axs[1, 0].legend()
  
  axs[1, 1].plot(Rank_EC['iteration'], Rank_EC['Nb_infected'], label="EC")
  axs[1, 1].plot(Rank_Topsis['iteration'], Rank_Topsis['Nb_infected'], label="Topsis")
  axs[1, 1].set_title("EC vs W-Topsis")
  axs[1, 1].legend()
  
  for ax in axs.flat:
      ax.set(xlabel='t', ylabel='F(t)')
  plt.show()
  