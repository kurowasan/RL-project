import numpy as np

class SparseGraphEnvironment(object):
    """Environment with a sparse path through a graph"""
    def __init__(self, state_dim=10, correct_path_proportion = 0.1,
                branching_prob = 1.0, nb_actions = 4, arborescent = False):
        super(SparseGraphEnvironment, self).__init__()
        self.nb_nodes = state_dim*state_dim
        #self.correct_path_proportion = correct_path_proportion
        self.correct_path_proportion = (np.log2(self.nb_nodes))/self.nb_nodes
        self.branching_prob = branching_prob
        self.nb_actions = nb_actions
        self.start = 0
        self.end = self.nb_nodes-1
        self.transition = np.zeros((self.nb_nodes, self.nb_nodes, self.nb_actions))

        ## fixing the first and last nodes as start and end
        ## making a random path of 60% of other nodes - this is the path to learn
        self.correct_path = np.random.permutation(np.arange(1,self.nb_nodes-1))[:int(self.correct_path_proportion*self.nb_nodes)]
        self.correct_path = np.hstack(([0],self.correct_path))
        self.correct_path = np.hstack((self.correct_path,self.nb_nodes-1))

        ## creating the adjacency matrix
        self.adjacency = np.zeros((self.nb_nodes,self.nb_nodes))

        if arborescent:
            self.make_tree()
        else:
            self.make_paths()

        self.make_transition()

    def add_edges(self,from_node,to_node):
        for i,j in zip(from_node, to_node):
            self.adjacency[i,j]=1
            self.adjacency[j,i]=1

    def get_usable_nodes(self,adjacency):
        usable = [i for i in np.arange(self.nb_nodes-1) if np.sum(self.adjacency[i,:])< self.nb_actions and np.sum(self.adjacency[i,:])> 0 ]
        return usable

    def check_if_same(self,from_node,to_node):
        for i,j in zip(from_node,to_node):
            if i==j:
                return True
        return False

    def make_paths(self):

        self.add_edges(self.correct_path[:-1], self.correct_path[1:])


        ### creating the first level of alternative paths
        used_nodes = [i for i in np.arange(self.nb_nodes) if i in self.correct_path]
        not_used_yet = not_used_yet = np.arange(self.nb_nodes)[(np.sum(self.adjacency,axis=1)==0)]
        print(self.nb_nodes)

        while np.sum(np.sum(self.adjacency,axis=1)==0)>0:
            potential_acceptors = self.get_usable_nodes(self.adjacency)    
            to_add = np.random.permutation(not_used_yet)[:int(np.ceil(len(not_used_yet)*0.8))]
            acceptors = np.random.permutation(potential_acceptors)[:len(to_add)]

            reshuffle = self.check_if_same(to_add, acceptors) #checking that the edges are between different nodes! (non-absorbing)
            while reshuffle: 
                acceptors = np.random.permutation(potential_acceptors)[:len(to_add)]
                reshuffle = self.check_if_same(to_add, acceptors)

            self.add_edges(to_add,acceptors)
            used_nodes = used_nodes+list(acceptors)
            used_nodes = used_nodes+list(to_add)
            used_nodes = list(set(used_nodes))

            not_used_yet = np.arange(self.nb_nodes)[(np.sum(self.adjacency,axis=1)==0)]
        return self.adjacency

    def make_tree(self):
        pass

        self.add_edges(self.correct_path[:-1], self.correct_path[1:])

        ### creating the first level of alternative paths
        used_nodes = [i for i in np.arange(self.nb_nodes) if i in self.correct_path]
        not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]
        

        current_tree_level = 0
        current_level_nodes = []
        while len(not_used_yet)>0:
            #import pdb; pdb.set_trace()
            current_level_nodes.append(self.correct_path[current_tree_level])
            donors = []
            acceptors = []

            for node in current_level_nodes:
                if len(not_used_yet)==0:
                    break
                ### pick children
                nb_children = np.random.randint(2,self.nb_actions-1)
                children = np.random.permutation(not_used_yet)[:nb_children]
                
                ### add children to the acceptor node list

                acceptors+=list(children)
                donors+=list(np.ones(nb_children).astype('int') * node)
                used_nodes+=list(donors)
                used_nodes+=list(acceptors)
                used_nodes = list(set(used_nodes))
                not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]    
            self.add_edges(donors,acceptors)
            not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]
            current_level_nodes = acceptors
            current_tree_level+=1 #keeping track of the correct path

        return self.adjacency


    def make_transition(self):

        for i in np.arange(self.adjacency.shape[0]):
            j_nodes = np.nonzero(self.adjacency[i])[0]
            
            for a in np.arange(self.nb_actions):
                
                if a < len(j_nodes): ### checking if it's a valid action
                    j = j_nodes[a]
                    if len(j_nodes)==1: ### if the total number of actions is one
                        self.transition[i,j,a]+=1
                    else: ### more than one action possible
                        correct_prob = np.random.uniform(0.9,1)
                        self.transition[i,j,a]+=correct_prob
                        for b in j_nodes:
                            if not j_nodes[a]==b:
                                self.transition[i,b,a]+=((1-correct_prob)/(len(j_nodes)-1))
                else:
                    ### return to same node
                    self.transition[i,i,a]+=1
        self.transition/=np.sum(self.transition)        
