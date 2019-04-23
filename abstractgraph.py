import numpy as np

class SparseGraphEnvironment(object):
    """Environment with a sparse path through a graph"""
    def __init__(self, state_dim=10, correct_path_proportion = 0.2, branching_prob = 1.0, nb_actions = 4, arborescent = False):
        super(SparseGraphEnvironment, self).__init__()
        self.nb_nodes = state_dim*state_dim
        self.correct_path_proportion = correct_path_proportion
        self.branching_prob = branching_prob
        self.nb_actions = nb_actions
        self.start = 0
        self.end = self.nb_nodes-1
        self.transition = np.zeros((nb_nodes, nb_nodes, nb_actions))

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
        usable = [i for i in np.arange(self.nb_nodes-1) if np.sum(self.adjacency[i,:])< self.nb_actions]
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
        not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]


        while len(not_used_yet)>0:
            potential_acceptors = self.get_usable_nodes(self.adjacency)    
            to_add = np.random.permutation(not_used_yet)[:int(np.ceil(len(not_used_yet)*self.branching_prob))]
            used_nodes = used_nodes+list(to_add)
            acceptors = np.random.permutation(potential_acceptors)[:len(to_add)]

            reshuffle = self.check_if_same(to_add, acceptors) #checking that the edges are between different nodes! (non-absorbing)
            while reshuffle: 
                acceptors = np.random.permutation(potential_acceptors)[:len(to_add)]
                reshuffle = self.check_if_same(to_add, acceptors)

            self.add_edges(to_add,acceptors)    
            not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]
        return self.adjacency

    def make_tree(self):
        pass

        self.add_edges(self.correct_path[:-1], self.correct_path[1:])

        ### creating the first level of alternative paths
        used_nodes = [i for i in np.arange(self.nb_nodes) if i in self.correct_path]
        not_used_yet = [i for i in np.arange(self.nb_nodes-1) if i not in used_nodes]
        current_level_nodes = [correct_path[0]]

        for current_level in xrange(len(self.correct_path)):
            for node in current_level_nodes:
                number_of_children = np.random.randint(2,self.nb_actions-1)

        ###creating a tree of 

        return self.adjacency


    def make_transition(self):

        for i in np.arange(self.adjacency.shape[0]):
            
            j_nodes = np.nonzero(self.adjacency[i])[0]
            max_actions = len(j_nodes)
            for a in np.arange(max_actions):
                j = j_nodes[a]
                self.transition[i,j,a]+=1











                    
