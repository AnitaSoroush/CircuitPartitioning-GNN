import os 
import numpy as np
from collections import defaultdict

from cirpart.globals import *
from cirpart.utils.tools import dict_to_lists

get_fp = lambda x: os.path.join(DATA_DIR, x + ".dat")


def get_netlist(circuit):
    fp = get_fp(circuit)
    netlist = Netlist()
    netlist.load_dat(fp)

    return netlist

class Netlist:
    def __init__(self):
        self._pass = False
        self.cutsize = None
        self.balanced = None

        # self.part0 = None
        # self.part1 = None

        self.Mcut = defaultdict(lambda: 0)
    
    def load_dat(self, fp):
        self.__init__()
        self.Nnets, self.Nmods, self.net_to_mods, self.mod_to_nets = Netlist.read_data(fp)
        self.create_netlist(self.Nnets, self.Nmods, self.net_to_mods, self.mod_to_nets)
    
    def save_dat(self, fp):
        """Save the current netlist to a .dat file"""
        with open(fp, "w") as f:
            f.write("Dimensions\n")
            f.write(f"\tnets = {self.Nnets} : 1.0\n")
            f.write(f"\tmodules = {self.Nmods} : 1.0\n")
            f.write(";\n\n")
            f.write("Connections\n")
            # for net, mods in self.net_to_mods.items():
            for net in range(1, self.Nnets+1):
                mods = self.net_to_mods[net]
                f.write(f"\t{net} : ")
                for mod in mods:
                    f.write(f"{mod} ")
                f.write("-1\n")
            f.write(";\n")
            f.write("END")
    
    def read_data(fp):
        """Read the data from the .dat file
        
        Return: netlist object
        """
        print(fp)
        with open(fp, "r") as f:
            lines = f.readlines()
            lines = [line.strip().replace("\t", "").replace("\n", "") for line in lines]

        dimension_line = -1
        connection_line = -1
        column_line = []
        for idx, line in enumerate(lines):
            if line == "Dimensions":
                dimension_line = idx
            elif line == "Connections":
                connection_line = idx
            elif line == ";":
                column_line.append(idx)

        dimension_info = lines[dimension_line+1:column_line[0]]
        connection_info = lines[connection_line+1:column_line[1]]

        Nnets = -1
        Nmods = -1

        # read from the format "net = a : b"
        net_info = dimension_info[0].split("=")[1].split(":")
        mod_info = dimension_info[1].split("=")[1].split(":")
        Nnets = int(net_info[0])
        # Wnets = float(net_info[1].strip())
        Nmods = int(mod_info[0])

        net_to_mods = {}

        line_need_fix = []
        for idx, line in enumerate(connection_info):
            if ":" not in line:
                line_need_fix.append(idx)
        for idx in line_need_fix[::-1]:
            connection_info[idx-1] += " " + connection_info[idx]
            connection_info.pop(idx)

        for line in connection_info:
            if ":" not in line:
                print(line)
            net, mods = line.split(":")
            net = int(net.strip())
            mods = mods.strip().split(" ")
            mods = [i for i in mods if i != ""]
            mods = [int(i) for i in mods][:-1]
            net_to_mods[net] = mods
        
        mod_to_nets = dict()
        # reverse the net_to_mods to mod_to_nets
        for net, mods in net_to_mods.items():
            for mod in mods:
                if mod not in mod_to_nets:
                    mod_to_nets[mod] = []
                mod_to_nets[mod].append(net)
                
        # print("Number of modules", Nmods)
        # print("Number for nets", Nnets)
        # print("connections", net_to_mods)
        
        return Nnets, Nmods, net_to_mods, mod_to_nets

    def create_netlist(self, Nnets, Nmods, net_to_mods, mod_to_nets):
        self.Nnets = Nnets
        self.Nmods = Nmods
        self.net_to_mods = net_to_mods
        self.mod_to_nets = mod_to_nets    # key: mod_id, value: net_id
        # self.mod_to_mods = dict()    # key: mod_id, value: mod_id
        self.edge_list = {}
        self.group_dict = dict()    # key: mod_id, value: group_id
        self.hyper_edges = [set(module - 1 for module in modules) for modules in net_to_mods.values()]
        # print(net_to_mods)
        # print(self.hyper_edges)

        # # reverse the net_to_mods to mod_to_nets
        # for net, mods in self.net_to_mods.items():
        #     for mod in mods:
        #         if mod not in self.mod_to_nets:
        #             self.mod_to_nets[mod] = []
        #         self.mod_to_nets[mod].append(net)
        
        # بخش پایین مربوط به ادجیسنسی بود قبلا که کامنت شده اینو بایداستفاده کنی
        
        # print(Nnets)
        # print(self.net_to_mods)
        for net_index in range(Nnets):
            net = self.net_to_mods[net_index+1]
            net_to_edge_weight = round(1/(len(net)-1),3)
            for mod1 in net:
                for mod2 in net:
                    if mod1 == mod2:
                        continue
                    if (mod1-1, mod2-1) in self.edge_list:
                        self.edge_list[(mod1-1, mod2-1)] += net_to_edge_weight
                    else:
                        self.edge_list[(mod1-1, mod2-1)] = net_to_edge_weight
                        
        self.edge_index = [[e[0] for e in list(self.edge_list.keys())],
                           [e[1] for e in list(self.edge_list.keys())]]
        # print(self.edge_index)
        self.edge_weights = list(self.edge_list.values())
        
        # print(len(self.edge_list))
            
        # print(self.net_to_mods.items())
        
        # پایین

        # graph = np.zeros((Nmods, Nmods), dtype=int)
        # for mod, nets in self.mod_to_nets.items():
        #     for net in nets:
        #         for mod2 in net_to_mods[net]:
        #             graph[mod-1, mod2-1] += 1
        # for i in range(Nmods):
        #     graph[i, i] = 0
        # self.graph = graph
        
        # بالا

        self.rand_bipart()      # Initalize self.group_dict

        self.last_move = {"mod": None, "cutsize": self.cutsize}
        self._pass = True
    
    def calculate_Mcut(self):
        Mcut = defaultdict(lambda: 0)
        for _, mods in self.net_to_mods.items():
            group_ids = [self.group_dict[mod] for mod in mods]

            if len(set(group_ids)) > 1:
                for mod in mods:
                    Mcut[mod] = 1
        
        for i in range(self.Nmods):
            if i+1 not in Mcut:
                Mcut[i+1] = 0
                
        self.Mcut = Mcut

    def rand_bipart(self):
        """Given a list of integer, evenly split them into 2 groups, two groups have the same length"""
        self.part0 = []
        self.part1 = []

        # arr = np.arange(1, self.num_modules+1)
        ids = np.arange(1, self.Nmods+1)
        np.random.shuffle(ids)

        group_size = len(ids) // 2

        for i in ids[:group_size]:
            self.group_dict[i] = 0
            # self.part0.append(i)
        for i in ids[group_size:]:
            self.group_dict[i] = 1
            # self.part1.append(i)
        
        self.cutsize = self.calculate_cutsize()
        self.calculate_Mcut()
        self.balanced = True
    
    def is_balanced(self):
        group0, group1 = self.get_groups()
        is_balanced = abs(len(group0) - len(group1)) <= 1
        self.balanced = is_balanced
        return is_balanced
    
    def abs_group_diff(self):
        group0, group1 = self.get_groups()
        return abs(len(group0) - len(group1))   

    def get_groups(self):
        """Get the group member of each current partition

        return: group0, group1
                the two groups are lists of module ids
        """
        
        group0 = []
        group1 = []
        for mod_id, group_id in self.group_dict.items():
            if group_id == 0:
                group0.append(mod_id)
            else:
                group1.append(mod_id)
        return group0, group1

    def get_group_id(self):
        """ Get the list of group id, the list is ordered by module id

        Returns:
            group_id: index by module ID - 1
        """
        
        return np.array([1 if self.group_dict[i+1] == 1 else -1 for i in range(self.Nmods)])

    def calculate_cutsize(self):
        """Given a partition, calculate the cutsize"""
        cutsize = 0
        for _, mods in self.net_to_mods.items():
            group_ids = [self.group_dict[mod] for mod in mods]
            if len(set(group_ids)) > 1:
                cutsize += 1
        return cutsize
        

    def move(self, mod_id):
        if self.cutsize is None:
            self.cutsize = self.calculate_cutsize()

        # prev_cutsize = self.last_move["cutsize"]
        prev_cutsize = self.cutsize

        # Flip the group of the module
        prev_group = self.group_dict[mod_id]
        new_group = 1 - prev_group

        self.group_dict[mod_id] = new_group
        self.last_move["mod"] = mod_id

        # If the module is not connected to any net, then the cutsize will not change
        if mod_id not in self.mod_to_nets:
            return
        
        # update cutsize here to avoid redundant calculation
        nets_connected_to_mod = self.mod_to_nets[mod_id]
        for net in nets_connected_to_mod:
            groups = set([self.group_dict[mod] for mod in self.net_to_mods[net] if mod != mod_id])
            groups = list(groups)
            if len(groups) == 1:
                if groups[0] == new_group:
                    prev_cutsize -= 1
                    # self.Mcut[mod_id] = 0
                else:
                    prev_cutsize += 1
                    # self.Mcut[mod_if] = 1
            else:
                pass
                # self.Mcut[mod_id] = 1

        self.last_move["cutsize"] = prev_cutsize
        self.cutsize = prev_cutsize
    
    def move_mod(self, mod_id):
        self.move(mod_id)

    def swap(self, mod_id):
        self.move(mod_id)
    
    def rand_swap(self):
        mod_id = self.get_rand_mod(2)
        self.move(mod_id[0])
        self.move(mod_id[1])

    def move_net(self, net_id):
        """ Move a net
            1/3 chance to move to the left
            1/3 chance to move to the right
            1/3 chance to stay at middle
        """

        mods_connected = self.net_to_mods[net_id]

        dice = np.random.rand()
        if dice < 1/3:
            # move to left
            for mod in mods_connected:
                if self.group_dict[mod] != 0:
                    self.move(mod)
        elif dice < 2/3:
            # move to right
            for mod in mods_connected:
                if self.group_dict[mod] != 1:
                    self.move(mod)
        else:
            # randomly choose half of the modules to move to the left
            # and the other half to move to the right
            half_n = len(mods_connected) // 2
            larger_half_n = half_n + len(mods_connected) % 2
            idx = np.arange(len(mods_connected))
            np.random.shuffle(idx)
            for i in idx[:larger_half_n]:
                if self.group_dict[mods_connected[i]] != 0:
                    self.move(mods_connected[i])
            for i in idx[larger_half_n:]:
                if self.group_dict[mods_connected[i]] != 1:
                    self.move(mods_connected[i])

    def reverse_net(self, net_id):
        """Reverse the group of all the modules connected to the net_id"""

        connected_mods = self.net_to_mods[net_id]

        for i in connected_mods:
            self.move_mod(i)
        # if this net is being cut
        # assumed
        
        # if not
        # ??? probably do nothing

    def get_rand_net(self):
        """Get a random net id"""
        return np.random.choice(list(self.net_to_mods.keys()))

    def get_rand_mod(self, n=1):
        """Get one or more random module id, sampled equally from both partitions
        Args:
            n: number of modules id to be returned
        """

        # Cost change when we change this moduel to swap
        mods = []
        
        half_n = n // 2
        larger_half_n = half_n + n % 2

        assert half_n + larger_half_n == n

        group0, group1 = self.get_groups()

        if len(group0) == len(group1) or len(group0) > len(group1):
            # sample larger_half_n ids from group0
            mods += list(np.random.choice(group0, larger_half_n, replace=False))
            mods += list(np.random.choice(group1, half_n, replace=False))
        else:
            mods += list(np.random.choice(group1, larger_half_n, replace=False))
            mods += list(np.random.choice(group0, half_n, replace=False))
    
        return mods
    
    def get_rand_mod_id(self, part=0):
        """Get a random module id given a partition id
        Args:
            part: partition id (0/1)
        """
        group0, group1 = self.get_groups()
        if len(group0) == 0:
            return np.random.choice(group1)
        elif len(group1) == 0:
            return np.random.choice(group0)
        
        if part == 0:
            return np.random.choice(group0)
        else:
            return np.random.choice(group1)
    
    def get_rand_mod_by_type(self, mod_type):
        """Get a random module id given a module type
        Args:
            mod_type: module type
                        - "C" (Cut). "C0" module with net cut, "C1" module with no net cut
                        - "D" (degree)
        """
        if mod_type[0] not in ["C"]:
            raise ValueError("mod_type not found")

        mod_number = int(mod_type[1:])
        mod_type = mod_type[0]

        if mod_type == "C":
            groups = dict_to_lists(self.Mcut)
            return np.random.choice(groups[mod_number])
        elif mod_type == "D":
            import networkx as nx

            # Create a sample graph (you can replace this with your own graph)
            G = nx.erdos_renyi_graph(100, 0.1)

            # Classify nodes based on degree
            high_degree_nodes = [node for node, degree in G.degree() if degree > 5]
            low_degree_nodes = [node for node, degree in G.degree() if degree <= 5]

            print("High-Degree Nodes:", high_degree_nodes)
            print("Low-Degree Nodes:", low_degree_nodes)

    
    def get_rand_mod_by_net_size(self, lo, hi):
        """ Give two integers lo and hi, find all the nets with connected modules in the range [lo, hi], 
            and then randomly choose a module connected to these nets

        Args:
            lo: lower bound of the number of nets that the module is connected to
            hi: upper bound of the number of nets that the module is connected to

        return: a random module id from the current partition
        """
        net_to_mods = self.net_to_mods
        net_candidate_pool = {net: len(mods) for net, mods in net_to_mods.items() if lo <= len(mods) <= hi}

        mod_candidate_pool = []
        for net, _ in net_candidate_pool.items():
            mod_candidate_pool += net_to_mods[net]
        mod_candidate_pool = list(set(mod_candidate_pool))

        mod_candidate_from_group0 = [i for i in mod_candidate_pool if self.group_dict[i] == 0]
        mod_candidate_from_group1 = [i for i in mod_candidate_pool if self.group_dict[i] == 1]

        group0, group1 = self.get_groups()

        if len(group0) == 0 or len(group1) == 0:
            return -1

        if len(group0) == len(group1) or len(group0) > len(group1):
            return np.random.choice(mod_candidate_from_group0)
        else:
            return np.random.choice(mod_candidate_from_group1)


def process_cplex_fil_result(fil_fp):
    """Get all the variables with solution=1 from the .fil file generated by CPLEX
    
    returns:
        (solution, group_dict)
        
        solution: list of variable names that is equal to 1
        group_dict: key = module ID, value = partition ID (0/1)
    """
    with open(fil_fp, "r") as f:
        fil_content = f.readlines()

    # Find the line with the keyword "Incumbent solution"
    for i, line in enumerate(fil_content):
        if "Incumbent solution" in line and "Variable Name" in fil_content[i+1]:
            break

    i = i + 2
    solution = []
    while "All other variables in the range" not in fil_content[i]:
        current_line = fil_content[i].split()
        if float(current_line[1]) != 1:
            i += 1
            continue
        variable = current_line[0].strip()
        solution.append(variable)
        i += 1

    # Initalize group dict: mod_id: group number
    x_var = [i for i in solution if i.startswith('x')]
    group_dict = dict()
    for i in x_var:
        mod_id, group_id = i[1:].split("_")
        group_dict[int(mod_id)] = int(group_id)-1

    return solution, group_dict

def get_net_class(fil_fp, Nnets):
    """
    Args:
        fil_fp: file path to the .fil file generated by CPLEX
        Nnets: number of nets in the netlist
    
    Returns:
        net_dict: key = net ID, value = (0/1/2) 
                                        0: net is inside partition 0
                                        1: net is inside partition 1
                                        2: being cut
    """
    
    solution, _ = process_cplex_fil_result(fil_fp)

    y_var = [i for i in solution if i.startswith("y")]
    y_var.sort(key=lambda x: int(x[1: x.find("_")]))

    net_dict = {}
    prev_id = 1

    for i in range(len(y_var)):
        net_id, group = y_var[i][1:].split("_")
        net_id = int(net_id)
        group = int(group)-1

        for i in range(prev_id, net_id):
            net_dict[i] = 2
        
        net_dict[net_id] = group

        prev_id = net_id + 1

    for i in range(prev_id, Nnets + 1):
        net_dict[i] = 2
    
    return net_dict

