import numpy as np

class ConvergeTest:
    """ A class to test whether the algorithm has converged. It stores the last n cost values and check whether the standard deviation is smaller than the threshold
    """
    def __init__(self, threshold=0.01, max_n=100, method="avg"):
        """
        Args:
            threshold (float): The threshold to stop the algorithm
            max_n (int): The maximum number of items to store in the queue
        """
        self.threshold = threshold
        self.max_n = max_n
        self.method = method
        self.queue = []
        self.n = 0

        self.list_sum = 0
        
    def update(self, item):
        """Append an item to the queue"""
        self.queue.append(item)
        self.list_sum += item
        self.n += 1

        if self.n > self.max_n:
            first_item = self.queue.pop(0)
            self.sum = self.list_sum - first_item + item
            self.n -= 1

    def is_converged(self):
        """Check convergence with STD"""
        if self.n < self.max_n:
            return False
        else:
            if self.method == "avg":
                return self.list_sum/self.n < self.threshold
            elif self.method == "cmp":
                return self.queue[0] == self.queue[-1]
            return np.std(self.queue) < self.threshold

def dict_to_lists(dictionary):
    """ Convert a dictionary to one or more lists. Number of lists is equal to the number of unique values in the dictionary
    """
    unique_values = list(set(dictionary.values()))
    unique_values.sort()
    lists = [[] for _ in range(len(unique_values))]
    for key, value in dictionary.items():
        lists[unique_values.index(value)].append(key)
    return lists

def generate_lp(netlist):
    y_var = []
    for i in range(netlist.Nnets):
        for j in [1, 2]:
            y_var.append(f"y{i+1}_{j}")

    x_var = []
    for i in range(netlist.Nmods):
        for j in [1, 2]:
            x_var.append(f"x{i+1}_{j}")

    x_placement_constraints = []
    for i in range(netlist.Nmods):
        x_placement_constraints.append(f"x{i+1}_1 + x{i+1}_2 = 1")

    block2_size = netlist.Nmods//2
    block1_size = netlist.Nmods - block2_size

    block_size_constraints = []
    block_size_constraints.append(" + ".join([f"x{i+1}_1" for i in range(netlist.Nmods)]) + f" <= {block1_size}")
    block_size_constraints.append(" + ".join([f"x{i+1}_2" for i in range(netlist.Nmods)]) + f" <= {block2_size}")

    y_placement_constraints = []
    y_1_placement_constraints = []
    y_2_placement_constraints = []
    for i in range(netlist.Nnets):
        connected_mods = netlist.net_to_mods[i+1]
        for mod in connected_mods:
            y_1_placement_constraints.append(f"y{i+1}_1 - x{mod}_1 <= 0")
            y_2_placement_constraints.append(f"y{i+1}_2 - x{mod}_2 <= 0")
    y_placement_constraints = y_1_placement_constraints + y_2_placement_constraints

    bounds = []
    for i in range(netlist.Nmods):
        bounds.append(f"0 <= x{i+1}_1 <= 1")
        bounds.append(f"0 <= x{i+1}_2 <= 1")

    integers = []
    for i in range(netlist.Nmods):
        integers.append(f"x{i+1}_1")
        integers.append(f"x{i+1}_2")
    integers = " ".join(integers)

    objective = " + ".join(y_var)

    objective = objective
    x_placement_constraints = x_placement_constraints
    block_size_constraints = block_size_constraints
    y_placement_constraints = y_placement_constraints
    bounds = bounds
    integers = integers

    result = dict()
    result['obj'] = objective
    result['x_c'] = x_placement_constraints
    result['b_c'] = block_size_constraints
    result['y_c'] = y_placement_constraints
    result['bounds'] = bounds
    result['int'] = integers

    return result

def write_lp(lp, fp):
    obj = lp['obj']
    x_c = lp['x_c']
    b_c = lp['b_c']
    y_c = lp['y_c']
    bounds = lp['bounds']
    integers = lp['int']

    with open(fp, 'w') as f:
        f.write(f"Maximize\n")
        f.write(f"  {obj}\n")
        f.write(f"Subject To\n")
        for c in x_c:
            f.write(f"  {c}\n")
        for c in b_c:
            f.write(f"  {c}\n")
        for c in y_c:
            f.write(f"  {c}\n")
        f.write(f"Bounds\n")
        for b in bounds:
            f.write(f"  {b}\n")
        f.write(f"Integers\n")
        f.write(f"  {integers}\n")
        f.write(f"End\n")


def get_sanchis_label(circuit):
    import cirpart.globals as globals
    label_str = getattr(globals, f"{circuit}_sanchis")
    label_str = label_str.replace(" ", "").replace("\n", "")
    part_dict = {i+1: int(label_str[i]) for i in range(len(label_str))}
    return part_dict