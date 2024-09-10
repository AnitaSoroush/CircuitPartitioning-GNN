Circuit clustering and circuit partitioning are essential tasks in the Very Large Scale Integration (VLSI) design flow and are usually solved using algorithmic techniques. Graph Neural Networks (GNNs), a branch of artificial neural networks, leverage the predictive power of deep learning on graph-structured data. Given the circuits’ graphical nature, GNNs are increasingly applied to various VLSI design tasks. This research investigates applying GNNs to solve circuit clustering and partitioning. Moreover, a hierarchical framework is suggested, linking the clustering GNN to the partitioning GNN. The proposed method improved the quality of the partitioning solution, in terms of cut size, by, on average, 5.6%.

This is my MSc thesis project, published in the [University of Guelph library](https://atrium.lib.uoguelph.ca/items/627e8d6b-9297-4a3d-a3a9-72e309856c80).

[CircuitPartitioning_GNN](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/blob/main/CircuitPartitioning_GNN) consists of 1 Python file and 3 folders:
* [KPartitions.py](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/blob/main/CircuitPartitioning_GNN/KPartitions.py) is the main partitioning code to run (you can adjust what combination of initial features you want to use)
* [Dataset](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/tree/main/CircuitPartitioning_GNN/Dataset/NET_dr) folder contains a vesy simple sample of the dataset acceptable to [cirpart](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/tree/main/CircuitPartitioning_GNN/cirpart).
* [cirpart](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/tree/main/CircuitPartitioning_GNN/cirpart) folder contains some necessary tools and Python packages developed for this project, mainly to address reading the netlist circuit datasets and turning them to manipulable graphs.
* and finally the results are pickled in [results](https://github.com/AnitaSoroush/CircuitPartitioning-GNN/tree/main/CircuitPartitioning_GNN/results) folder.

