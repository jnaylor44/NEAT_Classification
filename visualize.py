import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

def plot_stats(stats, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt.gcf():
        plt.close()
    
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()
    
    plt.close()

def plot_species(stats, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt.gcf():
        plt.close()
    
    species_sizes = stats.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(10, 6))
    plt.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)
    if view:
        plt.show()
    
    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    try:
        import graphviz
    except ImportError:
        warnings.warn("Graphviz not found. Using matplotlib for network visualization.")
        return _draw_net_matplotlib(config, genome, view, filename, node_names, node_colors)
    
    try:
        dot = graphviz.Digraph(format=fmt)
        dot.attr(rankdir='LR')

        if node_names is None:
            node_names = {}
        
        if node_colors is None:
            node_colors = {}

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            dot.node(str(k), label=name, _attributes={'shape': 'circle',
                                       'fontsize': '9',
                                       'height': '0.2',
                                       'width': '0.2',
                                       'style': 'filled',
                                       'fillcolor': node_colors.get(k, '#A0CBE2')})

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            dot.node(str(k), label=name, _attributes={'shape': 'circle',
                                       'fontsize': '9',
                                       'height': '0.2',
                                       'width': '0.2',
                                       'style': 'filled',
                                       'fillcolor': node_colors.get(k, '#FF5733')})

        for n in genome.nodes.keys():
            if n in inputs or n in outputs:
                continue
            
            attrs = {'shape': 'circle',
                    'fontsize': '9',
                    'height': '0.2',
                    'width': '0.2',
                    'style': 'filled',
                    'fillcolor': node_colors.get(n, '#A0CBE2')}
            dot.node(str(n), _attributes=attrs)

        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input, output = cg.key
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(str(input), str(output), _attributes={'style': style, 'color': color, 'penwidth': width})

        dot.render(filename, view=view)
        return dot
    except Exception as e:
        warnings.warn(f"Graphviz failed: {str(e)}. Using matplotlib for network visualization.")
        return _draw_net_matplotlib(config, genome, view, filename, node_names, node_colors)

def _draw_net_matplotlib(config, genome, view=False, filename=None, node_names=None, node_colors=None):
    """ Fallback network visualization using matplotlib """
    import networkx as nx
    
    if plt.gcf():
        plt.close()

    if node_names is None:
        node_names = {}
    
    if node_colors is None:
        node_colors = {}

    G = nx.DiGraph()
    
    # Add nodes
    for k in config.genome_config.input_keys:
        name = node_names.get(k, str(k))
        G.add_node(k, layer=0, color=node_colors.get(k, '#A0CBE2'), name=name)
        
    for k in config.genome_config.output_keys:
        name = node_names.get(k, str(k))
        G.add_node(k, layer=2, color=node_colors.get(k, '#FF5733'), name=name)
        
    for k in genome.nodes.keys():
        if k not in config.genome_config.input_keys and k not in config.genome_config.output_keys:
            G.add_node(k, layer=1, color=node_colors.get(k, '#A0CBE2'), name=str(k))

    # Add edges
    for cg in genome.connections.values():
        if cg.enabled:
            G.add_edge(cg.key[0], cg.key[1], weight=cg.weight)

    plt.figure(figsize=(12, 8))
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # Draw nodes
    for layer in range(3):
        nodes = [n for n, d in G.nodes(data=True) if d['layer'] == layer]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=[G.nodes[n]['color'] for n in nodes],
                              node_size=500)
    
    # Draw edges
    edges = G.edges()
    colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in edges]
    widths = [abs(G[u][v]['weight']) for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths)
    
    # Draw labels
    labels = {n: G.nodes[n]['name'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.axis('off')
    if filename:
        plt.savefig(filename)
    if view:
        plt.show()
    plt.close()