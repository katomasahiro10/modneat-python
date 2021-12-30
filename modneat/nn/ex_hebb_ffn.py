from modneat.graphs import feed_forward_layers
from modneat.genome import IndExHebbGenome


class ExHebbFFN(object):
    def __init__(self, inputs, outputs, node_evals, global_params):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.global_params = global_params
    
    @staticmethod
    def genome_type():
        return IndExHebbGenome

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                update_val = self.global_params['eta'] * \
                            (
                                self.global_params['a'] * self.values[i] * self.values[node] + \
                                self.global_params['b'] * self.values[i] + \
                                self.global_params['c'] * self.values[node] + \
                                self.global_params['d'] \
                            )
                self.weight_change(i, node, update_val)

        return [self.values[i] for i in self.output_nodes]

    def weight_change(self, input_node, output_node, value):
        """ Add value to connection weight between input_node and output_node. """
        node_loop_counter = -1

        for _output_node, act_func, agg_func, bias, response, links in self.node_evals:
            node_loop_counter += 1
            connection_loop_counter = -1
            for _input_node, _weight, a, b, c, d in links:
                connection_loop_counter += 1
                if(input_node == _input_node and output_node == _output_node):
                    listed_node_and_weight = list(self.node_evals[node_loop_counter][5][connection_loop_counter])
                    listed_node_and_weight[1] += value
                    self.node_evals[node_loop_counter][5][connection_loop_counter] = tuple(listed_node_and_weight)
                    #print('ノード{},{}の間の重みを {} だけ変更しました'.format(input_node, output_node, value))
                else:
                    pass

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight, cg.a, cg.b, cg.c, cg.d))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        global_params = genome.global_params[0].__dict__

        return ExHebbFFN(config.genome_config.input_keys, config.genome_config.output_keys, node_evals, global_params)
