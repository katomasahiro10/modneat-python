import copy
import math
from modneat.graphs import required_for_output
from modneat.genome import ModGenome
from modneat.nn import Recurrent
from modneat.nn.utils import weight_change

class ModRecurrent(Recurrent):
    def __init__(self, inputs, outputs, node_evals, global_params, config):
        super().__init__(inputs, outputs, node_evals)
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.original_node_evals = copy.deepcopy(self.node_evals)
        self.global_params = global_params

        self.values = [{}, {}]
        for v in self.values:
            for k in list(inputs) + list(outputs):
                v[k] = 0.0

            for node, ignored_modulatory_ratio, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links: #NOTE: linksは対応する各ノードに対する入力リンクのリスト. iは入力ノード, wは重み
                    # links = [
                    #    (input_node_id, weight),
                    #    (input_node_id, weight),
                    #    ...
                    # ]
                    v[i] = 0.0

        self.modulate_values = copy.copy(self.values[0])
        self.modulated_values = copy.copy(self.values[0])

        self.activate = 0


    @staticmethod
    def genome_type():
        return ModGenome

    def reset(self):
        self.node_evals = copy.deepcopy(self.original_node_evals)
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.modulate_values = copy.deepcopy(self.values[0])
        self.modulated_values = copy.deepcopy(self.values[0])
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, modulatory, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            if(not modulatory):
                ovalues[node] = activation(bias + response * s)
                self.modulate_values[node] = 0.0
            elif(modulatory):
                ovalues[node] = 0.0
                self.modulate_values[node] = activation(bias + response * s)

        # Caliculate modulated_values of each node
        for node, modulatory, act_func, agg_func, bias, response, links in self.node_evals:
            self.modulated_values[node] = self.global_params['m_d']
            for i, w in links:
                self.modulated_values[node] += self.modulate_values[i] * w

        for node, modulatory, activation, aggregation, bias, response, links in self.node_evals:
            for i, w in links:
                update_val = math.tanh (self.modulated_values[node] / 2.0) * \
                            self.global_params['eta'] * \
                            (
                                self.global_params['a'] * ivalues[i] * ovalues[node] + \
                                self.global_params['b'] * ivalues[i] + \
                                self.global_params['c'] * ovalues[node] + \
                                self.global_params['d'] \
                            )
                weight_change(self, i, node, update_val)

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, node.modulatory, activation_function, aggregation_function, node.bias, node.response, inputs))

        global_params = genome.global_params[0].__dict__

        return ModExHebbRNN(genome_config.input_keys, genome_config.output_keys, node_evals, global_params)
