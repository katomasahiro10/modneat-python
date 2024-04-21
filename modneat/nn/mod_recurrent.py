import copy
import math
from modneat.graphs import required_for_output
from modneat.genome import ModGenome
from modneat.nn import Recurrent
from modneat.nn.utils import weight_change

class ModRecurrent:
    def __init__(self, inputs, outputs, node_evals, global_params, config):
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

    def assert_type(self):
        #a, b, c, d, etaをglobalに設定するか、localに設定するかに関するassrsion
        if self.config.evoparam_mode == 'local':
            assert self.config.genome_config.compatibility_global_param_coefficient == 0.0, "ERROR:evoparam_mode is 'local', but compatibility_global_param_coefficient is not 0.0"
        elif self.config.evoparam_mode == 'global':
            assert self.config.genome_config.compatibility_local_param_coefficient == 0.0, "ERROR:evoparam_mode is 'global', but compatibility_local_param_coefficient is not 0.0"

    def reset(self):
        self.node_evals = copy.deepcopy(self.original_node_evals)
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.modulate_values = copy.copy(self.values[0])
        self.modulated_values = copy.copy(self.values[0])
        self.active = 0

    def activate(self, inputs, is_update = True):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, modulatory_ratio, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)

            assert modulatory_ratio >= 0.0 and modulatory_ratio <= 1.0, "ERROR:modulatory_ratio must be between 0.0 and 1.0"

            if(self.config.modulatory_mode == 'bool'):
                if(modulatory_ratio > 0.5):
                    ovalues[node] = 0.0
                    self.modulate_values[node] = activation(bias + response * s)
                else:
                    ovalues[node] = activation(bias + response * s)
                    self.modulate_values[node] = 0.0
            elif(self.config.modulatory_mode == 'float'):
                ovalues[node] = activation(bias + response * s) * (1.0 - modulatory_ratio)
                self.modulate_values[node] = activation(bias + response * s) * modulatory_ratio
            else:
                raise RuntimeError("modulatory_mode must be 'bool' or 'float'")

        # Caliculate modulated_values of each node
        for node, modulatory_ratio, act_func, agg_func, bias, response, links in self.node_evals:
            self.modulated_values[node] = 0.0
            for i, w, eta, a, b, c, d, m_d in links:
                self.modulated_values[node] += self.modulate_values[i] * w
            if(self.config.evoparam_mode == 'global'):
                self.modulated_values[node] += self.global_params['m_d']
            elif(self.config.evoparam_mode == 'local'):
                self.modulated_values[node] += m_d

        if(is_update):
            for node, modulatory_ratio, activation, aggregation, bias, response, links in self.node_evals:
                for i, w, eta, a, b, c, d, m_d in links:
                    if(self.config.evoparam_mode == 'global'): #グローバル値を利用するモードならば、各パラメータをグローバル値で上書きして計算に用いる。ローカル値利用モードならばこの処理は不要なのでスキップする
                        a, b, c, d, eta = self.global_params['a'], self.global_params['b'], self.global_params['c'], self.global_params['d'], self.global_params['eta']
                    #Soltoggioの設定に基づいて重みを更新
                    update_val = math.tanh (self.modulated_values[node] / 2.0) * \
                                eta * \
                                (
                                    a * ivalues[i] * ovalues[node] + \
                                    b * ivalues[i] + \
                                    c * ovalues[node] + \
                                    d \
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
            node_evals.append((node_key, node.modulatory_ratio, activation_function, aggregation_function, node.bias, node.response, inputs))

        global_params = genome.global_params[0].__dict__

        return ModRecurrent(genome_config.input_keys, genome_config.output_keys, node_evals, global_params, config)
