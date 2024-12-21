from re import fullmatch
from math import exp
from statistics import mean
from random import sample, randint
import networkx as nx
import matplotlib.pyplot as plt
from typing import List


#########################
# Message-RetCode table #
#########################

status_to_msg = {
        401: '[ The response should not contain spaces ]',
        402: '[ The responce should not be empty and should only contain ASCII a-z characters ]',
        404: '[ The word should not be presented in the riddle ]',
        405: '[ Please solve one more riddle ]',
        406: '[ Pass ]',
        407: '[ Fail ]',
        408: '< "Get" responce status >'
}


###################
# Hyperparameters #
###################

PATTERN = 'single_dense'
assert PATTERN in ('single_dense', 'single_rand'), 'pattern not supported'

WEIGHT_ELASTICITY = 0.1
WEIGHT_LIMIT = 5
WORD_SET_SIZE = 5
THRESH = 0.5
DEFUALT_EDGE_WEIGHT = 0
HEURISTIC_RATE = 8
DENSE_SAMPLING_RATE = 50


###################
# Backend section #
###################

def graph_db_import() -> nx.Graph:
        return nx.read_gml('database/graph.gml')

def graph_db_save(graph: nx.Graph) -> None:
        nx.write_gml(graph, 'database/graph.gml')

def create_default_edges(graph: nx.Graph, word: str) -> None:
        for node in graph.nodes().keys():
                if node != word:
                        graph.add_edge(node, word, weight=DEFUALT_EDGE_WEIGHT)

def graph_db_add(graph: nx.Graph, word: str) -> bool:
        if graph.has_node(word):
                return True
        else:
                graph.add_node(word)
                create_default_edges(graph, word)
                return False

def graph_db_remove(graph: nx.Graph, word: str) -> bool:
        if graph.has_node(word):
                graph.remove_node(word)
                return True
        else:
                return False

def graph_db_get_edge(graph: nx.Graph, word_1: str, word_2: str) -> float:
        return graph[word_1][word_2]['weight']

def graph_db_set_edge(graph: nx.Graph, word_1: str, word_2: str,
                      new_weight: float) -> None:
        graph[word_1][word_2]['weight'] = new_weight

def graph_db_get_all_nbrs(graph: nx.Graph, word: str) -> List:
        center = graph[word]
        item_list = []
        for nbr in center.items():
                item_list.append([nbr[0], nbr[1]['weight']])
        return item_list

def graph_db_get_all_words(graph: nx.Graph) -> List:
        return list(graph.nodes().keys())

def sigm_dist(edge_weight: float) -> float:
        return 1.0 / (1 + exp(-1.0 * edge_weight))

def build_subgraph_branch(graph: nx.Graph, center: str, subgraph: nx.Graph,
                          heur_thresh: float, restrictions: List) -> None:
        for node, weight in graph_db_get_all_nbrs(graph, center):
                dist = sigm_dist(weight)
                if dist < heur_thresh:
                        if subgraph.has_node(node):
                                next_step = True
                        else:
                                subgraph.add_node(node)
                                next_step = False

                        # even if the edge already exists:
                        if ((center not in restrictions) or
                            (node not in restrictions)):
                                subgraph.add_edge(center, node, weight=dist)

                        if next_step:
                                build_subgraph_branch(graph, node, subgraph,
                                                      heur_thresh, restrictions)

def build_subgraph(graph: nx.Graph, word_1: str, word_2: str,
                   restrictions: List) -> nx.Graph:
        subgraph = nx.Graph()
        subgraph.add_node(word_1)
        heur_thresh = (HEURISTIC_RATE *
                       sigm_dist(graph_db_get_edge(graph, word_1, word_2)))
        build_subgraph_branch(graph, word_1, subgraph, heur_thresh,
                              restrictions)
        return subgraph

def res_dist(graph: nx.Graph, word_1: str, word_2: str,
             restrictions: List = []) -> float:
        subgraph = build_subgraph(graph, word_1, word_2, restrictions)
        return nx.resistance_distance(subgraph, word_1, word_2)

def mean_res_dist(graph: nx.Graph, center: str, word_set: List) -> float:
        if PATTERN == 'single_dense':
                return mean([res_dist(graph, center, word, word_set)
                             for word in word_set])
        else: # PATTERN == 'single_rand'
                return mean([res_dist(graph, center, word)
                             for word in word_set])

def enhance_humanity(graph: nx.Graph, center: str, word_set: List) -> None:
        for word in word_set:
                edge_weight = graph_db_get_edge(graph, center, word)
                new_edge_weight = (edge_weight - WEIGHT_ELASTICITY
                                   if edge_weight > -1.0 * WEIGHT_LIMIT
                                   else edge_weight)
                graph_db_set_edge(graph, center, word, new_edge_weight)

def enhance_machinery(graph: nx.Graph, center: str, word_set: List) -> None:
        for word in word_set:
                edge_weight = graph_db_get_edge(graph, center, word)
                new_edge_weight = (edge_weight + WEIGHT_ELASTICITY
                                   if edge_weight < WEIGHT_LIMIT
                                   else edge_weight)
                graph_db_set_edge(graph, center, word, new_edge_weight)

def make_verdict(graph: nx.Graph, responce: str, word_set: List) -> bool:
        return bool(mean_res_dist(graph, responce, word_set) < THRESH)

def make_postponed_enhancements(graph: nx.Graph, human: bool) -> None:
        center = ''
        word_set = []
        after_file = open('artifacts/to_be_decided.dat', 'r')
        for line in after_file:
                word = line.strip()
                if word[0] == '*':
                        center = word[1:]
                elif word[0] == '-':
                        if human:
                                enhance_humanity(graph, center, word_set)
                        else:
                                enhance_machinery(graph, center, word_set)
                        word_set = []
                else:
                        word_set.append(word)
        after_file.close()
        open('artifacts/to_be_decided.dat', 'w').close()

def generate_word_set_dense(graph: nx.Graph) -> List:
        all_words = graph_db_get_all_words(graph)

        sampled_idx = randint(0, len(all_words) - 1)
        word_set = [all_words[sampled_idx]]
        all_words.pop(sampled_idx)

        for i in range(min(WORD_SET_SIZE, graph.number_of_nodes()) - 1):
                word_sample_set = sample(all_words,
                                         min(DENSE_SAMPLING_RATE,
                                             len(all_words)))
                res_dists = [mean_res_dist(graph, word, word_set)
                             for word in word_sample_set]
                best_idx = res_dists.index(max(res_dists))
                word_set.append(word_sample_set[best_idx])
                all_words.remove(word_sample_set[best_idx])
        
        return word_set

def generate_word_set_rand(graph: nx.Graph) -> List:
        all_words = graph_db_get_all_words(graph)
        return sample(all_words, min(WORD_SET_SIZE, graph.number_of_nodes()))

def generate_word_set(graph: nx.Graph) -> List:
        if PATTERN == 'single_dense':
                return generate_word_set_dense(graph)
        else: # PATTERN == 'single_rand'
                return generate_word_set_rand(graph)

def postpone_enhancement(resp: str, word_set: List) -> None:
        after_file = open('artifacts/to_be_decided.dat', 'a')
        after_file.write('*' + resp + '\n')
        for word in word_set:
                after_file.write(word + '\n')
        after_file.write('-\n')
        after_file.close()

def remember_word_set(word_set: List) -> None:
        word_file = open('artifacts/actual_word_set.dat', 'w')
        for word in word_set:
                word_file.write(word + '\n')
        word_file.close()

def retrieve_word_set() -> List:
        word_file = open('artifacts/actual_word_set.dat', 'r')
        word_set = []
        for line in word_file:
                word_set.append(line.strip())
        word_file.close()
        return word_set

def process_get(graph: nx.Graph) -> List:
        word_set = generate_word_set(graph)
        remember_word_set(word_set)
        return word_set

def process_post(graph: nx.Graph, post_text: str, mode: str = None) -> int:
        if post_text.find(' ') != -1:
                return 401

        post_text = post_text.lower()

        if not bool(fullmatch(r'[a-z]+', post_text)):
                return 402
        
        word_set = retrieve_word_set()

        if post_text in word_set:
                return 404

        if graph_db_add(graph, post_text) == False:
                postpone_enhancement(post_text, word_set)
                return 405

        if mode is not None:
                assert mode in ('hum', 'mac'), 'oops, wrong "post" mode used'
                if mode == 'hum':
                        enhance_humanity(graph, post_text, word_set)
                        make_postponed_enhancements(graph, True)
                        return 406
                elif mode == 'mac':
                        enhance_machinery(graph, post_text, word_set)
                        make_postponed_enhancements(graph, False)
                        return 407

        if make_verdict(graph, post_text, word_set) == True:
                enhance_humanity(graph, post_text, word_set)
                make_postponed_enhancements(graph, True)
                return 406
        else:
                enhance_machinery(graph, post_text, word_set)
                make_postponed_enhancements(graph, False)
                return 407

def app(env, start_responce) -> List:
        request_method = env['REQUEST_METHOD']

        responce_body = ''
        status = 408

        if request_method == 'GET':
                graph = graph_db_import()
                responce_body = process_get(graph)
        elif request_method == 'POST':
                graph = graph_db_import()
                content_length = int(env.get('CONTENT_LENGTH', 0))
                post_data = env['wsgi.input'].read(content_length)
                status = process_post(graph, post_data.decode('utf-8'))
                graph_db_save(graph)
        else:
                # please change the exception to whatever you like
                raise NameError('request method unsupported')

        responce_headers = [('Content-Type', 'text/plain')]
        start_responce(status, responce_headers)

        return responce_body


#########################
# CLI interface section #
#########################

await_state = 'await_get'
learn_state = 'test'
busy = False

GRAPH = graph_db_import()

while True:
        assert await_state in ('await_get', 'await_post'), 'await state fault'
        assert learn_state in ('learn', 'test'), 'learn/test state fault'

        inp = input('\n   > ')
        split_inp = inp.strip().split(' ')

        if inp == 'get':
                if await_state != 'await_get':
                        print('\n        answer the question, please.')
                        continue
                print('\n        ' + ', '.join(process_get(GRAPH)))
                await_state = 'await_post'
                busy = True

        elif split_inp[0] == 'post':
                if await_state != 'await_post':
                        print('\n        request a question, please.')
                        continue

                if learn_state == 'learn':
                        if split_inp[1] == 'mac':
                                sts = process_post(GRAPH,
                                                   split_inp[2],
                                                   'mac')
                                assert sts in status_to_msg.keys(), 'wrong status'
                                msg = status_to_msg[sts]
                                print('\n        ' + msg)
                        elif split_inp[1] == 'hum':
                                sts = process_post(GRAPH,
                                                   split_inp[2],
                                                   'hum')
                                assert sts in status_to_msg.keys(), 'wrong status'
                                msg = status_to_msg[sts]
                                print('\n        ' + msg)
                        else:
                                print('\n        ...')
                                continue
                else: # learn_state == 'test'
                        sts = process_post(GRAPH, split_inp[1])
                        assert sts in status_to_msg.keys(), 'wrong status'
                        msg = status_to_msg[sts]
                        print('\n        ' + msg)

                await_state = 'await_get'
                if msg in ('[ Fail ]', '[ Pass ]'):
                        busy = False

        elif inp == 'quit':
                graph_db_save(GRAPH)
                break
        elif inp == 'test':
                print('\n        switched to test mode.')
                learn_state = 'test'
        elif inp == 'learn':
                print('\n        switched to learn mode.')
                learn_state = 'learn'
        elif inp == 'print':
                fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
                dot_graph = nx.draw_networkx(GRAPH, node_size=0.2,
                                             with_labels=True, width=0.05,
                                             font_size=4, ax=ax,
                                             font_family='monospace',
                                             node_color='#ff0000',
                                             alpha=0.5)
                fig.savefig('graph.png', format='png')
                print('\n        .png graph saved in the current dir.')
        elif inp == 'stat':
                print('\n        %d nodes available.' % GRAPH.number_of_nodes())
        elif inp == 'save':
                graph_db_save(GRAPH)
                print('\n        database state commited to graph.gml file.')

        elif split_inp[0] == 'insert':
                if busy:
                        print('\n        database update in progress, denied.')
                        continue

                if len(split_inp) < 2:
                        print('\n        at least one word should be inserted.')
                        continue

                for word in split_inp[1:]:
                        if word.strip() == '':
                                continue
                        if not bool(fullmatch(r'[a-z]+', word)):
                                print('\n        "%s" was skipped ' % word +
                                      'due to an inappropriate format.')
                                continue
                        if graph_db_add(GRAPH, word) == True:
                                print('\n        "%s" is already ' % word +
                                      'present in the database.')
                        else:
                                print('\n        "%s" added to ' % word +
                                      'the database.')

        elif split_inp[0] == 'remove':
                if busy:
                        print('\n        database update in progress, denied.')
                        continue

                if len(split_inp) < 2:
                        print('\n        at least one word should be removed.')
                        continue

                for word in split_inp[1:]:
                        if word.strip() == '':
                                continue
                        if not bool(fullmatch(r'[a-z]+', word)):
                                print('\n        "%s" was skipped ' % word +
                                      'due to an inappropriate format.')
                                continue
                        if graph_db_remove(GRAPH, word) == True:
                                print('\n        "%s" removed ' % word +
                                      'from the database.')
                        else:
                                print('\n        "%s" is not ' % word +
                                      'present in the database.')

        else:
                print('\n        ...')

print('')
