import ast
import numpy as np
import random
import matplotlib
import javalang.ast
import javalang.util

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def show_score(score_list, yname, train_name, color, save_path):
    score_list = np.array(score_list)
    plt.xlabel('Epoch')
    plt.ylabel(yname)
    plt.plot(score_list, color=color, label=train_name)
    plt.xlim((-0.5, len(score_list) - 0.5))
    plt.xticks(np.arange(0, len(score_list), 1))
    plt.legend(loc=1)
    plt.grid()
    plt.savefig(save_path)
    plt.show()
    plt.clf()


def show_2scores(score_list1, score_list2, yname, train_name1, train_name2, color1, color2, save_path):
    score_list1 = np.array(score_list1)
    score_list2 = np.array(score_list2)
    plt.xlabel('Epoch')
    plt.ylabel(yname)
    plt.plot(score_list1, color=color1, label=train_name1)
    plt.plot(score_list2, color=color2, label=train_name2)
    plt.xlim((-0.5, len(score_list2) - 0.5))
    plt.xticks(np.arange(0, len(score_list2), 1))
    plt.legend(loc=1)
    plt.grid()
    plt.savefig(save_path)
    plt.show()
    plt.clf()

def pre_walk_tree(node, index, edge_index):
    types = []
    features = []
    edge_types = []
    edge_in_out_indexs_s, edge_in_out_indexs_t = [], []
    edge_in_out_head_tail = []

    child_index = index + 1
    types.append("ast")

    features.append(str(type(node)))

    for field_name, field in ast.iter_fields(node):
        if isinstance(field, ast.AST):
            edge_types.append(field_name)
            edge_in_out_indexs_s.extend([edge_index, edge_index])
            edge_in_out_indexs_t.extend([index, child_index])
            edge_in_out_head_tail.extend([0, 1])
            child_edge_index = edge_index + 1
            child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree(
                field, child_index, child_edge_index)
            types.extend(child_types)
            features.extend(child_features)
            edge_types.extend(child_edge_types)
            edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
            edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
            edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, list) and field and isinstance(field[0], ast.AST):
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            child_edge_index = edge_index + 1
            for item in field:
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree(
                    item, child_index, child_edge_index)
                types.extend(child_types)
                features.extend(child_features)
                edge_types.extend(child_edge_types)
                edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
                edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
                edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, list) and field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            for item in field:
                types.append("ident")
                features.append(str(item))
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index += 1
            edge_index += 1
        elif field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            types.append("ident")
            features.append(str(field))
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(child_index)
            edge_in_out_head_tail.append(1)
            child_index += 1
            edge_index += 1

    return child_index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail

def pre_walk_tree_java(node, index, edge_index):
    types = []
    features = []
    edge_types = []
    edge_in_out_indexs_s, edge_in_out_indexs_t = [], []
    edge_in_out_head_tail = []

    child_index = index + 1
    types.append("ast")

    features.append(str(type(node)))

    for field_name in node.attrs:
        field = getattr(node, field_name)
        if isinstance(field, javalang.ast.Node):
            edge_types.append(field_name)
            edge_in_out_indexs_s.extend([edge_index, edge_index])
            edge_in_out_indexs_t.extend([index, child_index])
            edge_in_out_head_tail.extend([0, 1])
            child_edge_index = edge_index + 1
            child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree_java(
                field, child_index, child_edge_index)
            types.extend(child_types)
            features.extend(child_features)
            edge_types.extend(child_edge_types)
            edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
            edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
            edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, (list, tuple)) and field and isinstance(field[0], javalang.ast.Node):
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            child_edge_index = edge_index + 1
            for item in field:
                if isinstance(item, javalang.ast.Node):
                    edge_in_out_indexs_s.append(edge_index)
                    edge_in_out_indexs_t.append(child_index)
                    edge_in_out_head_tail.append(1)
                    child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree_java(
                        item, child_index, child_edge_index)
                    types.extend(child_types)
                    features.extend(child_features)
                    edge_types.extend(child_edge_types)
                    edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
                    edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
                    edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
                elif isinstance(item, (list, tuple)):
                    for it in item:
                        edge_in_out_indexs_s.append(edge_index)
                        edge_in_out_indexs_t.append(child_index)
                        edge_in_out_head_tail.append(1)
                        child_index, child_edge_index, child_types, child_features, child_edge_types, child_edge_in_out_indexs_s, child_edge_in_out_indexs_t, child_edge_in_out_head_tail = pre_walk_tree_java(
                            it, child_index, child_edge_index)
                        types.extend(child_types)
                        features.extend(child_features)
                        edge_types.extend(child_edge_types)
                        edge_in_out_indexs_s.extend(child_edge_in_out_indexs_s)
                        edge_in_out_indexs_t.extend(child_edge_in_out_indexs_t)
                        edge_in_out_head_tail.extend(child_edge_in_out_head_tail)
            edge_index = child_edge_index
        elif isinstance(field, (list, tuple)) and field and field[0] and not isinstance(field[0], list):
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            for item in field:
                types.append("ident")
                features.append(str(item))
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index += 1
            edge_index += 1
        elif isinstance(field, set) and field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            for item in field:
                types.append("ident")
                features.append(str(item))
                edge_in_out_indexs_s.append(edge_index)
                edge_in_out_indexs_t.append(child_index)
                edge_in_out_head_tail.append(1)
                child_index += 1
            edge_index += 1
        elif field:
            edge_types.append(field_name)
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(index)
            edge_in_out_head_tail.append(0)
            types.append("ident")
            features.append(str(field))
            edge_in_out_indexs_s.append(edge_index)
            edge_in_out_indexs_t.append(child_index)
            edge_in_out_head_tail.append(1)
            child_index += 1
            edge_index += 1

    return child_index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail