import ast
import javalang.parse
from collections import Counter
import json
from utils import pre_walk_tree, pre_walk_tree_java


class VocabEntry:
    def __init__(self, word2id=None, entry_class="not_label"):
        if word2id:
            self.word2id = word2id
        elif entry_class == "labels" or entry_class == "types":
            self.word2id = {}
        else:
            self.word2id = {'<pad>': 0, '<unk>': 1}

        self.id2word = {v: k for k, v in self.word2id.items()}

    def add(self, word):
        if word not in self.word2id:
            wid = self.word2id[word] = len(self.word2id)
            self.id2word[wid] = word
            return wid
        else:
            return self.word2id[word]


class Vocab:
    def __init__(self, vocab):
        self.vocab = vocab

    @staticmethod
    def build_for_ast(paths_file_path):
        paths_file = open(paths_file_path)

        tokens = {"types": [], "edge_types": [], "labels": []}
        for file_path in paths_file:
            file = open(file_path[3:-1], encoding="utf-8")
            code = file.read()
            try:
                root = ast.parse(code)
                index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = pre_walk_tree(
                    root, 0, 0)
                for (type, feature) in zip(types, features):
                    if type in tokens:
                        tokens[type].append(feature)
                    else:
                        tokens[type] = [feature]
                tokens["types"].extend(types)
                tokens["edge_types"].extend(edge_types)
                tokens["labels"].append(file_path[34:40])

            except SyntaxError:
                pass

            file.close()

        vocab = {}
        for f in tokens:
            tokens_count = dict(Counter(tokens[f]).most_common())
            token_vocab = VocabEntry(entry_class=f)
            for token in tokens_count:
                if tokens_count[token] > 1:
                    token_vocab.add(token)
            vocab[f] = token_vocab

        paths_file.close()
        return Vocab(vocab)

    @staticmethod
    def build_for_ast_java(paths_file_path):
        paths_file = open(paths_file_path)

        tokens = {"types": [], "edge_types": [], "labels": []}
        for file_path in paths_file:
            file = open(file_path[3:-1], encoding="utf-8")
            code = file.read()
            try:
                root = javalang.parse.parse(code)
                index, edge_index, types, features, edge_types, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = pre_walk_tree_java(
                    root, 0, 0)
                for (type, feature) in zip(types, features):
                    if type in tokens:
                        tokens[type].append(feature)
                    else:
                        tokens[type] = [feature]
                tokens["types"].extend(types)
                tokens["edge_types"].extend(edge_types)
                tokens["labels"].append(file_path[32:38])

            except javalang.tokenizer.LexerError:
                pass

            file.close()

        vocab = {}
        for f in tokens:
            tokens_count = dict(Counter(tokens[f]).most_common())
            token_vocab = VocabEntry(entry_class=f)
            for token in tokens_count:
                if tokens_count[token] > 1:
                    token_vocab.add(token)
            vocab[f] = token_vocab

        paths_file.close()
        return Vocab(vocab)

    def save(self, file_path):
        json.dump({t: self.vocab[t].word2id for t in self.vocab}, open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        vocab = {}
        for e in entry:
            vocab[e] = VocabEntry(entry[e])
        return Vocab(vocab)


def main():
    print("start building vocab for python")
    v = Vocab.build_for_ast("data/train_files_paths.txt")
    v.save("data/vocab4ast.json")
    print("start building vocab for java")
    v2 = Vocab.build_for_ast_java("data/train_files_paths_java.txt")
    v2.save("data/vocab4ast_java.json")
    print("finish")

if __name__ == '__main__':
    main()
