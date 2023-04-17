import ast
import javalang
import javalang.parse
import os
from sklearn.model_selection import train_test_split

def splitdata():
    dir_path = "data/Project_CodeNet_Python800"

    files_paths = []
    labels = []
    for root, dir, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                code = open(file_path, encoding='utf-8').read()
                try:
                    ast.parse(code)
                    files_paths.append('../' + file_path)
                    labels.append(root)
                except SyntaxError:
                    pass

    train_files_paths, vt_files_paths, train_labels, vt_labels = train_test_split(files_paths, labels, test_size=0.4,
                                                                                  stratify=labels)
    valid_files_paths, test_files_paths, valid_labels, test_labels = train_test_split(vt_files_paths, vt_labels,
                                                                                      test_size=0.5, stratify=vt_labels)
    train_file = open("data/train_files_paths.txt", "w+")
    valid_file = open("data/valid_files_paths.txt", "w+")
    test_file = open("data/test_files_paths.txt", "w+")
    for train_file_path in train_files_paths:
        train_file.write(train_file_path)
        train_file.write("\n")

    for valid_file_path in valid_files_paths:
        valid_file.write(valid_file_path)
        valid_file.write("\n")

    for test_file_path in test_files_paths:
        test_file.write(test_file_path)
        test_file.write("\n")

    train_file.close()
    valid_file.close()
    test_file.close()
    print("finish")

def splitdata_java():
    dir_path = "data/Project_CodeNet_Java250"

    files_paths = []
    labels = []
    for root, dir, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith('.java'):
                file_path = os.path.join(root, file_name)
                code = open(file_path, encoding='utf-8').read()
                try:
                    javalang.parse.parse(code)
                    files_paths.append('../' + file_path)
                    labels.append(root)
                except javalang.tokenizer.LexerError:
                    pass

    train_files_paths, vt_files_paths, train_labels, vt_labels = train_test_split(files_paths, labels, test_size=0.4,
                                                                                  stratify=labels)
    valid_files_paths, test_files_paths, valid_labels, test_labels = train_test_split(vt_files_paths, vt_labels,
                                                                                      test_size=0.5, stratify=vt_labels)
    train_file = open("data/train_files_paths_java.txt", "w+")
    valid_file = open("data/valid_files_paths_java.txt", "w+")
    test_file = open("data/test_files_paths_java.txt", "w+")
    for train_file_path in train_files_paths:
        train_file.write(train_file_path)
        train_file.write("\n")

    for valid_file_path in valid_files_paths:
        valid_file.write(valid_file_path)
        valid_file.write("\n")

    for test_file_path in test_files_paths:
        test_file.write(test_file_path)
        test_file.write("\n")

    train_file.close()
    valid_file.close()
    test_file.close()
    print("finish")

if __name__ == '__main__':
    splitdata()
    splitdata_java()
