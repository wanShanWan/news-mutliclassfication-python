# -*- coding: utf-8 -*-
# author: Wanshan

import os


CUR_DIR = os.path.dirname(__file__)


def read_data(file_path):
    file_path = os.path.join(CUR_DIR, file_path)

    data = []
    label = []
    with open(file_path, 'r') as file:
        read_lines = file.readlines()

        for line in read_lines:
            pre_line = line.strip().split('\t')
            print(pre_line)
            if len(pre_line) == 2:
                label.append(int(pre_line[0]))
                data.append(str(pre_line[1]))

    return data, label


def content_word_segment(content):
    import jieba

    for i in range(0, len(content)):
        segment_str = ''
        segment = jieba.lcut(content[i])
        for j in range(0, len(segment)):
            segment_str += str(segment[j] + ' ')
        content[i] = segment_str
    return content


if __name__ == '__main__':
    data, label = read_data('./data/data_labeled.txt')
    data = content_word_segment(data)
