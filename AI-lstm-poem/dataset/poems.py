
import collections
import os
import sys
import numpy as np

start_token = 'G'
end_token = 'E'


def process_poems(file_name):
    # 诗集 处理好的结果定为list
    poems = []
    # 指定路径 读 打开方式utf8 得到了打开
    # 一行一行读
    # 第一步 对词进行分割 诗名 冒号 内容
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                # 去空格 按冒号分
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                # 有一些字符时 字太多或者太少 不要他们
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序 l是长度
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率 方便过滤偏僻字
    counter = collections.Counter(all_words)
    # item让他可便利 key是出现次数 x-1还是-x1
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字 这里取了全部
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID 计算机不认识汉字 给每一个词一个标示符 复杂：wordvec paragrapvec 在gensim 诗句和诗句p3 12:34
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    # 看数据量有多大 决定了一个epoch里多少batch
    n_chunk = len(poems_vec) // batch_size
    # 填充数据
    x_batches = []
    y_batches = []
    # 对一个epoch 做nchunk个batch
    for i in range(n_chunk):
        # 在原始数据中取索引
        start_index = i * batch_size
        end_index = start_index + batch_size
        # 每个市的长度不一样 对神经网络 用前要固定好结构 保证所有输入格式长度一样 输入不可变 因为是全连接操作？ 看谁最大
        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号 现在是全控
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        # y是标签
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
