import numpy as np
import tensorflow as tf
from models.model import rnn_model
# 一会要写的model 执行model 计算图？在这里写
from dataset.poems import process_poems, generate_batch



# 预处理操作和生成batch的操作

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size = ?')
#
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')
# 学习率 默认从0.01开始  大了下降loss慢
tf.app.flags.DEFINE_string('check_pointss_dir', './model/', 'check_pointss_dir')
# 保存模型位置
tf.app.flags.DEFINE_string('file_path', './data/.txt', 'file_path')
# 数据位置
tf.app.flags.DEFINE_integer('epoch', 50, 'train epoch')
# 4w唐诗 学50遍

start_token = 'G'
end_token = 'E'
# 指定开头和结尾 方便评测 让诗不会无限预测 预测出来等于E就结束

# 核心
def run_training():
    # 预处理 把话转化为向量 文字转化为整数 返回语料库
    poems_vector,word_to_int,vocabularies = process_poems(FLAGS.file_path)
    batch_inputs,batch_outputs = generate_batch(FLAGS.batch_size,poems_vector,word_to_int)
    # inout携程placehoder none 输出的结果
    # 之后要做交叉熵 target
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size,None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size,None])
    # 定义模型
    # lstm效果更好
    # run size rnn有中间隐层 有多少神经元2
    end_points = rnn_model(model='lstm',input=input_data,output_data = output_targets,vocab_size = len(vocabularies)
                           ,run_size = 32,num_layers = 2,batch_size = 10,learning_rate = 0.01)
    
# 如果是train/test
def main(is_train):
    if is_train:
        print ('training')
        run_training()
    else:
        print ('test')
        begin_word = input('word')

    # 以什么开头
        
if __name__ == '__main__': #main函数
    tf.app.run()