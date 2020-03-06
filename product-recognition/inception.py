import tensorflow as tf
import numpy as np
#import re
import os

model_dir = './product-recognition/inception'
image = './product-recognition/pic/靴子/238320.png'

#将类别ID转换为人类易读的标签
class NodeLookup(object):
    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            # 加载“label_lookup_path”文件
            # 此文件将数据集中所含类别（1-1000）与一个叫做target_class_string的地址对应起来
            # 其地址编码为“n********”星号代表数字
            label_lookup_path = os.path.join(
                    model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            # 加载“uid_lookup_path”文件
            # 此文件将数据集中所含类别具体名称与编码方式为“n********”的地址/UID一一对应起来
            uid_lookup_path = os.path.join(
                    model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
            # 预先检测地址是否存在
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            # 预先检测地址是否存在
            tf.logging.fatal('File does not exist %s', label_lookup_path)


        # Loads mapping from string UID to human-readable string
        # 加载编号字符串n********，即UID与分类名称之间的映射关系（字典）：uid_to_human
        
        # 读取uid_lookup_path中所有的lines
        # readlines(): Returns all lines from the file in a list.
        # Leaves the '\n' at the end.
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        
        # 创建空字典uid_to_human用以存储映射关系
        uid_to_human = {}
# =============================================================================
#         # 使用正则化方法处理文件：
#         p = re.compile(r'[n\d]*[ \S,]*')
#         for line in proto_as_ascii_lines:         
#              = p.findall(line)
#             uid = parsed_items[0]
#             human_string = parsed_items[2]
#             uid_to_human[uid] = human_string
# =============================================================================
        # 使用简单方法处理文件：
        # 一行行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符
            line = line.strip('\n')
            # 按照‘\t’分割，即tab，将line分为两个部分
            parse_items = line.split('\t')
            # 获取分类编码，即UID
            uid = parse_items[0]
            # 获取分类名称
            human_string = parse_items[1]
            # 新建编号字符串n********，即UID与分类名称之间的映射关系（字典）：uid_to_human
            uid_to_human[uid] = human_string
            

        # Loads mapping from string UID to integer node ID.
        # 加载编号字符串n********，即UID与分类代号，即node ID之间的映射关系（字典）
        
        # 加载分类字符串n********，即UID对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        # 创建空字典node_id_to_uid用以存储分类代码node ID与UID之间的关系
        node_id_to_uid = {}
        for line in proto_as_ascii:
            # 注意空格
            if line.startswith('  target_class:'):
                # 获取分类编号
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                # 获取UID（带双引号，eg："n01484850"）
                target_class_string = line.split(': ')[1]
                # 去掉前后的双引号，构建映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]
    
        # Loads the final mapping of integer node ID to human-readable string
        # 加载node ID与分类名称之间的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 假如uid不存在于uid_to_human中，则报错
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            # 获取分类名称
            name = uid_to_human[val]
            # 构建分类编号1-1000对应分类名称的映射关系：key为node_id；val为name
            node_id_to_name[key] = name
    
        return node_id_to_name

    # 传入分类编号1-1000，返回分类具体名称
    def id_to_string(self, node_id):
        # 若不存在，则返回空字符串
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

#读取图片
image_data = tf.gfile.FastGFile(image, 'rb').read()

#创建graph
create_graph()

# 创建会话，因为是从已有的Inception_v3模型中恢复，所以无需初始化
with tf.Session() as sess:
    # Inception_v3模型的最后一层softmax的输出
    # 形如'conv1'是节点名称，而'conv1:0'是张量名称，表示节点的第一个输出张量
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 输入图像（jpg格式）数据，得到softmax概率值（一个shape=(1,1008)的向量）
    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
    # 将结果转为1维数据
    predictions = np.squeeze(predictions)
    # 新建类：ID --> English string label.
    node_lookup = NodeLookup()
    # 排序，取出前5个概率最大的值（top-5)
    # argsort()返回的是数组值从小到大排列所对应的索引值
    top_5 = predictions.argsort()[-5:][::-1]
    for node_id in top_5:
        # 获取分类名称
        human_string = node_lookup.id_to_string(node_id)
        # 获取该分类的置信度
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))