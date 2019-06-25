# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 13:50
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : AFM.py
# @Software: PyCharm

from collections import OrderedDict, namedtuple
from itertools import chain, combinations
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

# for get reproducible results
seed = 2019
init_std = 1e-4
l2_reg_embedding = 1e-5
np.random.seed(seed)
tf.set_random_seed(seed)


class SparseFeat(namedtuple('SparseFeat', ["name", "dim", "hash_flag", "dtype"])):
    __slot__ = ()
    # __new__方法主要是当你继承一些不可变的class时(比如int, str, tuple)， 提供给你一个自定义这些类的实例化过程的途径。还有就是实现自定义的metaclass
    def __new__(cls, name, dim, hash_flag=False, dtype="float32"):
        return super(SparseFeat, cls).__new__(cls, name, dim, hash_flag, dtype)


class DenseFeat(namedtuple('DenseFeat', ["name", "dtype"])):
    __slot__ = ()
    def __new__(cls, name, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dtype)


class VarLenFeat(namedtuple('VarLenFeat', ["name", "vocab_len", "max_len", "hash_flag", "dtype"])):
    __slot = ()
    def __new__(cls, name, vocab_len, max_len, pool_flag="mean", hash_flag=False, dtype="float32"):
        return super(VarLenFeat, cls).__new__(cls, name, vocab_len, max_len, pool_flag, hash_flag, dtype)


# 定义hash层
class Hash(keras.layers.Layer):
    '''
    hash the input to [0,num_buckets)
    if mask = True, the output is 0, other will be [1, num_buckets)
    tensorflow对类别特征，会先转换成字符串，然后做hash
    不返回shape，hash之后没有shape， 对单个输入进行自定义层操作

    # 单个特征的情况
    inp = keras.layers.Input((1, ), dtype="string")
    out = Hash(100, True)(inp)
    m = keras.models.Model(inputs=inp, outputs=out)
    m.predict(np.array(["a"]))
    return int64
    array([[40]])

    # 多个特征的情况
    inp = keras.layers.Input((4, ), dtype="string")
    out = Hash(100, True)(inp)
    m = keras.models.Model(inputs=inp, outputs=out)
    t = np.array(["aaa", "bbb", "ccc", "ddd"]).reshape(1,4)
    x = m.predict(t)
    array([[42, 56,  6, 49]])
    '''
    def __init__(self, num_buckets, mask_zero=True, **kwargs):
        super(Hash, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x):
        if x.dtype != tf.string:
            x = tf.as_string(x)
        # 返回int64
        hash_x = tf.string_to_hash_bucket_fast(input=x, num_buckets=self.num_buckets if self.mask_zero else self.num_buckets - 1)

        if self.mask_zero:
            # 返回比较值，如果mask_zero为True，则0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
            mask_1 = tf.cast(tf.not_equal(x, "0"), "int64")
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), "int64")
            mask = mask_1 * mask_2
            hash_x = (1 + hash_x) * mask
        return hash_x


# 对变长特征使用pooling操作
class Sequence_Pooling_layer(keras.layers.Layer):
    '''
    Input shape: A list of two tensor [seq_value,seq_len]
    - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)```
    - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

    Output shape:
    - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

    Argument:
    - if supports_masking=True, the input nedd to support masking
    '''
    def __init__(self, pool_flag="mean", supports_masking=True, **kwargs):

        super(Sequence_Pooling_layer, self).__init__(**kwargs)
        self.pool_flag = pool_flag
        self.supports_masking = supports_masking
        if pool_flag not in ["sum", "mean", "max"]: raise ValueError("pool_flag should be sum、max、mean")

    def build(self, input_shape):
        if self.supports_masking:
            # input_shape: [batch_size, max_len, embed_size] => TensorShape([Dimension(batch_size), Dimension(max_len), Dimension(embed_size)])
            self.seq_len_max = input_shape[1]

        self.built = True

    def call(self, x, mask=None, **kwargs):
        if self.supports_masking:
            '''
            seq_emb_list = tf.random_normal([2, 6, 64])
            seq_len = tf.constant([[3], [4]])
            (<tf.Tensor 'random_normal_6:0' shape=(2, 6, 64) dtype=float32>,
            <tf.Tensor 'Const_7:0' shape=(2, 1) dtype=int32>)
            
            seq_len_max = seq_emb_list.shape[1].value
            seq_len_max  # 6
            
            mask = tf.sequence_mask(seq_len, seq_len_max, dtype=tf.float32)
            mask
            
            array([[[1., 1., 1., 0., 0., 0.]],
                    [[1., 1., 1., 1., 0., 0.]]], dtype=float32)
                    
            mask = tf.transpose(mask, (0, 2, 1))
            array([[[1.],
                     [1.],
                     [1.],
                     [0.],
                     [0.],
                     [0.]],
                
                    [[1.],
                     [1.],
                     [1.],
                     [1.],
                     [0.],
                     [0.]]], dtype=float32)
                     
            mask = tf.tile(mask, (1, 1, 64))
            mask    # <tf.Tensor 'Tile_3:0' shape=(2, 6, 64) dtype=float32>
            
            seq_emb_list *= mask    # <tf.Tensor 'mul_2:0' shape=(2, 6, 64) dtype=float32>
            '''
            # 参与mask计算的输入： 一个是本来的长度，另外一个是指定的最大长度
            seq_emb_list, seq_len = x
            mask = tf.sequence_mask(seq_len, self.seq_len_max, dtype=tf.float32)
            # 需要进行转置，参与正常计算
            mask = tf.transpose(mask, (0, 2, 1))
            embedding_size = seq_emb_list.shape[-1].value

            # 按照指定规则进行复制扩展
            mask = tf.tile(mask, (1, 1, embedding_size))

            seq_emb_list *= mask
            if self.pool_flag == "max":
                return tf.reduce_max(seq_emb_list, axis=1, keepdims=True)

            elif self.pool_flag == "mean":
                # 由于进行了mask，
                sum_all = tf.reduce_mean(seq_emb_list, axis=1, keepdims=False)  # 由于变长， 因此不能应用keepdims=True
                res = tf.div(sum_all, seq_len)
                return tf.expand_dims(res, axis=1)

            else:
                return tf.reduce_sum(seq_emb_list, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, input_shape[-1]

    def compute_mask(self, inputs, mask=None):
        return None

class FM(keras.layers.Layer):
    '''
    计算变量之间的二阶项
    input:
    - 3D tensor: [batch_size, feature_number, embedding_size]
    output:
    - 2D tensor: [batch_size, 1]
    '''
    def __init__(self):
        super(FM, self).__init__()

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("input dim should be 3 dims, but in fact {%s} dims" % (K.ndim(inputs)))

        square_of_sum = K.square(K.sum(inputs, axis=1))  # (batch_size, embedding_size)
        sum_of_square = K.sum(K.square(inputs), axis=1) # # (batch_size, embedding_size)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * K.sum(cross_term, axis=-1, keepdims=True) # # (batch_size, 1)
        return cross_term

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class AttentionLayer(keras.layers.Layer):
    '''
        基于attention来计算
        input:
        - a list of 3D tensor:  feature_size个[batch_size, 1, embedding_size]
        output:
        - a 2D tensor:  [batch_size, 1]
    '''
    def __init__(self, attention_units, seed=2019, l2_reg=1e-3, dropout_rate=0.1):
        super(AttentionLayer, self).__init__()
        self.attention_units = attention_units
        self.seed = seed
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError("input_shape should be list, and at least have 2 inputs")

        shape_set = set()
        for shape in input_shape:
            assert isinstance(input_shape, tf.TensorShape)
            shape_set.add(tuple(shape.as_list()))

        if len(shape_set) > 1:
            raise ValueError("inputs with same shapes")

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError("inputs of a list with same shape tensor like (None, 1, embedding_size)")

        embedding_size = input_shape[0][-1].value

        self.attention_W = self.add_weight(shape=(embedding_size, self.attention_units),
                                           initializer=keras.initializers.glorot_normal(seed=self.seed),
                                           regularizer=keras.regularizers.l2(self.l2_reg),
                                           name="attention_W")

        self.attention_b = self.add_weight(shape=(self.attention_units,),
                                           initializer=keras.initializers.zeros(),
                                           name="attention_b")

        self.projection_h = self.add_weight(shape=(self.attention_units, 1),
                                            initializer=keras.initializers.glorot_normal(seed=self.seed),
                                            name="projection_h")

        # dense layer
        self.projection_p = self.add_weight(shape=(embedding_size, 1),
                                            initializer=keras.initializers.glorot_normal(seed=self.seed),
                                            name="projection_p")

        self.dropout = keras.layers.Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, x):

        row = []
        col = []

        # 对特征进行两两组合
        for r, c in combinations(x, 2): # [field * (field - 1)] / 2
            row.append(r)
            col.append(c)

        p = K.concatenate(row, axis=1)  # [batch_size, [field * (field - 1)] / 2, embedding_size]
        q = K.concatenate(col, axis=1)

        inner_product = p * q   # 对应元素相乘
        # 添加非线性, 进行激活
        attention_tmp = K.relu(K.bias_add(K.dot(inner_product, self.attention_W), self.attention_b))
        # [batch_size, [field * (field - 1)] / 2, embedding_size] * [embedding_size, attention_units]  = > [batch_size, [field * (field - 1)] / 2, attention_units]

        # context 向量
        attention_tmp_dot = K.dot(attention_tmp, self.projection_h)  # [batch_size, [field * (field - 1)] / 2, 1]

        # 计算的是一个样本的sofmax， sum的是一个样本的所有特征
        attention_weight = K.softmax(attention_tmp_dot, axis=1)  # 等价于  K.exp(attention_tmp_dot) / K.sum(attention_tmp_dot, axis=1, keepdims=True)
        # [batch_size, [field * (field - 1)] / 2, 1]

        # 权重乘以内积
        attention_output = K.sum(inner_product * attention_weight, axis=1) # [batch_size, embedding_size]

        # 经过dropout操作
        attention_output = K.dropout(attention_output, self.dropout_rate) # [batch_size, embedding_size]

        # 等价于dense层
        afm_out = K.dot(attention_output, self.projection_p)    # [batch_size, 1]

        return afm_out


    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class AFM():
    '''构建AFM模型'''
    def __init__(self, feature_dim_dict, embedding_size=8, mask_zero=True, l2_rate=0.001, use_attention=True, attention_factor=8):
        '''
        定义输入， 定义了两类输入特征(sparse和dense)
        feature_dim_dict: {'sparse':{'field_1':4,'field_2':3,'field_3':2}, 'dense':['field_4','field_5']}
        {'sparse':['field_1','field_2','field_3'], 'dense':['field_4','field_5']}
        处理三种类型，一种是经过labelencoder之后的序列标签，一种是dense特征，还有一种padding之后的序列
        '''
        self.feature_dim_dict = feature_dim_dict
        self.embedding_size = embedding_size
        self.mask_zero = mask_zero
        self.l2_rate = l2_rate
        self.use_attention = use_attention
        self.attention_factor = attention_factor

        self.embedding_vec_list, self.dense_input, self.inputs_list = self.create_input()

    def create_input(self):
        embedding_vec_list = []   # 所有经过embedding的特征列表
        sparse_input = OrderedDict()

        for feat in self.feature_dim_dict["sparse"]:
            # 判断特征是不是SingleFeat()实例，同时代码提示功能
            assert isinstance(feat, SparseFeat)
            sparse_input[feat.name] = keras.layers.Input(shape=(1,), name="sparse_" + feat.name, dtype=feat.dtype)

        # 对sparse feature进行embedding操作
        sparse_embedding = {feat.name: keras.layers.Embedding(input_dim=feat.dim,
                                                              output_dim=self.embedding_size,
                                                              embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                              embeddings_regularizer=keras.regularizers.l2(l2_reg_embedding),
                                                              name="sparse_emb_" + feat.name) for feat in self.feature_dim_dict["sparse"]}
        # 哪些特征向量需要经过hash处理(有部分sparse feature指定了hash_flag=True, 因此为了减少特征数，需要进行特征hash)
        for feat in self.feature_dim_dict["sparse"]:
            if feat.hash_flag:
                lookup_idx = Hash(feat.dim, mask_zero=False)(sparse_input[feat.name])
            else:
                lookup_idx = sparse_input[feat.name]

            embedding_vec_list.append(sparse_embedding[feat.name](lookup_idx))


        dense_input = OrderedDict()

        for feat in self.feature_dim_dict["dense"]:
            assert isinstance(feat, DenseFeat)
            dense_input[feat.name] = keras.layers.Input(shape=(1,), name="dense_" + feat.name, dtype=feat.dtype)

        # 疑问？进行embedding
        continuous_embedding_list = list(map(keras.layers.Dense(self.embedding_size,
                                                           use_bias=False,
                                                           kernel_initializer=keras.regularizers.l2(self.l2_rate)
                                                           ), list(dense_input.values())))
        # 保持输出的维度
        continuous_embedding_list = list(map(keras.layers.Reshape((1, self.embedding_size)), continuous_embedding_list))
        embedding_vec_list += continuous_embedding_list


        seq_input = OrderedDict()

        seq_dict = self.feature_dim_dict.get("sequence", [])
        for feat in seq_dict:
            assert isinstance(feat, VarLenFeat)
            # padding之后的序列长度
            seq_input[feat.name] = keras.layers.Input(shape=(feat.max_len, ), name="seq_" + feat.name, dtype=feat.dtype)

        if self.mask_zero:
            seq_input_len_dict, seq_maxlen_dict = None, None
        else:
            seq_input_len_dict = {feat.name: keras.layers.Input(shape=(1,), name="seq_mask_flag_" + feat.name) for feat in seq_input}
            seq_maxlen_dict = {feat.name: feat.max_len for feat in seq_input}

        # 对sequence序列进行embedding操作
        for feat in seq_dict:
            # feat.vocab_len为hash指定的数目
            sparse_embedding[feat.name] = keras.layers.Embedding(input_dim=feat.vocab_len,
                                                                 output_dim=self.embedding_size,
                                                                 embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                                 embeddings_regularizer=keras.regularizers.l2(l2_reg_embedding),
                                                                 mask_zero=True,
                                                                 name="sparse_seq_emb_" + feat.name)
        # 合并经过embedding的sequence序列
        seq_emb_dict = {}
        if len(seq_input) > 0:
            # 如果特征太多，也需要进行特征hash
            for feat in seq_input:
                if feat.hash_flag:
                    # 此时返回的shape为[batch_size, max_len]
                    lookup_idx = Hash(feat.vocab_len, mask_zero=True)(seq_input[feat.name])
                else:
                    lookup_idx = seq_input[feat.name]
                # 此时返回的shape为[batch_size, max_len, embedding_size]
                seq_emb_dict[feat.name] = sparse_embedding[feat.name](lookup_idx)

        for feat in seq_dict:
            sequence_pool = Sequence_Pooling_layer(feat.pool_flag, supports_masking=True)([seq_emb_dict[feat.name], seq_input_len_dict[feat.name]])
            embedding_vec_list.append(sequence_pool)

        inputs_list = chain(*(map(list(lambda x: x.values(), [sparse_input, dense_input, seq_input]))))

        # 最终返回的是变量列表和对应的embedding序列
        return embedding_vec_list, dense_input, inputs_list

    def build_model(self):
        # 拼接所有的embedding向量
        fm_input = keras.layers.Concatenate(axis=1)(self.embedding_vec_list)
        if self.use_attention:
            fm_logit = AttentionLayer(self.attention_factor)(self.embedding_vec_list)
        else:
            fm_logit = FM()(fm_input)

        # 为了保证数据位于(0, 1)区间，且单个输出，还是需要进行sigmoid操作
        output = keras.layers.Dense(1, activation="sigmoid")
        model = keras.models.Model(inputs=self.inputs_list, outputs=output)

        return model