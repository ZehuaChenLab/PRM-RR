import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from utility.progress import WorkSplitter
from utility.model_helper import generate_nce_matrix, get_optimizer, calculate_mse, generate_batches_ids, convert_to_rating_matrix_from_lists
from scipy.sparse import csr_matrix
import time
from utility.predictor import predict
from utility.metrics import evaluate
import copy
import math

np.random.seed(47)
random.seed(47)
torch.random.manual_seed(47)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, rnn_type, concat_layer):
        super(BidirectionalRNN, self).__init__()#子类继承父类nn.Module且用nn.Module的初始化方法来初始化继承的属性
        self.num_layers = num_layers  #RNN层的个数
        self.dropout_rate = dropout_rate
        self.concat_layers = concat_layer    # 是否连接所有层的输出结果
        self.rnn_list = nn.ModuleList()     #存储所有的RNN层的列表
        for i in range(num_layers): # 循环num_layers次，添加num_layers个RNN层
            input_size = input_size if i == 0 else hidden_size * 2      # 如果是第一层，则输入大小为input_size，否则输入大小为hidden_size*2【前一层的输出数据size乘以2赋值给input_size】【双向RNN的每一层都由两个方向的RNN组成】
            self.rnn_list.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))   #添加一个RNN层，输入大小为input_size，输出大小为hidden_size，是一个双向RNN
            #这里的rnn_type参数并不是具体的RNN类，而是一个函数，表示要使用哪种RNN模型，比如nn.LSTM或nn.GRU等，这些都是继承自nn.RNN的具体模型类。

    def forward(self, x, x_mask):
        lengths = x_mask.data.eq(0).sum(1)  #eq函数是留下x_mask等于0的坐标，将x_mask中为0的位置标记出来，求和，得到变长序列的实际长度
        _, index_sort = torch.sort(lengths, dim=0, descending=True)  # 从大到小排序后的数据、排序后的索引
        _, index_unsort = torch.sort(index_sort, dim=0, descending=False) # 恢复一个原始序列的索引
        lengths = list(lengths[index_sort])     # 将长度按照排序后的顺序重新排列——从大到小，转换成普通列表
        x = x.index_select(dim=0, index=index_sort) #将输入数据x按照排序后的index_sort的顺序重新排列，将输入数据转置，变成TxBxC的形式
        x = x.transpose(0, 1) #转置
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)  #数据x的tensor和长度lengths的tensor打包成pack_padded_sequence格式合并到一个结构体rnn_input中，用作RNN输入
        outputs = [rnn_input]     #初始化一个列表outputs，并将第一层的输出结果rnn_input添加到列表中。
        for i in range(self.num_layers):
            rnn_input = outputs[-1]      # 取出上一层的输出结果
            if self.dropout_rate > 0.0:     # 如果dropout_rate大于0，则进行dropout操作
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)       # 将dropout后的数据再次打包成PackedSequence格式，重新得到一个结构体rnn_input，用于传给下一层的模型。
            outputs.append(self.rnn_list[i](rnn_input)[0])      # 计算当前层的输出结果，并将结果存入outputs列表中
            #其中[0]表示只取第一个元素，因为RNN模型的返回值是一个包含输出结果和隐状态的元组，而这里只需要输出结果。
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]     # 对outputs中的数据进行解包（pad_packed_sequence）操作，解压并填充成源序列的形状，然后取出输出结果赋值给outputs[i]
        if self.concat_layers:
            output = torch.cat(outputs[1:], dim=2)      # 如果concat_layers为True，将每一层的输出结果进行拼接，dim=2表示按照最后一个维度进行拼接。
        else:
            output = outputs[-1]        # 否则仅返回最后一层的输出结果
        output = output.transpose(0, 1)     # 将输出结果进行转置，变成BxTx(C或C*2)的形式，恢复原有的输入顺序
        output = output.index_select(dim=0, index=index_unsort)     # 将输出结果按照原始序列的顺序恢复，以恢复原有的输入顺序
        if output.size(1) != x_mask.size(1):    # 输出结果的长度不等于x_mask的长度
            padding = torch.zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(
                output.data.type())
            output = torch.cat([output, padding], 1)          #将填充tensor与输出结果进行拼接，dim=1表示在中间插入0
        # indices = x_mask.shape[1] - x_mask.sum(dim=1) - 1
        # output = output[torch.arange(output.shape[0]), indices, :]
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0):
        super(SelfAttention, self).__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)       #初始化一个线性层self.linear1，其中输入形状为input_size，输出形状为hidden_size。
        self.linear2 = nn.Linear(hidden_size, 1)        #初始化另外一个线性层self.linear2，其中输入形状为hidden_size，输出形状为1。

    def forward(self, input_encoded, input_mask):

        scores = F.dropout(F.tanh(self.linear1(input_encoded)), p=self.dropout_rate, training=self.training)
        # 首先将输入数据input_encoded传给self.linear1，得到线性变换后的输出结果，然后使用激活函数tanh进行非线性变换，并加上dropout，最终得到得分scores。
        scores = self.linear2(scores).squeeze(2)
        # 将得分scores传给self.linear2进行线性变换，并通过squeeze函数移除最后一个维度，得到一个形状为[batch_size, max_seq_len]的 得分矩阵 ，其中每个元素都表示对应位置的上下文质量。
        scores.data.masked_fill_(input_mask.data, -float('inf'))
        # 在得分矩阵的相应位置，使用掩码input_mask.data将无用的padding位置（值为0）替换成负无穷数。

        alpha = F.softmax(scores, dim=-1) #对得分矩阵进行softmax操作，将得分转化为标准化的概率分布，保存在alpha中。
        # 利用注意力权重alpha，对输入数据做一个加权平均，得到上下文向量。
        context = torch.bmm(alpha.unsqueeze(dim=1), input_encoded).squeeze(dim=1)
        # 形状为[batch_size, 1, max_seq_len]的tensor和input_encoded进行 矩阵乘法 得到的形状为[batch_size, max_seq_len, hidden_size]的tensor
        # 最后将第二个tensor沿着第1个维度与第1个tensor相乘，得到上下文向量context。

        return context   # 返回经过处理后的上下文向量context。

#class SelfAttentionWordFusion(nn.Module):
#    def __init__(self, input_size, hidden_size, num_topics, dropout_rate=0):
#        super(SelfAttentionWordFusion, self).__init__()
#
#       self.dropout_rate = dropout_rate
#      self.linear1 = nn.Linear(input_size + num_topics, hidden_size)     # Include num_topics in the input size
#     self.linear2 = nn.Linear(hidden_size, 1)        #初始化另外一个线性层self.linear2，其中输入形状为hidden_size，输出形状为1。
#        self.num_topics = num_topics
#
#    def forward(self, input_encoded, input_mask, x):
#        x = x.unsqueeze(2).repeat(1, 1, input_encoded.size(2))  # Repeat x to match the sequence length of input_encoded
#        input_combined = torch.cat((input_encoded, x), dim=2)   #  Concatenate input_encoded and x along the last dimension

#        scores = F.dropout(F.tanh(self.linear1(input_combined)), p=self.dropout_rate, training=self.training)
#        scores = self.linear2(scores).squeeze(2)
#        scores.data.masked_fill_(input_mask.data, -float('inf'))

#        alpha = F.softmax(scores, dim=-1)
#        context = torch.bmm(alpha.unsqueeze(dim=1), input_encoded).squeeze(dim=1)

#        return context


class SelfAttentionFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_topics, dropout_rate=0):
        #
        super(SelfAttentionFusion, self).__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        # self.linear3 = nn.Linear(705, 100)
        self.linear_x = nn.Linear(num_topics, hidden_size)

    def forward(self, input_encoded, input_mask, x):  # 已编码的输入序列（input_encoded）、输入序列的掩码（input_mask）和主题特征向量（x)

        input_x = self.linear_x(x).unsqueeze(1)    # 用linear_x映射主题特征x，然后在第1维上插入一个新维度以方便后续计算。
        # print(input_x.shape)
        # print(input_encoded.shape)
        # input_encoded = self.linear3(input_encoded)
        input_x = input_x.expand_as(input_encoded) # 将该新维度扩展到和input_encoded相同的形状

        input_concat = torch.cat((input_encoded, input_x), dim=2)   # 将input_encoded和input_x在第3维度上拼接起来，得到拼接后的输入张量input_concat

        scores = F.dropout(F.tanh(self.linear1(input_concat)), p=self.dropout_rate, training=self.training)
        scores = self.linear2(scores).squeeze(2)
        scores.data.masked_fill_(input_mask.data, -float('inf'))

        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(dim=1), input_encoded).squeeze(dim=1)

        return context


class Encoder(nn.Module):
    def __init__(self, vocabulary_size, glove_embedding_size, encoder_hidden_size,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics):
        # 词汇表大小（vocabulary_size）、GloVe嵌入的维度（glove_embedding_size）、编码器隐藏状态的维度（encoder_hidden_size）、
        # 编码器的层数（encoder_num_layers）、可选的dropout率（encoder_dropout_rate）、RNN类型（encoder_rnn_type）、连接层的数量（encoder_concat_layers）
        # 以及主题数量（num_topics）
        super(Encoder, self).__init__()

        hidden_size = encoder_hidden_size * 2  # 编码器的输出维度，是编码器隐藏单元数量的两倍【因为是双向RNN】

        # 初始化一个词嵌入层
        self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=glove_embedding_size)
        # vocabulary_size指定了词汇表中不同单词的数量，embedding_dim指定了每个单词的嵌入维度
        self.word_encoder = BidirectionalRNN(  # 接受单词嵌入作为输入，并输出一个双向RNN编码的张量
            input_size=glove_embedding_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers
        )
        self.word_attention = SelfAttention(input_size=hidden_size, hidden_size=hidden_size) # 在编码的单词序列上计算一个注意力向量，以便为后续的句子建模提取重要的单词部分
        # self.word_attention = SelfAttentionFusion(hidden_size, hidden_size, num_topics)

        self.sent_encoder = BidirectionalRNN(  # 输入特征的维度（hidden_size），然后类似于word_encoder的工作原理：它将对以前计算的word_encoded序列进行编码，以获取整个句子语境的向量表示。
            input_size=hidden_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers
        )
        self.sent_attention = SelfAttentionFusion(hidden_size, hidden_size, num_topics) # 在编码的句子序列上计算一个注意力向量

    def forward(self, word_indices, word_mask, sent_lengths, x):

        word_embedded = self.word_embeddings(word_indices)   # 将输入的word_indices通过嵌入层进行嵌入
        word_encoded = self.word_encoder.forward(x=word_embedded, x_mask=word_mask)  # 双向RNN 编码
        word_encoded = self.word_attention(input_encoded=word_encoded, input_mask=word_mask.bool())
        #word_encoded = self.word_attention2(input_encoded=word_encoded, input_mask=word_mask.bool(), x=x)  #词注意力，得到加权的张量


        sent_input, sent_mask = _align_sent(word_encoded, sent_lengths)  # 使用_align_sent函数来规范化编码后的单词序列，以及计算出新的掩码

        sent_encoded = self.sent_encoder.forward(x=sent_input, x_mask=sent_mask)  # 输入文本序列的句子数（100，13，128），
        sent_encoded = self.sent_attention(input_encoded=sent_encoded, input_mask=sent_mask.bool(), x=x)

        return sent_encoded

    def initialize(self, word_embeddings):
        self.word_embeddings.weight.data.copy_(word_embeddings)


def _align_sent(batch_input, sent_lenghts, sent_max=None):

    hidden_dim = batch_input.size(-1)  # 获取输入编码输出序列的最后一个维度大小，即 hidden_dim
    passage_num = len(sent_lenghts)    # 输入文本序列中的句子数

    if sent_max is not None:
        max_len = sent_max
    else:
        max_len = np.max(sent_lenghts)  #句子长度最大值:13

    sent_input = torch.zeros(passage_num, max_len, hidden_dim).cuda()  # 全0的张量 sent_input(100,13,128)，用于保存规范化后的编码序列
    sent_mask = torch.ones(passage_num, max_len).cuda()   # 全1的张量 sent_mask，用于保存规范化后的mask

    init_index = 0

    for index, length in enumerate(sent_lenghts):  # 遍历句子长度列表
        end_index = init_index + length  # 对于每个句子，计算该句子在 batch_input 中的起始和结束下标

        temp_input = batch_input[init_index:end_index, :]  #  获取该句子在 batch_input 中对应的编码输出序列

        if temp_input.size(0) > max_len:
            temp_input = temp_input[:max_len]  #如果 temp_input 的长度大于 max_len，则将其截断为 max_len 长度

        sent_input[index, :length, :] = temp_input
        sent_mask[index, :length] = 0     # 将 temp_input 拷贝到 sent_input 张量中对应位置上，并在 sent_mask 上相应位置进行标记

        init_index = end_index  # 更新下标，以便处理下一个句子

    return sent_input, sent_mask


class EncoderHAttn(nn.Module):
    def __init__(self, vocabulary_size, glove_embedding_size, encoder_hidden_size,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics):
        super(EncoderHAttn, self).__init__()

        self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=glove_embedding_size)
        self.context_encoder = BidirectionalRNN(
            input_size=glove_embedding_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers)
        self.fc1 = nn.Linear(2 * encoder_hidden_size, encoder_hidden_size * 2)  #全连接层

    def forward(self, word_indices, word_mask, x):    # x 额外输入的张量
        word_embedded = self.word_embeddings(word_indices)
        context_encoded = self.context_encoder.forward(x=word_embedded, x_mask=word_mask)
        context_encoded = self.fc1(context_encoded)
        return context_encoded

    def initialize(self, word_embeddings):   # 初始化的词嵌入权重
        self.word_embeddings.weight.data.copy_(word_embeddings)


class ModalityFusion(nn.Module):
    def __init__(self, md1_dim, md2_dim, attention_size=128):  # 输入向量维度大小、注意力机制向量维度大小
        super(ModalityFusion, self).__init__()
        self.md1_in = nn.Linear(md1_dim, attention_size)
        self.md2_in = nn.Linear(md2_dim, attention_size)

        self.linear_k = nn.Linear(attention_size, 1)  # 用于计算注意力分数的线性变换层，将注意力机制的向量维度大小 attention_size 转换为 1
        self.linear_q = nn.Linear(attention_size, attention_size)  # 用于加权向量的线性变换层，将注意力机制的向量维度大小 attention_size 转换为其他需要的维度大小

    def forward(self, md1_vec, md2_vec):
        batch_size = md1_vec.size(0)  # 获得三个维度大小
        md1_dim = md1_vec.size(1)
        md2_dim = md2_vec.size(1)

        md1_out = self.md1_in(md1_vec)  # 输入向量进行线性变换
        md2_out = self.md2_in(md2_vec)

        md1_out_attn = self.linear_k(torch.tanh(md1_out))   # 再进行tanh非线性变换，再线性变换——>得到两个注意力分数
        md2_out_attn = self.linear_k(torch.tanh(md2_out))

        attn_scores = torch.cat((md1_out_attn, md2_out_attn), dim=1)
        attn_scores = F.softmax(attn_scores, -1).unsqueeze(1)  # 拼接通过softmax归一化，得到各向量的注意力权重

        # md1_vec = self.linear_q(torch.tanh(md1_out))
        # md2_vec = self.linear_q(torch.tanh(md2_out))

        md1_vec = torch.tanh(self.linear_q(md1_out))  # 线性过的向量再线性+tanh
        md2_vec = torch.tanh(self.linear_q(md2_out))

        md_vecs = torch.stack((md1_vec, md2_vec), dim=1)  # 拼接

        out_vec = torch.bmm(attn_scores, md_vecs).squeeze()  # 加权后的向量

        return out_vec


class NCEAutoRecNLP(nn.Module):
    def __init__(self, num_users, num_items, num_topics, user, vocabulary_size, glove_embedding_size, encoder_hidden_size, attention_size, dropout_p,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, word_embeddings,
                 activation='relu', loss='mse', nce_head_positive_only=1, predict_head_positive_only=1):
        # 根据参数user的值，自动编码器将学习用户或物品的隐向量表示。如果user为True，则表示自动编码器是学习用户的特征；否则，自动编码器是学习物品的特征。

        # 初始化模型参数
        super(NCEAutoRecNLP, self).__init__()
        self.nce_head_positive_only = nce_head_positive_only
        self.predict_head_positive_only = predict_head_positive_only
        self.dropout_p = dropout_p
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        # 选择损失函数
        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if user:  #  如果是user模型，对物品进行编码
            self.encode = nn.Linear(num_items, num_topics) # 编码输入数据，将用户或物品表示为低维特征向量
            # self.nce_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_items)
            # self.predict_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_items)

            self.nce_decode = nn.Linear(attention_size, num_items) #将编码后的特征向量映射成NCE模型的负采样预测值，attention_size为模态融合后的特征向量的维度
            self.predict_decode = nn.Linear(attention_size, num_items)  # 预测值

            self.text_encoder = Encoder(vocabulary_size, glove_embedding_size, encoder_hidden_size,
                                    encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers,
                                    num_topics) #对评论进行编码

            self.modality_fusion = ModalityFusion(num_topics, encoder_hidden_size * 2,
                                                  attention_size=attention_size)

        else:
            self.encode = nn.Linear(num_users, num_topics)
            # self.nce_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_users)
            # self.predict_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_users)

            self.nce_decode = nn.Linear(attention_size, num_users)
            self.predict_decode = nn.Linear(attention_size, num_users)

            self.text_encoder = Encoder(vocabulary_size, glove_embedding_size, encoder_hidden_size,
                                    encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers,
                                    num_topics)

            self.modality_fusion = ModalityFusion(num_topics, encoder_hidden_size * 2,
                                                  attention_size=attention_size)

        self.text_encoder.initialize(word_embeddings)

    def get_feature(self, rating_matrix):  # 得到评分矩阵编码后的特征
        return self.encode(rating_matrix)

    def forward(self, rating_matrix):  #评分矩阵——>预测值
        x, y = matrix_factorization(rating_matrix, K=50, lr=0.001, reg=0.01, epochs=10000)
        x = self.encode(x)
        # x = self.encode(rating_matrix)
        x = self.activation(x)
        out = self.predict_decode(x)
        return out

    def forward_nce(self, rating_matrix):  # 返回负采样预测值
        x, y = matrix_factorization(rating_matrix, K=50, lr=0.001, reg=0.01, epochs=10000)
        x = self.encode(x)
        # x = self.encode(rating_matrix)
        x = self.activation(x)
        out = self.nce_decode(x)
        return out

    def forward_two_heads(self, rating_matrix):  # 返回NCE预测值和预测值
        x, y = matrix_factorization(rating_matrix, K=50, lr=0.001, reg=0.01, epochs=10000)
        x = self.encode(x)
        # x = self.encode(rating_matrix)
        x = self.activation(x)
        nce_out = self.nce_decode(x)
        predict_out = self.predict_decode(x)
        return nce_out, predict_out

    def forward_two_heads_language(self, rating_matrix_batch, language_inputs_batch): #带语言模态的双头模型前向传播函数

        x = F.dropout(rating_matrix_batch, p=self.dropout_p, training=self.training)
        x = self.encode(x)   # 降维处理得到特征向量x
        x = self.activation(x)

        [_, word_indices, word_mask, _, _, length_lists] = language_inputs_batch # 提取索引、掩码、长度等信息

        textual_features = self.text_encoder.forward(word_indices=word_indices, word_mask=word_mask, sent_lengths=length_lists, x=x)

        # predicted_features = torch.split(textual_features, length_lists)
        # mean_predicted_features = torch.stack([torch.mean(vec, dim=0) for vec in predicted_features], dim=0).cuda()

        modality_fusion = self.modality_fusion.forward(x, textual_features)

        # all_prediction = torch.cat((x, mean_predicted_features), 1)
        all_prediction = modality_fusion

        predict_out = self.predict_decode(all_prediction)
        nce_out = self.nce_decode(all_prediction)

        return nce_out, predict_out

    def forward_language(self, rating_matrix_batch, language_inputs_batch):  # 返回预测值
        x = self.encode(rating_matrix_batch)
        x = self.activation(x)

        [_, word_indices, word_mask, _, _, length_lists] = language_inputs_batch

        textual_features = self.text_encoder.forward(word_indices=word_indices, word_mask=word_mask, sent_lengths=length_lists, x=x)

        # predicted_features = torch.split(textual_features, length_lists)
        # mean_predicted_features = torch.stack([torch.mean(vec, dim=0) for vec in predicted_features], dim=0).cuda()

        modality_fusion = self.modality_fusion.forward(x, textual_features)

        # all_prediction = torch.cat((x, mean_predicted_features), 1)
        all_prediction = modality_fusion

        out = self.predict_decode(all_prediction)

        return out


class RNN_NCEAutoRec(object):
    def __init__(self, vocabulary_size, glove_embedding_size, elmo_embedding_size, encoder_hidden_size, attention_size, dropout_p,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics, max_len,
                 mc_times, separate, decoder_hidden_size, decoder_dropout_rate, word_dict, decoder_loss, num_users,
                 num_items, activation, autoencoder_loss,
                 nce_head_positive_only, predict_head_positive_only, word_embeddings):
        # num_topics：主题模型中主题的数量；max_len：输入序列的最大长度；mc_times：蒙特卡罗采样次数
        # separate：是否将用户和物品视为不同的训练样本；word_embeddings：预训练的词向量
        # nce_head_positive_only：是否只在 NCE 头部使用正采样；predict_head_positive_only：是否只在预测头部使用正采样

        self.user_autoencoder = NCEAutoRecNLP(num_users, num_items, num_topics, True, vocabulary_size, glove_embedding_size,
                                              encoder_hidden_size, attention_size, dropout_p, encoder_num_layers, encoder_dropout_rate, encoder_rnn_type,
                                              encoder_concat_layers, word_embeddings, activation, autoencoder_loss,
                                           nce_head_positive_only, predict_head_positive_only).cuda()

        self.item_autoencoder = NCEAutoRecNLP(num_users, num_items, num_topics, False, vocabulary_size, glove_embedding_size,
                                              encoder_hidden_size, attention_size, dropout_p, encoder_num_layers, encoder_dropout_rate, encoder_rnn_type,
                                              encoder_concat_layers, word_embeddings, activation, autoencoder_loss,
                                           nce_head_positive_only, predict_head_positive_only).cuda()

        self.max_len = max_len
        self.word_dict = word_dict

    @staticmethod
    def update_nce_autorec(model, max_autoencoder_iteration, lam, matrix, nce_matrix, mode, predict_optimizer,
                           step, documents, num_entries, autoencoder_batch_size, word_dict, max_len):
    # 训练模式下使用数据更新模型的方法
    # lam：正则化项中的系数；nce_matrix：目标矩阵；predict_optimizer：用于优化模型预测的优化器；num_entries：输入矩阵中条目的数量

        t1 = time.time()
        model.train()
        current_step = 0
        if max_autoencoder_iteration == 0:
            t2 = time.time()
            print(
                'finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                  mode))
            return step

        if mode == 'joint-1':
            for i in range(max_autoencoder_iteration):

                indices = list(range(num_entries))  # 由矩阵下标构成的索引列表
                random.shuffle(indices)   # 打乱该索引列表中的元素顺序

                for start_index in range(0, len(indices), autoencoder_batch_size):
                    end_index = min(start_index + autoencoder_batch_size, len(indices))

                    ids_batch = indices[start_index:end_index]

                    user_matrix_batch = matrix[ids_batch]  # 输入矩阵相应索引的行
                    user_nce_batch = nce_matrix[ids_batch]  # 目标矩阵中相应索引的行     用于计算损失

                    raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists = \
                        generate_batches_ids(documents, word_dict, max_len, ids_batch)
                    # 使用相应的 documents 生成用于自动编码器的原始文本、单词下标、单词掩码、索引指针、用户下标和长度列表

                    nce_out, predict_out = model.forward_two_heads_language(user_matrix_batch, [raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists])

                    reg_loss = (model.encode.weight ** 2).mean() * lam + (model.nce_decode.weight ** 2).mean() * lam + (
                            model.predict_decode.weight ** 2).mean() * lam
                    nce_loss = calculate_mse(nce_out, user_nce_batch, model.nce_head_positive_only)
                    predict_loss = calculate_mse(predict_out, user_matrix_batch, model.predict_head_positive_only)

                    loss = nce_loss + predict_loss + reg_loss  # 正则化损失、NCE损失、预测损失相加

                    predict_optimizer.zero_grad()
                    loss.backward()
                    predict_optimizer.step()
                    current_step += 1

        t2 = time.time()
        print('finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                mode))
        return current_step + step

    def update_nce_autorec_joint(self, user_autoencoder, item_autoencoder, max_autoencoder_iteration, lam, user_matrix,
                                 user_nce_matrix, item_matrix, item_nce_matrix, mode, predict_optimizer, step,
                                 user_documents, item_documents, num_users, num_items, autoencoder_batch_size,
                                 word_dict, max_len):
    # user_nce_matrix：用户目标矩阵

        t1 = time.time()
        user_autoencoder.train()
        item_autoencoder.train()

        current_step = 0

        if mode == 'joint-1':
            for i in range(max_autoencoder_iteration):

                user_indices = np.random.randint(0, num_users, autoencoder_batch_size).tolist()
                # 生成一个由用户输入矩阵的下标构成的索引列表
                # 从该列表中随机选取 autoencoder_batch_size 个下标，用于训练当前批次的自编码器。并将下标转换成列表类型方便后续处理

                user_matrix_batch = user_matrix[user_indices].cuda()
                user_nce_batch = user_nce_matrix[user_indices].cuda()  # 用于计算损失

                user_raw_texts, user_word_indices, user_word_mask, user_index_pointers, user_user_indices, user_length_lists\
                    = generate_batches_ids(user_documents, word_dict, max_len, user_indices)

                user_nce_out, user_predict_out = user_autoencoder.forward_two_heads_language(user_matrix_batch,
                                                                                             [user_raw_texts,
                                                                                              user_word_indices,
                                                                                              user_word_mask,
                                                                                              user_index_pointers,
                                                                                              user_user_indices,
                                                                                              user_length_lists])

                user_reg_loss = (user_autoencoder.encode.weight ** 2).mean() * lam + \
                                (user_autoencoder.nce_decode.weight ** 2).mean() * lam + \
                                (user_autoencoder.predict_decode.weight ** 2).mean() * lam

                user_nce_loss = calculate_mse(user_nce_out, user_nce_batch, user_autoencoder.nce_head_positive_only)

                user_predict_loss = calculate_mse(user_predict_out, user_matrix_batch,
                                                  user_autoencoder.predict_head_positive_only)

                user_loss = user_nce_loss + user_predict_loss + user_reg_loss

                loss = user_loss

                predict_optimizer.zero_grad()
                loss.backward()
                predict_optimizer.step()
                current_step += 1

        t2 = time.time()
        print('finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                mode))
        return current_step + step

    # 在测试集上进行预测，基本上和update_nce_autorec方法相同，只是在模型执行前调用model.eval()并关闭梯度更新。
    def inference_nce_autorec(self, model, matrix, nce_matrix, documents, num_entries, autoencoder_batch_size, word_dict, max_len):

        indices = list(range(num_entries))
        predict_out_list = []

        with torch.no_grad():  # 禁止求导
            model.eval()
            for start_index in range(0, len(indices), autoencoder_batch_size): #遍历所有下标批次，每次取出 autoencoder_batch_size 个下标
                end_index = min(start_index + autoencoder_batch_size, len(indices))

                ids_batch = indices[start_index:end_index]

                user_matrix_batch = matrix[ids_batch].cuda()  #用于计算预测输出
                # user_nce_batch = nce_matrix[ids_batch]

                raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists = \
                    generate_batches_ids(documents, word_dict, max_len, ids_batch)

                predict_out = model.forward_language(user_matrix_batch, [raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists]).cpu()
                ### 使用 cpu() 将张量转移到 CPU 上并将其添加到 predict_out_list 中
                predict_out_list.extend(predict_out)

        model.train()
        return torch.stack(predict_out_list)   # 使用 torch.stack() 将所有预测输出合并成一个张量返回

    @staticmethod
    def evaluate_model(prediction, matrix_train_csr, train_data, matrix_val_csr, val_data):
    # prediction：推断模型输出的预测值矩阵；matrix_train_csr：训练数据对应的稀疏矩阵；train_data：训练数据（用户 ID 列表、物品 ID 列表、评分列表）
    # matrix_val_csr：验证数据对应的稀疏矩阵； val_data：验证数据（用户 ID 列表、物品 ID 列表、评分列表）

        prediction_topK = predict(prediction, None, None, 50, matrix_train_csr) # 生成预测结果，只考虑前 50 个预测结果
        result = evaluate(prediction_topK, matrix_val_csr, ['R-Precision', 'NDCG', 'Precision', 'Recall'], [50])
        # training_users, training_items, training_ratings = train_data
        # validation_users, validation_items, validation_ratings = val_data
        # training_mse = evaluate_mse(prediction, training_users, training_items, training_ratings)
        # validation_mse = evaluate_mse(prediction, validation_users, validation_items, validation_ratings)
        # print(training_mse, validation_mse)
        # result['train_mse'] = [training_mse.item(), 0]
        # result['val_mse'] = [validation_mse.item(), 0]
        return result

    def train_model(self, train, val, user_documents, item_documents, iteration, lam, root, threshold, optimizer, momentum,
                    weight_decay, rec_learning_rate, autoencoder_batch_size, autoencoder_epoch,
                    mode, criteria, word_dict, max_len):
        # root：用于生成负采样嵌入的根节点；threshold：用于生成负采样嵌入的阈值；momentum：动量
        # weight_decay：权重衰减；rec_learning_rate：推荐器学习率；criteria：评价指标

        autoencoder_decoder_step = 0  # 记录解码步数

        (train_users, train_items, train_ratings) = train  # 将数据集转换为评分矩阵
        (val_users, val_items, val_ratings) = val
        num_users = len(user_documents)
        num_items = len(item_documents)
        matrix_train = convert_to_rating_matrix_from_lists(num_users, num_items, train_users, train_items,
                                                                 train_ratings, True)

        matrix_val = convert_to_rating_matrix_from_lists(num_users, num_items, val_users, val_items, val_ratings, True)
        matrix_train_csr = csr_matrix(matrix_train)   #  生成对应的稀疏矩阵
        matrix_val_csr = csr_matrix(matrix_val)

        user_ratings = torch.from_numpy(matrix_train).float()    # 将评分矩阵转成 PyTorch 张量，并生成对应的 NCE（负采样嵌入）矩阵
        user_nce_matrix = torch.from_numpy(generate_nce_matrix(matrix_train, root, threshold)).float()

        item_ratings = torch.from_numpy(matrix_train.T).float()
        item_nce_matrix = torch.from_numpy(generate_nce_matrix(matrix_train.T, root, threshold)).float()

        # 根据目标评价指标设置一个初始的最佳结果字典
        if 'mse' in criteria: # 将初始最佳结果字典的 mse 值设置为正无穷大，否则为[-1, 0]
            best_dict = {criteria: [float('inf'), 0]}
        else:
            best_dict = {criteria: [-1, 0]}

        num_users, num_items = matrix_train.shape

        predict_optimizer = get_optimizer(optimizer, [self.user_autoencoder, self.item_autoencoder], rec_learning_rate,
                                          momentum,
                                          weight_decay)

        for i in range(iteration):

            # 每次迭代更新模型
            autoencoder_decoder_step = self.update_nce_autorec_joint\
                    (self.user_autoencoder, self.item_autoencoder, autoencoder_epoch, lam, user_ratings, user_nce_matrix,
                     item_ratings, item_nce_matrix, mode, predict_optimizer, autoencoder_decoder_step, user_documents, item_documents,
                     num_users, num_items, autoencoder_batch_size, word_dict, max_len)

            # 计算推荐结果
            user_inference = self.inference_nce_autorec(self.user_autoencoder, user_ratings, user_nce_matrix, user_documents,
                                                        num_users, autoencoder_batch_size, word_dict, max_len)

            prediction = user_inference
            prediction_numpy = prediction.detach().cpu().numpy()
            # 模型评估：调用 evaluate_model 方法计算推荐结果的评价指标，并获取当前最佳的评估结果
            result = self.evaluate_model(prediction_numpy, matrix_train_csr, train, matrix_val_csr, val)

            # 当mse为评价指标时，mse更小时跟新最佳评估结果，并记录当前迭代次数和推荐结果
            if 'mse' in criteria and result[criteria][0] < best_dict[criteria][0]:
                best_dict = copy.deepcopy(result)
                best_dict['best_iteration'] = i + 1
                best_dict['best_prediction' ] = copy.deepcopy(prediction_numpy)
            elif 'NDCG' in criteria and result[criteria][0] > best_dict[criteria][0]:
                best_dict = copy.deepcopy(result)
                best_dict['best_iteration'] = i + 1
                best_dict['best_prediction'] = copy.deepcopy(prediction_numpy)
            # 打印当前的训练状态信息：
            print('current iteration is {0}'.format(i + 1))
            print('current {0} is {1}'.format(criteria, result[criteria]))
            print('best iteration so far is {0}'.format(best_dict['best_iteration']))
            print('best {0} is {1}'.format(criteria, best_dict[criteria]))
        return best_dict


def tafa1(train, val, document_data, iteration=15, lam=100, rank=100,
         optimizer='Adam', threshold=-1, root=1.1, mode='joint-1',
         rec_learning_rate=1e-4, activation_function='relu',
         loss_function='mse', nce_loss_positive_only=0, predict_loss_positive_only=0, momentum=0, weight_decay=0,
         glove_embedding_size=300, elmo_embedding_size=None, encoder_hidden_size=64, attention_size=256, dropout_p=0.5, encoder_num_layers=1,
         encoder_dropout_rate=0, encoder_rnn_type=nn.GRU, encoder_concat_layers=False, max_len=302,
         mc_times=5, separate=1, decoder_hidden_size=64, decoder_dropout_rate=0,
         decoder_loss=nn.CrossEntropyLoss(reduction='none'), rec_batch_size=100,
         rec_epoch=1, criteria='NDCG', **unused):
    # rank：自编码器隐层的节点数，默认为 100；nce_loss_positive_only：NCELoss 是否只对正样本进行计算，默认为 0（即计算所有样本）；
    # predict_loss_positive_only：PredictLoss 是否只对正样本进行计算，默认为 0（即计算所有样本）；
    # encoder_num_layers：编码器的 LSTM 层数，默认为 1；
    # encoder_concat_layers：是否将编码器各个层的输出拼接在一起作为自编码器的输入，默认为 False；
    # decoder_loss：解码器的损失函数，默认为交叉熵损失函数；rec_batch_size：推荐模型的批量大小，默认为 100；rec_epoch：推荐模型的迭代次数，默认为 -1；

    progress = WorkSplitter()  #创建了一个名为 progress 的进度条对象，用于显示训练进度
    (user_documents, item_documents, word_dict, word_embeddings) = document_data
    num_users = len(user_documents)
    num_items = len(item_documents)
    if rec_epoch == -1:
        rec_epoch = math.ceil(num_users / rec_batch_size)

    model = RNN_NCEAutoRec(vocabulary_size=len(word_dict), glove_embedding_size=glove_embedding_size, elmo_embedding_size=elmo_embedding_size,
                           encoder_hidden_size=encoder_hidden_size, attention_size=attention_size, dropout_p=dropout_p, encoder_num_layers=encoder_num_layers,
                           encoder_dropout_rate=encoder_dropout_rate, encoder_rnn_type=encoder_rnn_type,
                           encoder_concat_layers=encoder_concat_layers, num_topics=rank, max_len=max_len,
                           mc_times=mc_times, separate=separate, decoder_hidden_size=decoder_hidden_size,
                           decoder_dropout_rate=decoder_dropout_rate, word_dict=word_dict, decoder_loss=decoder_loss,
                           num_users=num_users, num_items=num_items, activation=activation_function,
                           autoencoder_loss=loss_function, nce_head_positive_only=nce_loss_positive_only,
                           predict_head_positive_only=predict_loss_positive_only, word_embeddings=word_embeddings)
    result = model.train_model(train, val, user_documents, item_documents, iteration, lam, root, threshold,
                               optimizer, momentum, weight_decay, rec_learning_rate,
                               rec_batch_size, rec_epoch, mode, criteria,
                               word_dict, max_len)
    return result
