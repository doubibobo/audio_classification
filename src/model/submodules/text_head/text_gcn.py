import dgl
import numpy
import torch
import torch.nn as nn


def load_vocab_edge_message(vocab, edge_weights, edge_matrix, vocab_embedding):
    vocab = numpy.load(vocab, allow_pickle=True).item()
    edge_matrix = numpy.load(edge_matrix)
    edge_weights = numpy.load(edge_weights)
    vocab_bert_embedding = numpy.load(vocab_embedding, allow_pickle=True).item()
    return len(edge_weights), edge_weights, edge_matrix, vocab, vocab_bert_embedding


class TextGCN(nn.Module):
    def __init__(self,
                 node_hidden_size,
                 vocab_path,
                 edge_weights_path,
                 edge_matrix_path,
                 vocab_embedding_path,
                 edge_trainable=False,
                 graph_embedding_drop_rate=0.1):
        """
        This is the note
        :param node_hidden_size:
        :param vocab_path:
        # :param max_length: this param will be determined in the dataset loader and config file, not need here
        :param edge_matrix_path:
        :param vocab_embedding_path: dict, key is the word, and value is the embedding_vector
        :param edge_trainable:
        :param graph_embedding_drop_rate:
        """
        super(TextGCN, self).__init__()
        self.node_hidden_size = node_hidden_size

        self.edge_number, self.edge_weights, self.edge_matrix, self.vocab, self.vocab_embedding = load_vocab_edge_message(
            vocab=vocab_path,
            edge_weights=edge_weights_path,
            edge_matrix=edge_matrix_path,
            vocab_embedding=vocab_embedding_path
        )
        #
        # self.edge_number = edge_number
        # self.edge_weights = edge_weights
        # self.edge_matrix = edge_matrix
        #
        # self.vocab = vocab
        # self.vocab_embedding = vocab_embedding

        # self.max_length_per_sample = max_length

        self.node_hidden = torch.nn.Embedding(len(self.vocab), node_hidden_size)
        self.node_hidden.weight.data.copy_(torch.tensor(self.load_vocab_embedding()))
        self.node_hidden.weight.requires_grad = True

        if edge_trainable:
            self.edge_hidden = torch.nn.Embedding.from_pretrained(torch.ones((self.edge_number, 1)), freeze=False)
        else:
            self.edge_hidden = torch.nn.Embedding.from_pretrained(torch.tensor(self.edge_weights, dtype=torch.float32), freeze=False)

        self.node_eta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.node_eta.data.fill_(0)
        # self.node_eta.requires_grad = True
        
        self.embedding_layer1 = nn.Linear(768, 768)
        # self.embedding_layer2 = nn.Linear(768, 768)
        # self.norm_layer = nn.LayerNorm(768)
        self.dropout_layer = nn.Dropout(p=graph_embedding_drop_rate)
        self.activation_layer = nn.ReLU()
        pass

    def load_vocab_embedding(self):
        # 这里要保证vocab embedding和word to id之间的对应关系
        vocab_embedding_matrix = numpy.zeros((len(self.vocab), self.node_hidden_size))
        for word, word_id in self.vocab.items():
            vocab_embedding_matrix[word_id] = self.vocab_embedding[word]
        print('THE SHAPE OF EMBEDDING MATRIX IS {}.\n'.format(vocab_embedding_matrix.shape))
        return vocab_embedding_matrix


    def create_graph_for_a_sample(self, token_id):
        """
        为样本构建图
        :param token_id: shape is <T>. Type is numpy.
        :return:
        """
        # if len(token_id) > self.max_length_per_sample:
        #     # TODO 数据加载时即可完成，这里不需要做额外处理
        #     pass
        local_vocab_id = set(token_id)
        old_to_new_vocab_id = dict(zip(local_vocab_id, range(len(local_vocab_id)))) # 从旧ID到新ID的转换

        # 1. 构建子图
        # sub_graph = dgl.DGLGraph().to(self.node_eta.device)
        sub_graph = dgl.graph(([], [])).to(self.node_eta.device)

        # 2. 构建节点
        sub_graph.add_nodes(len(local_vocab_id))  # 构建与文本长度大小一致的节点
        sub_graph.ndata['h'] = self.node_hidden(torch.Tensor(list(local_vocab_id)).int().to(self.node_eta.device))

        # 3. 构建边
        edges, old_edge_id = self.create_edges_for_a_sample(local_vocab_id, old_to_new_vocab_id)
        id_src, id_dst = zip(*edges)
        sub_graph.add_edges(id_src, id_dst)
        sub_graph.edata['w'] = self.edge_hidden(torch.Tensor(list(old_edge_id)).int().to(self.node_eta.device))

        return sub_graph


    def create_edges_for_a_sample(self, token_id, old_to_new_vocab_id):
        """
        为样本构建边，如何定义边是否存在？
        因为这里直接是构建子图，所以需要将大图节点信息转化为子图节点信息
        :param token_id: 样本数据ID
        :param old_to_new_vocab_id: 全局字典转化为局部字典
        :return:
            edges: 子图的边信息（使用子图的节点信息进行展示），格式为[new_id, new_id]
            old_edge_id: 子图边ID信息（使用大图的节点ID进行展示），格式为[message_for_old_edge_id]
        """
        edges = []
        old_edge_id = []
        new_token_id = []

        # TODO 如何处理全部为padding的情况
        for item_id in token_id:
            if item_id != 0:
                new_token_id.append(item_id)
            else:
                pass
        if len(new_token_id) == 0:
            new_token_id.append(0)

        for index, word_old_id in enumerate(new_token_id):
            new_token_id_src = old_to_new_vocab_id[word_old_id]
            for i in range(len(new_token_id)):
                new_token_id_dst = old_to_new_vocab_id[new_token_id[i]]
                # 新建一条边，已经包括了自环
                edges.append([new_token_id_src, new_token_id_dst])
                old_edge_id.append(self.edge_matrix[word_old_id, new_token_id[i]])

        return edges, old_edge_id


    def forward(self, token_ids):
        """
        图的输入：每个样本中可能含有的情感词，可能需要提前补零
        :param token_ids: shape is <B, T>. B and T denotes batch size and text length respectively. Type is torch.Tensor
        :return:
        """
        token_ids = token_ids.cpu().numpy().tolist()
        sub_graphs = [self.create_graph_for_a_sample(token_id) for token_id in token_ids]

        # TODO 这里需要改成单张图的信息传递过程

        # 1. 初始化大图
        batch_graph = dgl.batch(sub_graphs).to(self.node_eta.device)
        before_node_embedding = batch_graph.ndata['h']

        # 2. 完成大图的更新
        batch_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            reduce_func=dgl.function.max('weighted_message', 'h')
        )

        after_node_embedding = batch_graph.ndata['h']

        # 3. 处理聚合后和聚合前图的特征信息
        new_node_embedding = self.node_eta * before_node_embedding + (1 - self.node_eta) * after_node_embedding
        batch_graph.ndata['h'] = new_node_embedding

        # 计算整张图的特征
        graph_embedding = dgl.mean_nodes(batch_graph, feat='h')
        graph_embedding = self.embedding_layer1(graph_embedding)
        graph_embedding = self.activation_layer(graph_embedding)
        graph_embedding = self.dropout_layer(graph_embedding)
        # graph_embedding = self.norm_layer(graph_embedding)
        # graph_embedding = self.embedding_layer2(graph_embedding)
        return graph_embedding
