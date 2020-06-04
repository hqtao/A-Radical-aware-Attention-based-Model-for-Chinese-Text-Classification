# coding: utf-8
# Copyright @tongshiwei
"""
This file define the networks structure and
provide a simplest training and testing example.
"""
import logging
import os

import mxnet as mx
from longling.ML.MxnetHelper.gallery.layer.attention import \
    DotProductAttentionCell
from longling.ML.MxnetHelper.toolkit.ctx import split_and_load
from longling.ML.MxnetHelper.toolkit.viz import plot_network, VizError
from mxnet import gluon, nd, autograd
from tqdm import tqdm

# set parameters
try:
    # for python module
    from .etl import transform
    from .configuration import Configuration
except (ImportError, SystemError):
    # for python script
    from etl import transform
    from configuration import Configuration

from RAFG.share.etl import pseudo_data_generation
from RAFG.share.embedding import WCREmbedding

__all__ = ["WCRLSTM", "net_viz", "fit_f", "BP_LOSS_F", "eval_f"]


class WCRLSTM(gluon.HybridBlock):
    def __init__(self,
                 net_type,
                 class_num, lstm_hidden, embedding_dim,
                 word_embedding_size, word_radical_embedding_size,
                 char_embedding_size, char_radical_embedding_size,
                 embed_dropout=0.5, fc_dropout=0.5,
                 **kwargs):
        r"""Our method: 包含词和字，以及字、词部首的网络模型"""
        super(WCRLSTM, self).__init__(**kwargs)
        self.word_length = None
        self.character_length = None
        self.lstm_hidden = lstm_hidden
        self.net_type = net_type

        with self.name_scope():
            self.embedding = WCREmbedding(
                word_embedding_size=word_embedding_size,
                word_radical_embedding_size=word_radical_embedding_size,
                char_embedding_size=char_embedding_size,
                char_radical_embedding_size=char_radical_embedding_size,
                embedding_dim=embedding_dim,
                dropout=embed_dropout,
            )
            for i in range(4):
                if self.net_type in ("bilstm", "bilstm_att"):
                    setattr(self, "rnn%s" % i,
                            gluon.rnn.BidirectionalCell(
                                gluon.rnn.LSTMCell(lstm_hidden),
                                gluon.rnn.LSTMCell(lstm_hidden))
                            )
                elif self.net_type == "lstm":
                    setattr(
                        self, "rnn%s" % i, gluon.rnn.LSTMCell(lstm_hidden),
                    )
                else:
                    raise TypeError(
                        "net_type should be lstm, bilstm or bilstm_att,"
                        " now is %s" % self.net_type
                    )

            if self.net_type == "bilstm_att":
                self.word_attention = DotProductAttentionCell(
                    units=lstm_hidden, scaled=False
                )
                self.char_attention = DotProductAttentionCell(
                    units=lstm_hidden, scaled=False
                )

            self.dropout = gluon.nn.Dropout(fc_dropout)
            self.fc = gluon.nn.Dense(class_num)

    def hybrid_forward(self, F, word_seq, word_radical_seq, character_seq,
                       character_radical_seq, word_mask, charater_mask,
                       *args, **kwargs):
        word_length = self.word_length if self.word_length else len(word_seq[0])
        character_length = self.character_length if self.character_length else len(
            character_seq[0])

        word_embedding, word_radical_embedding, character_embedding, \
        character_radical_embedding = self.embedding(
            word_seq, word_radical_seq, character_seq,
            character_radical_seq
        )

        merge_outputs = True
        if F is mx.symbol:
            # 似乎 gluon 有问题， 对symbol
            word_mask, charater_mask = None, None

        if self.net_type == "bilstm_att":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn0").unroll(
                word_length, word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            word_radical_embedding = \
                self.word_attention(word_radical_embedding, w_e,
                                    mask=word_mask
                                    )[0]
            wr_e, (wr_lo, wr_ls, wr_ro, wr_rs) = getattr(self, "rnn1").unroll(
                word_length, word_radical_embedding,
                begin_state=(w_ls, w_ls, w_rs, w_rs),
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_lo, c_ls, c_ro, c_rs) = getattr(self, "rnn2").unroll(
                character_length, character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            character_radical_embedding = \
                self.char_attention(character_radical_embedding, c_e,
                                    mask=charater_mask
                                    )[0]
            cr_e, (cr_lo, cr_ls, cr_ro, cr_rs) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                begin_state=(c_ls, c_ls, c_rs, c_rs),
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )

            attention = F.concat(
                w_lo, w_ro, wr_lo, wr_ro, c_lo, c_ro, cr_lo, cr_ro
            )
        elif self.net_type == "bilstm":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn0").unroll(
                word_length, word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )

            wr_e, (wr_lo, wr_ls, wr_ro, wr_rs) = getattr(self, "rnn1").unroll(
                word_length, word_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_lo, c_ls, c_ro, c_rs) = getattr(self, "rnn2").unroll(
                character_length, character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            cr_e, (cr_lo, cr_ls, cr_ro, cr_rs) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )

            attention = F.concat(
                w_lo, w_ro, wr_lo, wr_ro, c_lo, c_ro, cr_lo, cr_ro
            )
        elif self.net_type == "lstm":
            w_e, (w_o, w_s) = getattr(self, "rnn0").unroll(
                word_length,
                word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            wr_e, (wr_o, wr_s) = getattr(self, "rnn1").unroll(
                word_length,
                word_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_o, c_s) = getattr(self, "rnn2").unroll(
                character_length,
                character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            cr_e, (cr_o, cr_s) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask)

            attention = F.concat(w_o, wr_o, c_o, cr_o)
        else:
            raise TypeError(
                "net_type should be lstm, bilstm or bilstm_att,"
                " now is %s" % self.net_type
            )
        attention = self.dropout(attention)
        # attention = self.dropout(
        #     F.Activation(self.nn(attention), act_type='softrelu')
        # )
        # fc_in = self.layers_attention(attention)
        fc_in = attention
        return self.fc(fc_in)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length
        self.character_length = character_length


def net_viz(net, cfg, view_tag=False, **kwargs):
    """visualization check, only support pure static network"""
    batch_size = cfg.batch_size
    model_dir = cfg.model_dir
    fixed_length = cfg.fixed_length
    logger = kwargs.get(
        'logger',
        cfg.logger if hasattr(cfg, 'logger') else logging
    )

    try:
        viz_dir = os.path.join(model_dir, "plot/network")
        logger.info("visualization: file in %s" % viz_dir)

        word_length, character_length = (1, 2) if not fixed_length else (
            fixed_length, fixed_length)
        from copy import deepcopy
        viz_net = deepcopy(net)
        viz_net.set_network_unroll(word_length, character_length)
        viz_shape = {
            'word_seq': (batch_size,) + (word_length,),
            'word_radical_seq': (batch_size,) + (word_length,),
            'character_seq': (batch_size,) + (character_length,),
            'character_radical_seq': (batch_size,) + (character_length,),
            # 'label': (batch_size,) + (1, )
        }
        word_seq = mx.sym.var("word_seq")
        word_radical_seq = mx.sym.var("word_radical_seq")
        character_seq = mx.sym.var("character_seq")
        character_radical_seq = mx.sym.var("character_radical_seq")
        word_mask = mx.sym.var("word_mask")
        char_mask = mx.sym.var("char_mask")
        sym = viz_net(
            word_seq, word_radical_seq, character_seq,
            character_radical_seq, word_mask, char_mask
        )
        plot_network(
            nn_symbol=sym,
            save_path=viz_dir,
            shape=viz_shape,
            node_attrs={"fixedsize": "false"},
            view=view_tag
        )
    except VizError as e:
        logger.error("error happen in visualization, aborted")
        logger.error(e)


def etl(cfg):
    return transform(pseudo_data_generation(), cfg)


def fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    word, word_radical, char, char_radical, word_mask, char_mask, label = _data
    # todo modify the input to net
    output = _net(word, word_radical, char, char_radical, word_mask, char_mask)
    bp_loss = None
    for name, func in loss_function.items():
        # todo modify the input to func
        loss = func(output, label)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = nd.mean(loss).asscalar()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def eval_f(_net, test_data, ctx=mx.cpu()):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    ground_truth = []
    prediction = []

    def evaluation_function(y_true, y_pred):
        evaluation_result = {}
        precsion, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred
        )
        evaluation_result.update(
            {"precision_%d" % i: precsion[i] for i in range(len(precsion))}
        )
        evaluation_result.update(
            {"recall_%d" % i: recall[i] for i in range(len(recall))}
        )
        evaluation_result.update(
            {"f1_%d" % i: f1[i] for i in range(len(f1))}
        )
        evaluation_result.update(
            {"Accuracy": accuracy_score(y_true, y_pred)}
        )
        return evaluation_result

    for batch_data in tqdm(test_data, "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (word, word_radical, char, char_radical, word_mask,
             char_mask, label) in ctx_data:
            output = _net(
                word, word_radical, char, char_radical, word_mask, char_mask
            )
            pred = mx.nd.argmax(output, axis=1)
            ground_truth.extend(label.asnumpy().tolist())
            prediction.extend(pred.asnumpy().tolist())

    return evaluation_function(ground_truth, prediction)


BP_LOSS_F = {"cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss()}


def numerical_check(_net, cfg):
    net.initialize()

    datas = etl(cfg)

    bp_loss_f = BP_LOSS_F
    loss_function = {}
    loss_function.update(bp_loss_f)
    from longling.ML.toolkit.monitor import MovingLoss
    from longling.ML.MxnetHelper.glue import module

    loss_monitor = MovingLoss(loss_function)

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=cfg.optimizer,
        optimizer_params=cfg.optimizer_params,
        select=cfg.train_select
    )

    for epoch in range(0, 100):
        for _data in tqdm(datas):
            with autograd.record():
                bp_loss = fit_f(
                    _net, _data, bp_loss_f, loss_function, loss_monitor
                )
            assert bp_loss is not None
            bp_loss.backward()
            trainer.step(cfg.batch_size)
        print(eval_f(_net, datas))
        print("epoch-%d: %s" % (epoch, list(loss_monitor.items())))


if __name__ == '__main__':
    cfg = Configuration()

    # generate sym
    net = WCRLSTM(
        class_num=cfg.class_num,
        word_embedding_size=cfg.word_embedding_size,
        word_radical_embedding_size=cfg.word_radical_embedding_size,
        char_embedding_size=cfg.char_embedding_size,
        char_radical_embedding_size=cfg.char_radical_embedding_size,
        **cfg.hyper_params
    )

    # # visualiztion check
    # net_viz(net, cfg, True)

    # numerical check
    numerical_check(net, cfg)
