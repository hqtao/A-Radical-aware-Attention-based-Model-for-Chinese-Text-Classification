# A-Radical-aware-Attention-based-Model-for-Chinese-Text-Classification
My research paper

Chinese, a language derived from pictographs, is essentially different from English or other phonetic languages. For Chinese, one of the most unique is that the character system of Chinese is based on hieroglyphics, which has the raw meanings. That is to say, not only words and characters can express specific meanings, but also radicals are important carriers of semantics. 

This is the implementation of  [A Radical-aware Attention-based Model for Chinese Text Classiﬁcation](http://home.ustc.edu.cn/~hqtao/index_files/RAFG_Tao.pdf).
After the publication, we went deeper into the designing of architecture 
and managed to make some improvement in promoting some models effectiveness.

- WCLSTM realizes the Baseline of CLSTM and CBLSTM;
- WCRLSTM realizes RAFG（BLSTM_attention）, FourLSTM and FourBLSTM;
- WRCLSTM realizes the concatenated model (including unidirectional and bidirectional LSTM versions) of three features: character, word and word-level radical, with an attention mechanism between word and word-level radical;



If you are interseted in Chinese natural language processing, welcome refer to and cite our paper which has been published in AAAI-19: https://aaai.org/ojs/index.php/AAAI/article/view/4446



## Tutorial

### Environment setting

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/RFAG
```

### Dependency

```python3
# longling
git clone https://github.com/tswsxk/longling
```

### Demo running commands

#### only use word and character features

```bash
# run on dataset1, use gpu(0), use lstm
python3 WCLSTM.py --workspace lstm --dataset dataset1 --caption "baseline for wc, lstm, on dataset1" --net_type lstm --ctx "gpu(0)"
# run on dataset2, use gpu(1), use lstm
python3 WCLSTM.py --workspace lstm --dataset dataset2 --caption "baseline for wc, lstm, on dataset2" --net_type lstm --ctx "gpu(1)"
# run on dataset1, use gpu(3), use bilstm, todo
python3 WCLSTM.py --workspace bilstm --dataset dataset1 --caption "baseline for wc, bilstm, on dataset1" --net_type bilstm --ctx "gpu(3)"
# run on dataset2, use gpu(3), use bilstm
python3 WCLSTM.py --workspace bilstm --dataset dataset2 --caption "baseline for wc, bilstm, on dataset2" --net_type bilstm --ctx "gpu(3)"
```

### with only word radical features

```bash
# run on dataset1, use gpu(4), use lstm
python3 WRCLSTM.py --workspace lstm --dataset dataset1 --caption "baseline for wrc, lstm, on dataset1" --net_type lstm --ctx "gpu(4)"
# run on dataset2, use gpu(5), use lstm
python3 WRCLSTM.py --workspace lstm --dataset dataset2 --caption "baseline for wrc, lstm, on dataset2" --net_type lstm --ctx "gpu(5)"
# run on dataset1, use gpu(6), use bilstm
python3 WRCLSTM.py --workspace bilstm --dataset dataset1 --caption "baseline for wrc, bilstm, on dataset1" --net_type bilstm --ctx "gpu(6)"
# run on dataset2, use gpu(7), use bilstm
python3 WRCLSTM.py --workspace bilstm --dataset dataset2 --caption "baseline for wrc, bilstm, on dataset2" --net_type bilstm --ctx "gpu(7)"
# run on dataset1, use gpu(4), use bilstm_att
python3 WRCLSTM.py --workspace bilstm_att --dataset dataset1 --caption "ours, bilstm_att, on dataset1" --net_type bilstm_att --ctx "gpu(4)"
# run on dataset2, use gpu(5), use bilstm_att
python3 WRCLSTM.py --workspace bilstm_att --dataset dataset2 --caption "ours, bilstm_att, on dataset2" --net_type bilstm_att --ctx "gpu(5)"
```

#### with radical features

```bash
# run on dataset1, use gpu(8), use lstm
python3 WCRLSTM.py --workspace lstm --dataset dataset1 --caption "baseline for wcr, lstm, on dataset1" --net_type lstm --ctx "gpu(8)"
# run on dataset2, use gpu(9), use lstm
python3 WCRLSTM.py --workspace lstm --dataset dataset2 --caption "baseline for wcr, lstm, on dataset2" --net_type lstm --ctx "gpu(9)"
# run on dataset1, use gpu(0), use bilstm
python3 WCRLSTM.py --workspace bilstm --dataset dataset1 --caption "baseline for wcr, bilstm, on dataset1" --net_type bilstm --ctx "gpu(0)"
# run on dataset2, use gpu(1), use bilstm
python3 WCRLSTM.py --workspace bilstm --dataset dataset2 --caption "baseline for wcr, bilstm, on dataset2" --net_type bilstm --ctx "gpu(1)"
# run on dataset1, use gpu(8), use bilstm_att
python3 WCRLSTM.py --workspace bilstm_att --dataset dataset1 --caption "ours, bilstm_att, on dataset1" --net_type bilstm_att --ctx "gpu(8)"
# run on dataset2, use gpu(9), use bilstm_att
python3 WCRLSTM.py --workspace bilstm_att --dataset dataset2 --caption "ours, bilstm_att, on dataset2" --net_type bilstm_att --ctx "gpu(9)"
```



