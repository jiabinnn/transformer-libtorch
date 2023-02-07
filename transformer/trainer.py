import torch
import torch.nn as nn
import os
from model import Transformer, TransformerWithoutMask
import data_utils
from config import MyConfig

if __name__ == '__main__':
    config = MyConfig('config/config.ini')

    batch_size = config.batch_size
    max_len = config.max_len
    epochs = config.epochs
    PAD_VALUE = config.pad_value
    lr = config.learning_rate
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = config.device
    
    num_hiddens = config.num_hiddens
    query_size = key_size = value_size = num_hiddens
    ffn_num_hiddens = config.ffn_num_hiddens
    num_heads = config.num_heads
    encoder_layers = config.encoder_layers
    decoder_layers = config.decoder_layers
    dropout = config.dropout
    use_bias = config.use_bias
    
    model1_entire_path = config.model1_entire_path
    # model1_stat_dict_path = config.model1_stat_dict_path
    train_data_path = config.train_data_path
    
    corpus = data_utils.Corpus(train_data_path)
    train_iter, source_vocab, target_vocab = corpus.get_dataset(batch_size=batch_size, num_steps=max_len, num_examples=100, shuffle=False)
    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)

    dirname = os.path.dirname(model1_entire_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # '''deal mask before transformer'''
    # model = TransformerWithoutMask(src_vocab=source_vocab_size,
    #                     tgt_vocab=target_vocab_size,
    #                     max_len=max_len,
    #                     pad_idx=1,
    #                     query_size=query_size,
    #                     key_size=key_size,
    #                     value_size=value_size,
    #                     num_hiddens=num_hiddens,
    #                     ffn_num_hiddens=ffn_num_hiddens,
    #                     num_heads=num_heads,
    #                     enc_layers=encoder_layers,
    #                     dec_layers=decoder_layers,
    #                     dropout=dropout,
    #                     use_bias=use_bias)
    # model = model.to(device=device)

    # print('source vocab size:', source_vocab_size)
    # print('target vocab size:', target_vocab_size)
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # print(*[(name, param.shape) for name, param in model.named_parameters()])
    
    # for epoch in range(epochs):
    #     i = 0
    #     for x, x_len, y, y_len in train_iter:
    #         x = x.to(device=device)
    #         y = y.to(device=device)
            
    #         bos = torch.tensor([target_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
    #         dec_input = torch.cat([bos, y], dim=1)
    #         dec_input = dec_input[:, :-1]
    #         # 处理mask
    #         enc_self_attn_mask = data_utils.get_attn_pad_mask(x, x, PAD_VALUE).to(device)
            
    #         dec_self_attn_mask = data_utils.get_attn_pad_mask(dec_input, dec_input, PAD_VALUE).to(device)
    #         dec_self_subsequent_attn_mask = data_utils.get_attn_subsequent_mask(dec_input)
    #         dec_self_attn_mask = dec_self_attn_mask + dec_self_subsequent_attn_mask
    #         dec_self_attn_mask[:] = dec_self_attn_mask.gt(0).to(device)

    #         dec_enc_attn_mask = data_utils.get_attn_pad_mask(dec_input, x, PAD_VALUE).to(device)
    #         # print(enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask)
    #         # print(source_vocab.to_token(list(x[0])))
    #         # print(target_vocab.to_token(list(dec_input[0])))
    #         optimizer.zero_grad()
    #         pred = model(x, dec_input, enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask)
    #         loss = criterion(pred.permute(0, 2, 1), y)
    #         loss.backward()
    #         optimizer.step()
            
    #         if i % 100 == 0:
    #             print('epoch:', epoch, 'step:', i, 'loss:', loss.item())
    #         i += 1
    #     pred_result = pred.argmax(dim=2)
    #     print("enc_input: ", source_vocab.to_token(list(x[0])))
    #     print("dec_input: ", target_vocab.to_token(list(dec_input[0])))
    #     print("pred: ", target_vocab.to_token(list(pred_result[0])))
    
    #     print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    #     if os.path.exists(model1_entire_path):
    #         os.remove(model1_entire_path)
    #     torch.save(model, model1_entire_path)

    
    
    model2_entire_path = config.model2_entire_path
    '''deal mask in transformer'''
    model = Transformer(src_vocab=source_vocab_size,
                        tgt_vocab=target_vocab_size,
                        max_len=max_len,
                        pad_idx=1,
                        query_size=query_size,
                        key_size=key_size,
                        value_size=value_size,
                        num_hiddens=num_hiddens,
                        ffn_num_hiddens=ffn_num_hiddens,
                        num_heads=num_heads,
                        enc_layers=encoder_layers,
                        dec_layers=decoder_layers,
                        dropout=dropout,
                        use_bias=use_bias)
    model = model.to(device=device)

    print('source vocab size:', source_vocab_size)
    print('target vocab size:', target_vocab_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # print(*[(name, param.shape) for name, param in model.named_parameters()])
    
    for epoch in range(epochs):
        i = 0
        for x, x_len, y, y_len in train_iter:
            x = x.to(device=device)
            y = y.to(device=device)
            bos = torch.tensor([target_vocab['<bos>']] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y], dim=1)
            dec_input = dec_input[:, :-1]
            # print(source_vocab.to_token(list(x[0])))
            # print(target_vocab.to_token(list(dec_input[0])))
            optimizer.zero_grad()
            pred = model(x, dec_input)
            loss = criterion(pred.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('epoch:', epoch, 'step:', i, 'loss:', loss.item())
            i += 1

        # print(source_vocab.to_token(list(x[0])))
        # print(target_vocab.to_token(list(dec_input[0])))
        # pred_result = pred.argmax(dim=2)
        # print(target_vocab.to_token(list(pred_result[0])))
        pred_result = pred.argmax(dim=2)
        print("enc_input: ", source_vocab.to_token(list(x[0])))
        print("dec_input: ", target_vocab.to_token(list(dec_input[0])))
        print("pred: ", target_vocab.to_token(list(pred_result[0])))

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        # torch.save(model.state_dict(), 'saved_models/state_dict_model.pt')
        if os.path.exists(model2_entire_path):
            os.remove(model2_entire_path)
        torch.save(model, model2_entire_path)