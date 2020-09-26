
#basics
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
import torch

# Feel free to add any new code to this script


def extract_features(data:pd.DataFrame, max_sample_length:int, id2w):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    
    # loop through df in an efficient way somehow
    
    sent_ids = [s for s in data["sentence_id"]]
    token_ids = [s for s in data["token_id"]]
    start_ids = [s for s in data["char_start_id"]]
    end_ids = [s for s in data["char_end_id"]]
    split = [s for s in data["split"]]
    all_rows = list(zip(sent_ids, token_ids, start_ids, end_ids, split))
    
    #pos_list = []
    #ordered_set = []
    #for i in sent_ids:
    #    if i not in ordered_set:
    #        ordered_set.append(i)
    #print(ordered_set)
    #for sent in ordered_set:
    #    sentence = data[data.sentence_id == sent]
    #    print(sent)
    #    sentence = [s for s in data[data.sentence_id == sent]]
    #    print(sentence)
    #    pos_list.append(nltk.pos_tag(sentence))
    #print(pos_list)
                
    
    train_features = []
    val_features = []
    test_features = []
    
    for i in range(len(all_rows)):
        row = all_rows[i]
        #print(row)
        if (i == 0) or (all_rows[i-1][0] != row[0]):
            at_start = 1
            prior_word = len(id2w)  # additional id meaning that no prior token in same sentence
        else:
            at_start = 0
            prior_word = all_rows[i-1][1]
        if (i == len(all_rows)-1) or (all_rows[i+1][0] != row[0]):
            at_end = 1
            next_word = len(id2w)  # additional id, as above
        else:
            at_end = 0
            next_word = all_rows[i+1][1]
        word_length = row[3] - row[2]
        if id2w[row[1]].isalpha:
            has_punct = 0
        else:
            has_punct = 1
        #pos = nltk.pos_tag((id2w[row[1]]))
        features = [at_start, at_end, prior_word, next_word, word_length, has_punct]
        if row[4] == "train":
            train_features.append(features)
        elif row[4] == "val":
            val_features.append(features)
        elif row[4] == "test":
            test_features.append(features)
    
    train_features = train_features[:(len(train_features)-len(train_features)%max_sample_length)]
    val_features = val_features[:(len(val_features)-len(val_features)%max_sample_length)]
    test_features = test_features[:(len(test_features)-len(test_features)%max_sample_length)]
    
    device = torch.device('cuda:0')
    train_f_tensor = torch.LongTensor(train_features)
    train_f_tensor = train_f_tensor.reshape([(len(train_f_tensor)//max_sample_length),max_sample_length, 6]).to(device) # number features hardcoded: 6
    val_f_tensor = torch.LongTensor(val_features)
    val_f_tensor = val_f_tensor.reshape([(len(val_f_tensor)//max_sample_length),max_sample_length, 6]).to(device)
    test_f_tensor = torch.LongTensor(test_features)
    test_f_tensor = test_f_tensor.reshape([(len(test_f_tensor)//max_sample_length),max_sample_length, 6]).to(device)
    
   
    
    # try padding instead of removing words
    
    # try POS again?
    
    # clean up all this mess and write readme!

    
    return train_f_tensor, val_f_tensor, test_f_tensor
    #pass
