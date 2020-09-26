
#basics
import random
import pandas as pd
import torch
#from lxml import etree  ## is it allowed to add things here?
import xml.etree.ElementTree as ET
from pathlib import Path ##?
import random
from collections import Counter
import matplotlib.pyplot as plt

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def fill_dfs(self, file_list):  # read data from xml into dataframes
        self.file_list = file_list
        self.data_list = []
        self.ner_list = []
        for file in file_list:
            if str(file).split("/")[2] == "Test":
                split = "test"
            else:
                split = random.choices(["train", "val"], weights = (80, 20), k = 1)[0]  # split train into train and eval 
            tree = ET.parse(file)
            root = tree.getroot()
            for elem in root:
                sent_id = elem.get("id")
                sentence = elem.get("text")
                tokens = self.get_tokens(sentence)
                for subelem in elem:
                    if subelem.tag == "entity":  # maybe try to split all multi-word entities?
                        ner_id = self.get_id(subelem.get("type"), self.ner2id)
                        if len(subelem.get("charOffset").split("-")) == 2:
                            start_char, end_char = subelem.get("charOffset").split("-")
                            self.ner_list.append([sent_id, ner_id, int(start_char), int(end_char)])
                            #self.y += 1
                        else:
                            for word in subelem.get("charOffset").split(";"):
                                start_char, end_char = word.split("-")
                                self.ner_list.append([sent_id, ner_id, int(start_char), int(end_char)])
                                #self.y += 1
                for token in tokens:
                    token_id = self.get_id(token[0], self.word2id)
                    start_char = token[1]
                    end_char = token[2]
                    self.data_list.append([sent_id, token_id, start_char, end_char, split])
                    #self.x += 1
                
            #print(self.x)
            #if self.x >= 20000:
            #    break
        #return self.data_df, self.ner_df
        pass                   
                        
                
                
    def get_id(self, token, dic):  
        self.token = token
        self.dic = dic
        if token in dic:  
            return dic[token]
        else:
            dic[token] = len(dic)
            return dic[token]
        
    
    def get_tokens(self, sentence):  # split the sentence into tokens with start and end characters
        self.sentence = sentence
        tokens = sentence.split(" ")  # really not sure what to do with punctuation - example: 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine -> so some punctuation should stay there?
        #tokens = sentence.strip(".,()").split(" ")  # remove . and , TODO: also remove ()? And what about stopwords, e.g., etc..?
        tokens_with_numbers = []
        i = 0
        for token in tokens: 
            if token != "":
                token = token.lower()  # maybe not ideal, capitalization might be useful for brand names...???
                start_char = i
                last_char = token[-1]  # actual last character, vs. end_char is position of that character in sentence
                #end_char = i + len(token)-1
                #i += len(token) + 1
                if last_char.isalnum():  # condition remove trailing punctuation in words
                    end_char = i + len(token)-1
                    i += len(token) + 1
                else:
                    end_char = i + len(token)-2 
                    token = token.replace(last_char, "")  # still not perfect, sometimes more than one character in end...
                tokens_with_numbers.append((token, start_char, end_char))
        return tokens_with_numbers
            
            
            
            
            
                    
                
    
    
    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        
        # initialize the data structures
        #self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        #self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) 
        self.word2id = {}
        self.ner2id = {}
        #self.x = 0  # count for data_df rows
        #self.y = 0  # count for ner_df rows
        # next steps: make more efficient!!
        
        
        # read in the files
        allfiles = Path(data_dir)
        file_list = [f for f in allfiles.glob('**/*.xml') if f.is_file()]
        self.fill_dfs(file_list)
        self.data_df = pd.DataFrame(self.data_list, columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(self.ner_list, columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) 
        
        # transpose token-id dicts:
        self.id2word = {y:x for x,y in self.word2id.items()}
        self.id2ner = {y:x for x,y in self.ner2id.items()}
        self.vocab = list(self.word2id.keys())
        
        #set max_sample_length
        self.max_sample_length = 50
        
        
        
       # for file in file_list:
       #     if str(file).split("/")[2] == "Test":
       #         split = "test"
       #     else:
       #         split = "train"
       #     print("file no", x)
       #     print(file)
       #     tree = ET.parse(file)
       #     root = tree.getroot()
       #     for elem in root:
       #         print("Elem:", elem.attrib)
       #         sent_id = elem.get("id")
       #         for subelem in elem:
       #             if subelem.tag == "entity":
       #                 word = subelem.get("text").lower()
       #                 if word in voc:  
       #                     token_id = voc[word]
       #                 else:
       #                     voc[word] = len(voc)
       #                     token_id = voc[word]
       #                 print("Subelem:", subelem.attrib)
       #                 if len(subelem.get("charOffset").split("-")) == 2:
       #                     char_start_id, char_end_id = subelem.get("charOffset").split("-")
       #                 else:
       #                     char_start_id, char_end_id = (subelem.get("charOffset").split("-")[0], subelem.get("charOffset").split("-")[1].split(";")[1]), (subelem.get("charOffset").split("-")[1].split(";")[0], subelem.get("charOffset").split("-")[1])  
                        # super ugly, used for discontinued 2-part words, tuberculosis (and something else) drugs, e.g., now represented as (40, 50), (45, 55) -> better to just take very first and very last?
        #                ner_id = subelem.get("type") # chose to take all 4 categories, because less information loss, and big meaning differences...
        #                self.data_df.loc[x] = [sent_id, token_id, char_start_id, char_end_id, split]
        #                self.ner_df.loc[x] = [sent_id, ner_id, char_start_id, char_end_id]
        #    print(voc)
        #    x+=1
            #if x >= 5:
             #   break
        print("dataframes are ready")
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        # divide df by splits
        df_train = self.data_df[self.data_df.split=="train"]
        df_val = self.data_df[self.data_df.split=="val"]
        df_test = self.data_df[self.data_df.split=="test"]
        print("split df")
        
        #get labels
        self.train_labels = self.get_labels(df_train)
        self.val_labels = self.get_labels(df_val)
        self.test_labels = self.get_labels(df_test)          
        
        device = torch.device('cuda:0')
        # put labels into tensors
        train_tensor = torch.LongTensor(self.train_labels)
        self.train_tensor = train_tensor.reshape([(len(self.train_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        val_tensor = torch.LongTensor(self.val_labels)
        self.val_tensor = val_tensor.reshape([(len(self.val_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        test_tensor = torch.LongTensor(self.test_labels)
        self.test_tensor = test_tensor.reshape([(len(self.test_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        
        print("got y")
        return self.train_tensor, self.val_tensor, self.test_tensor
                       
    def get_labels(self, df):
        self.df = df
        label_list = []
        
        sent_ids = [s for s in df["sentence_id"]]
        start_ids = [s for s in df["char_start_id"]]
        end_ids = [s for s in df["char_end_id"]]
        id_tuples = list(zip(sent_ids, start_ids, end_ids))
        
        label_sent_ids = [s for s in self.ner_df["sentence_id"]]
        label_start_ids = [s for s in self.ner_df["char_start_id"]]
        label_end_ids = [s for s in self.ner_df["char_end_id"]]
        labels = [s for s in self.ner_df["ner_id"]]
        label_tuples = list(zip(label_sent_ids, label_start_ids, label_end_ids))
        #print(label_tuples)
        
        for t in id_tuples:
            if t in label_tuples:
                #print("same")
                label = labels[label_tuples.index(t)]
            else:
                label = 4
            label_list.append(label)            
        
        #for i in range(len(df)):
         #   row = df.iloc[i]
          #  sent_id = row["sentence_id"]
           # char_onset = row["char_start_id"]
            #char_offset = row["char_end_id"]
    #        print(i)
     #       print(row)
      #      print(sent_id, char_onset)
            #same_sent = self.ner_df[self.ner_df["sentence_id"] == sent_id]
       #     rightline = self.ner_df.loc[(self.ner_df["sentence_id"] == sent_id) & (self.ner_df["char_start_id"] == char_onset) & (self.ner_df["char_end_id"] == char_offset)]
            #print(same_sent)
            #same_token = same_sent[same_sent["char_start_id"] == char_onset]
            #same_token = same_token[same_sent["char_end_id"] == char_offset]
        #    print(rightline)
            #print(same_token)
            #print("WOHOO!", len(same_token))
         #   if len(rightline) > 0:
          #      print(rightline["ner_id"])
           #     label = int(rightline["ner_id"])  ## this has some kind of problem...!!!
            #else:
    #            label = 4 # new label for "not an ner"             
     #       label_list.append(label)
        label_list = label_list[:(len(label_list)-len(label_list)%self.max_sample_length)]  ##?? really understand this&explain in readme!
        #print(label_list)
        return label_list              
                       
    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        self.get_y()
        
        # get label counts
        train_counts = Counter(self.train_labels)
        val_counts = Counter(self.val_labels)
        test_counts = Counter(self.test_labels)
        
        # put counts into a dataframe:
        counts_df = pd.DataFrame([train_counts, val_counts, test_counts], index=['train', 'val', 'test'])
        counts_df.plot(kind='bar')   # asked for histogram, but this seems to make more sense...
        plt.show()
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



