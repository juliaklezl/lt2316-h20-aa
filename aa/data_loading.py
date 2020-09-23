
#basics
import random
import pandas as pd
import torch
#from lxml import etree  ## is it allowed to add things here?
import xml.etree.ElementTree as ET
from pathlib import Path ##?


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
        for file in file_list:
            if str(file).split("/")[2] == "Test":
                split = "test"
            else:
                split = "train"
            tree = ET.parse(file)
            root = tree.getroot()
            for elem in root:
                sent_id = elem.get("id")
                sentence = elem.get("text")
                tokens = self.get_tokens(sentence)
                for token in tokens:
                    token_id = self.get_id(token, self.id2word)
                    start_char = token[1]
                    end_char = token[2]
                    self.data_df.loc[self.x] = [sent_id, token_id, start_char, end_char, split]
                    self.x += 1
                for subelem in elem:
                    if subelem.tag == "entity":
                        ner_id = self.get_id(subelem.get("type"), self.id2ner)
                        if len(subelem.get("charOffset").split("-")) == 2:
                            start_char, end_char = subelem.get("charOffset").split("-")
                            self.ner_df.loc[self.x] = [sent_id, ner_id, start_char, end_char]
                            self.x += 1
                        else:
                            for word in subelem.get("charOffset").split(";"):
                                start_char, end_char = word.split("-")
                                self.ner_df.loc[self.x] = [sent_id, ner_id, start_char, end_char]
                                self.x += 1
        return self.data_df, self.ner_df
                            
                        
                
                
    def get_id(self, token, dic):  ## !!!!!
        self.token = token
        self.dic = dic
        for key, value in dic.items():  
            if token == value:
                return key
            dic[len(dic)] = token
            return len(dic)
        
    
    def get_tokens(self, sentence):  # split the sentence into tokens with start and end characters
        self.sentence = sentence
        tokens = sentence.strip(".").strip(",").split(" ")  # remove . and , TODO: also remove ()? And what about stopwords, e.g., etc..?
        tokens_with_numbers = []
        i = 0
        for token in tokens:
            start_char = i
            end_char = i + len(token)-1
            i += len(token) + 1
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
        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) 
        self.id2word = {}
        self.id2ner = {}
        self.x = 1  # count for dataframe rows
        
        # next steps: split train into train and val, make more efficient!!
        
        
        # read in the files
        allfiles = Path(data_dir)
        file_list = [f for f in allfiles.glob('**/*.xml') if f.is_file()]
        self.fill_dfs(file_list)
       
        
        
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
        return 


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        pass

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
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



