# LT2316 H20 Assignment A1

Name: *Julia Klezl* 

## Notes on Part 1.

Overall, I just followed the functions provided in the assignment. First I use Path and ElementTree to read in parse all the files. I loop through each sentence of each document, tokenize the sentences, and get the tokens' character spans. I also loop through the entities in the sentence trees to get the actual named entities identified in the training data. The token and NER labels get encoded to integers. I collect all this information in a list of lists, which is later converted to a dataframe. 
In order to get_y, I extract all sentence_id's and character on- and offsets of the token-set and the label-set into a list of tuples (for efficiency/speed). These tuples are used to map the tokens to the appropriate labels where possible. When there is no label, I assign the entity type 0, signalling that a token is not a named entitiy. Finally, the lists of labels get converted into tensors of the correct shape and saved to the GPU. This is the overall structure I used - in the following I will just explain specific design decisions I made (and which I think need an explanation/justification):

location of dataset: My program assumes to find the dataset at the path given in the notebook. I saved the data in a folder together with the lt2316-h20-aa directory. 

train/val split: I used 20% of the files marked as training data for the development set. It is chosen by random weighted assignment of the labels "train" and "val". 

compound words: Since I do the manual tokenization by whitespaces, all compound words are separated into 2 individual tokens. Therefore, to keep the data as consistent as possible, I decided to also separate the already identified entities when they consist of more than one word. The only case where this is not reliable is when entities consist of more than 2 words separated by other words in between. In these cases, groupings of several words might persist. This is of course not ideal, but doesnt cause wider problems in the data as the character count is not affected, so it seems like a neglectable inconsistency. 

lists/dataframes: for performance reasons, I collect the data for the dataframe in a list of lists while adding to it, then convert it to a dataframe once its complete.

sample padding: I interpreted the instructions so that each sample is one sentence. To have equal length for all samples, I add padding lists of 0s except for the split to all samples, up to the length of the longest sentence in the data. I think 0 is a good choice for char_start_id and char_end_id since their being the same encodes that it is not an actual token, and 0 is not an id in the token-vocab, so it shouldnt create too much confusion here either. I'm not sure about whether 0 is a good id for sentence_id, but I need the sentence Ids to identify sentence borders in the feature extraction, so I didnt want to add the same sentence id as in the actual data for that sample. In the end, it shouldn't matter since the sentence_id isn't part of the final feature tensors anyway. 

tokenization and lemmatization: I was very unsure how to approach this since the drug names and chemical substances we're interested here don't always follow the usual patterns of language (c.f. "words" like: 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine). I tokenized by whitespaces, which is already problematic since it separates some tokens that are actually one word. Therefore, I didn't dare to do much more to the tokens except for removing trailing punctuation (esp. periods and commas) and lowercasing everything. (The trailing punctuation unfortunately isn't always fully removed when there are several punctuation marks in a row - I'm sure this can be solved, but for a 2-week assignment perfectionsm needs to be kept in check). I also split at semicolons and slashes, since these are often used to separate words (e.g. when listing synonyms or members of a group). I considered keeping the capital letters since it might be useful for identifying brand names, but they add so many duplicate words to the vocabulary simply due to sentence beginnings that I felt like it did more harm than good in the data. This cautious approach to lemmatization leads me to have a very large vocabulary (including many near-duplicates), but this seemed like the better choice than reducing it too much in this case.

entity groups: I decided to keep all 4 of the entity groups instead of combining drug and drug_n. Even though these 2 might appear in similar contexts, they do carry a significantly different meaning, so I decided it would impact the data quality negatively to combine them. Additionally, there is already a group imbalance (with the largest number of entities being drugs), so I didn't want to add to that by making this group even bigger. And even though this is just a university assignment, when considering possible real-world applications of such a program, it seems problematic to use the categories for drugs and substances explicitly not made for human consumption interchangeably.

## Notes on Part 2.

I chose to use the following 7 features for this task:

1 & 2) Sentence borders. While browsing through the data, I noticed that many sentences begin or end with a drug name (or group, brand, etc). So my first two features encode whether a word is located at the very beginning or at the very end of a sentence 

3 & 4) Neighboring words. Since we're looking at sentences, I think the context a word appears in is very important in defining/identifying it, and as we've learned in the past courses, the closer the neighbours, the more relevant they are. Therefore I picked the id of the preceding and following word in the sentence. As we are working with a collection of sentences that are not necessarily in a meaningful order, I'm only interested in neighbouring words within the sentence. If there is none, I added an additional token signifying that there is no direct neighbour.

5) Word length. Many of the drug names are unusually long, while brand names for example generally aren't (since people should remember their names). So I think this could be an interesting feature for both identifying an NER and classifying it correctly.

6) Punctuation and numbers. As mentioned above, many chemical names of drugs include numbers and commas, so this feature checks whether a token includes only alphabetic characters or not. 
 
7) POS-tags. Since most of the entities we are trying to identify in this task are nouns, I thought getting the POS tag for each token could be a useful feature. I used the nltk POS-tagger, and then encoded the resulting tags to integers again. For the dummy padding tokens in each sample, I added the additional id 0 (just to denote that it is not an actual POS),

## Notes on Part Bonus.

I was not completely sure what I was meant to do in the first plot. I interpreted it in a way that I counted how many sentences have x tokens. For the histogram, I chose to use 50 bins, since that results in a plot that's both fine-grained enough to be interesting and still big enough to be visible. I followed the same approach for the second plot. In the third one, I used matplotlib-venn to draw the diagram and pip installed this on mltgpu (pip install --user matplotlib-venn). Since this library can only draw 3 circles at most, I did combine the drug and drug_n category here.
UPDATE: Instead of matplotlib-venn I switched to using pyvenn, since this package allows diagrams with more than 3 groups. I installed it with pip (pip install --user venn). Now all 4 groups are displayed separately.