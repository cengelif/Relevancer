# Relevancer
This project is started by Ali Hürriyetoğlu and Elif Türkay from Radboud University Nijmegen and Ron Boortman from Floodtags.

# Relevancer: Finding and Labeling the Structure in a Tweet Set

 Ali Hürriyetoğlu, Elif Türkay, Antal van den Bosch, Mustafa Erkan Başar

Usually, collections of tweets gathered by means of keyword-based queries are overly rich in the sense that not all tweets are relevant for a task at hand. Making sense of that data for a particular scenario depends on the context of the person who needs the information. Since nobody practically has time to read the whole of a tweet set, it is advisable to use machine learning to facilitate the first information organization steps. In practice, the most useful first step is to identify tweets that are clearly irrelevant for a particular task, for instance because they are posted by non-human accounts, contain spam, or point to another sense of a polysemous keyword used at query time.

Available systems [1, 2, 3] were designed for certain domains, languages, and use cases. Moreover each of them suffers from some combination of the following restrictions. First, they are restricted to analyzing tweets that contain at least a certain number of well-formed key terms or content words in certain languages. Second, they do not take into account tweet characteristics such as emoticons and personal language use. Third, they rarely use attributes of a tweet other than the text. Finally, they assume the availability of a set of annotators or a crowd that is willing to label a sufficient number of tweets.

We developed our automatic Twitter filtering system Relevancer such that it does not suffer from the aforementioned restrictions. Relevancer facilitates exploring a tweet data set by dividing the data set into subsets to which the annotator can attach labels. Relevancer first employs an unsupervised clustering algorithm on a subset of the tweets to explore the structure of the tweet set, with explicit distinction of retweets and (near) duplicates. Subsequently, the annotator is asked to rate tweet clusters as being relevant and coherent. The clustering step is repeated for the remaining tweets until the domain expert stops the process, or a predefined target, such as the percentage of labeled tweets that should be labeled, is met. The type of clustering algorithm (currently, k-means and mini batch k-means are supported), and the annotation candidate group selection parameters are automatically adapted based on the size and the quality of the output respectively. The labeled tweet groups, the output of the unsupervised step, is used as training data for building automatic classifiers that can be applied 'in the wild' to any new tweets gathered with the same query.


## Acknowledgements
This project was supported by the Dutch national programme COMMIT as part of the Infiniti project.
