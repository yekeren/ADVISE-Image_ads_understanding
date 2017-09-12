# Learning ads embedding

This project is focusing on embedding learning of PITT ads dataset
(see [LINK](http://people.cs.pitt.edu/~kovashka/ads/)). In general, it utilizes
triplet loss to distinguish between related caption-image pair and unrelated
caption-image pair. By doing so, the model embeds both image patches and words
into a shared vector space, which could be later used for tasks such as
classification, captioning, vision question answering and so on.
