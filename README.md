# Learning ads embedding

This project is focusing on embedding learning of PITT ads dataset
(see [LINK](http://people.cs.pitt.edu/~kovashka/ads/)). In general, it utilizes
triplet loss to distinguish between related caption-image pair and unrelated
caption-image pair. By doing so, the model embeds both image patches and words
into a shared vector space, which could be later used for tasks such as
classification, captioning, vision question answering and so on.

## Method

There are mainly three components in the implementation: region proposal
network, image feature extractor, and text embedder. Generally speaking, region
proposal network is responsible to predict highly probable objects and provides
bounding box information of these objects. Given these boxes, image feature
extractor crops image patches from the input image and extract patch level
feature representation. The final image level representation is a combination of
the patch representations. At the meantime, text embedder encodes statements 
into the same feature space as the image representation. Triplet loss is used to
train the whole network.

### Region proposal network

### Image feature extractor

### Text embedder
