# ADVISE: Symbolism and External Knowledge for Decoding Advertisements.

The ADVISE project focuses on the embedding learning task of PITT ads dataset
(see [LINK](http://people.cs.pitt.edu/~kovashka/ads/)). We also use this
implementation to take part in the Automatic Understanding of Visual
Advertisements challenge (see 
[LINK](https://evalai.cloudcv.org/web/challenges/challenge-page/86/overview)). 
As the baseline approaches, the <a href="configs/vse++.pbtxt">VSE</a> model
achieves an accuracy of 62% and the <a href="configs/advise.kb.pbtxt">ADVISE</a>
model achieves an accuracy of 69% in the challenge.

In general, our model utilizes triplet ranking loss to distinguish between 
related caption-image pair and unrelated caption-image pair caption-image pair. 
By doing so, the model project both image patches and words into a shared 
vector space, which could be later used for tasks such as classification, 
captioning, vision question answering and so on. Beyond the traditional visual
semantic embedding, we found that using the 1) bottom-up attention mechanism, 2)
constraints via symbols and captions, 3) additive external knowledge, helps to
improve the final performance especially for the public service announcements 
(PSAs).

We provide both the baseline <a href="configs/vse++.pbtxt">VSE</a> model and 
our <a href="configs/advise.kb.pbtxt">ADVISE</a> model in this repository.
If you are using our implementation, please cite our paper:
```
Ye, Keren, and Adriana Kovashka. "ADVISE: Symbolism and External Knowledge for
Decoding Advertisements." arXiv preprint arXiv:1711.06666 (2017).
```
\[[link](https://arxiv.org/pdf/1711.06666.pdf)\]\[[bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:K2QWc_pL9-YJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWvsX4yeW9FRFUealOfUsxcfTEzOL2F4A&scisf=4&ct=citation&cd=-1&hl=en)\]

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

## Prerequisites
Tensorflow >= version 1.6

Disk space > 20G (in order to store the downloaded images and intermediate files)

## Getting start

* Clone this repository.
```
git clone https://github.com/yekeren/ADVISE.git
```

* Enter the ROOT of the local directory.
```
cd ADVISE
```

* Prepare the PITT Ads Dataset and pre-trained models. This step shall take 
a long time to proceed (3-4 hours in our enviroments using GPU). The 
"prepare\_data.sh" script shall guide you to to:

      - Download the PITT Ads Dataset (>= 11G)
      - Clone the "tensorflow/models" repository, in which we use the
      object\_detection API and the InceptionV4 slim model.
      - Download the pre-trained GloVe model.
      - Download the pre-traind InceptionV4 model.
      - Prepare the vocabulary and initial embedding matrix of the action-reason 
      annotations of the ads data.
      - Prepare the vocabulary and initial embedding matrix of the DenseCap
      annotations.
      - Prepare the vocabulary and initial embedding matrix of the ads symbols.
      - Extract both the image features and the regional features using
      InceptionV4 model. Note that we provide two types of region proposals. 
      <a href="output/symbol_box_test.json">One</a> extracted using the tensorflow
      object detection API (we trained the model on ads symbol boxes by ourselves), 
      and <a href="output/densecap_test.json">the other</a> extracted using the 
      DenseCap model. You could also provide region proposals extracted by 
      yourselves and encode them using the same JSON format. Note: it is not 
      necessary to extract features using both of the two region proposals. 
      So you can comment out either the symbol box or the densecap box 
      ("prepare_data.sh" line 144-166).


```
sh prepare_data.sh
```

* If you want to provide region proposals extracted by yourselves. You can 
still use the visualization tools provided by us:
```
cd visualization/data
ln -s ../../output/densecap_test.json .
ln -s ../../output/symbol_box_test.json .
ln -s ../../data/test_images/ ./images
cd ..
python -m SimpleHTTPServer 8009
```
Then checkout the contents from your web browswer
using <a href="http://localhost:8009/symbol_box.html">http://localhost:8009/symbol_box.html</a>
or <a href="http://localhost:8009/densecap_box.html">http://localhost:8009/densecap_box.html</a>.
You shall see results similar to:

![symbol box visualizationtext][symbol_box | width=200] 
![densecap box visualizationtext][densecap_box | width=200]

[symbol_box]: https://github.com/yekeren/ADVISE/blob/master/docs/symbol_box.png "Logo Title Text 2"
[densecap_box]: https://github.com/yekeren/ADVISE/blob/master/docs/densecap_box.png "Logo Title Text 2"


## References
Finally, special thanks to these authors, our implementation mainly depends 
on their efforts.
```
Hussain, Zaeem, et al. "Automatic understanding of image and video
advertisements." 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). IEEE, 2017.

Kiros, Ryan, Ruslan Salakhutdinov, and Richard S. Zemel. "Unifying
visual-semantic embeddings with multimodal neural language models." arXiv
preprint arXiv:1411.2539 (2014).

Faghri, Fartash, et al. "VSE++: Improved Visual-Semantic Embeddings." arXiv
preprint arXiv:1707.05612 (2017).

Huang, Jonathan, et al. "Speed/accuracy trade-offs for modern convolutional
object detectors." IEEE CVPR. 2017.

Anderson, Peter, et al. "Bottom-up and top-down attention for image captioning
and VQA." arXiv preprint arXiv:1707.07998 (2017).

Teney, Damien, et al. "Tips and Tricks for Visual Question Answering: Learnings
from the 2017 Challenge." arXiv preprint arXiv:1708.02711 (2017).

Johnson, Justin, Andrej Karpathy, and Li Fei-Fei. "Densecap: Fully convolutional
localization networks for dense captioning." IEEE CVPR. 2016.

Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified
embedding for face recognition and clustering." IEEE CVPR. 2015.

Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of
residual connections on learning." AAAI. Vol. 4. 2017.

Abadi, Martín, et al. "TensorFlow: A System for Large-Scale Machine Learning."
OSDI. Vol. 16. 2016.
```
