# RECK

Code for the IEEE TCSVT paper "[Towards Bridged Vision and Language: Learning Cross-modal Knowledge Representation for Relation Extraction](xxxxxx)" .


Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

Data Preprocess
==========
To extract visual objects, following xxxxx, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. For the  images, the extracted objects are stored in `img_vg`. The images of the object obey the following naming format: `imgname_pred_yolo_crop_num.png`, where `imgname` is the name of the image corresponding to the object, `num` is the number of the object predicted by the toolkit. 
3. For the images, we construct a dictionary to record the correspondence between the images and the objects. Taking `mre_train_dif.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['imgname_pred_yolo_crop_num0.png', 'imgname_pred_yolo_crop_num1.png', ...] }`, where key is the name of the images, value is a List of the objects.
4. We provide knowledge extracted from conceptnet, presented in the form of triples and paths in files from `./data/concept_data`.


Data Download
==========

+ MNRE

You need to download three kinds of data to run the code.

1. The images of MNRE.
2. The visual objects from the raw images.
3. The ConceptNet_vocab.json file for concept representation.

You can download these data by XXXXXXX, and then place folders `img_org`, `img_vg`, `diffusion_pic` in the "./data" path.
	
Files' Structure
==========

The expected structure of files is:

```
RECK
 |-- ckpt # save the check point
 |-- data
 |    |-- txt  # text data
 |    |    |-- ours_train.txt # input data
 |    |    |-- ours_val.txt
 |    |    |-- ours_test.txt
 |    |    |-- mre_train_dict.pth  # {imgname: [object-image]}
 |    |    |-- ...
 |    |    |-- mre_dif_train_dif.pth # {imgname: [coordinates]}
 |    |    |-- ...
 |    |-- img_org       # original image data
 |    |-- img_vg   # visual object image data for original image
 |    |-- concept_data # external knowledge
 |    |    |-- train_0-5_t2v_entity.json # the external knowledge from text to vision
 |    |    |-- train_0-5_v2t_entity.json # the external knowledge from vision to text
 |    |    |-- ...
 |    |-- ours_rel2id.json # target relations
 |-- opennre	# main framework 
 |    |-- encoder # main model
 |    |    |-- bert_encoder.py # TMR-RE
 |    |    |-- modeling_bert.py
 |    |-- framework # processing files
 |    |    |-- data_loader.py # data processor
 |    |    |-- sentence_re.py # trainer
 |    |    |-- utils.py
 |    |-- model # classifier
 |    |    |-- softmax_nn.py # main classifier
 |    |    |-- modeling_bert.py 
 |    |    |-- base_model.py # supporting the classifier, no modification required
 |    |-- tokenization # tokenizers, no modification required
 |    |-- pretrain.py # basic file
 |    |-- utils.py # basic file
 |-- opennre.egg-info
 |-- results # saving the results
 |    |-- test # results for test set
 |    |-- val # results for validation set
 |-- run.py   # main 
 |-- ConceptNet_vocab.json # vocabulary of concept to its features from ConceptNet
```

Combination of `bert_encoder.py` and `softmax_nn.py` becomes our TMR-NER model.

`data_loader.py` is the file for processing raw data.

`sentence_re.py` is thr file that sets up training, testing, and other processes.

`run.py` is used for running the whole program.


Train
==========

## RE Task

The data path and GPU related configuration are in the `run.py`. To train ner model, run this script:

```shell
python run.py \
      --ckpt='your_re_ckpt_path' \
      --max_epoch=20 \
      --batch_size=16 \
      --lr=1e-5 \
      --sample_ratio=1.0
```

Test
==========
## RE Task

To test ner model, you can use the tained model and set `load_path` to the model path, then run following script:


```shell
python run.py \
      --ckpt='your_re_ckpt_path' \
      --max_epoch=20 \
      --batch_size=16 \
      --lr=1e-5 \
	  --test_only \
      --sample_ratio=1.0 \
```

