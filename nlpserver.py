#
#	NLP Server with FastText: https://github.com/digitaldogsbody/nlpserver-fasttext
#
#	To run:
# 	$ nohup python3 nlpserver.py  >logs/nlpserver_out.log 2>logs/nlpserver_errors.log &
#
from flask import Flask, jsonify, abort, request, send_from_directory, render_template
import os

app = Flask(__name__)

#  configurations
#app.config['var1'] = 'test'
app.config["YOLO_MODEL_NAME"] = "yolov8m.pt"
app.config["MOBILENET_MODEL_NAME"] = "mobilenet_v3_small.tflite"
app.config["EFFICIENTDET_DETECT_MODEL_NAME"] = "efficientdet_lite2.tflite"
app.config["MOBILENET_DETECT_MODEL_NAME"] = "ssd_mobilenet_v2.tflite"
for variable, value in os.environ.items():
	if variable == "YOLO_MODEL_NAME":
		# Can be set via Docker ENV
		app.config["YOLO_MODEL_NAME"] = value
	if variable == "MOBILENET_MODEL_NAME":	
		app.config["MOBILENET_MODEL_NAME"] = value
	if variable == "EFFICIENTDET_MODEL_NAME":	
		app.config["EFFICIENTDET_MODEL_NAME"] = value	
	if variable == "MOBILENET_DETECT_MODEL_NAME":	
		app.config["MOBILENET_DETECT_MODEL_NAME"] = value		

default_data = {}
default_data['web64'] = {
		'app': 'nlpserver',
		'version':	'1.1.0',
		'last_modified': '2024-05-05',
		'documentation': 'https://github.com/esmero/nlpserver-fasttext/README.md',
		'github': 'https://github.com/esmero/nlpserver-fasttext',
		'endpoints': ['/status','/gensim/summarize', '/polyglot/neighbours', '/langid', '/polyglot/entities', '/polyglot/sentiment', '/newspaper', '/readability', '/spacy/entities', '/afinn', '/fasttext', '/image/yolo', 'image/mobilenet'],
	}

default_data['message'] = 'NLP Server by web64.com - with fasttext addition by digitaldogsbody'
data = default_data

@app.route("/")
def main():
	return render_template('form.html')
	#return jsonify(data)

@app.route('/status')
def status():
	data = dict(default_data)
	data['missing_libraries'] = []
	
	try:
		import textblob
	except ImportError:
		data['missing_libraries'].append('textblob')

	try:
		import spacy
	except ImportError:
		data['missing_libraries'].append('spacy')
	try:
		import gensim
	except ImportError:
		data['missing_libraries'].append('gensim')
	
	try:
		import newspaper
	except ImportError:
		data['missing_libraries'].append('newspaper')

	try:
		import langid
	except ImportError:
		data['missing_libraries'].append('langid')

	try:
		import readability
	except ImportError:
		data['missing_libraries'].append('readability')
	
	try:
		import bs4
	except ImportError:
		data['missing_libraries'].append('bs4')
	
	try:
		import afinn
	except ImportError:
		data['missing_libraries'].append('afinn')

	try:
		import polyglot
	except ImportError:
		data['missing_libraries'].append('polyglot')
	else:
		from polyglot.downloader import Downloader
		dwnld = Downloader()
		data['polyglot_lang_models'] = {}

		for info in sorted(dwnld.collections(), key=str):
			status = dwnld.status(info)
			if info.id.startswith('LANG:') and status != 'not installed':
				data['polyglot_lang_models'][info.id] = status
	
	try:
		import fasttext
	except ImportError:
		data['missing_libraries'].append('fasttext')

	return jsonify(data)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/spacy/entities", methods=['GET', 'POST'])
def spacy_entities():
	import spacy
	data = dict(default_data)
	data['message'] = "Spacy.io Entities (NER) - Usage: 'text' POST parameter, 'lang' POST parameter for Spacy model (lang=en by default)"
	params = {}

	if request.method == 'GET':
		return jsonify(data)

	params = request.form # postdata

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not 'text' in params:
		data['error'] = '[text] parameter not found'
		return jsonify(data)

	if not 'lang' in params:
		lang = 'en'
	else:
		lang = params['lang']

	nlp = spacy.load( lang )
	doc = nlp( params['text'] )
	data['entities']  = {}
	
	counters  = {}
	for ent in doc.ents:
		if not ent.label_ in data['entities']:
			data['entities'][ent.label_] = dict()
			counters[ent.label_] = 0
		else:
			counters[ent.label_] += 1
	
		data['entities'][ ent.label_ ][ counters[ent.label_] ] =  ent.text
		#data['entities'][ent.label_].add( ent.text )

	return jsonify(data)


@app.route("/gensim/summarize", methods=['GET', 'POST'])
def gensim_summarize():
	from gensim.summarization.summarizer import summarize
	data = dict(default_data)
	data['message'] = "Summarize long text - Usage: 'text' POST parameter"
	params = {}

	if request.method == 'GET':
		return jsonify(data)

	params = request.form # postdata

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not 'text' in params:
		data['error'] = '[text] parameter not found'
		return jsonify(data)

	if not 'word_count' in params:
		word_count = None
	else:
		word_count = int(params['word_count'])
	
	data['summarize'] = summarize( text=params['text'], word_count=word_count )

	return jsonify(data)


@app.route("/polyglot/neighbours", methods=['GET'])
def embeddings():
	from polyglot.text import Word
	data = dict(default_data)
	data['message'] = "Neighbours (Embeddings) - Find neighbors of word API - Parameters: 'word', 'lang' language (default: en)"
	params = {}
	
	params['word']= request.args.get('word')
	params['lang']= request.args.get('lang')

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['word']:
		data['error'] = '[word] parameter not found'
		return jsonify(data)

	if not params['lang']:
		# data['error'] = '[lang] parameter not found'
		# return jsonify(data)
		params['lang'] = 'en'

	data['neighbours'] = {}

	try:
		word = Word(params['word'], language=params['lang'])
	except KeyError:
		data['error'] = 'ERROR: word not found'
		return jsonify(data)

	if not word:
		data['error'] = 'word not found'
		return jsonify(data)

	data['neighbours'] = word.neighbors

	return jsonify(data)


@app.route("/langid", methods=['GET', 'POST'])
def language():
	import langid
	data = dict(default_data)
	data['message'] = "Language Detection API - Usage: 'text' GET/POST parameter "
	data['langid'] = {}
	params = {}
	

	if request.method == 'GET':
		params['text'] = request.args.get('text')
	elif request.method == 'POST':
		params = request.form # postdata
	else:
		data['error'] = 'Invalid request method'
		return jsonify(data)

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['text']:
		data['error'] = '[text] parameter not found'
		return jsonify(data)

	lang_data = langid.classify( params['text'] ) 
	data['langid']['language'] = lang_data[0]
	data['langid']['score'] = lang_data[1]

	data['message'] = "Detected Language: " + data['langid']['language']

	return jsonify(data)


@app.route("/polyglot/sentiment", methods=['GET','POST'])
def polyglot_sentiment():
	from polyglot.text import Text

	data = dict(default_data)
	data['message'] = "Sentiment Analysis API - POST only"
	data['sentiment'] = {}

	params = request.form # postdata

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['text']:
		data['error'] = 'Text parameter not found'
		return jsonify(data)

	if not 'lang' in params:
		language = 'en' # default language
	else:
		language = params['lang']


	polyglot_text = Text(params['text'], hint_language_code=language)
	data['sentiment'] = polyglot_text.polarity
	return jsonify(data)


@app.route("/polyglot/entities", methods=['GET','POST'])
def polyglot_entities():
	from polyglot.text import Text

	data = dict(default_data)
	data['message'] = "Entity Extraction and Sentiment Analysis API- POST only"
	data['polyglot'] = {}

	params = request.form # postdata

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['text']:
		data['error'] = 'Text parameter not found'
		return jsonify(data)

	if not 'lang' in params:
		language = 'en' # default language
	else:
		language = params['lang']
	
	
	polyglot_text = Text(params['text'], hint_language_code=language)

	data['polyglot']['entities'] = polyglot_text.entities
	try:
		data['polyglot']['sentiment'] = polyglot_text.polarity
	except:
		data['polyglot']['sentiment'] = 0
	# if len(params['text']) > 100:
	# 	data['polyglot']['sentiment'] = polyglot_text.polarity
	# else:
	# 	data['polyglot']['sentiment'] = 0


	

	data['polyglot']['type_entities']  = {}
	if polyglot_text.entities:
		counter = 0
		for entity in polyglot_text.entities:
			data['polyglot']['type_entities'][counter] = {}
			data['polyglot']['type_entities'][counter][entity.tag] = {}
			data['polyglot']['type_entities'][counter][entity.tag] = entity
			counter += 1

	return jsonify(data)


# https://github.com/buriy/python-readability
@app.route("/readability", methods=['GET', 'POST'])
def readability():
	import requests
	from readability import Document	
	from bs4 import BeautifulSoup 

	data = dict(default_data)
	data['message'] = "Article Extraction by Readability"
	data['params'] = {}
	data['error'] = ''
	data['readability'] = {}

	if request.method == 'GET':
		data['params']['url'] = request.args.get('url')
		if not data['params']['url']:
			data['error'] = '[url] parameter not found'
			return jsonify(data)

		response = requests.get( data['params']['url'] )
		doc = Document(response.text)

	elif request.method == 'POST':
		params = request.form # postdata

		if not params:
			data['error'] = 'Missing parameters'
			return jsonify(data)

		if not params['html']:
			data['error'] = 'html parameter not found'
			return jsonify(data)
	
		doc = Document( params['html'] )
	
	data['readability']['title'] = doc.title()
	data['readability']['short_title'] = doc.short_title()
	#data['readability']['content'] = doc.content()
	data['readability']['article_html'] = doc.summary( html_partial=True )

	soup = BeautifulSoup( data['readability']['article_html'] ) 
	data['readability']['text'] =  soup.get_text() 

	return jsonify(data)


@app.route("/afinn", methods=['GET', 'POST'])
def afinn_sentiment():
	data = dict(default_data)
	data['message'] = "Sentiment Analysis by afinn"

	from afinn import Afinn
	

	data['afinn'] = 0
	#data['afinn'] = afinn.score('This is utterly excellent!')

	params = request.form # postdata

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['text']:
		data['error'] = 'Text parameter not found'
		return jsonify(data)

	if not 'lang' in params:
		language = 'en' # default language
	else:
		language = params['lang']

	afinn = Afinn( language=language )
	data['afinn'] = afinn.score( params['text'] )

	return jsonify(data)


@app.route("/newspaper", methods=['GET', 'POST'])
def newspaper():
	from newspaper import Article
	import langid

	data = dict(default_data)
	data['message'] = "Article Extraction by Newspaper, and Language Detection by Langid"
	data['params'] = {}
	data['error'] = ''
	data['newspaper'] = {}
	data['langid'] = {}

	if request.method == 'GET':
		data['params']['url'] = request.args.get('url')
		if not data['params']['url']:
			data['error'] = '[url] parameter not found'
			return jsonify(data)

		article = Article(url=data['params']['url'],keep_article_html=True)
		article.download()
	elif request.method == 'POST':
		params = request.form # postdata

		if not params:
			data['error'] = 'Missing parameters'
			return jsonify(data)

		if not params['html']:
			data['error'] = 'html parameter not found'
			return jsonify(data)
	
		article = Article(url='',keep_article_html=True)
		article.set_html( params['html'] )
	else:
		data['error'] = 'Invalid request method'
		return jsonify(data)
	
	
	# Parse html 
	article.parse()

	data['newspaper']['article_html'] = article.article_html
	data['newspaper']['text'] = article.text
	data['newspaper']['title'] = article.title
	data['newspaper']['authors'] = article.authors
	data['newspaper']['top_image'] = article.top_image
	data['newspaper']['canonical_url'] = article.canonical_link
	data['newspaper']['meta_data'] = article.meta_data

	data['newspaper']['meta_description'] = article.meta_description
	if article.publish_date:
		data['newspaper']['publish_date'] = '{0:%Y-%m-%d %H:%M:%S}'.format(article.publish_date)

	data['newspaper']['source_url'] = article.source_url
	data['newspaper']['meta_lang'] = article.meta_lang

	#Detect language
	if len(article.text)  > 100:
		lang_data = langid.classify( article.title + ' ' + article.text ) 
		data['langid']['language'] = lang_data[0]
		data['langid']['score'] = lang_data[1]

	return jsonify(data)


@app.route("/fasttext", methods=['GET', 'POST'])
def fasttext():
	import fasttext

	data = dict(default_data)
	data['message'] = "FastText Language Detection -  Parameters: 'text', 'predictions' number of predictions to return (default: 1)"
	data['fasttext'] = {}
	params = {}
	

	if request.method == 'GET':
		params['text'] = request.args.get('text')
		params['predictions'] = request.args.get('predictions')
	elif request.method == 'POST':
		params = request.form # postdata
	else:
		data['error'] = 'Invalid request method'
		return jsonify(data)

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['text']:
		data['error'] = '[text] parameter not found'
		return jsonify(data)

	if not params['predictions']:
		params['predictions'] = 1
	else:
		try:
			if int(params['predictions']) < 1:
				data['error'] = '[predictions] parameter cannot be less than 1'
				return jsonify(data)
		except (TypeError, ValueError):
			data['error'] = '[predictions] parameter must be an integer'
			return jsonify(data)

	try:
		ft = fasttext.load_model("lid.176.bin")
	except ValueError:
		data['error'] = 'lid.176.bin model not found'
		return jsonify(data)
	
	if not ft:
		data['error'] = 'FastText model not initialised'
		return jsonify(data)

	lang_data = ft.predict( params['text'], k=int(params['predictions']) )
	
	langs = map(lambda x: x[9:], lang_data[0]) # remove __label__ prefix
	scores = lang_data[1]
	results = list(zip(langs, scores))

	data['fasttext']['language'] = results[0][0]
	data['fasttext']['score'] = results[0][1]
	data['fasttext']['results'] = results

	data['message'] = "Detected Language: " + data['fasttext']['language']

	return jsonify(data)

@app.route("/image/yolo", methods=['GET', 'POST'])
def yolo():
	# Import your Libraries 
	import torch
	from torchvision import transforms
	from PIL import Image, ImageDraw
	from pathlib import Path
	from ultralytics import YOLO
	import pandas as pd
	import numpy as np
	from pathlib import Path
	import requests
	from io import BytesIO
	from keras.preprocessing.image import img_to_array
	from sklearn import preprocessing
	import json

	intermediate_features = []

	def hook_fn(module, input, output):
		intermediate_features.append(output)

	def extract_features(intermediate_features, model, img, layer_index=20):##Choose the layer that fit your application
		hook = model.model.model[layer_index].register_forward_hook(hook_fn)
		print(hook)
		with torch.no_grad():
			model(img)
		hook.remove()
		return intermediate_features[0]  # Access the first element of the list
	def loadImage(url, size = 640):
		try:
			response = requests.get(url)
			response.raise_for_status()
		except requests.exceptions.RequestException as err:
			data['error'] =  err.strerror
			return jsonify(data)
	
		img_bytes = BytesIO(response.content)
		img = Image.open(img_bytes)
		img = img.convert('RGB')
		img = img.resize((size,size), Image.NEAREST)
		img = img_to_array(img)
		return img
	
	data = dict(default_data)
	data['message'] = "Yolo -  Parameters: 'iiif_image_url', 'labels' a list of valid labels for object detection (default: face)"
	data['yolo'] = {}
	params = {}
	

	if request.method == 'GET':
		params['iiif_image_url'] = request.args.get('iiif_image_url')
		params['labels'] = request.args.getlist('labels')
	elif request.method == 'POST':
		params = request.form # postdata
	else:
		data['error'] = 'Invalid request method'
		return jsonify(data)

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['iiif_image_url']:
		data['error'] = '[iiif_image_url] parameter not found'
		return jsonify(data)

	if not params['labels']:
		params['labels'] = ['face']

	try:
		model = YOLO('models/yolo/'+ app.config["YOLO_MODEL_NAME"]) 
	except ValueError:
		data['error'] = 'models/yolo/'+ app.config["YOLO_MODEL_NAME"] + ' not found'
		return jsonify(data)
	
	if not model:
		data['error'] = 'yolov8 model not initialized'
		return jsonify(data)

	img = loadImage(params['iiif_image_url'], 640)
	data['yolo']['objects']  = []
	data['yolo']['modelinfo']  = {}
	object_detect_results = model(img, conf=0.1) 
	# model.names gives me the classes.
	# We don't know if the user set tge obb model or the regular one, so we will have to iterate over both options, bbox and obb
	for object_detect_result in object_detect_results:
		if hasattr(object_detect_result, "obb") and object_detect_result.obb is not None:  # Access the .obb attribute instead of .boxes
			print('An obb model')
			data['yolo']['objects'] = json.loads(object_detect_result.tojson(True))
		elif hasattr(object_detect_result, "boxes") and object_detect_result.boxes is not None:
			print('Not an obb model')
			if type(object_detect_result) != 'NoneType':
				data['yolo']['objects'] = json.loads(object_detect_result.tojson(True))
		else:
			data['yolo']['objects'] = []
		
	data['yolo']['modelinfo'] = {'train_args': model.ckpt["train_args"], 'date': model.ckpt["date"], 'version': model.ckpt["version"]} 
	
	# features =  extract_features(intermediate_features=intermediate_features,model=model, img = img) // More advanced. Step 2
	# The embed method is pretty new. 
	vector = model.embed(img, verbose=False)[0]
	print(vector.shape[0])
	# Vector size for this layer (i think by default it will be numlayers - 2 so 20) is 576
	# array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample
	# This "should" return a Unit Vector so we can use "dot_product" in Solr
	X_l1 = preprocessing.normalize([vector.detach().tolist()], norm='l1')
	# see https://nightlies.apache.org/solr/draft-guides/solr-reference-guide-antora/solr/10_0/query-guide/dense-vector-search.html
	data['yolo']['vector'] = X_l1[0].tolist()
	data['message'] = 'done'

	return jsonify(data)
@app.route("/image/mobilenet", methods=['GET', 'POST'])
def mobilenet():
	# Import your Libraries 
	from PIL import Image
	from pathlib import Path
	import pandas as pd
	import numpy as np
	from pathlib import Path
	import requests
	from io import BytesIO
	from sklearn import preprocessing
	import mediapipe as mp
	from mediapipe.tasks import python
	from mediapipe.tasks.python import vision

	intermediate_features = []

	def loadImage(url, size = 480):
		try:
			response = requests.get(url)
			response.raise_for_status()
		except requests.exceptions.RequestException as err:
			data['error'] =  err.strerror
			return jsonify(data)
	
		img_bytes = BytesIO(response.content)
		img = Image.open(img_bytes)
		img = img.convert('RGB')
		img.thumbnail((size,size), Image.NEAREST)
		# Media pipe uses a different format than YOLO, img here is PIL
		img = np.asarray(img)
		return img
	
	data = dict(default_data)
	data['message'] = "mobilenet -  Parameters: 'iiif_image_url"
	data['mobilenet'] = {}
	data['efficientdet'] = {}
	params = {}
	objects = []
	

	if request.method == 'GET':
		params['iiif_image_url'] = request.args.get('iiif_image_url')
	elif request.method == 'POST':
		params = request.form # postdata
	else:
		data['error'] = 'Invalid request method'
		return jsonify(data)

	if not params:
		data['error'] = 'Missing parameters'
		return jsonify(data)

	if not params['iiif_image_url']:
		data['error'] = '[iiif_image_url] parameter not found'
		return jsonify(data)
	try:
		# Create options for Image Embedder
		base_options_embedder = python.BaseOptions(model_asset_path='models/mobilenet/' + app.config["MOBILENET_MODEL_NAME"])
		base_options_detected = python.BaseOptions(model_asset_path='models/mobilenet/' + app.config["MOBILENET_DETECT_MODEL_NAME"])
		l2_normalize = True #@param {type:"boolean"}
		quantize = True #@param {type:"boolean"}
		options_embedder = vision.ImageEmbedderOptions(base_options=base_options_embedder, l2_normalize=l2_normalize, quantize=quantize)
		options_detector = vision.ObjectDetectorOptions(base_options=base_options_detected, score_threshold=0.5)




# Create Image Embedder
		with vision.ImageEmbedder.create_from_options(options_embedder) as embedder:

  		# Format images for MediaPipe
			img = loadImage(params['iiif_image_url'], 640)
			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
			embedding_result = embedder.embed(image)
			with vision.ObjectDetector.create_from_options(options_detector) as detector:
				detector_results = detector.detect(image)

	except ValueError:
		data['error'] = 'models/mobilenet/' + app.config["MOBILENET_MODEL_NAME"] + ' not found'
		return jsonify(data)

	if not detector_results.detections:
		objects = []
	else:
		# make the coordinates percentage based.
		for ml_result_index in range(len(detector_results.detections)):
			detector_results.detections[ml_result_index].bounding_box.origin_x = detector_results.detections[ml_result_index].bounding_box.origin_x/image.width
			detector_results.detections[ml_result_index].bounding_box.origin_y = detector_results.detections[ml_result_index].bounding_box.origin_y/image.height
			detector_results.detections[ml_result_index].bounding_box.width = detector_results.detections[ml_result_index].bounding_box.width/image.width
			detector_results.detections[ml_result_index].bounding_box.height = detector_results.detections[ml_result_index].bounding_box.width/image.height
		objects = detector_results.detections
	vector = embedding_result.embeddings[0].embedding
	# print(embedding_result.embeddings[0].embedding.shape[0])
	# Vector size for this layer (inumlayers - 1) is 1024
	# This "should" return a Unit Vector so we can use "dot_product" in Solr
	X_l1 = preprocessing.normalize([vector], norm='l1')
	# see https://nightlies.apache.org/solr/draft-guides/solr-reference-guide-antora/solr/10_0/query-guide/dense-vector-search.html
	data['mobilenet']['vector'] = X_l1[0].tolist()
	
	data['mobilenet']['objects'] = objects
	data['message'] = 'done'
	return jsonify(data)

# @app.route("/tester", methods=['GET', 'POST'])
# def tester():
# 	return render_template('form.html')

app.run(host='0.0.0.0', port=6400, debug=False)


