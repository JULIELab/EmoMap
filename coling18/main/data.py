import framework.prepare_data as data
import framework.models
from framework.reference_methods.aicyber import MLP_Ensemble
from keras.optimizers import Adam



VLIMIT=None

class Setting():
	def __init__(self, name, language, data):
		'''
		Args:
			name 			String (describing results_file)
			language 		String (will be used to load embeddings)
			data 			function
		'''
		self.name=name
		self.language=language.lower()
		self.load_data=data

SETTINGS=[
			Setting('English_ANEW_Stevenson',
					'english',
					data.get_english_anew),
			Setting('English_Warriner_Stevenson',
					'english',
					data.get_english_warriner),
			Setting('Spanish_Redondo',
					'spanish',
					data.get_spanish_redondo),
			Setting('Spanish_Hinojosa',
					'spanish',
					data.get_spanish_hinojosa),
			Setting('Spanish_Stadthagen',
					'spanish',
					data.get_spanish_stadthagen),
			Setting('German_BAWL',
					'german',
					data.get_german_bawl),
			Setting('Polish_NAWL',
					'polish',
					data.get_polish_nawl),
			Setting('Polish_Imbir',
					'polish',
					data.get_polish_imbir)
			]

IN_PAPER_NAMES={'English_ANEW_Stevenson':'en_1',
				'English_Warriner_Stevenson':'en_2',
				'Spanish_Redondo':'es_1',
				'Spanish_Hinojosa':'es_2',
				'Spanish_Stadthagen':'es_3',
				'German_BAWL':'de_1',
				'Polish_NAWL':'pl_1',
				'Polish_Imbir':'pl_2'}

SHORT_COLUMNS={'Valence':'Val',
				'Arousal':'Aro',
				'Dominance':'Dom',
				'Joy':'Joy',
				'Anger':'Ang',
				'Sadness':'Sad',
				'Fear':'Fea',
				'Disgust':'Dsg'}

LANGUAGES={setting.language for setting in SETTINGS}

def GET_EMBEDDINGS(language, vocab_limit=VLIMIT):
	return data.get_facebook_fasttext_wikipedia(language, vocab_limit)

VAD=['Valence', 'Arousal', 'Dominance']
VA=['Valence', 'Arousal']
BE5=['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']


DIRECTIONS={'vad2be':{'source':VAD, 'target':BE5},
			'be2vad':{'source':BE5, 'target':VAD}}

MY_MODEL={	'activation':'relu',
			'dropout_hidden':.2,
			'train_steps':10000,
			'batch_size':128,
			'optimizer':Adam(lr=1e-3),
			'hidden_layers':[128,128]}

KFOLD=10
