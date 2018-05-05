import prepare_data as data
import pandas as pd

def print_min_max(df, message):
	print(message)
	print(df.max(axis=0))
	print(df.min(axis=0))
	print()

print('VAD scores should be in range [1,9], Basic Emotions in range [1,5].')
print()
print()
print_min_max(df=data.get_english(),
			  message='"English data set (anew 2010 and stevenson 2007)"')
print_min_max(df=data.get_spanish(),
			  message="Spanish data set (Redondo 2007 and Ferre 2016")
print_min_max(df=data.get_polish(),
			  message="Polish data set (Imibir 2016 and Wierzba 2015")
print_min_max(df=data.get_german(), 
			  message='German data set (Schmidtke 2014 and Briesemeister 2011)')
print_min_max(df=data.load_hinojosa16(), 
			  message='Second Spanish data set (Hinojosa 2016a+b)')
print_min_max(df=data.load_stadthagen16(),
			  message='Stadthagen-Gonzales 2016')
print_min_max(df=data.load_kanske10(),
			  message='Kanske & Kotz, 2010.')
print_min_max(df=data.load_guasch15()),
			  message='Guasch et al., 2015'