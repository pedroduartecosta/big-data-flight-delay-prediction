import pandas as pd
import numpy as np 
import matplotlib as plt
import matplotlib.pyplot as pyplot
import seaborn as sns
import ast, json
import glob


#--------------------------------------CORRELATION MATRIX--------------------------------------#
def matrixCorr(df):
	corr = df.corr()

	# Generate a mask for the upper triangle
	#mask = np.zeros_like(corr, dtype=np.bool)
	#mask[np.triu_indices_from(mask)] = True

	# Set up the matplotlib figure
	f, ax = plt.pyplot.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, cmap=cmap, annot=True, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')
	pyplot.show()

def scatterPlot(df):
	sns.regplot(x=df["ArrDelay"], y=df["Distance"])
	pyplot.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
pd.set_option('display.max_columns', 19)
sns.set(style="white")
sns.set(color_codes=True)

df = pd.read_csv('../flightdelaypredictor/data/2008.csv', sep=",")

delCol = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 
		'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

df = df.drop(delCol, axis=1)

#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.corr(method ='pearson'))
#print(df.head())

#matrixCorr(df)
scatterPlot(df)