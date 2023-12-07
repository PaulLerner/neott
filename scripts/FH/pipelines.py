
from .basis_funcs import *;

from .process_xml import loadAllSources, load_articles_franceTerme, load_wiktionnaire_df
from .base_models import get_encoding_len, apply_models
symbols_before = dir()
from .analyze_base_models import *
symbols_after = dir()
from inspect import getmembers, isfunction
from .daille import analyzeClassicalFixDist, get_daille_type
from .derif import df_to_derif, incorporate_derif_etym
from .fine_tune import fine_tune_models
from .visualizations import *;
from .markov import initMarkov

anaFuncs = [("classicalFixes",analyzeClassicalFixDist)]



def base_models_pipeline(useCorpera = ("franceTerme","wiktionnaire"),load_from_store=False,quickie=False, modelNames = [], genDerif = False, type2Quickie=False, skipInit = False, evaluate = True, fineTune = False, bigQuickie=False, analysis = True, total_ret = 1000, doMarkov = True, loadFinetunedModels = False, large = True, evalName = "",startingChunk = 0,endingChunk = 100):

	print("here we are in le pipeline!")
	corpusName = "combined"
	suffix = ""
	if skipInit:
		if quickie:
			if bigQuickie:
				suffix = "_bigQuickie"
			else:
				suffix = "_quickie"
		df = loadVar("dfFinal","combined",suffix)
		df = clean_df(df);
		if doMarkov:
			markovModel = loadVar("markovModel", "terms", "")
		else:
			markovModel = None
		if type2Quickie:
			df = pd.concat([df[:quickieNumArts], df[-1*quickieNumArts:]]).reset_index()
		print("using",df.columns)
		if large:
			suffix += "_large"
		# df = get_daille_type(df, suffix=suffix)
		# # df, markovModel = initMarkov(df)
		# dumpVar(df, "dfFinal", corpusName, suffix)
	else:
		df, suffix = loadAllSources(load_from_store, quickie, type2Quickie=type2Quickie, bigQuickie=bigQuickie, startingChunk=startingChunk, endingChunk=endingChunk)
		df = incorporate_derif_etym(df,corpusName, genDerif=genDerif, suffix=suffix)
		df = get_encoding_len(df, modelNames=modelNames, suffix=suffix);
		df = get_daille_type(df, suffix=suffix)
		df, markovModel = initMarkov(df)
		dumpVar(df, "dfFinal", corpusName, suffix)
		dumpVar(markovModel, "markovModel", "terms", "")
		if not doMarkov:
			markovModel = None
	print("setty set",df, df.columns)
	# if quickie:
	# 	df = pd.concat((df[:10],df.sample(quickieNumArts)))
	if loadFinetunedModels:
		suffix += "_fineTunedMods"
	if type(markovModel) == type(None):
		suffix += "_noMarkov"
	print("aint no cols yet",df.columns)
	if fineTune:
		fine_tune_models(df, modelNames,markovModel = markovModel, quickie=quickie, total_ret=total_ret, loadFinetunedModels = loadFinetunedModels, suffix = suffix, large = large);
	if evaluate:
		dfEval, _ = apply_models(df, quickie=quickie, markovModel = markovModel,modelNames=modelNames, loadFinetunedModels = loadFinetunedModels, testAndTrain=False, incTerm=True,retWholeDF=False)
		dumpVar(dfEval, evalName,"",suffix)
	# dumpVar(df, "dfFinal", corpusName, suffix)
	if analysis:
		for anaFunc in anaFuncs:
			print("doing ana",anaFunc[0])
			anaFunc = anaFunc[1]
			anaFunc(df=df, suffix=suffix);

