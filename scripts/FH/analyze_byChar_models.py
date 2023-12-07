from .basis_funcs import *
import Levenshtein as lev

from .daille import getMorphoType

try:
	allTerms = loadVar("wordSet", "", "_allWords");
	knownClassicalFixes = getKnownClassicalFixes()
except:
	print("mac problemas ehh")

def get_daille_type_row(term, df):
	'''
	A simplified version of daille-typing, when no morphological information is available (for constructed terms)
	see associated comments in daille@get_daille_type
	:param term:
	:param df:
	:return:
	'''
	if term in df.term:
		return df[df.term==term].iloc[0].daille_type

	morpho_tok, morphoType = getMorphoType(term)
	if type(morphoType) == type(None):
		if len(term) > 0 and term[-1] == "é" and term[:-1] + "er" in allTerms:
			retName = "conv"
		else:
			retName = "UNKNOWN"
	elif morpho_tok == "-":
		toks = term.split("-");
		fixes = {toks[0] + "-", "-" + toks[1]}
		if len(fixes.intersection(knownClassicalFixes)) > 0:
			retName = "neoClass"
		elif len(fixes.intersection(noNeoFixes)) > 0:
			retName = "affix";
		else:
			retName = "native"
	else:
		retName = "syntag"
	return retName

nameDict = {"byChar_daillePrediction_revise_hundo":"dai1Cam0",
				"byChar_revise_hundo":"dai0Cam0",
				"byChar_camemLayer_daillePrediction_revise_hundo":"dai1Cam1",
				"byChar_camemLayer_revise_hundo":"dai0Cam1"}

byCharRoots = sorted(list(nameDict.keys()))


def visualize_byChar_results(resDF = None, recalc = False):

	totalRoot = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/"
	try:
		os.listdir(totalRoot)
	except:
		totalRoot = "/Users/f002nb9/Documents/f002nb9/saclay/m1/stage/dump/final"

	fig, axes = plt.subplots(2,2, figsize=(14, 10), dpi=80);

	for runNameIndex, runName in enumerate(byCharRoots):
		print("run",runName)
		ax = axes[runNameIndex % 2][(runNameIndex//2)%2]

		if "f002nb9" in totalRoot:
			losses = pickLoad(totalRoot + "/losses.pickle");
		else:
			losses = scrapeLosses(totalRoot + "/../slurm/"+nameDict[runName].lower() + ".out",
								  totalRoot + "/"+runName + "/losses_rev.pickle", recalc = recalc);
		# else:
		# 	losses = pickLoad(totalRoot + "/" + runName + "/losses.pickle");

		allLosses = []
		prevHighWaterMark = 0
		for loss in losses:
			if loss["epoch"] > prevHighWaterMark:
				allLosses.append(loss)
				prevHighWaterMark = loss["epoch"]

		trainLoss = [float(x["train_loss"]) for x in allLosses]
		validLoss = [float(x["valid_loss"]) for x in allLosses]
		epoch = [float(x["epoch"]) for x in allLosses]
		tLoss = ax.plot(epoch, trainLoss,label="train loss")
		vLoss = ax.plot(epoch, validLoss, label="valid loss")
		ax.set_title(nameDict[runName]);

		print(epoch)
		print(trainLoss)
		print(validLoss);

		if not resDF is None:
			twinx = ax.twinx()
			thrinx = ax.twinx()#twinx.get_grid_helper().new_fixed_axis
			# twinx.axis["right"] = thrinx(loc="right", axes=twinx,
			# 									offset=(60, 0))
			# twinx.axis["right"].toggle(all=True)
			thrinx.spines['right'].set_position(('outward', 60))

			levCols = sorted([c for c in resDF.columns if ("lev_" in c and nameDict[runName] in c)],key = lambda x: int(x.split("lev_")[1].split("_")[0]))
			levMeans = resDF[levCols].mean().values

			dailleCols = sorted([c for c in resDF.columns if ("dailleCorrect_" in c and nameDict[runName] in c)], key=lambda x: int(x.split("dailleCorrect_")[1].split("_")[0]))
			dailleMeans = resDF[dailleCols].mean().values

			xVals = np.arange(len(levMeans))*5
			print("manus", levMeans, dailleMeans, xVals)

			levPlt = thrinx.plot(xVals,levMeans,label="lev mean",c="red")
			daillePlt = twinx.plot(xVals,dailleMeans,label="daille mean",c="green")

			lns = tLoss + vLoss + levPlt + daillePlt
			labs = [l.get_label() for l in lns]
			if runNameIndex == 0:
				ax.legend(lns, labs, loc=0)
			ax.set_xlabel("Epoch")
			ax.set_ylabel("CE-Loss")
			twinx.set_ylabel("Percentage of Proper Daille Classification")
			thrinx.set_ylabel("Mean Levenshtein Distance")

	print("plottinger")
	fig.tight_layout()
	print("taight")
	if "f00" in totalRoot:
		plt.show()
	else:
		plt.savefig(totalRoot + "/byCharIllustration.png")
	print("nu?");


def considerBaselineModels():
	useColsBaseline = ["term", "flaub_topNPreds_laDefDeTokEst_nonMarkov_preFineTune", "daille_type",
					   "flaub_topNPreds_laDefDeTokEst_nonMarkov_postFineTune",
					   "flaub_levSimTop1_laDefDeTokEst_preFineTune", "flaub_levSimTop1_laDefDeTokEst_postFineTune"]
	dfBaseline = pd.read_csv("/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/combined_dfBaseline.csv",
							 usecols=useColsBaseline) \
		.rename(columns={"flaub_topNPreds_laDefDeTokEst_nonMarkov_preFineTune": "pred_preTune",
						 "flaub_topNPreds_laDefDeTokEst_nonMarkov_postFineTune": "pred_postTune",
						 "flaub_levSimTop1_laDefDeTokEst_preFineTune": "pre_lev",
						 "flaub_levSimTop1_laDefDeTokEst_postFineTune": "post_lev"})
	dfBaseline["pred_preTune"] = dfBaseline.pred_preTune.apply(lambda x: x.split(";")[0])
	dfBaseline["pred_postTune"] = dfBaseline.pred_preTune.apply(lambda x: x.split(";")[0])

	dfBaseline["daille_pre"] = dfBaseline.apply(lambda row: get_daille_type_row(row.pred_preTune, dfBaseline), axis=1);
	dfBaseline["daille_post"] = dfBaseline.apply(lambda row: get_daille_type_row(row.pred_postTune, dfBaseline), axis=1);

	dfBaseline["dailleCorrect_pre"] = dfBaseline.apply(lambda row: row.daille_type == row.daille_pre, axis=1)
	dfBaseline["dailleCorrect_post"] = dfBaseline.apply(lambda row: row.daille_type == row.daille_post, axis=1)

	results = []
	d = {"model": "flaub", "Lev. Mean Pre-": dfBaseline.pre_lev.mean(),
		 "Lev. Mean Post-": dfBaseline.post_lev.mean(),
		 "Daille Acc. Pre-": dfBaseline[dfBaseline.daille_type != "UNKNOWN"].dailleCorrect_pre.mean(),
		 "Daille Acc. Post-": dfBaseline[dfBaseline.daille_type != "UNKNOWN"].dailleCorrect_post.mean()}
	results.append(d);
	return results

def combineAllByCharRuns(recalc = False):
	'''
	Combine all by-char runs, evaluate statistics on them
	:return:
	'''

	if not recalc:
		runningDf = pd.read_csv("/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/runningResults.csv")
		return visualize_byChar_results(runningDf, recalc = recalc)

	print("we be combinin")

	###For future visualization
	results = considerBaselineModels()


	runningDf = None
	allGoods = []
	for root in byCharRoots:
		name = nameDict[root];
		d = {"model": name}
		rootPath = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/" + root
		print("namer",name)
		postDfs = sorted([path for path in os.listdir(rootPath) if "allRes100" in path],key = lambda x: int(x.split("allRes100_")[1].split("_")[0]))
		for postDfPath in postDfs:
			epochNum = postDfPath.split("_")[1]


			postDf = pickLoad(rootPath + "/" + postDfPath)

			predCol = "pred_"+epochNum + "_" + name
			levCol = "lev_"+epochNum + "_" + name
			dailleColl = "daille_"+epochNum + "_" + name
			daille_corrColl = "dailleCorrect_" + epochNum + "_" + name

			postDf = postDf[["term","byChar__epoch_" + epochNum,"index","daille_type"]].rename(columns = {"byChar__epoch_"+epochNum:predCol})

			#get Levenshtein distance between prediction and gold
			postDf[levCol] = postDf.apply(lambda row: lev.distance(row.term, row[predCol]),axis=1);
			if False:
				df = pd.merge(on=["index","term","daille_type"],left=preTune,right=postTune,suffixes = ("_preTune","_postTune")).drop(columns = ["index"])
			# df = df.rename(columns = {c: c + "_" + name for c in df.columns if not c == "term"})

			#get daille prediction
			postDf[dailleColl] = postDf.apply(lambda row: get_daille_type_row(row[predCol], postDf), axis=1);
			postDf[daille_corrColl] = postDf.apply(lambda row: row.daille_type == row[dailleColl], axis=1)

			d["Lev. Mean " + epochNum] =  postDf[levCol].mean()
			d["Daille Acc. " + epochNum] = postDf[postDf.daille_type != "UNKNOWN"][daille_corrColl].mean()


			goods = postDf[(postDf[levCol] <= 2)][["term",predCol]]
			goods["model"] = name + "_" + epochNum
			print("goods",goods);
			allGoods.append(goods);

			if not runningDf is None:
				runningDf = runningDf.merge(postDf, on=["index","term","daille_type"])
			else:
				runningDf = postDf

		results.append(d)


	### The following two files are out-dated/incomplete
	resDF = pd.DataFrame(results)
	outPath = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/descriptiveResults.csv"
	print(resDF)
	print("find it at",outPath)
	resDF.to_csv(outPath,index=False);

	outPathGoods = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/goodResults.csv"
	goodsDf = pd.concat(allGoods);
	print("all goods",goodsDf, outPathGoods)
	goodsDf.to_csv(outPathGoods, index=False)

	### This is the important file
	outPathRunning = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/runningResults.csv"
	print("running by char at", runningDf, outPathRunning)
	runningDf.to_csv(outPathRunning, index=False)

	print(resDF);
	visualize_byChar_results(runningDf, recalc=recalc)


def scrapeLosses(logFilePath, lossOutPath, recalc = False):
	print("nose",logFilePath)
	print("face",lossOutPath)
	if not recalc and os.path.exists(lossOutPath):
		return pickLoad(lossOutPath)
	else:
		print("dont exist ne")
	logs = open(logFilePath,"r", encoding="ISO-8859-1").readlines()
	highWaterMark = 0
	losses = []
	first10 = False
	reset = False
	for line in logs:
		epochMatch = re.match("batch ([0-9]+) ([0-9]+)",line)
		if not epochMatch is None:
			epoch = int(epochMatch.group(1))
			batch = int(epochMatch.group(2))

			if epoch >= 10:
				first10 = True
			if epoch == 0 and first10:
				reset = True
			if not reset:
				continue;
			epoch = epoch + batch / 768
			if epoch < highWaterMark:
				continue
			elif epoch > 100:
				break
			else:
				highWaterMark = epoch
				continue;
		if not reset: continue;
		lossMatch = re.match("trainLoss ([0-9]+\.[0-9]+) walidLoss ([0-9]+\.[0-9]+)",line)
		if not lossMatch is None:
			trainLoss = float(lossMatch.group(1))
			validLoss = float(lossMatch.group(2))

			d = {"epoch":highWaterMark,"train_loss":trainLoss,"valid_loss":validLoss}
			losses.append(d)

	with open(lossOutPath,"wb") as fp:
		pickle.dump(losses, fp);
	return losses











'''
README which describes:
	• data structure
	• important functions in code
		• (particularly evaluation code)
'''

'''

'''