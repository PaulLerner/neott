
from .basis_funcs import *;

def saveImg(path, final):
	path = path.replace(" ","_");
	if final:
		plt.savefig(path,dpi=80);
	else:
		plt.savefig(path)
	print("saved img to",path);


def basicPosDomainGraph(fullDF, final = False):
	print("running basicPosDomainGraph")
	outLoc = imgOutLoc + "/" + "basicVis"
	pathlib.Path(outLoc).mkdir(exist_ok=True,parents=True);
	my_cmap = cm.get_cmap('jet')
	nameSources = (("ft","FranceTerme Dataset"),("wiktionnaire","Wiktionnaire Dataset"),("Combined","Combined Dataset"))
	for sourceIndex, (techNameSource, formalNameSource) in enumerate(nameSources):
		print("doing for",formalNameSource)
		if techNameSource == "Combined":
			df = fullDF
		else:
			df = fullDF[fullDF.source==techNameSource]
		fig, axes = plt.subplots((2 if techNameSource != "Combined" else 3),1,figsize=(14, (10 if techNameSource != "Combined" else 15)), dpi=80);
		axNum = 0
		for dataIndex, ((techNameData, formalNameData)) in enumerate((("basic_pos","Part of Speech"),("daille_type","Daille Type"),("Domain","Domain"))):
			if techNameSource != "Combined" and techNameData == "daille_type": continue;
			srs = df.groupby(techNameData)[techNameData].count().sort_values(ascending=False)[:20]
			print("srs",srs)
			my_norm = Normalize(vmin=0, vmax=max(srs.values))
			color = my_cmap(my_norm(srs.values))
			ax = axes[axNum]
			axNum += 1;
			ax.bar(np.arange(len(srs)),srs,tick_label=srs.index, color=color)
			xticks = [x.replace(" ","\n") for x in srs.index]
			xticks = [x if x != "None" else "(From FT)" for x in xticks]
			ax.set_title("Distribution of " + formalNameData, fontsize=20)
			fs = 14
			if formalNameData == "Domain":
				ax.set_xticklabels(xticks,rotation=90, fontsize=fs)
			else:
				ax.set_xticklabels(xticks, fontsize=fs)
			for index, val in enumerate(srs):
				ax.annotate(
					round(val/len(df),2), xy=(index, val+ax.get_ylim()[1]*0.02),ha='center', fontsize=fs
				)
			ax.set_ylim(0,ax.get_ylim()[1]*1.05)
		plt.suptitle("Domain and POS Distribution for " + formalNameSource, fontsize=22)
		plt.tight_layout()
		imgLoc = outLoc + "/" + formalNameSource + ".png"
		saveImg(imgLoc, final)


def visualizeLosses():
	outLoc = imgOutLoc + "/" + "trainingVis"
	pathlib.Path(outLoc).mkdir(exist_ok=True, parents=True);
	import ast
	fig, axes = plt.subplots(3, 1, figsize=(12, 9))
	print("visualizing loss")
	for modelIndex, modelName in enumerate(baseModelNames):
		df = pickLoad('/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/_intermediateEvaluations_' + modelName + '.pickle')[modelName]
		print("working with",df)
		epochs = [round(x,2) for x in df.epoch]
		ax = axes[modelIndex]
		axPerp = ax.twinx()
		for name, _ in queryNamesAndConsts:
			levSimTop1 = df[name + "_levSimTop1"].apply(lambda x: x.mean())
			levSimTop5 = df[name + "_levSimTop5"].apply(lambda x: x.mean())
			# levSimTop1 = df[name+"_levSimTop1"].apply(lambda x: np.mean(ast.literal_eval(re.sub("(?<!\[)\s+",",",x))))
			# levSimTop5 = df[name + "_levSimTop5"].apply(lambda x: np.mean(ast.literal_eval(re.sub("(?<!\[)\s+", ",", x))))
			ax.plot(epochs, levSimTop1,label=name + "_levSimTop1");
			ax.plot(epochs, levSimTop5, label=name + "_levSimTop5", linestyle='dashed');
		axPerp.plot(epochs, df.eval_loss,label="Eval Loss",linestyle="dashdot")
		lines, labels = ax.get_legend_handles_labels()
		lines2, labels2 = axPerp.get_legend_handles_labels()
		ax.legend(lines + lines2, labels + labels2, ncol=2,loc="upper right")
		ax.set_ylabel("Mean Cross-Entropy")
		ax.set_xlabel("Epoch")
		ax.set_title(modelName)
	plt.suptitle("Loss evolution over epochs for fine-tuning of baseline models",fontsize=20)
	plt.tight_layout()
	plt.savefig(outLoc + "/fineTuneBaseline.png")


def visualizeCharLosses():
	np.random.seed(69)
	for name, losses in (("No Camembert Layer",lossesCam), ("Camembert Layer",lossesNoCam)):
		validLoss = [x["valid_loss"] for x in losses]
		trainLoss = [x["train_loss"] for x in losses]
		epochs = [x["epoch"] for x in losses]
		# c = generateRandomColor()
		plt.scatter(epochs, trainLoss,alpha=0.4,s=0.5)
		plt.plot(epochs, validLoss, label="Train (Scatter)/Valid Loss " + name)#, alpha=0.4)
		#, alpha=0.4)#, color=c)
	plt.legend()
	plt.title("Mean Cross-entropy for Batch")
	plt.ylabel("Cross-Entropy Loss")
	plt.xlabel("Epoch")
	plt.show()
