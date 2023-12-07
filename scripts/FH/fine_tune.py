from functools import reduce
from .basis_funcs import *
from .base_models import evaluatePredictions, getWordsAndProbs, get_top_predictions_for_torch_output, apply_models
import pathlib
from torch.nn.functional import softmax
'''
Some helpful commands for Felix that he regularly forgets

GPU interactice sesh:
srun -N 1 --gres=gpu:1 --pty bash --nodelist=n51 
--mem-per-gpu=16G
p3
import torch
torch.cuda.is_available()
'''

import torch
from datasets import load_metric, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer, TrainerCallback
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = "1";

def correctEpochs(dList, seenEpochs):
	print("correctin those epochs wut",seenEpochs)
	corredOnes = 0
	for d in dList:
		if not d["epochCorrected"]:
			print("oh yeah",corredOnes)
			corredOnes += 1
			d["epoch"] += seenEpochs
			d["epochCorrected"] = True
			print("truthee")
		else:
			print("already corrected ya")

def fine_tune_models(df, modelNames = [], markovModel = None, quickie=False, batch_size = 5, total_ret = 15, loadFinetunedModels = False, suffix = "",large = False, applyToo = False):
	'''
	TODO@efeh: finetune only for one corpus, valid on other, or further strategy for train/valid splitting
	:param df:
	:param modelNames:
	:param quickie:
	:param batch_size:
	:return:
	'''

	if suffix == "":
		suffix = getSuffix(quickie, loadFinetunedModels)
	if len(modelNames) == 1:
		suffix += "_"+modelNames[0];
		# df = df[:quickieNumArts]

	print("fine tuning le models now")
	if len(modelNames) == 0:
		modelNames = list(baseModelNames)



	df = df.drop(columns = [x for x in df.columns if len([hfKw for hfKw in huggingfaceKeyWords if hfKw in x]) > 0])
	if not "subset" in df:
		print("should not be le case!")
		df["subset"] = df.apply(lambda _: ttSplitFunc(np.random.rand()), axis=1)
	print("starting out finetuning with",df);
	if applyToo:
		dfPrev, preAppParams = apply_models(df,markovModel=markovModel, top_n=5, total_ret=total_ret,
							  quickie=False, manualModels=modelsForFT, prefix="preFineTune_",
							retWholeDF=False,incTerm=True, testAndTrain=False, fakeRun= False)
		try:
			print("post initial application", dfPrev, list(dfPrev.columns))
		except:
			pass
		dumpVar(dfPrev, "dfPrev", "", suffix=suffix)

	shouldBroke = 0
	perplexities = {}
	trainingLossesByEpoch = {}
	intermediateEvaluations = {model: [] for model in modelNames}
	for modelName in modelNames:
	# for modelName, tokenizer, model in modelsForFT:
	# 	modelsForFT = [(name, loadTokenizerAndModel(name, loadFinetunedModels, large=large)) for name in modelNames]
	# 	modelsForFT = [(tup[0], tup[1][0], tup[1][1]) for tup in modelsForFT]
		tokenizer, model = loadTokenizerAndModel(modelName, loadFinetunedModels, large=large)
		if not torch.cuda.is_available():
			raise Exception("No GPU available hoss")
		model = model.to("cuda:0")

		allTrains = []
		allTests = []
		queryStrs = []

		for queryName, const in queryNamesAndConsts:
			queryStr = queryName + "_query"
			queryStrs.append(queryStr);
			dfThis = df[["term","subset","defn", modelName+"_termLen"]]#queryStr
			if len(dfThis) == 0: continue;
			dfThis["queryString"+"_unmasked"] = dfThis.apply(lambda row: const.format(maskStr = row.term, defn = row.defn),axis=1)
			dfThis["queryString"] = dfThis.apply(lambda row: getQuery(const, row.defn, modelName, row[modelName+"_termLen"]), axis=1)

			train, valid = dfThis[dfThis.subset=="train"], dfThis[dfThis.subset=="valid"]
			allTrains.append(train)
			allTests.append(valid);

		train = pd.concat(allTrains)
		valid = pd.concat(allTests)
		print("trainy train",train);
		dumpVar(train,"trainyTrain","hack",modelName)

		dsTrain = Dataset.from_pandas(train)
		dsTest = Dataset.from_pandas(valid)
		dsDict = DatasetDict({"train":dsTrain,"valid":dsTest})

		def shouldBreak(myl):
			try:
				print("should I break it up", myl[-3:], sorted(myl[-3:], reverse=True), (myl[0] - myl[-1]) / myl[0])
				if (myl[-1] - myl[0]) / myl[-1] > 0.05:
					# loss at end of epoch was significantly greater than beginning of epoch; that's no good at all!
					return True;
				if (max(myl) - min(myl)) / max(myl) < 0.05:
					# no loss fluctuation at all
					return True
				return False
			except:
				return False

		def tokenize_function(examples):
			result = tokenizer(examples["queryString"], padding='max_length',)
			unmaskedResult = tokenizer(examples["queryString"+"_unmasked"], padding='max_length',);
			# print("result",result,"unmas",unmaskedResult)
			result["labels"] = [unmaskedResult["input_ids"][i] for i in range(len(unmaskedResult["input_ids"]))]
			if tokenizer.is_fast:
				result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
			return result

		removeColumns = ["__index_level_0__","defn","subset", modelName+"_termLen","queryString","queryString_unmasked"]

		tokenized_datasets = dsDict.map(
			tokenize_function, batched=True, remove_columns=removeColumns
		)

		logging_steps = len(tokenized_datasets["train"]) // (batch_size*5)
		print("slogging steps",logging_steps, batch_size,len(tokenized_datasets["train"]))

		largeSuffix = ""
		if large:
			largeSuffix = "_large"
		outDir = dumpPath + "/models/" + modelName + largeSuffix + "-finetuned-tech" + ("_quickie" if quickie else "")
		pathlib.Path(outDir).mkdir(exist_ok=True,parents=True)

		initialEpochs = 1
		if quickie:
			initialEpochs = 1
		training_args = TrainingArguments(
			output_dir=outDir,
			overwrite_output_dir=True,
			evaluation_strategy="steps",
			eval_steps=logging_steps,
			learning_rate=2e-5,
			save_steps = (5 if quickie else 5000),
			weight_decay=0.01,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			fp16=True,
			num_train_epochs=initialEpochs,
			logging_steps=logging_steps,
			logging_dir=outDir+"/log",
			resume_from_checkpoint=True
		)

		class IntermediateEvalCallback(TrainerCallback):
			def on_evaluate(self, args, state, control, model=None, **kwargs):
				model.eval()
				epoch = state.epoch
				if epoch is None:
					epoch = 0;
				interDF, _ = apply_models(df, markovModel=markovModel, top_n=5, total_ret=total_ret,
							 quickie=False, manualModels=[(modelName, tokenizer, model)], prefix="",
							 retWholeDF=False, incTerm=True, testAndTrain=False, on_evaluate_take_perc=0.05)
				d = {"epoch":epoch}
				for query, _ in queryNamesAndConsts:
					d[query + "_" + "levSimTop1"] = interDF[modelName + "_" + "levSimTop1" + "_" + query + "_nonMarkov"].values
					d[query + "_" + "levSimTop5"] = interDF[modelName + "_" + "levSimTop5" + "_" + query + "_nonMarkov"].values
				print("obstetrics",kwargs["metrics"])
				d["eval_loss"] = kwargs["metrics"]["eval_loss"]
				d["epochCorrected"] = False
				intermediateEvaluations[modelName].append(d);

				'''
				d[prefix + modelName + "_" + "levSimTop1" + "_" + queryName + suffix] = levDist1
                    d[prefix + modelName+"_"+"groundTruthPosition" + "_" + queryName + suffix] =  positionOfGroundTruth
                    d[prefix + modelName+"_"+"topNPreds"+"_"+queryName + suffix] = top_n_preds
                    d[prefix + modelName + "_" + "levSimTop5" + "_" + queryName + suffix] = levDist5
                    d[prefix + modelName + "_" + "rejex" + "_" + queryName] = rejex
                '''
				model.train()

				#apply_models(df,markovModel = None, top_n = 5, total_ret = 100, quickie = False, modelNames = [], prefix = "", onlyCorpusName = "", onlyQueryName = "", onlyModelName = "", retWholeDF = True, incTerm = False, manualModels = [], loadFinetunedModels = False, testAndTrain = True, fakeRun = False):

		# calculate loss here


		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=tokenized_datasets["train"],
			eval_dataset=tokenized_datasets["valid"],
			callbacks=[IntermediateEvalCallback]
		)

		print("burr",dir(trainer.lr_scheduler))

		test_resultsPrev = trainer.evaluate()
		perplexityPrev = np.exp(test_resultsPrev["eval_loss"])
		print("perplexity Prev", perplexityPrev)
		# preParams = list(model.parameters())[0].detach().cpu().numpy()
		losses = []
		seenEpochs = 0

		# epochsLastRound = initialEpochs
		while True:
			trainer.train()
			myl = [x for x in trainer.state.log_history if "eval_loss" in x]
			myl.sort(key=lambda x: x["epoch"]);
			myl = [{"epoch":+x["epoch"]+seenEpochs, "eval_loss":x["eval_loss"]} for x in myl]
			print("mylin", );
			losses.extend(myl)
			myl = [x["eval_loss"] for x in myl];

			print("hoober",intermediateEvaluations[modelName])
			correctEpochs(intermediateEvaluations[modelName], seenEpochs);

			if shouldBreak(myl):
				shouldBroke += 1
				if shouldBroke == 3:
					print("progress has stopped; breaking")
					break;
			else:
				shouldBroke = 0;
				print("Nah, we ain't breaking up shit")

			epochAdd = 1
			seenEpochs += 1

			# epochsLastRound = epochAdd
			print("durr",dir(trainer),dir(trainer.lr_scheduler));
			trainer.num_train_epochs = epochAdd
			print("training again! gadzuks",losses)


		print("'valu8n")
		test_resultsPost = trainer.evaluate()
		perplexityPost = np.exp(test_resultsPost["eval_loss"])
		correctEpochs(intermediateEvaluations[modelName], seenEpochs);
		print("perplexity post",perplexityPost)
		trainingLossByEpoch = pd.DataFrame([x for x in trainer.state.log_history if "eval_loss" in x])
		print("hoy bloy ans bevirance",trainer.state.log_history)
		perplexities[modelName] = (perplexityPrev,perplexityPost)
		trainingLossesByEpoch[modelName] = trainingLossByEpoch
		intermediateEvaluations[modelName] = pd.DataFrame(intermediateEvaluations[modelName])

	if applyToo:
		dfFineTune, postAppParams = apply_models(df, markovModel=markovModel, top_n=5, total_ret=total_ret,
								  quickie=False, manualModels=modelsForFT, prefix="postFineTune_",
								retWholeDF=False, incTerm=True,testAndTrain=False)
		dumpVar(dfFineTune, "dfFineTune", "", suffix=suffix)

		print("post finetuning", dfFineTune, list(dfFineTune.columns))
		print("final samesies?", np.all(preAppParams[modelName] == postAppParams[modelName]), np.any(preAppParams[modelName] == postAppParams[modelName]));
		df = df.merge(dfPrev, on=["repn", "term", "source"]);
		df = df.merge(dfFineTune, on=["repn", "term", "source"]);
		dumpVar(df, "dfFineTuneCombined", "", suffix=suffix)
	for modelName in modelNames:
		prevPerplex, postPerplex = perplexities[modelName]
		print("perplexity differences",prevPerplex, postPerplex)
		print("lossinger",trainingLossesByEpoch)

	print("fine tuned",df)

	dumpVar(perplexities,"perplexities","",suffix)
	dumpVar(trainingLossesByEpoch,"trainingLossesByEpoch","",suffix)
	dumpVar(intermediateEvaluations, "intermediateEvaluations","",suffix)




'''
MT-5, m-bart (encoder-decoder?)
is camembert trained with whole words or subwords? flaubert pareil? xlmRob?
consider options with only 1,2,3 masks
daille-type as token in m-bart
'''