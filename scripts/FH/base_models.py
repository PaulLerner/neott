import torch

from .basis_funcs import *
from scipy.special import softmax as softmax_cpu
from torch.nn.functional import softmax as softmax_gpu
import Levenshtein as lev
from .markov import MarkovModel

def getPredictions(term, definition, modelsList = [], total_ret = 10):
    for query, str_exm in queryNamesAndConsts:
        for modelName in baseModelNames:
            if len(modelsList) > 0 and not modelName in modelsList: continue;
            mod, tok = loadTokenizerAndModel(modelName)
            numToks = len(tok.encode(term))
            getQuery(str_exm, definition, modelName, numToks)
            preds, rejex, predsNonMarkov = applyModel(mod, tok, query, None, total_ret=total_ret, isCuda=False, softmax=softmax_cpu)
            print("preds for",term,query,modelName)
            for x in preds:
                print(preds);

def get_encoding_len(df, modelNames = [], suffix = ""):
    print("getting encoding len monsieur")
    from transformers import AutoTokenizer

    modelsList = []
    for modelName in baseModelNames:
        if len(modelNames) > 0 and not modelName in modelNames: continue;
        print(modelName, modelNames)
        tok, _ = loadTokenizerAndModel(modelName)
        modelsList.append((modelName, tok, None));
        if modelName + "_termLen" in df.columns:
            df = df.drop(columns = [modelName + "_termLen"]);

    print("modelling pains",modelsList)
    def encode_len_term(term):
        enc_lens = {}
        for modelName, tokenizer, _ in modelsList:
            enc_lens[modelName + "_termLen"] = len(tokenizer.encode(term))-2
        return enc_lens

    applied_df = df.apply(lambda row: encode_len_term(row.term), axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    dumpVar(df, "dfFinal", "combined", suffix)
    return df

def get_top_predictions_for_torch_output(tokenizer, wordsAndProbs, total_ret, markovModel, minMarkovLikelihood = minMarkovLikelihood):
    #todo@feh: can I make the models better by using isValid?
    def getProbForIdx(wordsAndProbs, idx):
        return np.sum([wordsAndProbs[i][idx[i]][1] for i in range(len(wordsAndProbs))])

    # def isValid(config):
    #     idx = getIdxForIndices(config);
    #     if hasBigRept(idx): return False
    #     words = tokenizer.decode(idx).strip()
    #     if type(re.search("[^\w\s]{2,}",words)) != type(None):
    #         print("rejing",words, config);
    #         return False
    #     return True


    def hasBigRept(nextChild):
        for index in range(2, len(nextChild)):
            if len(set(nextChild[index-2:index+1])) == 1:
                return True
        return False

    def getIdxForIndices(nextIndices):
        return [wordsAndProbs[i][nextIndices[i]][0] for i in range(len(nextIndices))]

    consideredChildren = set()
    nextParentList = []
    preds = []
    # for seed in range(np.power(3,len(wordsAndProbs))):
    #     config = []
    #     for index in range(len(wordsAndProbs)):
    #         config.append(seed%3)
    #         seed //= 3
    #     config = tuple(config)
    #     if not isValid(config): continue;
    #     initialProb = getProbForIdx(wordsAndProbs,config)
    #     nextParentList.append((config, initialProb))
    #     consideredChildren.add(config)
    initConfig = tuple([0]*len(wordsAndProbs))
    initProb =  getProbForIdx(wordsAndProbs,initConfig)
    nextParentList.append((initConfig, initProb))
    rejex = []
    nextParentList.sort(key=lambda tup: tup[1]);
    testedChildren = 0;
    while len(preds) < total_ret and testedChildren < 3*total_ret:
        testedChildren += 1
        try:
            nextIndices,_  = nextParentList.pop()
        except:
            print("not enough valid predictions! gadzuks")
            break;
        decodeIdx = getIdxForIndices(nextIndices)
        words = tokenizer.decode(decodeIdx)
        pred = words.strip()
        childAppd = False
        for i in range(len(wordsAndProbs)):
            nextChild = nextIndices[:i]+(nextIndices[i]+1,)+nextIndices[i+1:]
            # print("taking next chald",nextChild)
            try:
                probForChild = getProbForIdx(wordsAndProbs,nextChild);
                if (not nextChild in consideredChildren):
                    consideredChildren.add(nextChild);
                    if True:#isValid(nextChild):
                        nextParentList.append((nextChild, probForChild))
                        childAppd = True
            except:
                print("probability tree doesn't extend for this child!", nextChild)
                continue
        if childAppd:
            nextParentList.sort(key = lambda tup: tup[1]);
        nextChildIdx = getIdxForIndices(nextIndices)
        if len(words) < 3:
            # print("rejecting",words,"on account of it being too short to be a word.")
            continue;
        elif type(markovModel) == type(None):
            preds.append(pred)
        else:
            wordProb = markovModel.getLikelihood(pred, upToOrder = 2)
            #represented by the 3rd percentile markov probability
            if wordProb < minMarkovLikelihood:
                preds.append(pred)
            else:
                print("rejecting!",wordProb, pred, decodeIdx)
                rejex.append(pred)
    while len(preds) < total_ret:
        preds.append("");
    return preds, rejex

def getWordsAndProbs(last_hidden_state, masked_pos, total_ret, softmax, isCuda = True):
    wordsAndProbs = []
    for index, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        if isCuda:
            sftTime = time.time()
            probs = softmax(mask_hidden_state).cpu().numpy();
            # print("cudalicious",time.time()-sftTime)
        else:
            try:
                probs = softmax(mask_hidden_state.detach().numpy());
            except:
                probs = softmax(mask_hidden_state.numpy());

        idx = torch.topk(mask_hidden_state, k=10 * total_ret + 1, dim=0)[1]
        idxTime = time.time()
        wordsAndProbs.append([(x.item(), probs[x.item()]) for x in idx])
        idxTime = time.time() - idxTime
        # print("wordsAndProbs", idxTime)
    return wordsAndProbs


def queryToOutput(query):
    token_ids = tokenizer.encode(query, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    with torch.no_grad():
        if torch.cuda.is_available():
            output = model(token_ids.cuda())
        else:
            output = model(token_ids)
    return output

def applyModel(model,tokenizer, query, markovModel, total_ret = 10, isCuda = False, softmax = softmax_cpu, minMarkovLikelihood=minMarkovLikelihood, skipMarkov = False):
    '''
    todo@feh: note that <mask>tok, giving half a word, for example, interpolates a space in tokenization, might remove that
    :param model:
    :param tokenizer:
    :param query:
    :param total_ret:
    :return:
    '''

    token_ids = tokenizer.encode(query, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    with torch.no_grad():
        if isCuda:
            output = model(token_ids.cuda())
        else:
            output = model(token_ids)
    last_hidden_state = output[0].squeeze()
    # print("interstitial",token_ids, masked_pos,type(model),type(tokenizer));
    getWPTime = time.time()
    wordsAndProbs = getWordsAndProbs(last_hidden_state, masked_pos, total_ret, softmax, isCuda = True)
    getWPTime = time.time()-getWPTime

    print("predictin for",query);

    getTopWithMarkovTime = time.time()
    if not skipMarkov:
        preds, rejex = get_top_predictions_for_torch_output(tokenizer, wordsAndProbs, total_ret, markovModel, minMarkovLikelihood=minMarkovLikelihood)
    else:
        preds, rejex = None, None
    getTopWithMarkovTime = time.time() - getTopWithMarkovTime
    getTopSansMarkovTime = time.time()
    predsNonMarkov, _ = get_top_predictions_for_torch_output(tokenizer, wordsAndProbs, total_ret, None, minMarkovLikelihood=minMarkovLikelihood)
    getTopSansMarkovTime = time.time() - getTopSansMarkovTime
    if not skipMarkov:
        print("enfin",len(preds),preds[:5])
    return preds,rejex, predsNonMarkov, getWPTime, getTopWithMarkovTime, getTopSansMarkovTime

def evaluatePredictions(term, preds,total_ret, top_n=5):
    '''
    Evaluate the predictions of baseline models

    :param term:
    :param preds:
    :param total_ret:
    :param top_n:
    :return: Lev distance of 1 best, of closest in 5 best, index of ground truth (or total_ret), the top_n predictions
    '''
    top_n_preds = "; ".join(preds[:top_n]);

    try:
        positionOfGroundTruth = preds.index(term);
    except:
        positionOfGroundTruth = total_ret

    levDist5 = np.infty
    for rankIndex in range(5):
        levDist = lev.distance(term, preds[rankIndex]);
        if rankIndex == 0:
            levDist1 = levDist
        levDist5 = min(levDist, levDist5)

    return levDist1, levDist5, positionOfGroundTruth, top_n_preds


def getBestGuessQuotient(dfFinal):
    colsEq = [c for c in dfFinal.columns if "_levSimTop1_" in c and "postFineTune" in c];
    colsLev = [c for c in dfFinal.columns if "_topNPreds_" in c and "postFineTune" in c];
    for colsIndex, cols in enumerate([colsEq, colsLev]):
        quotients = []
        markQuotients = []
        for c in cols:
            if colsIndex == 0:
                quotients.append((c, dfFinal[c].mean()))  # (dfFinal[dfFinal.term == dfFinal.bestGuess]) / len(dfFinal)))
            else:
                dfFinal["bestGuess"] = dfFinal[c].apply(lambda x: x.split(";")[0].strip())
                quotients.append((c, len(dfFinal[dfFinal.term==dfFinal.bestGuess])/len(dfFinal)))
                # dfFinal[c + "_markov"] = dfFinal.bestGuess.apply(mm.getLikelihood)
                markovMean = dfFinal[c + "_markov"].mean()
                markovRejectPerc = len(dfFinal[dfFinal[c + "_markov"] > 10.63])/len(dfFinal)
                print("markovMean",markovMean, c);
                print("markovRejectPerc",markovRejectPerc,c);
                markQuotients.append((c, markovMean))
        print(sorted(quotients, key = lambda tup: tup[1]));
        print(sorted(markQuotients, key=lambda tup: tup[1]));


def apply_models(df,markovModel = None, top_n = 5, total_ret = 100, quickie = False, modelNames = [], prefix = "", onlyCorpusName = "", onlyQueryName = "", onlyModelName = "", retWholeDF = True, incTerm = False, manualModels = [], loadFinetunedModels = False, testAndTrain = False, fakeRun = False, on_evaluate_take_perc = 1):
    '''

    Take DF, apply it to all models passed in either manualModels of modelNames
    multi-word masking inspired by
    https://ramsrigoutham.medium.com/sized-fill-in-the-blank-or-multi-mask-filling-with-roberta-and-huggingface-transformers-58eb9e7fb0c
    :param df:
    :param top_n:
    :param total_ret:
    :return:
    '''

    if fakeRun: return None, None

    wholeEvalTime = time.time()

    if not testAndTrain:
        df = df[df.subset=="test"];

    skipMarkov = False
    if on_evaluate_take_perc < 1:
        sampleSize = max(min(50,len(df)),int(len(df)*on_evaluate_take_perc));
        df = df.sample(sampleSize)
        print("it's an interstitial boi!",on_evaluate_take_perc)
        skipMarkov = True
    df = df.reset_index()
    if "index" in df.columns:
        df = df.drop(columns = ["index"]);

    print("mannyMod",manualModels)
    if len(manualModels) > 0:
        modelsList = manualModels
    else:
        modelsList = []
        for modelName in baseModelNames:
            if (len(modelNames) > 0 and not modelName in modelNames) or (len(onlyModelName) > 0 and onlyModelName != modelName): continue;
            tok, mod = loadTokenizerAndModel(modelName, loadFinetunedModels)
            mod.eval()
            modelsList.append((modelName, tok, mod));


    isCuda = modelsList[0][2].device.type == "cuda"
    if isCuda:
        softmax = softmax_gpu
    else:
        softmax = softmax_cpu

    postParams = {}
    loopTimes = []
    evalTimes = []
    appTimes = []
    termTimes = []
    withMarkovTimes = []
    sansMarkovTimes = []
    wpTimes = []

    def apply_OOTB_model(line):
        '''
        Apply an out-of-the-box model to a row of a DF
        :param line:
        :return:
        '''

        if line.name % 10 == 0:
            print("perc done",line.name/len(df))
        term = line.term.strip()
        termTime = time.time()
        d = {}
        for queryName, str_exm in queryNamesAndConsts:
            if len(onlyQueryName) > 0 and not queryName == onlyQueryName: continue;
            for modelName, tokenizer, model in modelsList:
                loopTime = time.time()
                if len(onlyModelName) > 0 and not modelName == onlyModelName: continue;
                if len(onlyCorpusName) > 0 and not line.source==onlyCorpusName: continue;

                defnString = line["defn"]
                numToks = line[modelName + "_termLen"]
                query = getQuery(str_exm, defnString, modelName, numToks)
                if type(defnString) == float:
                    continue;

                #prepare masked string
                if modelName == "flaub":
                    query = query.replace("<mask>",maskTokDict[modelName])

                if not modelName in postParams:
                    postParams[modelName] = (list(model.parameters())[0].detach().cpu().numpy())[0]

                appTime = time.time()
                #actually apply model to row
                preds, rejex, predsNonMarkov, getWPTime, getTopWithMarkovTime, getTopSansMarkovTime = applyModel(model, tokenizer, query,markovModel, total_ret=total_ret, isCuda=isCuda, softmax=softmax, skipMarkov=skipMarkov)
                appTime = time.time()-appTime
                appTimes.append(appTime)
                print("predicated", round(appTime,2),line.term, (preds[:5] if not skipMarkov else None));

                wpTimes.append(getWPTime)
                if not skipMarkov:
                    withMarkovTimes.append(getTopWithMarkovTime)
                sansMarkovTimes.append(getTopSansMarkovTime)
                # print("wpt",getWPTime, "withMarkov",getTopWithMarkovTime,"sansMarkov",getTopSansMarkovTime)

                for predsIndex, predsList in enumerate((preds, predsNonMarkov)):
                    if predsIndex == 0 and skipMarkov:
                        continue;
                    evalTime = time.time()
                    levDist1, levDist5, positionOfGroundTruth, top_n_preds = evaluatePredictions(term, predsList, total_ret, top_n=top_n)
                    evalTime = time.time()-evalTime
                    evalTimes.append(evalTime)
                    print("evalTime",round(evalTime,2))
                    if predsIndex == 1:
                        suffix = "_nonMarkov"
                    else:
                        suffix = ""

                    ### assign results of prediction
                    d[prefix + modelName + "_" + "levSimTop1" + "_" + queryName + suffix] = levDist1
                    d[prefix + modelName+"_"+"groundTruthPosition" + "_" + queryName + suffix] =  positionOfGroundTruth
                    d[prefix + modelName+"_"+"topNPreds"+"_"+queryName + suffix] = top_n_preds
                    d[prefix + modelName + "_" + "levSimTop5" + "_" + queryName + suffix] = levDist5
                    d[prefix + modelName + "_" + "rejex" + "_" + queryName] = rejex
                    loopTime = time.time() - loopTime
                    loopTimes.append(loopTime)

        if incTerm:
            d["term"] = term
        d["repn"] = line.repn
        d["source"] = line.source

        termTime = time.time()-termTime
        print("loopTime",termTime)
        termTimes.append(round(termTime,2))
        return d

    try:
        from pandarallel import pandarallel
        pandarallel.initialize()
        appFunc = df.parallel_apply
    except:
        print("no panderos :(")
        appFunc = df.apply

    resDF = appFunc(lambda row: apply_OOTB_model(row),axis=1,result_type='expand')

    if retWholeDF:
        df = pd.concat([df, resDF],axis="columns");
    else:
        df = resDF

    avgLoopTime = np.mean(loopTimes)
    avgEvalTime = np.mean(evalTimes)
    avgAppTime = np.mean(appTimes)
    avgTermTimes = np.mean(termTimes)
    print("avgLoopTime",avgLoopTime,"avgEvalTime",avgEvalTime,"avgAppTime",avgAppTime,"avgTermTimes",avgTermTimes)
    if on_evaluate_take_perc < 1:
        dumpVar(loopTimes,"loopTimes","","")
        dumpVar(appTimes, "appTimes", "", "")
        dumpVar(evalTimes, "evalTimes", "", "")

    evalTime = time.time()-wholeEvalTime
    print("whole honkin eval time",evalTime)
    return df, postParams


def correctLoss(loss):
    toAddEpoch = 0;
    newLoss = []
    prevEpoch = -1
    for tup in loss:
        if tup["epoch"] < prevEpoch:
            toAddEpoch += 1;
        prevEpoch = tup['epoch']
        tup["epoch"] += toAddEpoch
        newLoss.append(tup)
    for tup in newLoss:
        if float(int(tup["epoch"])) == float(tup["epoch"]):
            print(tup["epoch"]);
    return newLoss



    '''
    todo@feh:
    transform to notebook
    calculate correlations between various features:
        • do spacy-distance and lev distance correlate?
        • do mean-ranking and {lev, spacy} distance correlate?
        • which query is better?
        • which model is better?
    '''



# print ("Original Sentence: ", sentence)
# sentence = sentence.replace("___", "<mask>")
# print ("Original Sentence replaced with mask: ", sentence)
# print ("\n")
#
# predicted_blanks = get_prediction(sentence)
# print ("\nBest guess for fill in the blank :::", predicted_blanks)
#
'''
todo@feh:
    * fix the camembert masked business: https://camembert-model.fr/
    * read the morphology book, use derif/the other software to extract tokenizations
'''

# newCols = {}
# for x in cols:
#     newCol = x.replace("xlm_rob","xlmRob")
#     newCol = newCol.replace("Sim_top", "SimTop1")
#     newCol = newCol.replace("Sim_", "SimTop5_")
#     newCol = newCol.replace("la_def_de_tok_est","laDefDeTokEst")
#     newCol = newCol.replace("veut_dire", "veutDire")
#     newCol = newCol.replace("plain_colon", "plainColon")
#     newCols[x] = newCol
#
#
# df.rename(newCols, axis=1,inplace=True)
