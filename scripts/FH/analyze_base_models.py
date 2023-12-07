
from .basis_funcs import *
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr
from seaborn import violinplot as violaplot
import matplotlib.patches as mpatches

def analyze_ret_position(df, corpusName, suffix="",total_ret = 10000):

    #3 models, 3 definitions
    models = set([name for x in df.columns for name in x.split("_")]).intersection(set(baseModelNames))
    definitions = set([name for x in df.columns for name in x.split("_")]).intersection(set([tup[0] for tup in queryNamesAndConsts]))
    rows = []

    groupbys = ("basic_pos","termLen")
    for groupbyCol in groupbys:
        fig, axes = plt.subplots(len(models), len(definitions), figsize=(12, 9))
        plt.suptitle(groupbyCol + " vs Term Retrieval Rank" + (" (corr. coef./p-val)" if groupbyCol == "basic_pos" else ""), fontsize=20)
        for modelIndex, model in enumerate(models):
            for definitionIndex, definition in enumerate(definitions):

                try:
                    ax = axes[modelIndex][definitionIndex]
                except:
                    ax = axes[definitionIndex]
                colTopN = [col for col in df.columns if "_pos_" in col and definition in col and model in col][0]
                print("boop",model, definition, axes[modelIndex],colTopN)
                # col5 = [x for x in df.columns if "Top5" in col and definition in col and model in col]
                vals = df.groupby(groupbyCol)[colTopN].apply(list).reset_index()

                percsSub10000 = []
                meanPositions = []
                def plotRow(row):
                    rowLabel = str(row[groupbyCol])
                    if groupbyCol == "basic_pos" and len(rowLabel) < 2:
                        rowLabel = "UNKNOWN"
                    if (groupbyCol == "termLen" and row[groupbyCol] <= 3) or groupbyCol == "basic_pos":
                        perc = (np.array(row[colTopN]) < 10000).mean()
                        print("perc",perc)
                        percsSub10000.append(str(round(perc,3)))
                        meanPosition = np.array(row[colTopN]).mean()
                        meanPositions.append(str(round(meanPosition,2)))
                    if groupbyCol == "basic_pos":
                        others = [x for l in vals[vals.basic_pos != row.basic_pos][colTopN].values for x in l]
                        yess = row[colTopN] + others
                        xess = [-1] * len(row[colTopN]) + [0] * len(others);
                        corrCoeff, p_val = pearsonr(xess, yess);
                        rowLabel += " " + str(round(corrCoeff, 2)) + "/" + str(round(p_val, 2))
                    ax.hist(row[colTopN], alpha=1 - 0.5 * (1 - (row[groupbyCol] if groupbyCol == "termLen" else 0) / len(vals)), weights=np.ones(len(row[colTopN])) / len(row[colTopN]), bins=np.arange(0, 10000, 400), label=rowLabel)
                vals.apply(lambda row: plotRow(row),axis=1)
                ax.set_title(model + "; " + definition)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                if groupbyCol == "termLen" and modelIndex == len(models)-1 and definitionIndex == len(definitions) -1:
                    fig.tight_layout()
                    ax.legend(bbox_to_anchor=(-1, -0.2), ncol = len(vals),loc='upper center')
                    fig.tight_layout()
                elif groupbyCol == "basic_pos":
                    ax.legend(loc="upper right")
                if definitionIndex == 0:
                    ax.set_ylabel("Count")
                if  modelIndex == len(models)-1:
                    ax.set_xlabel("Retrieval rank")
                print("kk",percsSub10000)
                if groupbyCol == "termLen":
                    row = {"model":model,"definition":definition,"sub 1000 perc.":"/".join(percsSub10000), "mean position":"/".join(meanPositions)}
                    rows.append(row);
        dfRanks = pd.DataFrame(rows)
        dfRanks.to_csv(dumpPath + "/"  +  corpusName + "_" + "dfRetrievalFoundRanks" + groupbyCol + suffix +  ".csv",index=None)
        print(dfRanks)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12, wspace=0.33,hspace=0.33)
        plt.savefig(imgPath + "/" +  corpusName + "_" + groupbyCol + "_vs_ret_rank" + suffix + ".png");

def analyze_nearness(df, corpusName, total_ret = 10000, suffix=""):

    #3 models, 3 definitions

    rows = {}
    groupbys = ("termLen","basic_pos","daille_type")
    for groupbyCol in groupbys:

        for distType in ("lev_Norm","lev","spacy"):
            if groupbyCol != "termLen" and distType == "lev_Norm": continue;

            models = set([name for x in df.columns for name in x.split("_")]).intersection(set(baseModelNames))
            definitions = set([name for x in df.columns for name in x.split("_")]).intersection(set([tup[0] for tup in queryNamesAndConsts]))
            fig, axes = plt.subplots(len(models)*2, len(definitions),figsize=(13, 9))
            supTit = corpusName + " Term-length vs " + ("Nearness" if distType == "spacy" else "Distance") + " ("+distType+") of Top Retrievals" + (" (corr. coef./p-val)" if groupbyCol == "basic_pos" else "")
            print("supTit",supTit,groupbyCol,distType)
            plt.suptitle(supTit,fontsize = 20)

            for modelIndex, model in enumerate(models):
                for definitionIndex, definition in enumerate(definitions):
                    if not (model, definition) in rows:
                        row = {"model":model,"definition":definition}
                        rows[(model, definition)] = row
                    else:
                        row = rows[(model, definition)]
                    for topNumIndex, topNum in enumerate(("1","5")):

                        ax = axes[modelIndex*2+topNumIndex][definitionIndex]
                        print("essaying",topNum,definition,model,distType)
                        colTop = [col for col in df.columns if "Top" + topNum + "_" in col and definition in col and model in col and distType.split("_")[0] in col][0]
                        print("hoop",colTop,groupbyCol)
                        # col5 = [x for x in df.columns if "Top5" in col and definition in col and model in col]
                        vals = df.groupby(groupbyCol)[colTop].agg(lambda col: list(col)).reset_index()
                        if distType == "lev_Norm":
                            vals[colTop] = vals.apply(lambda row: [x/row[groupbyCol] for x in row[colTop]],axis=1)

                        xPos = np.arange(len(vals))
                        ax.boxplot(vals[colTop], positions=xPos, widths=0.8)
                        ax.set_xticks(xPos)

                        if groupbyCol == "termLen":
                            corrXes, corrYes = [], []
                            def rowToCorrLists(row):
                                corrYes.extend(row[colTop])
                                corrXes.extend([row[groupbyCol]]*len(row[colTop]))
                            vals.apply(lambda row: rowToCorrLists(row),axis=1);
                            corrCoeff, p_val = pearsonr(corrXes, corrYes);
                            corrCoeffString = "c.c. " + str(round(corrCoeff,2)) + " (" + str(round(p_val,2)) + ")"
                            if modelIndex * 2 + topNumIndex == len(axes) - 1:
                                ax.set_xlabel("Num. of Chars in Token")
                        elif groupbyCol == "basic_pos":
                            corrCoeffArr = []
                            def rowToCorrLists(row):
                                others = [x for l in vals[vals.basic_pos != row.basic_pos][colTop].values for x in l]
                                yess = row[colTop] + others
                                xess = [1]*len(row[colTop]) + [0]*len(others);
                                corrCoeff, p_val = pearsonr(xess, yess);
                                basicPosName = str(row.basic_pos)
                                if len(basicPosName) < 2:
                                    basicPosName = "UNKNOWN"
                                corrCoeffArr.append(basicPosName + "\n" + str(round(corrCoeff,2)) + "/" + str(round(p_val,2)))
                            vals.apply(lambda row: rowToCorrLists(row),axis=1);
                            ax.set_xticklabels(corrCoeffArr)
                            corrCoeffString = ""
                            if modelIndex * 2 + topNumIndex == len(axes) - 1:
                                ax.set_xlabel("POS")
                        ax.set_title(model + "; " + definition + " (top-" + topNum + "); " + corrCoeffString)

                        if definitionIndex == 0:
                            ax.set_ylabel("Distance")

                        topMeans = vals.apply(lambda row: np.mean(row[colTop]),axis=1)
                        print("schlop means",topMeans,vals)
                        if groupbyCol == "termLen":
                            ax.plot(np.arange(len(topMeans)),topMeans)
                            threshLox = sorted(list(set([min(len(topMeans)-1,x) for x in [0,2,4,6]])))
                            print("threshLox",threshLox)
                            topMeans = [str(round(x,2)) for x in topMeans.iloc[threshLox].values]
                        else:
                            topMeans = [str(round(x, 2)) for x in topMeans.values]
                        row["Top "+topNum + " " + distType + " " + groupbyCol + " Distance"] = "/".join(topMeans)
                        #todo@feh: finish paragraph on this in doc


            fig.tight_layout()
            fig.subplots_adjust(bottom=0.12, wspace=0.05)
            plt.savefig(imgPath + "/" + corpusName + "_" + groupbyCol + "_vs_distance" + distType + suffix +  ".png")

    rows = list(rows.values())
    dfRanks = pd.DataFrame(rows)
    dfRanks.to_csv(dumpPath + "/" +  corpusName + "_" + "dfNearness" + suffix + ".csv", index=None)
    print(dfRanks)


def analyzeLevSpacAmeliorationByModel(df, evalMarkov = True):
    '''
    For model type
        For prompt
            For nearness, lev distance
                Plot mirror histogram of distances
    Make table of changes
    :param df:
    :return:
    '''
    #todo@feh: implement me
    models = set([name for x in df.columns for name in x.split("_")]).intersection(set(baseModelNames))
    prompts = set([name for x in df.columns for name in x.split("_")]).intersection(set([tup[0] for tup in queryNamesAndConsts]))
    # metrics = ("basic_pos", "termLen")
    if evalMarkov:
        evalTypes = ("nonMarkov_postFineTune", "postFineTune")
    else:
        evalTypes = ("preFineTune","postFineTune")#,"postFineTuneNoMarkov")
    metrics = [("groundTruthPosition","position of gold in results"), ("levSimTop5","mean Levenshtein distance with top 5 results")]
    for metric, plainText in metrics:
        fig, axes = plt.subplots(len(models), len(prompts), figsize=(12, 9))
        rows = []
        plt.suptitle("Model improvement in "+plainText + ("with Markov model" if evalMarkov else " after fine-tuning"), fontsize=20)
        for modIndex, model in enumerate(models):
            for promptIndex, prompt in enumerate(prompts):
                ax = axes[modIndex][promptIndex]
                distancesPre = np.array([min(x,100) for x in df[model + "_" + metric + "_" + prompt + "_" + evalTypes[0]]])
                distancesPost = np.array([min(x,100) for x in df[model + "_" + metric + "_" + prompt + "_" +  evalTypes[1]]])
                meanImprovement = np.mean(distancesPost) - np.mean(distancesPre)
                ledge = (modIndex == len(models)-1 and promptIndex == len(prompts)-1)
                if evalMarkov:
                    beforeText = "Sans Markov"
                    afterText = "With Markov"
                else:
                    beforeText = "Pre Fine-tuning"
                    afterText = "Post Fine-tuning"
                plot_mirror_hist(distancesPre, beforeText, distancesPost, afterText, ax, model + "; " + prompt + " ("+str(round(meanImprovement,2)) + ")", ledge = ledge, nBoxes = 20, lhs = 0)
                rows.append({"model":model,"prompt":prompt,beforeText.replace(" ","-"):round(np.mean(distancesPre),2),
                             afterText.replace(" ","-"):round(np.mean(distancesPost),2),"mean improvement":round(meanImprovement,2)})
        # outBase = "/Users/f002nb9/Documents/f002nb9/saclay/m1/stage/dump/slurm"
        outBase = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/img/baseline";
        imgOutPath = outBase + "/prePostHistos_" + metric + ("_markov" if evalMarkov else "") + ".png"
        fig.tight_layout()
        plt.savefig(imgOutPath);
        # plt.show()
        plt.close()
        print(imgOutPath)
        summaryDF = pd.DataFrame(rows)
        summaryDF = summaryDF.sort_values("mean improvement")
        summaryDF.to_csv(outBase + "/" + "summaryDF"+metric + ("_markov" if evalMarkov else "") + ".csv",index=False)
        print(summaryDF.to_latex())



def otherTypeAnalyzeLevSpacAmelioration(df):
    '''
    For model type
        For prompt
            For nearness, lev distance
                Plot mirror histogram of distances
    Make table of changes
    :param df:
    :return:
    '''
    #todo@feh: implement me
    dfValid = df[df.subset == "valid"]
    models = set([name for x in df.columns for name in x.split("_")]).intersection(set(baseModelNames))
    otherVars = ["daille_type","basic_pos"]
    numAxes = np.sum([len(set(df[var])) for var in otherVars])
    numRows = int(np.sqrt(numAxes));
    numCols = int(np.ceil(numAxes/numRows))
    # metrics = ("basic_pos", "termLen")
    evalTypes = ("preFineTune","postFineTune")#,"postFineTuneNoMarkov")
    metrics = [("groundTruthPosition","position of gold in results"), ("levSimTop5","mean Levenshtein distance with top 5 results")]
    for metric, plainText in metrics:
        fig, axes = plt.subplots(numRows, numCols, figsize=(12, 9))
        rows = []
        plt.suptitle("Model improvement in "+plainText + " after fine-tuning", fontsize=20)
        col = 0
        row = 0
        for var in otherVars:
            for index, varChoice in enumerate(list(set(df[var]))):
                print(col, row, numCols, numRows)
                ax = axes[row][col]
                ledge = (col == 0 and row == 0)
                if col == numCols -1:
                    col = 0
                    row += 1
                else:
                    col += 1
                dfForVarChoice = df[df[var] == varChoice]
                preVals = np.ndarray.flatten(dfForVarChoice[[col for col in dfForVarChoice.columns if ("_" + evalTypes[0]) in col and ("_" + metric + "_") in col]].values)
                postVals = np.ndarray.flatten(dfForVarChoice[[col for col in dfForVarChoice.columns if ("_" + evalTypes[1]) in col and ("_" + metric + "_") in col]].values)
                distancesPre = np.array([min(x,100) for x in preVals])
                distancesPost = np.array([min(x,100) for x in postVals])
                meanImprovement = np.mean(distancesPost) - np.mean(distancesPre)
                plot_mirror_hist(distancesPre, "Pre Fine-tuning", distancesPost, "Post Fine-tuning", ax, var + "; " + varChoice + " ("+str(round(meanImprovement,2)) + ")", ledge = ledge, nBoxes = 20, lhs = 0)
                rows.append({"variable":var,"type":varChoice,"pre-fine-tune":round(np.mean(distancesPre),2),
                             "post-fine-tune":round(np.mean(distancesPost),2),"mean improvement":round(meanImprovement,2)})
        # outBase = "/Users/f002nb9/Documents/f002nb9/saclay/m1/stage/dump/slurm"
        outBase = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/img/baseline/";
        imgOutPath = outBase + "prePostHistosVars_" + metric + ".png"
        fig.tight_layout()
        plt.savefig(imgOutPath);
        # plt.show()
        plt.close()
        print(imgOutPath)
        summaryDF = pd.DataFrame(rows)
        summaryDF = summaryDF.sort_values("post-fine-tune")
        summaryDF.to_csv(outBase + "/" + "summaryDF"+metric + ".csv",index=False)
        print(summaryDF.to_latex())




def termLenAnalyzeLevSpacAmelioration(df):
    '''
    For model type
        For prompt
            For nearness, lev distance
                Plot mirror histogram of distances
    Make table of changes
    :param df:
    :return:
    '''
    #todo@feh: implement me
    dfValid = df[df.subset == "valid"]
    models = set([name for x in df.columns for name in x.split("_")]).intersection(set(baseModelNames))
    prompts = set([name for x in df.columns for name in x.split("_")]).intersection(set([tup[0] for tup in queryNamesAndConsts]))
    # metrics = ("basic_pos", "termLen")
    evalTypes = ("preFineTune", "postFineTune")  # ,"postFineTuneNoMarkov")
    metrics = [("levSimTop5", "mean Levenshtein distance with top 5 results"),("groundTruthPosition", "position of gold in results"),]
    for metric, plainText in metrics:
        fig, axes = plt.subplots(len(models), len(prompts), figsize=(12, 9))
        rows = []
        plt.suptitle("Model improvement in " + plainText + " after fine-tuning", fontsize=20)
        for modIndex, model in enumerate(models):
            for promptIndex, prompt in enumerate(prompts):
                ax = axes[modIndex][promptIndex]
                dfPre = df[[model+"_" + "termLen",model + "_" + metric + "_" + prompt + "_" + evalTypes[0]]].copy()
                dfPre = dfPre.rename(columns={model + "_" + metric + "_" + prompt + "_" + evalTypes[0]:"metric"})
                dfPre["pre"] = "Pre-Fine-tune"
                dfPost = df[[model + "_" + "termLen",  model + "_" + metric + "_" + prompt + "_" + evalTypes[1]]].copy()
                dfPost = dfPost.rename(columns={ model + "_" + metric + "_" + prompt + "_" + evalTypes[1]: "metric"})
                dfPost["pre"] = "Post-Fine-tune"
                dfThis = pd.concat(([dfPre, dfPost]))
                violaplot(x=model + "_" + "termLen",y="metric",hue="pre",data=dfThis,split=True,ax=ax)
                dfThis = df.groupby(model + "_" + "termLen")[[model + "_" + metric + "_" + prompt + "_" + evalTypes[i] for i in range(len(evalTypes))]]
                dfThis = dfThis.agg(lambda col: list(col)).reset_index().sort_values(model + "_" + "termLen")
                dfThis["improvement"] = dfThis.apply(lambda row: np.mean(row[ model + "_" + metric + "_" + prompt + "_" + evalTypes[1]])-np.mean(row[model + "_" + metric + "_" + prompt + "_" + evalTypes[0]]),axis=1)
                dfThis["meanPost"] = dfThis.apply(lambda row: np.mean(row[ model + "_" + metric + "_" + prompt + "_" + evalTypes[1]]),axis=1)
                corrCoffPost = pearsonr(dfThis[model + "_" + "termLen"],dfThis.meanPost)
                axImp = ax.twinx()
                axImp.plot(dfThis.sort_values(model + "_" + "termLen").improvement,color="green")
                if min(dfThis.improvement) < 0:
                    print("nose",min(dfThis.improvement))
                    rect = mpatches.Rectangle((0, min(dfThis.improvement)), len(dfThis), -1*min(dfThis.improvement), linewidth=0, alpha=0.19, color="black")
                    axImp.add_patch(rect)
                # ax.set_xticklabels
                ax.set_title(model + "; " + prompt + (" (" + str(round(corrCoffPost[0],2)) + ")" if corrCoffPost[1] < 0.05 else " (p.v. > 0.05)"))
                ax.set_ylabel(metric)
                # axImp.set_ylabel("Mean Improvement")
                ledge = (modIndex == len(models) - 1 and promptIndex == len(prompts) - 1)
                if not ledge:
                    ax.get_legend().remove()
                else:
                    patch = mpatches.Patch(color='green', label='Improvement')
                    handles, _ = ax.get_legend_handles_labels()
                    handles.append(patch)
                    ax.legend(handles=handles,loc='upper left')
                    # plot_mirror_hist(distancesPre, "Pre Fine-tuning", distancesPost, "Post Fine-tuning", ax, model + "; " + prompt + " (" + str(round(meanImprovement, 2)) + ")", ledge=ledge, nBoxes=20, lhs=0)
                # rows.append({"model": model, "prompt": prompt, "pre-fine-tune": round(np.mean(distancesPre), 2),
                #              "post-fine-tune": round(np.mean(distancesPost), 2), "mean improvement": round(meanImprovement, 2)})
        # outBase = "/Users/f002nb9/Documents/f002nb9/saclay/m1/stage/dump/slurm"
        outBase = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/img/baseline";
        imgOutPath = outBase + "/prePostHistosLen_" + metric + ".png"
        fig.tight_layout()
        plt.savefig(imgOutPath);
        # plt.show()
        plt.close()
        print(imgOutPath)
        # summaryDF = pd.DataFrame(rows)
        # summaryDF = summaryDF.sort_values("post-fine-tune")
        # summaryDF.to_csv(outBase + "/" + "summaryDF" + metric + ".csv", index=False)
        # print(summaryDF.to_latex())




def plot_mirror_hist(posVals, posLabel, negVals, negLabel, ax, title, xLabel="", yLabel="", ledge=False, nBoxes=20, percLedge=False, shadeCorners=False, lhs = None):
    # print("neggington",posVals,posLabel, negVals,negLabel, ax, title, xLabel, yLabel)
    cmPos = plt.cm.get_cmap('viridis')
    cmNeg = plt.cm.get_cmap('magma')
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    ax.set_title(title)
    monoSide = False
    if len(negVals) == 0:
        negVals = posVals
        monoSide = True
    limit1 = np.fabs(min(np.min(posVals), np.min(negVals)))
    rhs = max(limit1, np.fabs(max(np.max(posVals), np.max(negVals))))
    if type(lhs) == type(None):
        lhs = rhs * -1
    bins = np.arange(lhs, rhs, (rhs - lhs) / nBoxes)
    posHeights, bins, patches = ax.hist(x=posVals, bins=bins, density=True, edgecolor='black', histtype='bar', ec="black", label=posLabel)
    print("binnington", bins, lhs, rhs, nBoxes, patches, posHeights);
    # do coloring
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmPos(c))
    # end coloring
    if not monoSide:
        negHeights, bins = np.histogram(negVals, density=True, bins=bins)
        negHeights *= -1
        bin_width = np.diff(bins)[0]
        bin_pos = (bins[:-1] + bin_width / 2) * -1
        ax.bar(bin_pos * -1, negHeights, width=bin_width, edgecolor='black', label=negLabel, color=cmNeg(rescale(-1 * negVals)))
    else:
        negHeights = None
    if len(yLabel) > 0:
        ax.set_ylabel(yLabel)
    if len(xLabel) > 0:
        ax.set_xlabel(xLabel)
    boxPos = 0.45
    for index, (vals, heights) in enumerate(((posVals,posHeights), (negVals, negHeights))):
        x = bins[int(nBoxes * boxPos)]
        if index == 0:
            perc = posLabel
        else:
            perc = negLabel
        y = ((-1) ** index) * max(np.fabs(heights)) * boxPos
        print("y diddle",x, y, perc)
        ax.annotate(
            perc, xy=(x, y),
        )
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    ax.set_title(title)
    monoSide = False
    if len(negVals) == 0:
        negVals = posVals
        monoSide = True
    limit1 = np.fabs(min(np.min(posVals), np.min(negVals)))
    rhs = max(limit1, np.fabs(max(np.max(posVals), np.max(negVals))))
    if type(lhs) == type(None):
        lhs = rhs * -1
    bins = np.arange(lhs, rhs, (rhs - lhs) / nBoxes)
    posHeights, bins, patches = ax.hist(x=posVals, bins=bins, density=True, edgecolor='black', histtype='bar', ec="black", label=posLabel)
    print("binnington", bins, lhs, rhs, nBoxes, patches, posHeights);
    # do coloring
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmPos(c))
    # end coloring
    if not monoSide:
        negHeights, bins = np.histogram(negVals, density=True, bins=bins)
        negHeights *= -1
        bin_width = np.diff(bins)[0]
        bin_pos = (bins[:-1] + bin_width / 2) * -1
        ax.bar(bin_pos * -1, negHeights, width=bin_width, edgecolor='black', label=negLabel, color=cmNeg(rescale(-1 * negVals)))
    else:
        negHeights = None
    if len(yLabel) > 0:
        ax.set_ylabel(yLabel)
    if len(xLabel) > 0:
        ax.set_xlabel(xLabel)
    boxPos = 0.45
    for index, (vals, heights) in enumerate(((posVals,posHeights), (negVals, negHeights))):
        x = bins[int(nBoxes * boxPos)]
        if index == 0:
            perc = posLabel
        else:
            perc = negLabel
        y = ((-1) ** index) * max(np.fabs(heights)) * boxPos
        print("y diddle",x, y, perc)
        ax.annotate(
            perc, xy=(x, y),
        )


def analyzeGoodResults(df):
    cols = [col for col in df.columns if "postFineTune" in col and "groundTruthPosition" in col]
    df[["retAvg","retMin"]] = df.apply(lambda row: (np.mean(row[cols]),np.min(row[cols])), axis=1)
    df["retAvg"] = df.apply(lambda row: np.mean(row[cols]), axis=1)
    test[test.retAvg < 10][["term", "postFineTune_flaub_topNPreds_laDefDeTokEst"] + cols]
