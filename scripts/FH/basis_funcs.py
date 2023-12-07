# coding=utf-8
from builtins import *
import os
import pathlib
import re
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import time

import matplotlib
def pickLoad(pth):
    with open(pth,"rb") as fpp:
        return pickle.load(fpp)

import pandas as pd
import pickle
import sys

randomC = lambda: np.random.randint(0,255)
def generateRandomColor():
    return '#%02X%02X%02X' % (randomC(), randomC(), randomC())

from dotenv import load_dotenv
load_dotenv()
rawWiktionnairePath = os.getenv("raw_wiktionnaire_path")
rawFranceTermePath = os.getenv("raw_franceTerme_path")
dumpPath= os.getenv("dump_out_loc")

imgOutLoc = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/img"

if type(dumpPath) == type(None):
    dumpPath = "/mnt/beegfs/projects/neo_scf_herron/stage/out/dump"
    print("dumppath none!")

imgPath = os.getenv("img_out_loc")
if type(imgPath) == type(None):
    imgPath = "/mnt/beegfs/projects/neo_scf_herron/stage/out/img"
    print("imgpath none!")
dataPath = os.getenv("data_loc")
spacyLibPath = os.getenv("spacy_lib_path")
minMarkovLikelihood = 10.625025814492451

baseModelNames = "xlmRob","flaub","camem"
corpusAbbs = "wikt","ft"

maskTokDict = {"camem":"<mask>","xlmRob":"<mask>","flaub":"<special1>"}

huggingfaceKeyWords = ["levSimTop","groundTruthPosition","topNPreds"]

queryNamesAndConsts = (("laDefDeTokEst","La définition de «{maskStr}» est {defn}"),("plainColon","«{maskStr}»: {defn}"),("veutDire","Le terme «{maskStr}» veut dire {defn}"))

def getQuery(baseQuery, defn, modelName, numToks):
    maskTok = maskTokDict[modelName]
    maskStr = "".join([maskTok] * numToks)
    query = baseQuery.format(defn=defn, maskStr=maskStr);
    return query

try:
    canShowPlots = bool(int(os.getenv("can_show_plots")))
except:
    canShowPlots=False

if not canShowPlots:
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

quickieNumArts = 15

dirs = [dumpPath, imgPath]
for eachDir in dirs:
    print("dir",eachDir)
    try:
        pathlib.Path(eachDir).mkdir(parents=True,exist_ok=True);
    except:
        print("ok so something things are bogus")

def dumpVar(var, varName, corpusName, suffix):
    varDumpPath = dumpPath + "/" + get_var_path_name(varName, corpusName, suffix)
    print("dumping to",varDumpPath, suffix)
    with open(varDumpPath,"wb") as fp:
        pickle.dump(var, fp)
    return varDumpPath


def loadVar(varName, corpusName, suffix):
    varPath = dumpPath + "/" + get_var_path_name(varName, corpusName, suffix)
    print("loading var",varName,varPath)
    with open(varPath, "rb") as fp:
        return pickle.load(fp)

def loadData(varName, corpusName):
    with open(dataPath + "/" + varName + ("_" + corpusName if len(corpusName) > 0 else ""),"rb") as fp:
        return pickle.load(fp)

def get_var_path_name(varName, corpusName, suffix):
    return re.sub("_{1,}","_",corpusName + "_" + varName + ( "_" + suffix if len(suffix) > 0 else "") + ".pickle")


def getOrigLang(dfWiktionnaire, term):
    try:
        row = dfWiktionnaire[dfWiktionnaire.term == term].iloc[0]
        descLang = row.descLang
        return descLang
    except:
        return None

import unicodedata
def strip_accents(s):
    '''
    from @oefe on stackoverflow
    :param s: 
    :return: 
    '''
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
    s = s.replace("œ","oe").replace("æ","ae");
    return s;

def loadNlp():
    print("loading nlp sry this takes a smidge")
    import spacy
    nlp = spacy.load(spacyLibPath)
    return nlp

def getSuffix(quickie, loadFinetunedModels):
    suffix = ""
    if quickie:
        suffix += "_quickie"
    if loadFinetunedModels:
        suffix += "_fineTunedMods"
    return suffix

def clean_df(df):
    df = df[df.Domain != "Toponymie"];
    df = df[(~df.term.isna()) & (~df.defn.isna())]
    df = df[df.term.apply(lambda x: len(x) > 3)]
    df = df[df.defn.apply(lambda d: len(d) >= 25)]
    df = df[(df.term.str[0] != "-") & (df.term.str[-1] != "-")]
    df = df[(~df.defn.str.contains("{")) & (~df.defn.str.contains("}"))]
    df = df[df.basic_pos!="UNKNOWN"]
    df = df[df.term.apply(lambda term: re.match("^[a-zA-Z\-\s\'’]+$",strip_accents(term)) is not None)]
    df["Domain"] = df.Domain.apply(lambda d: (d if len(d) > 0 else "None"))
    df = df[df.Domain != "None"];
    return df

def ttSplitFunc(x):
    if x < 0.8:
        return "train"
    elif x < 0.9:
        return "test"
    return "valid"

def loadTokenizerAndModel(name, loadFinetunedModels = False, large = False):
    import torch
    techName = ""
    if name == "xlmRob":
        if large:
            techName = "xlm-roberta-large"
        else:
            techName = "xlm-roberta-base"
    if name == "camem":
        if large:
            techName = "camembert/camembert-large"
        else:
            techName = "camembert-base"
    if name == "flaub":
        if large:
            techName = "flaubert/flaubert_large_cased"
        else:
            techName = "flaubert/flaubert_base_cased"
    print("loading",techName)
    proxDict = {"http": "http://webproxy.lab-ia.fr:8080", "https": "http://webproxy.lab-ia.fr:8080"}
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    try:
        tok = AutoTokenizer.from_pretrained(techName)
    except:
        tok = AutoTokenizer.from_pretrained(techName, proxies=proxDict)
    if loadFinetunedModels:
        largeSuffix = ""
        if large:
            largeSuffix = "_large"
        rootPath = "/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/models/"+name + largeSuffix +  "-finetuned-tech/"
        checkpoints = [x for x in os.listdir(rootPath) if os.path.isdir(rootPath + "/" + x) and "checkpoint-" in x]
        checkpoints.sort(key = lambda cp: int(cp.split("-")[1]))
        latestCheckpoint = rootPath + "/" + checkpoints[-1]
        print("loading model from",latestCheckpoint)
        model = AutoModelForMaskedLM.from_pretrained(latestCheckpoint)
    else:
    # if name in tokModDict:
    #     return tokModDict[name]["tok"],tokModDict[name]["model"]
        try:
            model = AutoModelForMaskedLM.from_pretrained(techName,proxies=proxDict)
        except:
            model = AutoModelForMaskedLM.from_pretrained(techName)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # tokModDict[techName] = {}
    # tokModDict[techName]["tok"] = tok
    # tokModDict[techName]["model"] = model
    return tok, model


def getKnownClassicalFixes():
    allKnownWiktFixes = loadVar("allFixesWikt", "", "")
    knownClassicalFixes = set({k: v for (k, v) in allKnownWiktFixes.items() if ("la" in v or "grc" in v)}.keys())
    #sourced from http://cm1cm2.ceyreste.free.fr/paulbert/prefix.html
    classicalFixesOtherSrc = set(loadVar("classicalPrefixes", "", ""))
    knownClassicalFixes = knownClassicalFixes.union(classicalFixesOtherSrc).difference(noNeoFixes)
    return knownClassicalFixes


spacyPOSDict = {"NOUN":"NOM","VERB":"VERBE","ADJ":"ADJ","ADV":"ADV"}

def normalizeSpacyPOS(doc):
    pos = doc[0].pos_
    if pos in spacyPOSDict:
        return doc[0].pos_.lower()
    return "UNKNOWN"


noNeoFixes = set(["dé-","dés-","im-","in-","ir-","il-","-if","pro-",
                  "em-","en-","pré-","sur-","sub-","sous-",
                  "co-","com-","-teur","-ateur","-trice","-eur","-euse","-asse",
                  "-son","-tion","-ion","-sion","-ation","-ome",
                  "-some","-ité","ibilité","-able","-esse","ab-","-ème",
                  "-type","-ier","-ière","-atif","-ant","-ante",
                  "-eau","-ique","a-","-al","-ale","-ent","-ment","-ement",
                  "-age","-et","-ille","-elle","-aux","ad-","-aire","-air",
                  "-té","-tée","-é","-ée","-el",'-e', '-o', '-s', 'é-', '-é',
                  '-er', '-in', 'un-','-en', 'an-', '-el',
                  're-', '-ie', '-on', '-at', 'ob-', 'mi-',
                  '-it', '-ît', '-an', '-ée', '-té', '-ez', 'ra-', '-im', '-ay', '-il',
                  '-eux', '-ois', '-ais', '-ain', '-ace',
                   'des-', '-ure', '-oir', '-ise', '-ate', 'per-',
                  '-idé', '-ère', '-èle', '-ail', '-aie',
                  '-acé', '-one',"-ible","-ance","-iste","-ette","-iser","-rice",
                  "-ification","-fication","-page","-ibilité","-esque","-fier","-ifier",
                  "-ieux"])

noNeoFixesLen = set([x for x in noNeoFixes if len(x) > 2]);


def isFix(fix, rest, classicalFixes, allTerms, fixType):
    if len(rest) <= 2: return None
    fixes = set()
    if (fix in classicalFixes or fix in allTerms):
        if rest in allTerms:
            fixes.add(fix)
        if len(fix) > 4:
            fixes.add(fix)
    if fixType == "prefix":
        pureFix = fix[:-2]+fix[-1]
    else:
        pureFix = fix[0]+fix[2:]
    if (len(pureFix) > 3 and pureFix in classicalFixes):
        fixes.add(fix)
    if len(fixes) == 0:
        return None
    return fixes


def findExtraToks(term, allTerms, classicalFixes):
    '''
    use prefix dict, some diddling, to find possible classical prefixes and suffixes
    todo@feh: idea, if language of origin is latin or greek and no related words, then it's an affix?
    todo@feh: what is dé1? déqualification, for example, <AsItWas> 3,VERBE/ion/suf/NOM+déqualifier/VERBE,VERBE/dé1/pre/VERBE+qualifier/VERBE&quot;
    https://fr.wiktionary.org/wiki/audio
    :param term:
    :return: hyphenated fixes!
    '''
    extraFixes = set()
    for index in range(1,len(term)-1):
        fixes = isFix("-"+term[index:],term[:index], classicalFixes, allTerms, "suffix")
        if not fixes is None:
            for fix in fixes:
                extraFixes.add(fix)
        fixes = isFix(term[:index]+"-",term[index:], classicalFixes, allTerms, "prefix")
        if not fixes is None:
            for fix in fixes:
                extraFixes.add(fix)
    print("extraFixes",term,extraFixes)
    return extraFixes