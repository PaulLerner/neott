import time


import wikitextparser as wtp
import matplotlib.pyplot as plt

import Levenshtein as lev
from .basis_funcs import *
import xml.etree.ElementTree as ET

techRoots = {"informatique", "tech", "compu","internet","télécom", "couche ","scien","médecine","nucléaires","anglicisme",'botanique', 'linguistique', 'mécanique', 'minéralogie', 'agriculture', 'géologie', 'anatomie', 'biologie', 'finance', 'architecture', 'mathématiques', 'psychologie', 'commerce', 'métrologie', 'économie', 'médecine', 'informatique', 'chimie', 'physique', 'biochimie', 'zoologie', 'programmation', 'génétique'}
#todo@feh: scrape all categories

def terms_to_txt(df):

    textPath = dumpPath + "/" + "terms.txt"
    fp = open(textPath,"w")
    df.apply(lambda row: fp.write(row.term+"\n"),axis=1);
    fp.close()
    return textPath

def get_mainUnits(mainUnitName, xml_path = None):
    if type(xml_path) == type(None):
        xml_path = rawFranceTermePath
    print("xmlPath",xml_path)
    root = ET.parse(xml_path).getroot()
    articles = root.findall(mainUnitName)
    return articles

def clean(term):
    term = term.split(",")[0]
    term = re.sub("\s+\(.*\)","",term)
    return term

def load_articles_franceTerme(quickie = False, load_from_store = False,isolation = True, noShareds = False):

    suffix = ""
    if quickie:
        n = quickieNumArts
        suffix += "_quickie"
    else:
        n = np.power(10,10)


    if load_from_store:
        print("loadin that bad boi from store")
        df = loadVar("df","franceTerme",suffix);
        print("gone and loaded")
        return df, suffix;


    print("processing articles",n);
    articles = get_mainUnits("Article")

    nonEnglishSource = 0
    rows = []
    for artIndex, article in enumerate(articles[:n]):
        if artIndex%100 == 0:
            print(artIndex/len(articles[:n]))
        term = clean(article.find("Terme_balise").text.strip())
        definition = article.find("Definition").text.strip()
        dom = article.find("Domaine").find("Dom").text.strip();
        pos = article.find("Terme").get("categorie")
        equivs = article.findall("Equivalent")


        transl = None
        for e in equivs:
            if e.get("langue") == "en":
                transl = clean(e.find("Equi_prop").text.strip())
                break;

        #    print("term",term,"trad",transl)

        rows.append({"term": term,"defn":definition, "Domain": dom, "englishTrans": transl, "POS":pos,"source":"ft","isNew":True})
    df = pd.DataFrame(rows)
    print("df",df,len(rows))
    df = df.fillna({"POS":"UNKNOWN"})
    df = df.fillna({"Domain":"missing"})
    nlp = loadNlp()
    df["basic_pos"] = df.apply(lambda row: mapBasicPOSFranceTerme(row, nlp), axis=1)
    repRows = {}
    df["repn"] = df.apply(lambda row: numberRepRows(row, repRows),axis=1);
    if not noShareds:
        df["termLen"] = df.apply(lambda row: row.term.count(" "), axis=1)
    df = clean_df(df);
    dumpVar(df,"df", "franceTerme",suffix)
    return df, suffix

def mapBasicPOSFranceTerme(row, nlp):
    if row.POS == "missing": return "UNKNOWN";
    if row.Domain == "Toponymie": return "UNKNOWN";
    if row.POS in ("loc.adj.", 'adj.', 'adj. ou n.m.', "n. ou adj.", "loc.adj.inv.", 'n.f. ou adj', 'n.m. ou adj.inv.', 'n.m. ou adj.'): return "adj"
    if row.POS in ("n.", "n.m.pl.", "n.f.pl.", "n. ou adj.", "loc.n.f.", "n.m.", "n.f.", 'n.m. ou adj.inv.', 'n.f. ou adj', 'n.m.inv.', 'loc.n.m.', 'n.m. ou adj.'): return "noun"
    if row.POS in ("v.intr.", "v.", "v.tr."): return "verb"
    if row.POS in ("adv."): return "adv"
    return normalizeSpacyPOS(nlp(row.term))

def mapBasicPOSWikt(row, nlp):
    posTok = "POS"
    if "nom" in row[posTok]: return "noun"
    if "adj" in row[posTok]: return "adj"
    if "adv" in row[posTok]: return "adv"
    if "verb" in row[posTok]: return "verb"
    return normalizeSpacyPOS(nlp(row.term))

def accordBasicPos(row, nlp):
    if row.basic_pos_ft == row.basic_pos_wikt:
        ret= row.basic_pos_ft
    elif type(row.basic_pos_ft) == str and (row.basic_pos_ft != "UNKNOWN" or row.basic_pos_wikt == 'UNKNOWN'):
        ret = row.basic_pos_ft
    else:
        ret=row.basic_pos_wikt
    if type(ret) != str and np.isnan(ret):
        ret = normalizeSpacyPOS(nlp(row.term))
    print("they out here accordin",row.term,row.basic_pos_wikt,row.basic_pos_ft,ret);
    return ret

def processTraduction(term, sections):
    try:
        tradSection = [s for s in sections if type(s.title) != type(None) and "traduction" in s.title][0]
    except:
        return None
    try:
        term = strip_accents(term)
        postDec = re.split("=+\s*\n{1,}:{0,1}", tradSection.string)[1]
        engTrads = [x.group(1) for x in re.finditer("{{trad\+{0,1}\|en\|([^}]+)}}", postDec)]
        for x in engTrads:
            if lev.distance(x.lower(), term) <= 2:
                return x
    except:
        pass
    return None

def processEtymology(sections):
    ancestor = ""
    firstLanguage = ""

    try:
        etymSection = [s for s in sections if type(s.title) != type(None) and "étymologie" in s.title][0]
    except:
        print("no etymology on this mother")
        return "", False, False, [], [], "", [], [], False
    postDec = re.split("=+\s*\n{1,}:{0,1}", etymSection.string)[1]
    isMotValise = re.search("{{mot-valise\|([^}]+)}}",postDec) is not None;
    subToks = re.search("{{composé de\|([^}]+)}}", postDec)
    for subTokString in ["composé de", "cf"]:
        if type(subToks) == type(None):
            subToks = re.search("{{" + subTokString + "\|([^}]+)}}", postDec)
            if type(subToks) != type(None): break;
    if type(subToks) != type(None):
        subToks = [x for x in subToks.group(1).split("|") if not "=" in x]
    else:
        subToks = []

    etyls = list(re.finditer("{{étyl\|([^}]+)}}", postDec))
    firstLans = []

    if "latin" in postDec:
        firstLans.append("la")
    if "grec" in postDec:
        firstLans.append("grc");



    relatedTerms = re.findall("\[\[([^\]]+)\]\]", postDec) + re.findall("''{{lien\|([^\|]+)\|fr}}''", postDec)
    for x in relatedTerms:
        if "#la" in x.lower():
            firstLans.append("la")
        if "#grc" in x:
            firstLans.append("grc")
    print("pre", relatedTerms)
    relatedTerms = [x.split("#")[0].split("|")[0] for x in relatedTerms]
    relatedTerms = [x for x in relatedTerms if len(x) > 0]
    print("post", relatedTerms)
    fixes = [x for x in relatedTerms if x[0] == "-" or x[-1] == "-"]

    for etyl in etyls:
        firstLanguage = ""
        for tok in etyl.group(1).split("|"):
            if firstLanguage == "" and not "=" in tok and not "fr" == tok:
                firstLanguage = tok
            elif "mot=" in tok:
                ancestor = tok[4:]
                relatedTerms.append(ancestor)
        print("we be addin", firstLanguage)
        firstLans.append(firstLanguage)
    if len(firstLans) > 1:
        if "la" in firstLans or len([x for x in firstLans if "latin" in x]) > 0:
            firstLanguage = "la"
        elif "gr" in firstLans:
            firstLanguage = "gr"
        else:
            firstLanguage = firstLans[0]
    elif len(firstLans) == 1:
        firstLanguage = firstLans[0]

    isBorrowed = False
    if not firstLanguage in ("la","grc") and "emprunt" in postDec.lower():
        isBorrowed = True

    dates = [int(date) for date in re.findall("\(([0-9]{4})\)", postDec) if int(date) >= 1900]
    if len(dates) > 0:
        isNew = True
    else:
        isNew = False
        dates = list(re.finditer("{{(?:date\|([0-9]{4}))|(?:([0-9]{4})\|date)}}", postDec))
        dates = [int(d.group(1)) for d in dates if int(d.group(1)) >= 1900]
        if len(dates) > 0:
            isNew = True
    return firstLanguage, isBorrowed, isNew, subToks, fixes, ancestor, relatedTerms, etyls, isMotValise


def retainTitle(title, termList):
    ret = False
    if (title in termList): ret = True
    elif title.replace("'","’") in termList or title.replace("’","'") in termList: ret = True
    return ret

def processPoses(poses, justFixes, txt, title, allFixes, firstLanguage, termList, allWords, allCats, termListExclusive, breakOnWord, isNew, getSections=False):

    domainAndExamples = {}
    for posIndex, posMatch in enumerate(poses):
        storageForPOS = []
        pos = posMatch.group(1)
        if justFixes:
            if "fix" in pos and "-" in title:
                if not title in allFixes:
                    allFixes[title] = set()
                allFixes[title].add(firstLanguage)
            continue;
        if not pos in domainAndExamples:
            domainAndExamples[pos] = []
        startFrenchIndex = re.search("{{langue\|fr}}", txt)  #
        if type(startFrenchIndex) == type(None): continue;
        txt = txt[startFrenchIndex.end():]
        endFrenchIndex = re.search("{{langue\|\w+}}", txt)
        if not type(endFrenchIndex) == type(None):
            txt = txt[:endFrenchIndex.start()]
        if allWords:
            allDefs = list(re.finditer("\n#+\**\s+[^\']{1}[^\n]+", txt))
            for eachDef in allDefs:
                defString = re.split("\n#\**\s*",eachDef.group(0))[1];
                if "|exemple" in defString or "exemple|" in defString: continue;
                defPieces = re.split("}}\s+(?<=[^{])", defString)
                if len(defPieces) == 1:
                    #no categories, no problem
                    goodCat = ""
                    defn = defString
                else:
                    defn = defPieces[-1]
                    allCatStrings = [x.strip()[2:] for x in defPieces[:-1]]
                    goodCat = ""
                    for catName in allCatStrings:
                        catNamePieces = [x for x in catName.split("|") if 2 < len(x) < 50 and not "lexique" in x and not x == "info lex" and not "exemple" in x and re.match("^[\sa-zA-Z0-9\-,.\'’!]+$", strip_accents(x))]
                        if len(catNamePieces) > 0:
                            goodCat = catNamePieces[0]
                            break;
                defn = wtp.parse(defn).plain_text().strip()
                defn = re.sub("'{2,}", "", defn);
                storageForPOS.append((defn,"", goodCat))
                print("david deffin",defn);
            continue;


        # find all occurences of \n#\s
        categoriesAndDefs = list(re.finditer("\n#+\**\s+(?={{[^}]*\w+}}\s*)+[^\']{1}[^\n]+", txt))
        allDefs = list(re.finditer("\n#+\**\s+[^\']{1}[^\n]+", txt))
        numOtherDefs = len(allDefs) - len(categoriesAndDefs)
        if False and retainTitle(title, termList) and len(categoriesAndDefs) == 0:
            categoriesAndDefs = list(re.finditer("\n#\s+(?={{[^}]*\w+}}\s*)*[^\n]+\n", txt))
        if breakOnWord:
            print("catsAndDefs")
            for x in categoriesAndDefs:
                print(x)
            print("otha")
            for x in allDefs:
                print(x)
            with open(dumpPath + "/txt.dump", "wb") as fp:
                pickle.dump(txt, fp);


        numNonTechCats = numOtherDefs;
        for catIndex, catMatch in enumerate(categoriesAndDefs):
            catString = catMatch.group(0).replace("#", "").strip()[2:];
            print("catString",catString)
            try:
                if True:
                    definitionStart = re.search("}}\s*[^{]{1}", catString).end() - 1
            except:
                # definition defies normal formatting
                print("definition formatting flub?", catString)
                continue;
            # catNames = list(re.finditer("#\s+(?:{{([^}]+)}}\s*)+",catString))
            allCatStrings = re.split("}}\s*{{", re.split("}}\s+[^{]", catString)[0])
            goodCat = ""
            nonExmplCats = [catName for catName in allCatStrings if not ("|exemple" in catName or "exemple|" in catName)]
            if breakOnWord is not None:
                print("nonExmpleCats")
                for x in nonExmplCats: print(x)
            for catName in nonExmplCats:
                if "|exemple" in catName or "exemple|" in catName:
                    print("filtering out an exemple!", title, catName)
                    continue;
                if not "lex" in catName:
                    print("no lex, probably not a cat..",catName)
                    continue;
                catNamePieces = [x for x in catName.split("|") if 2 < len(x) < 50 and not "lexique" in x and not x == "info lex" and not "exemple" in x and re.match("^[\sa-zA-Z0-9\-,.\'’!]+$",strip_accents(x))]
                for catNamePiece in catNamePieces:
                    if getSections:
                        allCats.append({"cat":catNamePiece,"src":title});
                        print("nadin",catNamePiece)
                        continue;
                    if allWords:
                        goodCat = catNamePiece
                        break;
                    for techPiece in techRoots:
                        if techPiece in catNamePiece.lower():
                            goodCat = catNamePiece
                            print("goodCatting", title, catNamePieces, "thisPiece", catNamePiece, techPiece, "\nkittykat", catString)
                            break;
                    if goodCat != "": break;
                if goodCat != "": break;
            # print("cattington",goodCat)
            if goodCat == "":
                numNonTechCats += 1
            if isNew and len(goodCat) == 0:
                goodCat = "isNew"
            if len(goodCat) > 0 or (len(termList) > 0 and termListExclusive and catIndex == 0) or retainTitle(title, termList):
                # print("taking because", allWords, goodCat, retainTitle(title, termList))
                definition = wtp.parse(catString[definitionStart:]).plain_text().strip()
                definition = re.sub("'{2,}", "", definition);
                if catIndex == (len(categoriesAndDefs) - 1):
                    endMatchSearchString = "\n={2,}|\n{1,}|{{source|\[\[Fichier"
                    endMatch = re.search(endMatchSearchString, txt[catMatch.end():])
                    if type(endMatch) != type(None):
                        endIndex = catMatch.end() + endMatch.start()
                    else:
                        endIndex = len(txt)
                else:
                    endIndex = categoriesAndDefs[catIndex + 1].start()
                if "{{exemple|" in txt[catMatch.end():endIndex]:
                    examples = [x for x in list(re.finditer("{{exemple\|([^}]+)}}", txt[catMatch.end():endIndex])) if not ("lang=" in x.group(0) and not "lang=fr" in x.group(0))]
                    goodExamples = []
                    for exm in examples:
                        exmPieces = [piece for piece in exm.group(1).split("|") if not "=" in piece]
                        if len(exmPieces) > 0:
                            goodExamples.append(exmPieces[0])
                    examples = [wtp.parse(exm).plain_text().strip() for exm in goodExamples]
                else:
                    exampleText = re.sub("[\*\#]", "", txt[catMatch.end():endIndex])
                    exampleText = re.sub("'{2,}", "", exampleText);
                    earlyEndMatch = re.search("\n={2,}|\n\n", exampleText)
                    if not type(earlyEndMatch) == type(None):
                        exampleText = exampleText[:earlyEndMatch.start()]
                    examples = [wtp.parse(exm).plain_text().strip() for exm in exampleText.split("\n") if len(exm) > 5]
                    examples = [exm for exm in examples if title.lower() in exm.lower()]
                storageForPOS.append((definition,"\t".join(examples), goodCat))


        for tup in storageForPOS:
            domainAndExamples[pos].append(tup + (numNonTechCats,))
            # print("appendin",tup,numNonTechCats)
    if breakOnWord is not None:
        print("domainAndExamps",domainAndExamples)
    return domainAndExamples

def load_wiktionnaire_df(chunkSize = 100000000, startingChunk = 0,endingChunk = np.infty, breakOnWord = None,stopAfter = np.infty, allWords = False, numWords = np.infty, load_from_store = False, quickie=False, termList = set(),termListExclusive = True,isolation = True,noDump = False,justFixes=False, noShareds = True, bigQuickie = False, getSections = False):
    '''
    TODO:
        •take only terms s.t. lang=fr
        •count terms
        •isolate techy/science terms
        •export to common format with franceTerm

    :param wiktionaryPath:
    :return:
    '''

    import time
    import wikitextparser as wtp
    import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET

    suffix = ""
    if stopAfter == np.infty:
        if quickie:
            stopAfter = quickieNumArts
            if bigQuickie:
                stopAfter *= 80
            suffix += "_quickie"
        else:
            stopAfter = np.power(10,10)

    if allWords:
        suffix += "_allWords"
    print("stippity stop",stopAfter)

    if len(termList) > 0 and termListExclusive:
        suffix += "_termList"


    if load_from_store:
        df = loadVar("df","wiktionnaire",suffix);
        return df, suffix;


    flubbedTitles = set()
    allCats = []

    print("termite",termListExclusive,termList)


    allFixes = {}
    wiktionaryPath = rawWiktionnairePath
    goodDocs = []
    print("reading from doc",wiktionaryPath, rawWiktionnairePath)
    fpInput = open(wiktionaryPath,"r")
    fpInput.seek(startingChunk*chunkSize)
    lines = fpInput.read(chunkSize)
    print("mines",lines[:100])
    numChunks = startingChunk
    while len(lines) > 0 and numChunks <= endingChunk:
        numChunks += 1
        pageMarks = list(re.finditer("page>",lines))
        if lines[pageMarks[0].start()-1] == "/":
            pageMarks = pageMarks[1:]
        numPages = len(pageMarks) //2
        pages = [lines[pageMarks[2*pageIndex].start()-1:pageMarks[2*pageIndex+1].end()] for pageIndex in range(numPages)]
        for index, page in enumerate(pages):
            pageXML = ET.fromstring(page);
            title = pageXML.find("title").text
            # print("absolut",title);
            if not "{{langue|fr}}" in page:
                continue;
            if "Wiktionnaire:" in title:
                #print("wiktionnaire!:",title)
                continue;
            if "Modèle:" in title:
                #print("modèle!",title)
                continue
            if len(goodDocs) > stopAfter:
                print("goodDocs says break")
                break;
            if len(termList) > 0 and termListExclusive and title not in termList:
                continue;

            print("titleist",title,title==breakOnWord)

            englishEquiv = None
            if "{{langue|en}}" in page:
                englishEquiv = title

            if type(breakOnWord) == str and not (type(re.search("^" + breakOnWord, title)) != type(None) or retainTitle(title, set([breakOnWord]))): continue;
            try:
                print("processing",title)
                if "MediaWiki:" in title: continue;
                # print("windex",index, round(index/len(pages),2))
                tx = pageXML.find("revision")
                txt = tx.find("text").text

                parseText = wtp.parse(txt)
                sections = parseText.get_sections()

                firstLanguage, isBorrowed, isNew, subToks, fixes, ancestor, relatedTerms, etyls, isMotValise = processEtymology(sections)

                if englishEquiv is None:
                    englishEquiv = processTraduction(title.lower(), sections)

                #todo@feh: take if techroot OR in frenchTerme OR has year in etymologie and that year is recent (> 1900)

                poses = list(re.finditer("=={2,}\s+{{S\|([^\|}]+)\|fr[^}]*}}\s+=={2,}",txt))

                domainAndExamples = processPoses(poses, justFixes, txt, title, allFixes, firstLanguage, termList, allWords, allCats, termListExclusive, breakOnWord, isNew, getSections=getSections)
                if getSections:
                    continue;

                if False and all([len(pos) == 0 for pos in domainAndExamples.keys()]) and allWords:
                    pass
                    # goodDocs.append({"term": title, "Domain": "", "POS": "\t".join(domainAndExamples.keys()),
                    #                  "termLen": termLen, "defn": "",
                    #                  "exmpls": "", "numChunks": numChunks,
                    #                  "source": "wiktionnaire", "ancestor": ancestor, "descLang": firstLanguage,
                    #                  "subToks": subToks, "relatedTerms": relatedTerms, "fixes": fixes, "isNew": isNew})
                elif allWords or not all([len(pos) == 0 for pos in domainAndExamples.keys()]):
                    # print("domine jseu",domainAndExamples)
                    domainAndExamples = {eachPos: domainAndExamples[eachPos] for eachPos in domainAndExamples.keys() if len(domainAndExamples[eachPos]) > 0}
                    for pos in domainAndExamples.keys():
                        for (definition,examples, goodCat, nonExmplCatLens) in domainAndExamples[pos]:
                            # print("adding to dict",title, definition, examples, goodCat)
                            goodDocs.append({"term": title, "Domain": goodCat, "POS": pos, "defn": definition,
                                     "exmpls": "\t".join(examples), "numChunks": numChunks,"otherCatLens":nonExmplCatLens,"isMotValise":isMotValise,
                                     "source": "wiktionnaire", "ancestor": ancestor, "descLang": firstLanguage,"englishEquiv":englishEquiv,
                                     "subToks_wikt": subToks, "relatedTerms_wikt": relatedTerms, "fixes_wikt": fixes, "isNew": isNew, "isBorrowed": isBorrowed})


            except Exception as exp:
                print("flub!",title)
                flubbedTitles.add((title,numChunks))
                raise(exp)
            if type(breakOnWord) == str and (type(re.search("^" + breakOnWord, title)) != type(None) or retainTitle(title, set([breakOnWord]))):
                print("yaa")
                print(page)
                print(firstLanguage,etyls)
                print(relatedTerms)
                print(numChunks)
                print(poses)
                print(title)
                break;
        if type(breakOnWord) == str and (type(re.search("^" + breakOnWord, title)) != type(None) or retainTitle(title, set([breakOnWord]))):
            break;
        if "</mediawiki>" in lines:
            break;
        print("seen",len(goodDocs), "already", numChunks)
        if len(goodDocs) > stopAfter:
            print("we're just about done here")
            break;
        oldLines = lines[pageMarks[numPages*2-1].end():]
        newLines = fpInput.read(chunkSize)
        lines = oldLines + newLines

    if justFixes:
        dumpVar(allFixes,"allFixesWikt","",suffix);
        print("we did fixes",allFixes)
        return;

    if getSections:
        catDF = pd.DataFrame(allCats)
        # catDF = catDF.groupby("cat")["cat"].count().sort_values()
        dumpVar(allCats, "allCats", "wiktionnaire", suffix)
        dumpVar(catDF, "allCatsDF", "wiktionnaire", suffix)
        return;

    if not breakOnWord is None:
        return page, goodDocs[-1]
    df = pd.DataFrame(goodDocs)


    nlp = loadNlp()
    df["basic_pos"] = df.apply(lambda row: mapBasicPOSWikt(row,nlp), axis=1)
    repRows = {}
    df["repn"] = df.apply(lambda row: numberRepRows(row, repRows),axis=1);

    if not noShareds:
        df["termLen"] = df.apply(lambda row: row.term.count(" "), axis=1)
    print("all about that action boss")

    if allWords:
        wordSet = set(df.term);
        wordSet = [x for x in wordSet if len(x) > 3]
        dumpVar(wordSet,"wordSet","",suffix);
        print("dumped that")

    df = clean_df(df);

    if allWords:
        df["subset"] = df.apply(lambda _: ttSplitFunc(np.random.rand()), axis=1)
        dumpVar(df,"allWiktionaryDF","",suffix);
        print("dumped that")

    if not noDump:
        print("dumpin")
        dumpVar(df,"df", "wiktionnaire",suffix)
        dumpVar(flubbedTitles,"flubbedTitles", "wiktionnaire",suffix)
    print("crikey", df)
    return df, suffix

def numberRepRows(row, repRows):
    term = row.term
    #term = row.veutDire_query_ft
    if term in repRows:
        repRows[term] += 1;
    else:
        repRows[term] = 0
    return repRows[term]

def loadAllSources(load_from_store=False, quickie=False, type2Quickie = False, bigQuickie = False, startingChunk=0, endingChunk=100):
    '''
    todo@feh: concatenate rather than merge! oy
    :param load_from_store:
    :param quickie:
    :param type2Quickie:
    :param bigQuickie:
    :return:
    '''
    print("church street blues",load_from_store, quickie and not type2Quickie)
    franceTermeDF, _ = load_articles_franceTerme(quickie=quickie and not type2Quickie, load_from_store=load_from_store,isolation=False, noShareds = True);
    wiktDF, suffix = load_wiktionnaire_df(quickie=quickie and not type2Quickie,load_from_store=load_from_store,termList=set(franceTermeDF.term.values),termListExclusive=False,isolation = False, bigQuickie=bigQuickie,startingChunk=startingChunk, endingChunk=endingChunk)
    if quickie and len(suffix) == 0:
        if bigQuickie:
            suffix += "_bigQuickie"
        else:
            suffix += "_quickie"
    # franceTermeDF = franceTermeDF.rename(columns={col: col + "_ft" for col in franceTermeDF.columns if not col in sharedCols});
    # wiktDF = wiktDF.rename(columns={col: col + "_wikt" for col in wiktDF.columns if not col in sharedCols});
    wiktDF["has_wikt"] = True
    franceTermeDF["has_ft"] = True

    # dfFinal = pd.merge(franceTermeDF,wiktDF, on="term", how="outer")#, suffixes=("", "_wikt"));
    dfFinal = pd.concat(([wiktDF, franceTermeDF]));
    # dfFinal["basic_pos"] = dfFinal.apply(lambda row: accordBasicPos(row, nlp), axis=1)
    dfFinal = dfFinal.fillna({"basic_pos": "UNKNOWN"})
    dfFinal = dfFinal.fillna({col: "UNKNOWN" for col in dfFinal.columns if "_wikt" in col})
    dfFinal = dfFinal.fillna("")

    dfFinal["termLen"] = dfFinal.apply(lambda row: row.term.count(" ")+1, axis=1)
    if type2Quickie:
        dfFinal = dfFinal[:quickieNumArts*(10 if bigQuickie else 1)]

    if not "subset" in dfFinal:
        dfFinal["subset"] = dfFinal.apply(lambda _: ttSplitFunc(np.random.rand()), axis=1)

    print("finally",dfFinal,dfFinal.columns)
    dfFinal = clean_df(dfFinal);
    dumpVar(dfFinal,"dfFinal","combined",suffix)
    return dfFinal, suffix


'''
Notes:
*la plupart n'ont pas de POS associé
* touts les mots sauf 13 ont une traduction anglaise

=====================

take all terms, not just those with tech
fine-tune by pre-segmenting letters individually
or segment according to morphological decomposition?


TODO today:
• Fine-tune with Slurm
• Classify terms by Daille (using DeriF results, Online Etymology Dictionary)

Big TODO:
• For each word:
    • If not exist, make node for it
    • If language of origin is known, annotate; else,
        • For each token in its derivation, create arc from word to token, recurse with token
After tree is fully constructed (no more words in cloud), classify each word by its tokenization and Daille classificaiotion 


TODO (next project lol):
• Train model to spit out structured etymology based on wiktionary entries

Bloom (Blume?), Mt5
End Of Term token?

Letter by letter?
'''


def considerCategories(df):
    dumpPath = "/vol/work3/yvon/ScienceTerms/herron_stage/out/dump"
    try:
        with open(dumpPath+"/seenCats.pickle","rb") as fp:
            seenCats = pickle.load(fp);
        with open(dumpPath+"/takeCats.pickle","rb") as fp:
            takeCats = pickle.load(fp);
    except:
        seenCats = set()
        takeCats = set()
    allCats = df.groupby("cat")["cat"].count().sort_values(ascending=False).index
    for cat in allCats[:50]:
        if cat in seenCats: continue;
        samps = df[df.cat==cat].sample(min(50,len(df[df.cat==cat]))).src
        print(cat)
        print(samps);
        take = input("");
        if take == "y":
            takeCats.add(cat)
        seenCats.add(cat)
        with open(dumpPath+"/seenCats.pickle","wb") as fp:
            pickle.dump(seenCats,fp);
        with open(dumpPath+"/takeCats.pickle","wb") as fp:
            pickle.dump(takeCats,fp);

    #produces: {'botanique', 'linguistique', 'mécanique', 'minéralogie', 'agriculture', 'géologie', 'Internet', 'anatomie', 'biologie', 'finance', 'architecture', 'mathématiques', 'psychologie', 'commerce', 'métrologie', 'économie', 'médecine', 'informatique', 'chimie', 'physique', 'biochimie', 'technique', 'zoologie', 'programmation', 'génétique', 'télécommunications'}


