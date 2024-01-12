#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
from jsonargparse import CLI

import pandas as pd


COLS="""CUI	Unique identifier for concept
LAT	Language of term
TS	Term status
LUI	Unique identifier for term
STT	String type
SUI	Unique identifier for string
ISPREF	Atom status - preferred (Y) or not (N) for this string within this concept
AUI	Unique identifier for atom - variable length field, 8 or 9 characters
SAUI	Source asserted atom identifier [optional]
SCUI	Source asserted concept identifier [optional]
SDUI	Source asserted descriptor identifier [optional]
SAB	Abbreviated source name (SAB). Maximum field length is 20 alphanumeric characters. Two source abbreviations are assigned:
TTY	Abbreviation for term type in source vocabulary, for example PN (Metathesaurus Preferred Name) or CD (Clinical Drug). Possible values are listed on the Abbreviations Used in Data Elements page.
CODE	Most useful source asserted identifier (if the source vocabulary has more than one identifier), or a Metathesaurus-generated source entry identifier (if the source vocabulary has none)
STR	String
SRL	Source restriction level
SUPPRESS	Suppressible flag. Values = O, E, Y, or N
CVF	Content View Flag. Bit field used to flag rows included in Content View. This field is a varchar field to maximize the number of bits available for use."""

SOURCES= """AIR 	AI/RHEUM 	1995AA 	ENG 	0
AOD 	Alcohol and Other Drug Thesaurus 	2002AC 	ENG 	0
ALT 	Alternative Billing Concepts 	2009AA 	ENG 	3
ATC 	Anatomical Therapeutic Chemical Classification System 	2023AB 	ENG 	0
AOT 	Authorized Osteopathic Thesaurus 	2006AD 	ENG 	0
BI 	Beth Israel Problem List 	1999AA 	ENG 	2
CDT 	CDT 	2022AB 	ENG 	3
HCDT 	CDT in HCPCS 	2023AA 	ENG 	3
CCC 	Clinical Care Classification 	2018AA 	ENG 	1
CCS 	Clinical Classifications Software 	2005AC 	ENG 	0
CCSR_ICD10CM 	Clinical Classifications Software Refined for ICD-10-CM 	2023AA 	ENG 	0
CCSR_ICD10PCS 	Clinical Classifications Software Refined for ICD-10-PCS 	2023AA 	ENG 	0
RAM 	Clinical Concepts by R A Miller 	2000AA 	ENG 	0
CCPSS 	Clinical Problem Statements 	2000AA 	ENG 	3
JABL 	Congenital Mental Retardation Syndromes 	2000AA 	ENG 	1
CHV 	Consumer Health Vocabulary 	2012AA 	ENG 	0
COSTAR 	COSTAR 	2002AD 	ENG 	0
CST 	COSTART 	1999AA 	ENG 	0
CPT 	CPT - Current Procedural Terminology 	2023AA 	ENG 	3
HCPT 	CPT in HCPCS 	2023AA 	ENG 	3
CPTSP 	CPT Spanish 	2001AC 	SPA 	3
CSP 	CRISP Thesaurus 	2006AB 	ENG 	0
DSM-5 	Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition 	2016AB 	ENG 	3
UWDA 	Digital Anatomist 	2003AC 	ENG 	0
DDB 	Diseases Database 	2001AA 	ENG 	3
DRUGBANK 	DrugBank 	2023AB 	ENG 	0
DXP 	DXplain 	1995AA 	ENG 	0
MTHSPL 	FDA Structured Product Labels 	2023AB 	ENG 	0
NDDF 	FDB MedKnowledge 	2023AB 	ENG 	3
FMA 	Foundational Model of Anatomy 	2019AB 	ENG 	0
GO 	Gene Ontology 	2023AB 	ENG 	0
MCM 	Glossary of Clinical Epidemiologic Terms 	1992AA 	ENG 	0
GS 	Gold Standard Drug Database 	2023AB 	ENG 	3
HCPCS 	HCPCS - Healthcare Common Procedure Coding System 	2023AA 	ENG 	0
HL7V2.5 	HL7 Version 2.5 	2005AC 	ENG 	0
HL7V3.0 	HL7 Version 3.0 	2023AB 	ENG 	0
HGNC 	HUGO Gene Nomenclature Committee 	2023AB 	ENG 	0
HPO 	Human Phenotype Ontology 	2023AB 	ENG 	0
DMDICD10 	ICD-10 German 	1997AA 	GER 	1
ICD10PCS 	ICD-10 Procedure Coding System 	2023AB 	ENG 	0
ICD10AE 	ICD-10, American English Equivalents 	1998AA 	ENG 	3
ICD10AM 	ICD-10, Australian Modification 	2000AB 	ENG 	3
ICD10AMAE 	ICD-10, Australian Modification, Americanized English Equivalents 	2002AD 	ENG 	3
MTHICD9 	ICD-9-CM Entry Terms 	2015AA 	ENG 	0
ICD10DUT 	ICD10, Dutch Translation 	2004AB 	DUT 	3
ICPCBAQ 	ICPC Basque 	2000AA 	BAQ 	0
ICPCDAN 	ICPC Danish 	1999AA 	DAN 	0
ICPCDUT 	ICPC Dutch 	1999AA 	DUT 	0
ICPCFIN 	ICPC Finnish 	1999AA 	FIN 	0
ICPCFRE 	ICPC French 	2000AA 	FRE 	0
ICPCGER 	ICPC German 	2000AA 	GER 	0
ICPCHEB 	ICPC Hebrew 	2000AA 	HEB 	0
ICPCHUN 	ICPC Hungarian 	1999AA 	HUN 	0
ICPCITA 	ICPC Italian 	1999AA 	ITA 	0
ICPCNOR 	ICPC Norwegian 	1999AA 	NOR 	0
ICPCPOR 	ICPC Portuguese 	1999AA 	POR 	0
ICPCSPA 	ICPC Spanish 	2000AA 	SPA 	0
ICPCSWE 	ICPC Swedish 	2000AA 	SWE 	0
ICPC2P 	ICPC-2 PLUS 	2006AB 	ENG 	3
ICPC2ICD10ENG 	ICPC2-ICD10 Thesaurus 	2005AB 	ENG 	3
ICPC2ICD10DUT 	ICPC2-ICD10 Thesaurus, Dutch Translation 	2005AB 	DUT 	3
MTHICPC2EAE 	ICPC2E American English Equivalents 	2004AB 	ENG 	3
ICPC2EDUT 	ICPC2E Dutch 	2004AB 	DUT 	3
HLREL 	ICPC2E ICD10 Relationships 	2001AA 		3
MTHICPC2ICD10AE 	ICPC2E-ICD10 Thesaurus, American English Equivalents 	2005AB 	ENG 	3
ICNP 	International Classification for Nursing Practice 	2023AA 	ENG 	3
ICD10 	International Classification of Diseases and Related Health Problems, Tenth Revision 	2004AB 	ENG 	3
ICD9CM 	International Classification of Diseases, Ninth Revision, Clinical Modification 	2015AA 	ENG 	0
ICD10CM 	International Classification of Diseases, Tenth Revision, Clinical Modification 	2023AB 	ENG 	4
ICF 	International Classification of Functioning, Disability and Health 	2009AA 	ENG 	4
ICF-CY 	International Classification of Functioning, Disability and Health for Children and Youth 	2009AA 	ENG 	4
ICPC 	International Classification of Primary Care 	1998AA 	ENG 	0
ICPC2EENG 	International Classification of Primary Care, 2nd Edition, Electronic 	2004AB 	ENG 	3
KCD5 	Korean Standard Classification of Disease Version 5 	2009AA 	KOR 	3
LCH 	Library of Congress Subject Headings 	1992AA 	ENG 	0
LCH_NW 	Library of Congress Subject Headings, Northwestern University subset 	2014AB 	ENG 	0
LNC 	LOINC 	2023AB 	ENG 	0
LNC-ZH-CN 	LOINC Linguistic Variant - Chinese, China 	2023AB 	CHI 	0
LNC-NL-NL 	LOINC Linguistic Variant - Dutch, Netherlands 	2023AB 	DUT 	0
LNC-ET-EE 	LOINC Linguistic Variant - Estonian, Estonia 	2023AB 	EST 	0
LNC-FR-BE 	LOINC Linguistic Variant - French, Belgium 	2023AB 	FRE 	0
LNC-FR-CA 	LOINC Linguistic Variant - French, Canada 	2023AB 	FRE 	0
LNC-FR-FR 	LOINC Linguistic Variant - French, France 	2023AB 	FRE 	0
LNC-DE-AT 	LOINC Linguistic Variant - German, Austria 	2023AB 	GER 	0
LNC-DE-DE 	LOINC Linguistic Variant - German, Germany 	2023AB 	GER 	0
LNC-EL-GR 	LOINC Linguistic Variant - Greek, Greece 	2023AB 	GRE 	0
LNC-IT-IT 	LOINC Linguistic Variant - Italian, Italy 	2023AB 	ITA 	0
LNC-KO-KR 	LOINC Linguistic Variant - Korea, Korean 	2023AB 	KOR 	0
LNC-PL-PL 	LOINC Linguistic Variant - Polish, Poland 	2023AB 	POL 	0
LNC-PT-BR 	LOINC Linguistic Variant - Portuguese, Brazil 	2023AB 	POR 	0
LNC-RU-RU 	LOINC Linguistic Variant - Russian, Russia 	2023AB 	RUS 	0
LNC-ES-AR 	LOINC Linguistic Variant - Spanish, Argentina 	2023AB 	SPA 	0
LNC-ES-MX 	LOINC Linguistic Variant - Spanish, Mexico 	2023AB 	SPA 	0
LNC-ES-ES 	LOINC Linguistic Variant - Spanish, Spain 	2023AB 	SPA 	0
LNC-TR-TR 	LOINC Linguistic Variant - Turkish, Turkey 	2023AB 	TUR 	0
LNC-UK-UA 	LOINC Linguistic Variant - Ukrainian, Ukraine 	2023AB 	UKR 	0
MVX 	Manufacturers of Vaccines 	2023AB 	ENG 	0
MEDCIN 	MEDCIN 	2023AB 	ENG 	3
MDR 	MedDRA 	2023AB 	ENG 	3
MDRARA 	MedDRA Arabic 	2023AB 	ARA 	3
MDRBPO 	MedDRA Brazilian Portuguese 	2023AB 	POR 	3
MDRCZE 	MedDRA Czech 	2023AB 	CZE 	3
MDRDUT 	MedDRA Dutch 	2023AB 	DUT 	3
MDRFRE 	MedDRA French 	2023AB 	FRE 	3
MDRGER 	MedDRA German 	2023AB 	GER 	3
MDRGRE 	MedDRA Greek 	2023AB 	GRE 	3
MDRHUN 	MedDRA Hungarian 	2023AB 	HUN 	3
MDRITA 	MedDRA Italian 	2023AB 	ITA 	3
MDRJPN 	MedDRA Japanese 	2023AB 	JPN 	3
MDRKOR 	MedDRA Korean 	2023AB 	KOR 	3
MDRLAV 	MedDRA Latvian 	2023AB 	LAV 	3
MDRPOL 	MedDRA Polish 	2023AB 	POL 	3
MDRPOR 	MedDRA Portuguese 	2023AB 	POR 	3
MDRRUS 	MedDRA Russian 	2023AB 	RUS 	3
MDRSPA 	MedDRA Spanish 	2023AB 	SPA 	3
MDRSWE 	MedDRA Swedish 	2023AB 	SWE 	3
CPM 	Medical Entities Dictionary 	2003AC 	ENG 	2
MED-RT 	Medication Reference Terminology 	2023AB 	ENG 	0
MEDLINEPLUS 	MedlinePlus Health Topics 	2023AA 	ENG 	0
MEDLINEPLUS_SPA 	MedlinePlus Spanish Health Topics 	2023AA 	SPA 	0
MSH 	MeSH 	2023AB 	ENG 	0
MSHSCR 	MeSH Croatian 	2019AA 	SCR 	3
MSHCZE 	MeSH Czech 	2023AA 	CZE 	3
MSHDUT 	MeSH Dutch 	2005AB 	DUT 	3
MSHFIN 	MeSH Finnish 	2008AA 	FIN 	3
MSHFRE 	MeSH French 	2023AA 	FRE 	3
MSHGER 	MeSH German 	2023AB 	GER 	0
MSHITA 	MeSH Italian 	2019AA 	ITA 	3
MSHJPN 	MeSH Japanese 	2015AB 	JPN 	3
MSHLAV 	MeSH Latvian 	2012AA 	LAV 	3
MSHNOR 	MeSH Norwegian 	2019AA 	NOR 	3
MSHPOL 	MeSH Polish 	2023AA 	POL 	3
MSHPOR 	MeSH Portuguese 	2023AA 	POR 	3
MSHRUS 	MeSH Russian 	2023AA 	RUS 	3
MSHSPA 	MeSH Spanish 	2023AA 	SPA 	3
MSHSWE 	MeSH Swedish 	2021AB 	SWE 	3
MTHCMSFRF 	Metathesaurus CMS Formulary Reference File 	2023AB 	ENG 	0
MTH 	Metathesaurus Names 	1990AA 	ENG 	0
MMX 	Micromedex 	2023AB 	ENG 	3
MTHMST 	Minimal Standard Terminology (UMLS) 	2002AA 	ENG 	0
MTHMSTFRE 	Minimal Standard Terminology French (UMLS) 	2002AA 	FRE 	0
MTHMSTITA 	Minimal Standard Terminology Italian (UMLS) 	2002AA 	ITA 	0
MMSL 	Multum 	2023AB 	ENG 	1
NANDA-I 	NANDA-I Taxonomy 	2018AA 	ENG 	3
VANDF 	National Drug File 	2023AB 	ENG 	0
NUCCHCPT 	National Uniform Claim Committee - Health Care Provider Taxonomy 	2023AB 	ENG 	3
NCBI 	NCBI Taxonomy 	2023AB 	ENG 	0
NCISEER 	NCI SEER ICD Mappings 	2002AD 		0
NCI 	NCI Thesaurus 	2023AB 	ENG 	0
NEU 	Neuronames Brain Hierarchy 	2023AB 	ENG 	3
NIC 	Nursing Interventions Classification 	2018AB 	ENG 	3
NOC 	Nursing Outcomes Classification 	2018AB 	ENG 	3
OMS 	Omaha System 	2007AC 	ENG 	1
OMIM 	Online Mendelian Inheritance in Man 	2023AB 	ENG 	0
ORPHANET 	ORPHANET 	2023AB 	ENG 	1
PCDS 	Patient Care Data Set 	1999AA 	ENG 	3
PNDS 	Perioperative Nursing Data Set 	2022AA 	ENG 	3
PPAC 	Pharmacy Practice Activity Classification 	1999AA 	ENG 	3
PDQ 	Physician Data Query 	2019AA 	ENG 	0
PSY 	Psychological Index Terms 	2005AB 	ENG 	3
QMR 	Quick Medical Reference 	1998AA 	ENG 	0
CDCREC 	Race & Ethnicity - CDC 	2021AB 	ENG 	0
RCD 	Read Codes 	2000AA 	ENG 	3
RCDAE 	Read Codes Am Engl 	2000AA 	ENG 	3
RCDSA 	Read Codes Am Synth 	2000AA 	ENG 	3
RCDSY 	Read Codes Synth 	2000AA 	ENG 	3
RXNORM 	RXNORM 	2023AB 	ENG 	0
SNM 	SNOMED 1982 	1991AA 	ENG 	9
SCTSPA 	SNOMED CT Spanish Edition 	2023AB 	SPA 	9
SNOMEDCT_US 	SNOMED CT, US Edition 	2023AB 	ENG 	9
SNOMEDCT_VET 	SNOMED CT, Veterinary Extension 	2023AB 	ENG 	9
SNMI 	SNOMED Intl 1998 	1999AA 	ENG 	9
SOP 	Source of Payment Typology 	2021AA 	ENG 	0
SRC 	Source Terminology Names (UMLS) 	1995AA 	ENG 	0
SPN 	Standard Product Nomenclature 	2004AA 	ENG 	0
TKMT 	Traditional Korean Medical Terms 	2011AB 	KOR 	0
ULT 	UltraSTAR 	1995AA 	ENG 	3
UMD 	UMDNS 	2023AA 	ENG 	1
DMDUMD 	UMDNS German 	1999AA 	GER 	1
USP 	USP Compendial Nomenclature 	2023AB 	ENG 	0
USPMG 	USP Model Guidelines 	2020AA 	ENG 	0
CVX 	Vaccines Administered 	2023AB 	ENG 	0
WHO 	WHOART 	1999AA 	ENG 	2
WHOFRE 	WHOART French 	1999AA 	FRE 	2
WHOGER 	WHOART German 	1999AA 	GER 	2
WHOPOR 	WHOART Portuguese 	1999AA 	POR 	2
WHOSPA 	WHOART Spanish 	1999AA 	SPA 	2"""

LANGUAGES = {}
for source in SOURCES.split("\n"):
    s, _, _, lang, _ = source.split("\t")
    s,lang = s.strip(), {"ENG":"en", "FRE": "fr"}.get(lang.strip())
    if lang is None:
        continue
    LANGUAGES.setdefault(lang, set())
    LANGUAGES[lang].add(s)
        
        
def get_glossary(path):
    names=[c.split("\t")[0].strip() for c in COLS.split("\n")]
    glossary = pd.read_csv(path,sep="|",names=names,dtype=str,index_col=False)
    glossary = glossary[glossary["LAT"].isin({'ENG', 'FRE'})]

    # only P or S. S seems to be acronyms
    glossary = glossary[glossary["TS"] == "P"]
    
    data = {}
    for _, term in glossary.iterrows():
        concept_id = term.CUI
        data.setdefault(concept_id, {"id": concept_id, "en":{"syns":[]}, "fr":{"syns":[]}})
        lang = {"ENG":"en", "FRE":"fr"}[term.LAT]
        if term.STT=="PF" and term.ISPREF=="Y":
            data[concept_id][lang]["text"] = term.STR
            data[concept_id][lang]["Dom"] = term.SAB
            data[concept_id][lang]["id"] = term.LUI
        else:
            data[concept_id][lang]["syns"].append({"text":term.STR, "source": term.SAB, "id": term.LUI})

    return data


def get_definitions(path, data):
    defs = pd.read_csv(path, sep="|", header=None)
    
    defs = defs[defs[4].isin((LANGUAGES["en"]|LANGUAGES["fr"]))]
    
    defs_per_source = Counter(defs[4])
        
    for concept_id, cdefs in defs.groupby(0):
        fr_defs, en_defs, = {}, {}
        for _, cdef in cdefs.iterrows():
            source = cdef[4]
            if source in LANGUAGES["en"]:
                en_defs[source] = {"text": cdef[5], "a_id": cdef[1], "at_id": cdef[2], "source": source}
            else:
                fr_defs[source] = {"text": cdef[5], "a_id": cdef[1], "at_id": cdef[2], "source": source}
        if en_defs:
            # prefer most frequent definitions for consistency
            best_def_en = sorted(en_defs, key=defs_per_source.get, reverse=True)[0]
            best_def_en = en_defs.pop(best_def_en)
        else:
            best_def_en = None
        if fr_defs:
            # prefer most frequent definitions for consistency
            best_def_fr = sorted(fr_defs, key=defs_per_source.get, reverse=True)[0]
            best_def_fr = fr_defs.pop(best_def_fr)
        else:
            best_def_fr = None
        data[concept_id]["en"]["def"] = best_def_en
        data[concept_id]["en"]["defs"] = en_defs
        data[concept_id]["fr"]["def"] = best_def_fr
        data[concept_id]["fr"]["defs"] = fr_defs

    return list(data.values())


def main(glossary_path: str, definitions_path: str, output: str):
    """
    Parse UMLS glossary and definitions
    
    Parameters
    ----------
    glossary_path: str
        Path to UMLS glossary (MRCONSO.RRF)
    definitions_path: str
        Path to UMLS definitions (MRDEF.RRF)
    output: str
    """
    data = get_glossary(glossary_path)
    data = get_definitions(definitions_path, data)
    with open(output, "wt") as file:
        json.dump(data, file)
    
    
if __name__ == "__main__":
    CLI(main)