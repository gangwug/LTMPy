# ######The sub-functions of LTMPy
import re
import pandas as pd
import mantel_test
import numpy as np
import scipy.stats as sps
from pybiomart import Server
from statistics import stdev, mean
from biothings_client import get_client


# ###sub-function of rename the first column of a data frame
def rename_column_fn(finputdf, column_index, new_name):
    finputdf = pd.DataFrame(finputdf)
    first_column_name = finputdf.columns[column_index]
    finputdf = finputdf.rename(columns={first_column_name: new_name})
    return finputdf


# ###the quantile normalization function
# ###a copy of the github version (https://github.com/ShawnLYU/Quantile_Normalize/blob/master/quantile_norm.py)
def quantile_normalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col: sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis=1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

# ###sub-function of mantel test; make sure finputD and fbenchCor with the same gene order
# #the parameter 'permutation_value' is not useful in the python version, so it is removed
def mantel_fn(finputdf, fbenchcordf, digits_num=3):
    # check whether id order is same between finputd and fbenchcor(does not have the gene symbol column)
    finputdf = pd.DataFrame(finputdf)
    fbenchcordf = pd.DataFrame(fbenchcordf)
    input_gene_list = list(finputdf.iloc[:, 0])
    bench_gene_list = list(fbenchcordf.columns)[1:]
    outs = []
    if input_gene_list == bench_gene_list:
        finputcordf = finputdf.iloc[:, 1:].transpose()
        finputcordf = finputcordf.corr(method="spearman")
        finputcordf.index = finputcordf.columns = list(finputdf.iloc[:, 0])
        outs = mantel_test.mant_test(finputcordf, fbenchcordf.iloc[:, 1:])
        outs["z.stat"] = round(outs["z.stat"], digits_num)
    else:
        print("The id order between finputD and fbenchCor is different. "
              "Please reset the id order.")
    return outs


# ###subfunction of running mantel test for each gene; zindexL is one gene with multiple sub-groups
def run_mantel_fn(fzindexls, fexpdf, fbenchcordf, fgorderls, fcvgenels):
    # zindexls: a list of data frames, containing column index for fexpdf, and 'gname', 'tag', 'zmean'
    # fexpdf: the expression matrix, a pandas data frame
    # benchcordf: the reference of correlation matrix, a pandas data frame (1st column is gene symbol)
    # gorderls: the order of clock gene list, a list
    # cvgenels: the input genes of nCV, a list
    fexpdf = pd.DataFrame(fexpdf)
    fexpdf.index = fexpdf.iloc[:, 0]
    fbenchcordf = pd.DataFrame(fbenchcordf)
    first_column_name = [fbenchcordf.columns[0]]
    fbenchcordf.index = fbenchcordf.iloc[:, 0]
    fbenchcordf = fbenchcordf.loc[fgorderls, (first_column_name + fgorderls)]

    def zstatd_sfn(sfzdf, sfexpdf, sfbenchcordf, sfgorderls, sfcvgenels):
        # sfzdf: a dataframe with columns as 'index', 'gname', 'tag', 'zmean'
        # sfexpdf: a data frame, with expression data
        tepdf = sfexpdf.loc[sfgorderls, ]
        tepindex = list(sfzdf.loc[:, "index"])
        checkdf = tepdf.iloc[:, tepindex]
        mant_checka = checkdf.apply(stdev, axis='columns')
        mant_checka = [num for num in list(mant_checka) if num == 0]
        if len(mant_checka) > 0:
            zstat = 0
            zcvdf = pd.DataFrame({"nCV": ["NA"]})
        else:
            zstat = mantel_fn(finputdf=tepdf.iloc[:, ([0] + tepindex)], fbenchcordf=sfbenchcordf)
            zexpdf = sfexpdf.iloc[:, tepindex]
            zsdv = zexpdf.apply(stdev, axis='columns')
            zmeanv = zexpdf.apply(mean, axis='columns')
            zcv = zsdv/zmeanv
            zncv = zcv/mean(zcv)
            zcvdf = pd.DataFrame({"geneSym": sfexpdf.iloc[:, 0], "cva": zcv})
            zcvdf.insert(2, "nCV", zncv)
            zcvdf.index = zcvdf.loc[:, "geneSym"]
            zcvdf = zcvdf.loc[sfcvgenels, ["geneSym", "nCV"]]
        # the unique gene symbol in sfzdf
        unigname = list(set(list(sfzdf.loc[:, "gname"])))
        unitag = list(set(list(sfzdf.loc[:, "tag"])))
        unimeanv = list(set(list(sfzdf.loc[:, "meanv"])))
        meanzncv = mean(zcvdf.loc[:, "nCV"])
        zout = pd.DataFrame({"gname": unigname, "tag": unitag, "zmean": unimeanv,
                             "zmantel": zstat['z.stat'], "zncv": meanzncv})
        zout.index = list(zout.loc[:, "tag"])
        zcvdf.insert(2, "rindex", ["0"]*zcvdf.shape[0])
        # pd.DataFrame.pivot() equal to tidyr::spread
        zcvdf = zcvdf.pivot(index='rindex', columns='geneSym', values='nCV')
        zcvdf.index = list(zout.loc[:, "tag"])
        zout = pd.concat([zout, zcvdf], axis=1, join="inner")      # axis=1, combine columns
        return zout

    zstatdf = [zstatd_sfn(sfzdf=sigindex, sfexpdf=fexpdf, sfbenchcordf=fbenchcordf, sfgorderls=fgorderls, sfcvgenels=fcvgenels) for sigindex in fzindexls]
    # zstatdf = map(partial(zstatd_sfn, sfexpdf=fexpdf, sfbenchcordf=fbenchcordf,
    #                       sfgorderls=fgorderls, sfcvgenels=fcvgenels), fzindexls)
    zstatdf = pd.concat(list(zstatdf), axis=0)
    return zstatdf


# ##subfunction of getting the correlation value
def top_tab_fn(fscreen_cordf, ftarget_measures, fonly_consistent_trend=False):
    # #screen_cordf: a pandas dataframe
    # #target_measure: a list
    fscreen_cordf = rename_column_fn(fscreen_cordf, 0, "geneSym")
    topdf = fscreen_cordf.loc[fscreen_cordf['measure'].isin(ftarget_measures)]
    top_groupls = list(topdf.groupby("geneSym"))

    # #top_groupdf is a list class, with each element as a tuple,
    # #within each tuple, the first element is geneSym, the second element is a dataframe
    # #to reach the sub-dataframe, trying top_groupdf[0][1], top_groupdf[1][1], ..., top_groupdf[n][1]
    def top_sfn(ztp, sf_octrend):
        zoutdf = ztp[1]
        # unique number of trend
        trendsigns = list(np.sign(list(zoutdf.loc[:, "corv"])))
        trendsigns = [str(sigt) for sigt in trendsigns]
        trendnum = len(set(trendsigns))
        uni_trend = "//".join(trendsigns)
        ltm_pre = mean(zoutdf.loc[:, "corv"])
        max_pva = max(zoutdf.loc[:, "p.value"])
        zoutdf['uniTrend'] = uni_trend
        zoutdf['LTM_pre'] = ltm_pre
        zoutdf['maxPva'] = max_pva
        if sf_octrend:
            zoutdf['trendNum'] = trendnum
            zoutdf = zoutdf[zoutdf['trendNum'] == 1]
            zoutdf = zoutdf.drop(columns=['trendNum'])
        return zoutdf

    # apply top_sfn to each element of top_groupls
    topdf = [top_sfn(ztp=tp, sf_octrend=fonly_consistent_trend) for tp in top_groupls]
    topdf = pd.concat(topdf, axis=0)
    topdf = topdf.drop(columns=['p.value'])
    mn_topdf = topdf.pivot(index='geneSym', columns=['measure'], values='corv')
    rm_topdf = topdf.drop(columns=['measure', 'corv'])
    rm_topdf = rm_topdf.drop_duplicates()
    rm_topdf.index = rm_topdf.loc[:, "geneSym"]
    topdf = pd.concat([rm_topdf, mn_topdf], axis=1, join='inner')
    topdf = topdf.sort_values(by=['uniTrend', 'LTM_pre'])
    target_cmns = ["geneSym", "uniTrend", "LTM_pre", "maxPva"]
    topdf = topdf.loc[:, (target_cmns + list(ftarget_measures))]
    return topdf


# ##subfunction of combing p-values by performing Fisher's method (source code in MetaCycle)
# #scipy.stats.combine_pvalues is python's Fisher's method;
# #the original zero_sub is 1e-100, change it to 1e-16; any pvalue less than 1e-16 means same
def fisher_method_fn(pvals, zero_sub=1e-16):
    cpvals = np.array(pvals)
    cpvals = np.where(cpvals < zero_sub, zero_sub, cpvals)
    rstat, rpva = sps.combine_pvalues(cpvals, method='fisher')
    return rpva


# ##subfunction of normalizing TCGA datasets
def tcga_tn_fn(finputdf, foutput_tn="tumor"):
    # the first column should be the gene symbol or other annotation name,
    # and other columns should be expression values
    finputdf = rename_column_fn(finputdf, 0, "Gene.Symbol")
    sampledf = pd.DataFrame({"sampleID": finputdf.columns[1:]})
    # get the tumor samples(outputTN = "tumor")/normal samples(outputTN = "normal")/
    # tumor-normal matched( outputTN = "paired")/all samples( outputTN = "all")
    if foutput_tn.lower() != "all":
        subjectid = [re.sub("(TCGA.\\S+.\\S+).\\d\\d\\w.\\d\\d\\w.\\S+", "\\1", sid) for sid in list(finputdf.columns[1:])]
        groupid = [re.sub("TCGA.\\S+.\\S+.(\\d\\d)\\w.\\d\\d\\w.\\S+", "\\1", sid) for sid in list(finputdf.columns[1:])]
        groupid = [int(sid) for sid in groupid]
        sampledf['subjectid'] = subjectid
        sampledf['groupid'] = groupid
        if foutput_tn.lower() == "tumor":
            sampledf = sampledf.loc[sampledf['groupid'] < 10]
        elif foutput_tn.lower() == "normal":
            sampledf = sampledf.loc[sampledf['groupid'] >= 10]
        elif foutput_tn.lower() == "paired":
            pairdict = {}
            pairkeys = list(sampledf.loc[:, "subjectid"])
            for sigkey in pairkeys:
                if sigkey in pairdict:
                    pairdict[sigkey] += 1
                else:
                    pairdict[sigkey] = 1
            pairdict = dict((k, v) for k, v in pairdict.items() if v == 2)
            sampledf = sampledf.loc[sampledf['subjectid'].isin(list(pairdict.keys()))]
        else:
            sampledf = pd.DataFrame([])
            print("Error: wrong parameter values for outputTN!!!"
                  "'outputTN' could be ('tumor', 'normal', 'paired', 'all')\n")
        finputdf = finputdf.loc[:, ["Gene.Symbol"]+list(sampledf.loc[:, "sampleID"])]
    return finputdf


# #####subfunctions of transforming the gene symbols to ensembl ID
# ##subfunction of mapping the ensembl and ensembl id
# ##modified the original version from https://autobencoder.com/2021-10-03-gene-conversion/
def get_ensembl_mappings(fhs_mm="Hs"):
    #server = biomart.BiomartServer(verbose=True)
    server = Server(host='http://www.ensembl.org')
    if fhs_mm.lower() == "hs":
        #mart = server.datasets['hsapiens_gene_ensembl']
        mart = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
        # List the targeted types of data
        fhs_attributes = ['ensembl_transcript_id', 'hgnc_symbol',
                      'ensembl_gene_id', 'ensembl_peptide_id']
    elif fhs_mm.lower() == "mm":
        #mart = server.datasets['mmusculus_gene_ensembl']
        mart = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['mmusculus_gene_ensembl'])
        # List the targeted types of data
        fhs_attributes = ['ensembl_transcript_id', 'mgi_symbol',
                      'ensembl_gene_id', 'ensembl_peptide_id']
    else:
        mart = []
        fhs_attributes = []
        print("Error: wrong parameter values for fhs_mm!!!"
              "'fhs_mm' could be ('Hs' or 'Mm')")
    # Get the mapping between the attributes
    #response = mart.search({'attributes': attributes})
    response = mart.query(attributes=fhs_attributes)
    #data = response.raw.data.decode('ascii')
    ensembl_to_genesymbol = {}
    genesymbol_to_ensembl = {}
    # Store the data in a dict
    for rnum in range(response.shape[0]):
        #line = line.split('\t')
        ## The entries are in the same order as in the `attributes` variable
        # transcript_id = line[0]
        # gene_symbol = line[1]
        # ensembl_gene = line[2]
        # ensembl_peptide = line[3]
        transcript_id = response.iloc[rnum, 0]
        gene_symbol = response.iloc[rnum, 1]
        ensembl_gene = response.iloc[rnum, 2]
        ensembl_peptide = response.iloc[rnum, 3]
        ensembl_to_genesymbol[transcript_id] = gene_symbol
        ensembl_to_genesymbol[ensembl_gene] = gene_symbol
        ensembl_to_genesymbol[ensembl_peptide] = gene_symbol
        if gene_symbol not in genesymbol_to_ensembl:
            genesymbol_to_ensembl[gene_symbol] = ensembl_gene
        else:
            genesymbol_to_ensembl[gene_symbol] = genesymbol_to_ensembl[gene_symbol] + "_" + ensembl_gene
    outdic = {"getGeneSym": ensembl_to_genesymbol, "getEnsemblId": genesymbol_to_ensembl}
    return outdic


# ##subfunction of mapping gene alias to official gene symbol
# ##not efficient, may need revise to improve the efficient
def alias_to_genesym_fn(fgname, sfspecies="hs"):
    outls = []
    mg = get_client('gene')
    if sfspecies.lower() == "hs":
        sfspecies = "human"
    elif sfspecies.lower() == "mm":
        sfspecies = "mouse"
    else:
        print("The 'sfspecies' should be set as 'hs' or 'mm'.")
    for sname in fgname:
        if isinstance(sname, float) or isinstance(sname, int):
            sname = str(sname) + '123456'
        tkey = re.sub("\\s+", "", sname)
        tkey = re.sub(r"\(", r"\\(", tkey)    # p33(CDK2) does not work, only p33\\(CDK2) or p33\(CDK2) work
        tkeysym = "symbol:" + tkey
        trysym = mg.query(tkeysym, species=sfspecies)
        if len(trysym['hits']) == 0:
            tkeyalias = "alias:" + tkey
            tryalias = mg.query(tkeyalias, species=sfspecies)
            if len(tryalias['hits']) == 0:
                outsym = ''
            else:
                outsym = [fhit['symbol'] for fhit in tryalias['hits']]
                outsym = "_".join(outsym)
        else:
            outsym = [fhit['symbol'] for fhit in trysym['hits']]
            outsym = "_".join(outsym)
        outls.append(outsym)
    return outls


# ######run above subfunctions to transform gene symbols to ensembl ID
def top_ensem_fn(findf, fspecies="Hs"):
    findf = rename_column_fn(findf, 0, "gname")
    topdf = findf
    topdf['geneSym'] = alias_to_genesym_fn(fgname=list(findf.loc[:, "gname"]), sfspecies=fspecies)
    # transform gene symbol to ENSEMBL_GENE_ID, which is much efficient for running DAVID online
    if (fspecies.lower() == "hs") or (fspecies.lower() == "mm"):
        ensembl = get_ensembl_mappings(fhs_mm=fspecies)
        ensembl = ensembl["getEnsemblId"]
        unigeneid = []
        for gsym in topdf['geneSym']:
            if gsym in ensembl.keys():
                unigeneid.append(ensembl[gsym])
            else:
                unigeneid.append("none")
        topdf['uniGeneID'] = unigeneid
        topdf = topdf[topdf['uniGeneID'] != "none"]
        return topdf
    else:
        print("This version only works for human and mouse. Please set 'fspecies' as 'Hs' or 'Mm'")
        return pd.DataFrame([])
