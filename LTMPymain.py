# ######The main function of LTMPy
import os
import sys
import time
import numpy as np
import pandas as pd
import LTMPysub as subf
import scipy.stats as sas
from multiprocessing import Pool
from functools import partial
from statistics import stdev, mean, median
from statsmodels.stats.multitest import multipletests


# ###Prepare the input data for LTM analysis
# ##This function can do quantile normalization,
# ##get the representative profile for rows with duplicated gene symbols,
# ##filter out genes with low expression values,
# ##and blunt the outliers of the given expression matrix.
def ltm_prep_fn(prepdf, quant_norm=True, uni_style="mad", remove_low_quant=0.1,
                blunt_low_quant=0.025, blunt_high_quant=0.975,
                digits_num=4, out_file_sym="", out_dir=""):
    # @param prepdf  a data frame. The first column is the gene symbol, and other columns are samples. One row per gene.
    # @param quant_norm logical. Set 'True' will do quantile normalization across samples.
    # @param uni_style a character string. Select the way of getting the representative profile for rows with duplicated gene symbols.
        # It must be "mad"(default), "sum", or "none", represent 'select the row with max mean absolute deviation', 'sum up duplicate rows'
        # or 'there is no duplicate gene name, and skip the step of getting the representative profile'.\
    # @param remove_low_quant a numeric value. Filter out genes based on their median expression value. If the median expression value locate below the 0.25(default) quantile,
        # this gene will be filtered out. Set it to 0 for keeping all the input genes.
    # @param blunt_low_quant a numeric value. Blunt the outlier value across samples gene by gene. If an expression value is below the \code{0.025}(default) quantile,
        # this outlier value will be blunted.
    # @param blunt_high_quant a numeric value. Blunt the outlier value across samples gene by gene. If an expression value is above the \code{0.975}(default) quantile,
        # this outlier value will be blunted.
    # @param digits_num an integer value. The integer indicates the number of decimal places to be used in the prepared data frame.
    # @param out_file_sym a character string. A prefix exists in the name of output file.
        # Set it \code{NULL}(default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store output file.

    # ##the first column should be the gene symbol or other annotation name, and other columns should be expression values
    prepdf = subf.rename_column_fn(prepdf, 0, "geneSym")
    inputdf = prepdf
    inputdf.index = list(inputdf["geneSym"])
    # ##########quantile normalization
    # #######Please note that there are minor differences for the normalization results between R and Python version
    # #######This needs to be fixed in the future (e.g., translate the original c code in preprocessCore package to the python code)
    if quant_norm:
        # #adjust 0 as related it to digitsNum
        prepm = inputdf.iloc[:, 1:] + 10**(-1*digits_num)
        prepm = np.log2(prepm)
        normdf = subf.quantile_normalize(df_input=prepm)
        normdf = 2**normdf
        normdf.insert(0, "geneSym", prepdf['geneSym'])
        inputdf = normdf
        #print("There may be some difference (estimated ~0.1% of all values) in the quantile normalization results from LTMPy compared to LTMR")
    # ##remove redundant symbols
    if uni_style.lower() == "sum":
        input_sdv = inputdf.iloc[:, 1:]
        input_sdv = input_sdv.apply(stdev, axis='columns')
        inputdf.insert(inputdf.shape[1], "stdv", input_sdv)
        # #filter out empty gene symbols
        inputdf = inputdf[inputdf["geneSym"] != "---"]
        inputdf = inputdf[inputdf["geneSym"] != ""]
        # #filter out gene symbols as nan
        inputdf = inputdf[inputdf["geneSym"].notnull()]
        # #filter out sdv as 0
        inputdf = inputdf[inputdf["stdv"] != 0]
        # #remove stdv colum
        inputdf = inputdf.drop(columns=["stdv"])
        # #get the sum value of rows with same gene symbol
        outdf = inputdf.groupby(by=["geneSym"]).sum()
        outdf.insert(0, "geneSym", outdf.index)
    elif uni_style.lower() == "mad":
        input_sdv = inputdf.iloc[:, 1:]
        input_sdv = input_sdv.apply(stdev, axis='columns')
        inputdf.insert(inputdf.shape[1], "stdv", input_sdv)
        # #filter out empty gene symbols
        inputdf = inputdf[inputdf["geneSym"] != "---"]
        inputdf = inputdf[inputdf["geneSym"] != ""]

        # #filter out gene symbols as nan
        inputdf = inputdf[inputdf["geneSym"].notnull()]
        # #filter out sdv as 0
        inputdf = inputdf[inputdf["stdv"] != 0]
        # #remove stdv colum
        inputdf = inputdf.drop(columns=["stdv"])
        # #get the mad value
        input_mad = inputdf.iloc[:, 1:]
        inputdf.insert(1, "madv", input_mad.mad(axis="columns"))
        # #select the max mad if redundant symbols exist
        outdc = {}
        for rowindex in range(inputdf.shape[0]):
            sigsym = inputdf.iloc[rowindex, 0]
            if sigsym not in outdc.keys():
                outdc[sigsym] = list(inputdf.iloc[rowindex, :])
            else:
                mada = outdc[sigsym][1]
                madb = inputdf.iloc[rowindex, 1]
                if madb > mada:
                    outdc[sigsym] = list(inputdf.iloc[rowindex, :])
        # #combine the dictionary as a data frame
        outdf = pd.DataFrame(outdc)
        # #transpose row to column, and rename the column names, then drop the madv column
        outdf = outdf.transpose()
        outdf.columns = inputdf.columns
        outdf = outdf.drop(columns=["madv"])
    elif uni_style.lower() == "none":
        outdf = inputdf
    else:
        print("Error:please set uni_style to 'mad', 'sum' or 'none'.\n")
    # ##remove values in the low quantile
    median_df = outdf.iloc[:, 1:]
    medv = median_df.apply(median, axis="columns")
    low_cut = np.quantile(medv, remove_low_quant)
    # #get the index above low quantile cut
    row_index = [tepi for tepi in range(len(medv)) if medv[tepi] > low_cut]
    outdf = outdf.iloc[row_index, :]

    # #the low blunt function
    def blow_fn(zv, quantv, digv, trendv):
        cutv = np.quantile(zv, quantv)
        # #blunt low if trendv is 0
        if trendv == 0:
            zv = [cutv if sigv < cutv else sigv for sigv in zv]
        # #blunt high if trendv is 0
        elif trendv == 1:
            zv = [cutv if sigv > cutv else sigv for sigv in zv]
        else:
            print("Please set trendv as 0 or 1.")
        zv = [round(sigv, digv) for sigv in zv]
        return zv

    # #get the dataframe without the gene symbol column
    outbf = outdf.iloc[:, 1:]
    # ##blunt low outliers
    if (blunt_low_quant > 0) & (blunt_low_quant < 1):
        # #result_type='broadcast' will ensure the same shape result
        outbf = outbf.apply(blow_fn, args=(blunt_low_quant, digits_num, 0), axis="columns", result_type='broadcast')
    # ##blunt high outliers
    if (blunt_high_quant > 0) & (blunt_high_quant < 1):
        outbf = outbf.apply(blow_fn, args=(blunt_high_quant, digits_num, 1), axis="columns", result_type='broadcast')
    # #add the first column
    outbf.insert(0, "geneSym", outdf.loc[:, "geneSym"])
    outdf = outbf
    # ##check and remove those rows with constant values again
    outbf = outbf.iloc[:, 1:]
    check_sdv = outbf.apply(stdev, axis="columns")
    # #get the index not with constant values
    index_row = [tepi for tepi in range(len(check_sdv)) if check_sdv[tepi] != 0]
    outdf = outdf.iloc[index_row, :]
    # ##output the file
    if len(out_file_sym) > 0:
        if len(out_dir):
            out_file_name = out_dir + "/" + out_file_sym + ".LTMprep.csv"
        else:
            out_file_name = out_file_sym + ".LTMprep.csv"
        outdf.to_csv(out_file_name, index=False)
    # ##output the data frame
    if len(out_file_sym) == 0:
        return outdf


# ###Prepare the input data for a quick run of LTMheat
# ##This function will select genes with high expression values
# ##and large expression variations across samples for a quick run of LTMheat.
def ltm_cut_fn(cutdf, exempt_genes, min_exp=-1e10,
               min_fold=1, out_file_sym="", out_dir=""):
    # @param cutdf  a data frame. The first column is the gene symbol, and other columns are samples. One row per gene.
    # @param exempt_genes a character vector. A list of clock genes that will be kept in the output data frame without getting into the filtration process.
    # @param min_exp a numeric value. The minimal median expression value in quantile group 1.
    # @param min_fold a numeric value. The minimal fold change between quantile group 4 and quantile group 1. Median value in each quantile group is used.
    # @param out_file_sym a character string. A prefix exists in the name of output file.
        # Set it ""(default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store output file.

    # ##rename the first column name
    cutdf = subf.rename_column_fn(cutdf, 0, "geneSym")
    # ##separate the data frame
    testdf = cutdf[~cutdf['geneSym'].isin(exempt_genes)]
    exemptdf = cutdf[cutdf['geneSym'].isin(exempt_genes)]
    out_index = []
    for rowi in range(testdf.shape[0]):
        qexpv = list(testdf.iloc[rowi, 1:])
        qexpv = np.quantile(qexpv, [0.125, 0.875])
        if (qexpv[0] >= min_exp) and (qexpv[1]/qexpv[0] >= min_fold):
            out_index.append(rowi)
    testdf = testdf.iloc[out_index, :]
    outdf = pd.concat([testdf, exemptdf], axis=0)
    # ##output the file
    if len(out_file_sym) > 0:
        if len(out_dir):
            out_file_name = out_dir + "/" + out_file_sym + ".LTMcut.csv"
        else:
            out_file_name = out_file_sym + ".LTMcut.csv"
        outdf.to_csv(out_file_name, index=False)
    # ##output the data frame
    if len(out_file_sym) == 0:
        return outdf


# ###Prepare the screen and sample data used for running LTMcook
# ##This function will separate the samples into given number of quantile groups,
# ##calculate the mantel's zstat value, nCV values of given clock gene list, gene by gene.
# ##It will output a list with screen and sample data frame used for running LTMcook.
def ltm_heat_fn(heatdf, benchdf, cv_genes, out_file_sym="",
                out_dir="", qnum=4, ncores=1, release_note=True):
    # @param heatdf  a data frame. The first column is the gene symbol, and other columns are samples. One row per gene.
    # @param benchdf a data frame. The expression correlation values of paired clock genes.
    # @param cv_genes a vector string. The list of clock genes representing the oscillation robustness of circadian clock.
    # @param out_file_sym a character string. A prefix exists in the name of output file.
        # Set it \code{NULL}(default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store output file.
    # @param qnum a numeric value. The number of quantile groups when separate samples based on expression level gene by gene.
    # @param ncores an integer. Number of cores to use. For parallel computing, set \code{nCores} a integer larger than 1.
    # @param release_note a logical value. If TRUE, time used during the analysis will be released on the screen.
    # Note: when there are tier values between quantile group boundaries (e.g. Q2 and Q3), there may cause difference in the output results
    #       the reason is that 'R:dplyr:ntile' treats tiers different with 'Python:pd.qcut'

    run_start = time.time()
    # #check the benchdf
    benchdf = subf.rename_column_fn(benchdf, column_index=0, new_name="geneSym")
    if benchdf.shape[0] != len(set(list(benchdf["geneSym"]))):
        sys.exit("Please make sure that there is no duplciate names in the first column of 'benchdf'.\n")
    benchdf.index = list(benchdf["geneSym"])
    # #the output data and sample data
    datadf = pd.DataFrame(heatdf)
    datadf = subf.rename_column_fn(datadf, column_index=0, new_name="geneSym")
    if datadf.shape[0] != len(set(list(datadf["geneSym"]))):
        sys.exit("Please make sure that there is no duplciate names in the first column of each data frame of the 'heatdf'.\n")
    datadf.index = list(datadf["geneSym"])
    cnames = datadf.columns
    commondf = datadf.merge(pd.DataFrame({"geneSym": list(benchdf["geneSym"])}), how='inner', on='geneSym')
    # #run gene one by one

    def get_index_fn(fza, fqcut):
        oria = fza
        fza = list(fza[1:])
        fza = [float(sigx) for sigx in fza]
        zout = pd.DataFrame({"expv": fza, "index": list(range(1, len(oria)))})
        tag = pd.qcut(fza, q=fqcut, labels=range(1, fqcut + 1))
        tag = ["Q"+str(sigv) for sigv in tag]
        zout['tag'] = tag
        zout['gname'] = str(oria[0])
        zout_grp = list(zout.groupby("tag"))

        # #zout_grp is a list class, with each element as a tuple,
        # #within each tuple, the first element is quantile group name, the second element is a dataframe
        # #to reach the sub-dataframe, trying zout_grp[0][1], zout_grp[1][1], ..., top_groupdf[n][1]
        def zout_sfn(ztp):
            szout = ztp[1]
            smean = mean(szout["expv"])
            szout["meanv"] = [smean]*(szout.shape[0])
            szout = szout.drop(columns=['expv'])
            return szout
        # #return the index data frame of quantile groups
        zout_indexl = [zout_sfn(ztp=tp) for tp in zout_grp]
        return zout_indexl

    # #the list of index data frame by applying get_index_fn function
    indexls = datadf.apply(get_index_fn, fqcut=qnum, axis='columns')
    # # the mantel test, cv calculation
    if ncores > 1:
        # the mantel test, cv calculation, parallel version
        with Pool(processes=ncores) as pool:
            # partial can be used to set constant values to all arguments which are not changed during parallel processing
            run_mantel_x = partial(subf.run_mantel_fn, fexpdf=datadf, fbenchcordf=benchdf,
                                   fgorderls=list(commondf["geneSym"]), fcvgenels=cv_genes)
            outputdf = pool.map(run_mantel_x, indexls)
            outputdf = pd.concat(list(outputdf), axis=0)
            pool.close()
            pool.join()
    else:
        outputdf = [subf.run_mantel_fn(fzindexls=sigls, fexpdf=datadf, fbenchcordf=benchdf,
                                         fgorderls=list(commondf["geneSym"]), fcvgenels=cv_genes) for sigls in indexls]
        # outputdf = map(partial(subf.run_mantel_fn, fexpdf=datadf, fbenchcordf=benchdf,
        #                        fgorderls=list(commondf["geneSym"]), fcvgenels=cv_genes), indexls)
        outputdf = pd.concat(list(outputdf), axis=0)
    # # get the sample list

    def sample_index_fn(zinls, zcnames):
        sample_list = list()
        for zdf in zinls:
            zindex = zdf['index']
            zsampledf = zdf
            zsampledf["sampleID"] = zcnames[zindex]
            zsampledf = zsampledf.drop(columns=["index", "meanv"])
            sample_list.append(zsampledf)
        zoutdf = pd.concat(sample_list, axis=0)
        return zoutdf
    sampledf = [sample_index_fn(zinls=sigls, zcnames=cnames) for sigls in indexls]
    sampledf = pd.concat(sampledf, axis=0)
    # # output the files
    if len(out_file_sym) > 0:
        if len(out_dir) > 0:
            out_file_name = out_dir + "/" + out_file_sym + ".Q" + str(qnum) + ".screen.csv"
            samp_file_name = out_dir + "/" + out_file_sym + ".Q" + str(qnum) + ".sample.csv"
        else:
            out_file_name = out_file_sym + ".Q" + str(qnum) + ".screen.csv"
            samp_file_name = out_file_sym + ".Q" + str(qnum) + ".sample.csv"
        outputdf.to_csv(out_file_name, index=False)
        sampledf.to_csv(samp_file_name, index=False)
    # # print the release note
    if release_note:
        print("The LTMheat analysis is done.\n")
        print("--- %s seconds ---" % (time.time() - run_start))
    # # return a dict instead of outputing files
    if len(out_file_sym) == 0:
        return {"screen": outputdf, "sample": sampledf}


# ###Prepare the data for running LTMdish
# ##This function calculates the correlation value between expression level and measures of clock strength, gene by gene.
# ##The input screen data is from LTMheat.
# ##test the ltm_cook step
def ltm_cook_fn(cookdf, cor_method="pearson", out_file_sym="", out_dir=""):
    # @param cookdf  a data frame. The screen data from LTMheat.
    # @param cor_method a character string. Select the correlation coefficient is to be used for
        # the correlation test between expression level and measures of clock strength.
        # It must be \code{"pearson"}(default), \code{"kendall"}, or \code{"spearman"}.
    # @param out_file_sym a character string. A prefix exists in the name of output file.
        # Set it \code{NULL}(default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store output file.

    # #check the correlation method
    if len(cor_method) and (cor_method.lower() in ["pearson", "kendall", "spearman"]):
        screendf = subf.rename_column_fn(cookdf, 0, "geneSym")
        screen_grpl = list(screendf.groupby("geneSym"))
        # #calculate the correlation value for each measure, screen_grpl is a tuple type

        def get_corv_fn(zdf, zcor_method):
            zcnames = list(zdf.columns)
            zcnames = [signame for signame in zcnames if signame not in ["geneSym", "tag", "zmean"]]
            zgenesym, zmeasure, zcorv, zpva = [[] for siga in range(4)]
            for sigmeasure in zcnames:
                zgenesym.append("".join(set(zdf["geneSym"])))
                zmeasure.append(sigmeasure)
                if zcor_method == "pearson":
                    sigcor, sigp = sas.pearsonr(np.array(zdf["zmean"]), np.array(zdf[sigmeasure]))
                elif zcor_method == "spearman":
                    sigcor, sigp = sas.spearmanr(np.array(zdf["zmean"]), np.array(zdf[sigmeasure]))
                elif zcor_method == "kendall":
                    sigcor, sigp = sas.kendalltau(np.array(zdf["zmean"]), np.array(zdf[sigmeasure]))
                else:
                    sigcor = sigp = np.nan
                zcorv.append(sigcor)
                zpva.append(sigp)
            zdcdf = pd.DataFrame({"geneSym": zgenesym, "measure": zmeasure,
                                  "corv": zcorv, "p.value": zpva})
            return zdcdf
        screen_cordf = [get_corv_fn(zdf=sigdf[1], zcor_method=cor_method) for sigdf in screen_grpl]
        screen_cordf = pd.concat(screen_cordf, axis=0)
        # #output the correlation file
        if len(out_file_sym):
            if len(out_dir):
                out_file_name = out_dir + "/" + out_file_sym + ".screenCorv.csv"
            else:
                out_file_name = out_file_sym + ".screenCorv.csv"
            screen_cordf.to_csv(out_file_name, index=False)
        else:
            return screen_cordf
    else:
        print("Please set 'cor_method' as 'pearson', 'kendall', or 'spearman'.\n")


# ###Generate the LTM analysis result for the input expression dataset
# ##This function calculates the LTM_abs value for each gene, which indicates the correlation between gene expression and clock strength.
def ltm_dish_fn(dishls, target_measures=("zmantel", "zncv"),
                only_consistent_trend=False, file_name="", out_dir=""):
    # @param dishls  a list of data frame. Each data frame is from LTMcook output.
    # @param target_measures a vector of character string. The target measures of clock strength used to calculate LTM_abs value for each gene.
    # @param only_consistent_trend logical. Set \code{TRUE} will only keep those genes show consistent trend across targetMeasures.
    # @param file_name a character string. The name of output file. Set it \code{NULL}(default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store output file.

    # ##get the LTM_pre value
    outdf = [subf.top_tab_fn(fscreen_cordf=sig_cordf, ftarget_measures=target_measures,
                               fonly_consistent_trend=only_consistent_trend) for sig_cordf in dishls]
    outdf = pd.concat(outdf, axis=0)
    outdf.index = list(range(outdf.shape[0]))
    # ##get the LTM_ori and LTM_abs value
    ltm_ori = outdf.groupby("geneSym")['LTM_pre'].mean()
    ltm_pva = outdf.groupby("geneSym")['maxPva'].max()
    ltm_bhq = multipletests(ltm_pva, method="fdr_bh")
    # #multipletests outputs a tuple data, with four elements, the second element is the bhq value
    ltm_bhq = ltm_bhq[1]
    ltm_abs = ltm_ori.abs()
    if list(ltm_ori.index) == list(ltm_pva.index):
        outdf = pd.DataFrame({"geneSym": list(ltm_ori.index),
                              "LTM_ori": ltm_ori, "LTM_pvalue": ltm_pva,
                              "LTM_BH.Q": ltm_bhq, "LTM_abs": ltm_abs})
        outdf = outdf.loc[:, ["geneSym", "LTM_ori", "LTM_abs"]]
        # #sort by LTM_abs, from high to low
        outdf = outdf.sort_values(by="LTM_abs", ascending=False)
    # ##output the file or return a data frame
    if len(file_name):
        if len(out_dir):
            out_file_name = out_dir + "/" + file_name
        else:
            out_file_name = file_name
        outdf.to_csv(out_file_name, index=False)
    else:
        return outdf


# ###Integrate the LTM analysis results of multiple datasets.
# ##This function get the meta values by integrating LTM analysis results of multiple datasets, e.g., multiple skin datasets.
def ltm_meta_fn(metals, file_name="", out_dir=""):
    # @param metals a list of data frame. Each data frame is from LTMdish output.
    # @param file_name a character string. The name of output file. Set it "" (default) if prefer to return a data frame.
    # @param out_dir a character string. The name of directory used to store the output file.

    tags = ["meta" + str(mid) for mid in range(len(metals))]
    metadf = []
    for mid, metal in enumerate(metals):
        ztepdf = metal
        ztepdf = subf.rename_column_fn(ztepdf, 0, "geneSym")
        ztepdf["grp"] = [tags[mid]]*(ztepdf.shape[0])
        metadf.append(ztepdf)
    metadf = pd.concat(metadf, axis=0)
    # #integrate result
    gene_sym = list(metadf["geneSym"])
    nona_index = np.where(pd.notna(gene_sym))
    combdf = metadf.iloc[list(nona_index[0]), :]
    sum_ltm_ori = combdf.groupby("geneSym")["LTM_ori"].sum()
    meta_ltm_ori = sum_ltm_ori / len(tags)
    meta_ltm_abs = meta_ltm_ori.abs()
    outdf = pd.DataFrame({"geneSym": meta_ltm_ori.index})
    outdf["meta_LTM_ori"] = list(meta_ltm_ori)
    outdf["meta_LTM_abs"] = list(meta_ltm_abs)
    # #arrange based on the meta_LTM_abs value, from high to low
    outdf = outdf.sort_values(by="meta_LTM_abs", ascending=False)
    # #output the file or return a data frame
    if len(file_name):
        if len(out_dir):
            out_file_name = out_dir + "/" + file_name
        else:
            out_file_name = file_name
        outdf.to_csv(out_file_name, index=False)
    else:
        return outdf


# ###Delete unnecessary files
# ##This function helps delete intermediate files generated during LTM analysis.
def ltm_wash_fn(wash_dir, wash_key=None, check_wash_list=False):
    # @param wash_dir a character string. The name of directory used to store intermediate files during LTM analysis.
    # @param wash_key a character string. The shared key word among targeted files under deletion.
    # @param check_wash_list logical. Set True will return a list of files.

    all_files = os.listdir(wash_dir)
    if wash_key is not None:
        del_files = [sigfile for sigfile in all_files if wash_key in sigfile]
    else:
        del_files = all_files
    del_files = [(wash_dir + "/" + sigfile) for sigfile in del_files]
    if len(del_files):
        zflag = []
        for sigfile in del_files:
            os.remove(sigfile)
            if os.path.isfile(sigfile):
                zflag.append(False)
            else:
                zflag.append(True)
        if check_wash_list:
            return pd.DataFrame({"file": del_files, "flag": zflag})
    else:
        print("No file is deleted.")
