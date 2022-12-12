######The main function of LTMPy
import pandas as pd
import LTMPymain as mainf

##test the ltm_prep step
exampledf = pd.read_csv("~/Desktop/example.csv")
out_prepdf = mainf.ltm_prep_fn(prepdf=exampledf, quant_norm=True, uni_style="mad", remove_low_quant=0.1,
                              blunt_low_quant=0.025, blunt_high_quant=0.975, out_dir="~/Desktop", out_file_sym="example")

##test the ltm_cut step
cutdf = pd.read_csv("~/Desktop/example.LTMprep.csv")
mclockdf = pd.read_csv("~/Desktop/mClockBench.csv")
mainf.ltm_cut_fn(cutdf, exempt_genes=list(mclockdf["geneSym"]), min_exp=10,
               min_fold=1.5, out_file_sym="example", out_dir="~/Desktop")

##test the ltm_heat step
outcutdf = pd.read_csv("~/Desktop/example.LTMcut.csv")
mclockdf = pd.read_csv("~/Desktop/mClockBench.csv")
mainf.ltm_heat_fn(heatdf=outcutdf, benchdf=mclockdf, cv_genes=list(mclockdf["geneSym"]),
                              out_file_sym="example", out_dir="~/Desktop", qnum=5, ncores=10, release_note=True)

###test the ltm_cook step
outheata = pd.read_csv("~/Desktop/example.Q4.screen.csv")
mainf.ltm_cook_fn(cookdf=outheata, cor_method="pearson", out_file_sym="example.Q4", out_dir="~/Desktop")
outheatb = pd.read_csv("~/Desktop/example.Q5.screen.csv")
mainf.ltm_cook_fn(cookdf=outheatb, cor_method="pearson", out_file_sym="example.Q5", out_dir="~/Desktop")

###test the ltm_dish step
outcooka = pd.read_csv("~/Desktop/example.Q4.LTMcook.csv")
outcookb = pd.read_csv("~/Desktop/example.Q5.LTMcook.csv")
mainf.ltm_dish_fn(dishls=[outcooka, outcookb], target_measures=("zmantel", "zncv"),
                only_consistent_trend=False, file_name="example_LTMdish.Py.csv", out_dir="~/Desktop")

###test the ltm_meta step
outdisha = pd.read_csv("~/Desktop/example_LTMdish.Py.csv")
outdishb = pd.read_csv("~/Desktop/example_LTMdish.Py.csv")
mainf.ltm_meta_fn(metals=[outdisha, outdishb], file_name="example.LTMmeta.Py.csv", out_dir="~/Desktop")
