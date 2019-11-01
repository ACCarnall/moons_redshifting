import numpy as np
import pandas as pd
import sys

run = sys.argv[1]

results = pd.read_csv("best_z_" + run + ".txt", sep="\t")

results["mag"] = np.zeros(results.shape[0])

data = results["#ID"].str.split("_")
results["type"] = data.str[0]
results["dust"] = data.str[1]
results["z_input"] = data.str[3].str[1:]
results["veldisp"] = data.str[4].str[1:]
results["mag"] = data.str[5].str[1:]

results["dust"] = results["dust"].str.replace("noAv", "0.0")
results["dust"] = results["dust"].str.replace("lowAv", "0.2")
results["dust"] = results["dust"].str.replace("midAv", "0.5")
results["dust"] = results["dust"].str.replace("highAv", "1.0")

results["#ID"] = results["#ID"].str[:-8]

results.to_csv("final_" + run + ".cat", sep="\t", index=False)
