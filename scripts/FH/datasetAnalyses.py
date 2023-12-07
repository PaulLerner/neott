from .basis_funcs import *


def analyze_term_dist(df, suffix=""):
    for corpusName in ("ft","wikt"):
        corsDF = df[(~df["has_"+corpusName].isna()) & (df.repn==0)]
        for name in ("basic_pos","Domain_"+corpusName):
            nameDist = corsDF.groupby(name)[name].count().sort_values(ascending=False)/len(corsDF)#.values()
            my_cmap = cm.get_cmap('jet')
            thresh = 0.0000001
            nameDistThresh = nameDist[nameDist > thresh]
            while len(nameDistThresh) > 25:
                nameDistThresh = nameDist[nameDist > thresh]
                thresh *= 2
                print(len(nameDistThresh),thresh)
            nameDist = nameDistThresh
            my_norm = Normalize(vmin=0, vmax=max(nameDist.values))
            color = my_cmap(my_norm(nameDist.values))
            fig = plt.figure(figsize=((14 if len(nameDist) > 10 else 10), 6), dpi=80)
            fig.tight_layout()
            plt.bar(np.arange(len(nameDist)),nameDist,color=color)
            plt.ylabel(name + " Distribution",fontsize=17)
            namesHacked = [x.replace(" ","\n") for x in nameDist.index]
            plt.xticks(np.arange(len(nameDist)),namesHacked,rotation=90,fontsize=(17 if len(nameDist) < 10 else 12))
            plt.title("Distribution of "+name+" for " + corpusName + " Science and Tech corpus",fontsize=20)# ("+str(100*round(percMissing,2))+"% missing)")
            fig.tight_layout()
            #plt.show()
            plt.savefig(imgPath + "/" + corpusName + "_" +name+"Dist" + suffix + ".png")
            plt.close()


def analyze_termLen_vs_POS(df,corpusName, suffix=""):
    vals = df.groupby("basic_pos")["termLen"].agg(lambda col: list(col)).reset_index()
    axes = fig, axes = plt.subplots(len(vals), 1, figsize=(8, 6))
    my_cmap = cm.get_cmap('jet')
    def histVal(row):

        my_norm = Normalize(vmin=0, vmax=max(row.termLen))
        color = my_cmap(my_norm(row.termLen))
        ax = axes[row.name];
        n, bins, patches = ax.hist(row.termLen)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', my_cmap(c))

        ax.set_title(row.basic_pos);
        ax.set_xticks(np.arange(max(row.termLen))+1)
        ax.set_xticklabels(np.arange(max(row.termLen))+1)
        ax.set_xlabel("Term Len")
        ax.set_ylabel("Count")
    vals.apply(histVal,axis=1);
    fig.tight_layout()

    plt.savefig(imgPath + "/" +  corpusName + "_" + "termLenVsPOS" + suffix + ".png")
