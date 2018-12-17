import numpy as np
import matplotlib.pyplot as plt

IMAGE_OUTPUT = "./RESULTS/IMAGES/"
DATA_OUTPUT = "./RESULTS/"

def dprime(gen_scores, imp_scores):
    x = np.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores))
    y = np.sqrt(np.power(np.std(gen_scores),2) + np.power(np.std(imp_scores),2))
    return x / y

def plot_scoreDist(gen_scores, imp_scores, run, show):
    plt.figure()
    plt.hist(gen_scores, color=['green', 'red'], lw=2, histtype='step', hatch='//', label='Genuine Scores')
    plt.hist(imp_scores, color='red', lw=2, histtype='step', hatch='\\', label='Impostor Scores')
    plt.legend(loc='best')
    dp = dprime(gen_scores, imp_scores)
    plt.xlim([0,1])
    plt.title('Score Distribution (d-prime= %.2f)' % dp)
    plt.savefig(IMAGE_OUTPUT + 'score_dist_ ' + str(run) + '.png', bbox_inches='tight')
    
    # show
    if show:
        plt.show()

def plot_det(far, frr, run, show):
    #compute eer
    far_minus_frr = 1
    eer = 0
    for i, j in zip(far, frr):
        if abs(i-j) < far_minus_frr:
            eer = i
            far_minus_frr = abs(i-j)
    plt.figure()
    plt.plot(far, frr, lw=2)
    plt.plot([0,1], [0,1], lw=1, color='black')
    plt.xlabel('false accept rate')
    plt.ylabel('false reject rate')
    plt.title('DET Curve (eer = %.2f)' % eer)
    plt.savefig(IMAGE_OUTPUT + 'det_' + str(run) + '.png', bbox_inches='tight')

    # show
    if show:
        plt.show()

def plot_roc(far, tpr, run, show):
    plt.figure()
    plt.plot(far, tpr, lw=2)
    plt.xlabel('false accept rate')
    plt.ylabel('true accept rate')
    plt.title('ROC Curve')
    plt.savefig(IMAGE_OUTPUT + 'roc_' + str(run) + '.png', bbox_inches='tight')

    # show
    if show:
        plt.show()

def perf_main(gen_scores, imp_scores, run, show=True, name=""):
    global IMAGE_OUTPUT
    IMAGE_OUPUT = IMAGE_OUTPUT + name

    # plot score distribution
    plot_scoreDist(gen_scores, imp_scores, run, show)

    # Separate into square
    thresholds = np.linspace(0, 1, 200)
    far = []
    frr = []
    tpr = []
    for t in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for g_s in gen_scores:
            if g_s <= t:
                tp += 1
            else:
                fn += 1
        for i_s in imp_scores:
            if i_s <= t:
                fp += 1
            else:
                tn += 1
        far.append(fp / (fp + tn))
        frr.append(fn / (fn + tp))
        tpr.append(tp / (tp + fn))

    # plot roc and det
    plot_roc(far, tpr, run, show)
    plot_det(far, frr, run, show)
    
    # return
    return debug
    
