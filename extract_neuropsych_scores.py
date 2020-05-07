# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import glob
import os
import os.path as op

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data_directory = "./data/cc700-scored"

scores = ['BentonFaces',
          'CardioMeasures',
          'Cattell',
          'EkmanEmHex',
          'EmotionRegulation',
          'EmotionalMemory',
          'FamousFaces',
          'ForceMatching',
          'Hotel',
          'MotorLearning',
          'PicturePriming',
          'Proverbs',
          'RTchoice',
          'RTsimple',
          'Synsem',
          'TOT',
          'VSTMcolour']


def read_data(score_name):
    fnames = glob.glob(
        op.join(data_directory, score_name, 'release001/data/*.txt'))

    dfs = list()
    for fname in sorted(fnames):
        df = pd.read_csv(fname, sep='\t')
        df['subject'] = fname.split('/')[-1].split('_')[1]
        dfs.append(df.set_index('subject'))
    return pd.concat(dfs)


results = list()
for score in scores:
    print(score)
    df = read_data(score)
    results.append(df)

all_subjects = list(set(sum([x.index.tolist() for x in results], [])))

out = pd.DataFrame(index=all_subjects)
for score, result in zip(scores, results):
    if score == 'BentonFaces':
        out[f"{score}_total"] = result['TotalScore']

    elif score == 'CardioMeasures':
        # Three physiologically distinct markers.
        for column in ('pulse_mean', 'bp_sys_mean', 'bp_dia_mean'):
            out[f"{score}_{column}"] = result[column]

    elif score == 'Cattell':
        out[f"{score}_total"] = result['TotalScore']

    elif score == 'EkmanEmHex':
        # facial expression discrimination and other
        # performance markers
        columns = ['MeanRT']
        tmp = pd.DataFrame(index=all_subjects)
        for column in columns:
            for emo in result['Type'].unique():
                tmp[f"{score}_{emo}_{column}"] = result.query(
                    f"Type == '{emo}'")[[column]]
        tmp = tmp.dropna()
        pca = PCA(n_components=1).fit(tmp)
        tmp['pc'] = pca.transform(tmp)
        out[f"{score}_pca1"] = tmp["pc"] 
        out[f"{score}_pca1_expv"] = pca.explained_variance_ratio_[0]
        print(
            score,
            pca.explained_variance_ratio_)

    elif score == 'EmotionalMemory':
        # compute sort of variance over valances
        # differences in neutral and emotional memory
        # are characteristic of age for the 3 different
        # memory types probed.
        for pattern in ('PriPr*', 'ValPr*', 'ObjPr*'):
            df_ = result.filter(regex=pattern)
            res = pd.DataFrame(index=df_.dropna().index)
            pca = PCA(n_components=1).fit(df_.dropna())
            res['pc'] = pca.transform(df_.dropna())
            out[f"{score}_{pattern[:-1]}_pca1"] = res['pc']
            print(
                score, 
                pattern,
                pca.explained_variance_ratio_)

    elif score == 'EmotionRegulation':
        # compute emotional reactivity by considering the difference
        # in responses to positive and negative films with regard
        # to positive or negative emotions. (should match).
        # Then compute regulation by assessing if people successfully
        # reppraised negative content by comparing negative emtions
        # in negative films naively watched and reappraised.

        out[f"{score}_reactivity_neg_emo"] = result.eval(
            "NegW_mean_neg - NegW_mean_pos")
        out[f"{score}_reactivity_pos_emo"] = result.eval(
            "PosW_mean_pos - PosW_mean_neg")
        out[f"{score}_regulation_neg_emo"] = result.eval(
            "NegW_mean_neg - NegR_mean_neg")

    elif score == 'FamousFaces':
        # check for capacity to remember details about familiar
        # persons. Account for total number of familiar persons
        # given max of 30 trials.
        df = result.query("Phase == 'FacesTest'")
        individual_max = 30 - df['FAMunfam']
        subcol = 'familiar_faces_details'

        details = pd.DataFrame(
            np.mean([df['FAMocc'] / individual_max,
                     df['FAMnam'] / individual_max], 0),
            index=df.index)
        out[f'{score}_details'] = details

    elif score == 'ForceMatching':
        # get force matching for slider (indirect) and lever (direct)
        # conditions. Both have different distributions
        out[f"{score}_force_match_direct"] = result[
            'FingerOverCompensationMean']

        out[f"{score}_force_match_indirect"] = result[
            'SliderOverCompensationMean']

    elif score == 'Hotel':
        # get difference from optimal time allocation for 5 uncompletable
        # tasks
        out[f"{score}_time"] = result['Time']

    elif score == 'MotorLearning':
        # motor learning
        # Trajectory Error is already calculated.
        # We can average over the phases of the experiment.
        out[f"{score}_trajectory_error_mean"] = result.filter(
            regex="TrajectoryErrorMean").mean(axis=1)

        out[f"{score}_trajectory_error_sd"] = result.filter(
            regex="TrajectoryErrorSD").std(axis=1)
    
    elif score == "PicturePriming":
        out[f"{score}_baseline_acc"] = result['ACC_baseline_all']
        out[f"{score}_baseline_rt"] = result['reITRT_baseline_all']

        sel1 = [
            'reITRT_baseline_low_phon',
            'reITRT_baseline_high_phon',
            'reITRT_baseline_low_sem',
            'reITRT_baseline_high_sem',
            'reITRT_baseline_unrel']

        sel2a = [
            'reITRT_priming_prime_low_phon',
            'reITRT_priming_prime_high_phon',
            'reITRT_priming_prime_low_sem',
            'reITRT_priming_prime_high_sem',
            'reITRT_priming_prime_unrel']

        sel2b = [
            'reITRT_priming_target_low_phon',
            'reITRT_priming_target_high_phon',
            'reITRT_priming_target_low_sem',
            'reITRT_priming_target_high_sem',
            'reITRT_priming_target_unrel']

        out[f"{score}_priming_prime"] = (
            result[sel1] - result[sel2a].values).mean(axis=1)
        out[f"{score}_priming_target"] = (
            result[sel1] - result[sel2b].values).mean(axis=1)

    elif score == "Proverbs":
        out[f"{score}"] = result["Score"]

    elif score == "RTchoice":
        out[score] = result["RTmean_all"]

    elif score == "RTsimple":
        out[score] = result["RTmean"]

    elif score == 'Synsem':
        out[f"{score}_prop_error"] = result.filter(
            regex='unacc_ERR').mean(axis=1)
        out[f"{score}_RT"] = result.filter(regex="_RT_").mean(axis=1)

    elif score == 'TOT':
        out[score] = result['ToT_ratio']

    elif score == 'VSTMcolour':
        # unclear how to interpret K1-4.
        # perceptual control block is maybe uninteresting
        result = result[~result.duplicated()]
        result = result[~result.index.duplicated()]
        out[f"{score}_K_mean"] = result.filter(regex="K").mean(axis=1)
        out[f"{score}_K_precision"] = result.filter(regex="Prcsn").mean(axis=1)
        out[f"{score}_K_doubt"] = result.filter(regex="Doubt").mean(axis=1)
        out[f"{score}_MSE"] = result.filter(regex="MSE").mean(axis=1)
    else:
        print("unmatched:", score)

for score in scores:
    print(score, out.filter(regex=score).shape[1])

out.to_csv("./data/neuropsych_scores.csv")