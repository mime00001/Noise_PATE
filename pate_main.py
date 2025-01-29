from pate_utils import analyze_multiclass_confident_gnmax, query
import numpy as np
import pandas as pd
import os

#taken from https://github.com/cleverhans-lab/PrivatePrompts/tree/main/PromptPATE/pate

def get_how_many_answered(predicted_labels):
    """
    This function counts how many labels we obtained. We provide the raw output from the PATE
    """
    total_len = len(predicted_labels)
    answered = np.mean(predicted_labels!=-1)

    return answered


def get_how_many_correctly_answered(predicted_labels, true_labels):
    """
    Given the noisy predictions from the teachers and the true validation labels, we get the number of correctly answered labels.
    """
    
    condition = lambda x : x == -1
    
    
    arr1 = [x for x in predicted_labels if not condition(x)]
    arr2 = [y for x, y in zip(predicted_labels, true_labels) if not condition(x)]
    
    correct_indices = np.equal(arr1, arr2)
    
    number_correct_indices = np.mean(correct_indices)
    return number_correct_indices


def get_final_noisy_labels(noisy_labels, indices_answered, max_num_query):
    """
    Based on the noisy labels obtained from the query to PATE, the information which indices were not rejected, and how
    many queries we can answer with the privacy budget, it returns an array that has -1 everywhere where we cannot answer
    (due to reject or budget exhausted) and the noisy label everywhere else.
    """

    actually_answered = np.ones(noisy_labels.shape) * -1

    # exclude the queries that we could not answer due to too confident gnmax pre-filtering:
    actually_answered[indices_answered] = noisy_labels[indices_answered]

    # exclude the queries that we cannot answer anymore because we ran out of budget:
    actually_answered[max_num_query:] = -1
    actually_answered = np.asarray(actually_answered, dtype=int)

    return actually_answered

def get_actually_consumed_epsilon(dp_eps):
    """
    This function tell us what epsilon we actually consumed.
    If the last number in the dp_eps is zero, this means that the budget was exhausted before the end of our queries.
    In this case, we need to take the second to last non-zero element. (Because the last element is above our target epsilon).
    In the case the last element is not zero, then we can return the last element.
    """
    if sum(dp_eps) == 0:
        print("No significant privacy costs incurred. Probably delta is quite large.")
        consumed_epsilon = 0.0
    elif dp_eps[-1] == 0.:
        consumed_epsilon = np.partition(dp_eps.flatten(), -2)[-2]
    else:
        consumed_epsilon = dp_eps[-1]

    return consumed_epsilon

def tune_pate(vote_array, threshold_list, sigma_threshold_list, sigma_gnmax_list, epsilon_list, delta_list, num_classes=2, savepath='', true_labels=None):
    """
    This function iterates through many parameter combinations for finding right PATE hyperparameters.
    It will create a csv file in which all parameters are logged together with how many queries they allow us to answer,
    and how many are correctly answered.
    """
    header = ['target_epsilon',  'achieved_eps', 'threshold', 'sigma_threshold', 'sigma_noise', 'delta', 'num_classes', 'num_answered',
              'num_correctly_answered']
 
    for threshold in threshold_list:
        print("Currently at threshold: {}".format(threshold))
        for sigma_threshold in sigma_threshold_list:
            print("Currently at sigma_threshold: {}".format(sigma_threshold))
            for sigma_gnmax in sigma_gnmax_list:
                for epsilon in epsilon_list:
                    for delta in delta_list:
                        # this part is for the privacy accounting:
                        max_num_query, dp_eps, partition, answered, order_opt = analyze_multiclass_confident_gnmax(
                            votes=vote_array,
                            threshold=threshold,
                            sigma_threshold=sigma_threshold,
                            sigma_gnmax=sigma_gnmax,
                            budget=epsilon,
                            delta=delta,
                            file=None)

                        # this is for getting the labels
                        noisy_labels, indices_answered = query(vote_array, threshold, sigma_threshold, sigma_gnmax,
                                                               num_classes)

                        final_labels = get_final_noisy_labels(noisy_labels, indices_answered, max_num_query)

                        num_answered = get_how_many_answered(final_labels)
                        if true_labels != None:
                            num_correctly_answered = get_how_many_correctly_answered(final_labels, true_labels)
                        else:
                            num_correctly_answered = 0
                        achieved_epsilon = get_actually_consumed_epsilon(dp_eps)

                        write_results = [epsilon, achieved_epsilon, threshold, sigma_threshold, sigma_gnmax, delta, num_classes, num_answered, num_correctly_answered]
                        results_df = pd.DataFrame(write_results).T

                        results_df.columns = header
                        # this appends to the csv file that we have with mode 'a'
                        results_df.to_csv(savepath, mode='a', index=False,
                                                 header=not os.path.isfile(savepath))


def inference_pate(vote_array, threshold, sigma_threshold, sigma_gnmax, epsilon, delta, num_classes=10, savepath='', save=True):
    """
    This function, given a vote array and the best found hyperparameters infers the teacher's final aggregated private votes.
    It saves them at the save path.
    """
    # this part is for the privacy accounting:
    max_num_query, dp_eps, partition, answered, order_opt = analyze_multiclass_confident_gnmax(votes=vote_array,
                                                                                               threshold=threshold,
                                                                                               sigma_threshold=sigma_threshold,
                                                                                               sigma_gnmax=sigma_gnmax,
                                                                                               budget=epsilon,
                                                                                               delta=delta,
                                                                                               file=None)
    # this is for getting the labels
    noisy_labels, indices_answered = query(vote_array, threshold, sigma_threshold, sigma_gnmax, num_classes)
    achieved_epsilon = get_actually_consumed_epsilon(dp_eps)
    print(f"achieved epsilon: {achieved_epsilon} \trequested epsilon: {epsilon}")
    final_labels = get_final_noisy_labels(noisy_labels, indices_answered, max_num_query)
    if save:
        np.save(savepath, final_labels)
    
    return achieved_epsilon, final_labels
    #pd.DataFrame(final_labels).to_csv(savepath, index=False, header=None)


def main(vote_array, threshold, sigma_threshold, sigma_gnmax, epsilon, delta, num_classes, savepath='', tune_hyper=False, true_labels=None):


    # print(f"how many queries could we answer: {max_num_query}")
    # print(f"list of our (accumulated) epsilon: {dp_eps}")
    # #print(partition) # not really important for us
    # print(f"how many query do we expect to answer at any given point: {answered}")
    # print(f"what is the optimal RDP order at every step: {order_opt}")

    """
    @user, you can use this function for two purposes, either finding best hyperparameters of PATE, or once you
    know which hyperparameters you want, you can just infer the labels.
    """

    # To do an inference, do this here:
    if not tune_hyper:
        final_label_path = 'data_anthropic/agnews/public_labels_agnews.txt'
        final_labels = inference_pate(vote_array, threshold, sigma_threshold, sigma_gnmax, epsilon, delta, num_classes=num_classes, savepath=final_label_path)
        final_index = np.arange(len(final_labels))
        #final_index = np.loadtxt("data/trec/public_index_trec.txt")
        mask = (final_labels != -1)
        final_labels = final_labels[mask]
        final_index = final_index[mask]
        print(len(final_labels))
        #final_index = final_index[mask]
        print(len(final_labels), len(final_index))
        np.savetxt(final_label_path, final_labels, fmt="%i")
        np.savetxt("data_anthropic/agnews/public_index_agnews.txt", final_index, fmt="%i")
    # this is the other option: you tune PATE with a range of parameters.
    else:

        tune_pate(vote_array, threshold, sigma_threshold, sigma_gnmax, epsilon, delta,
                  num_classes=num_classes, savepath=savepath, true_labels=true_labels)
