"""
Evaluation functions 
"""

from utils.metrics import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
import itertools as it
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import pandas as pd
from utils.torch_utils import move_batch_to_gpu


def disc_eval_method(label, prediction, num_class, metric="auc"):
    """
    Discriminator eval method

    Args:
        label (_type_): _description_
        prediction (_type_): _description_
        num_class (_type_): _description_
        metric (str, optional): _description_. Defaults to "auc".

    Raises:
        ValueError: _description_

    Returns:
        float: score
    """
    if metric == "auc":
        if num_class == 2:
            score = roc_auc_score(label, prediction, average="micro")
            score = max(score, 1 - score)
            return score
        else:
            lb = LabelBinarizer()
            classes = [i for i in range(num_class)]
            lb.fit(classes)
            label = lb.transform(label)
            score = roc_auc_score(label, prediction, multi_class="ovo", average="macro")
            score = max(score, 1 - score)
            return score
    else:
        raise ValueError("Unknown evaluation metric in disc_eval_method().")


def eval_multi_combination(model, data, metrics, fix_one=False):
    """
    Evaluate model on validation/test dataset under different filter combinations.
    The output is the averaged result over all the possible combinations.
    :param model: trained model
    :param data: validation or test data (not train data)
    :param fix_one: if true, only evaluate on one feature instead of all the combinations (save running time)
    :return: averaged evaluated result on given dataset
    """
    n_features = model.num_features
    feature_info = model.data_processor_dict["train"].feature_info

    if not fix_one:
        mask_list = [list(i) for i in it.product([0, 1], repeat=n_features)]
        mask_list.pop(0)
        # mask_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        feature_range = np.arange(n_features)
        shape = (feature_range.size, feature_range.max() + 1)
        one_hot = np.zeros(shape).astype(int)
        one_hot[feature_range, feature_range] = 1
        mask_list = one_hot.tolist()
        mask_list = [mask_list[1]]
    result_dict = {}
    acc_result = None
    final_grp_wise_result=defaultdict(list)

    for mask in mask_list:
        mask = np.asarray(mask)
        feature_idx = np.where(mask == 1)[0]
        f_name_list = [feature_info[i + 1].name for i in feature_idx]
        f_name = " ".join(f_name_list)

        cur_result,grp_wise_result = (
            evaluate(model, data, metrics, mask)
            if data is not None
            else [-1.0] * len(metrics)
        )
        acc_result = (
            np.array(cur_result)
            if acc_result is None
            else acc_result + np.asarray(cur_result)
        )

        result_dict[f_name] = cur_result
        for feature in ['age','gender','occupation']:
                final_grp_wise_result[feature]= np.array(grp_wise_result[feature]) if final_grp_wise_result[feature] ==[] else np.add(final_grp_wise_result[feature] , np.asarray(grp_wise_result[feature]))
        # print('multi_eval')
        # print(final_grp_wise_result)

    for feature in ['age','gender','occupation']:
        final_grp_wise_result[feature]/= len(mask_list)


    if acc_result is not None:
        acc_result /= len(mask_list)

    return list(acc_result), result_dict, final_grp_wise_result


@torch.no_grad()
def evaluate(model, batches, metrics, mask=None):
    """
    evaluate recommendation performance
    :param model:
    :param batches: data batches, each batch is a dict.
    :param mask: filter mask
    :param metrics: list of str
    :return: list of float number for each metric
    """
    model.eval()

    if mask is None:
        mask = [0] * model.filter_num
        mask = np.asarray(mask)

    result_dict = defaultdict(list)
    for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc="Predict"):
        batch = move_batch_to_gpu(batch)
        out_dict = model.predict(batch, mask)
        prediction = out_dict["prediction"]
        labels = batch["label"].cpu()
        #[gender, age, occ]
        genders = batch['features'][:, 0].cpu()
        ages = batch['features'][:, 1].cpu()
        occs = batch['features'][:, 2].cpu()
        users = batch['X'][:, 0].cpu()
        sample_ids = batch["X"][:, 2].cpu()
        assert len(labels) == len(prediction)
        assert len(sample_ids == len(prediction))
        prediction = prediction.cpu().numpy()
        # data_dict = {"label": labels, "sample_id": sample_ids}
        data_dict = {"label": labels, "sample_id": sample_ids, 'users': users, 'genders': genders,'ages': ages, 'occs': occs}
        results = evaluate_method(prediction, data_dict, metrics=metrics)
        for key in results:
            result_dict[key].extend(results[key])

    group_wise_df = pd.DataFrame.from_dict(result_dict)
    group_scores=defaultdict(list)
    # age-wise calculation
    age_df = group_wise_df.groupby('ages')
    for uid, group in age_df:
        group_scores['age'].append([group['ndcg@5'].agg(np.mean),group['ndcg@10'].agg(np.mean),group['hit@5'].agg(np.mean),group['hit@10'].agg(np.mean)])

    #gender-wise calculations
    gender_df = group_wise_df.groupby('genders')
    for uid, group in gender_df:
        group_scores['gender'].append([group['ndcg@5'].agg(np.mean),group['ndcg@10'].agg(np.mean),group['hit@5'].agg(np.mean),group['hit@10'].agg(np.mean)])

    #occupation-wise calculations
    occ_df = group_wise_df.groupby('occs')
    for _ , group in occ_df:
        group_scores['occupation'].append([group['ndcg@5'].agg(np.mean),group['ndcg@10'].agg(np.mean),group['hit@5'].agg(np.mean),group['hit@10'].agg(np.mean)])

    # print(group_scores)
    group_wise_df = pd.DataFrame(None)
    age_df = pd.DataFrame(None)
    gender_df = pd.DataFrame(None)
    occ_df = pd.DataFrame(None)
        
    evaluations = []
    for metric in metrics:
        evaluations.append(np.average(result_dict[metric]))

    return evaluations,group_scores


def evaluate_method(p, data, metrics):
    """
    Evaluate model predictions. Helper function to evaluate function
    :param p: predicted values, np.array
    :param data: data dictionary which include ground truth labels
    :param metrics: metrics list
    :return: a list of results. The order is consistent to metric list.
    """
    label = data["label"]
    evaluations = {}
    for metric in metrics:
        if metric == "rmse":
            evaluations[metric] = [np.sqrt(mean_squared_error(label, p))]
        elif metric == "mae":
            evaluations[metric] = [mean_absolute_error(label, p)]
        elif metric == "auc":
            evaluations[metric] = [roc_auc_score(label, p)]
        else:
            k = int(metric.split("@")[-1])
            df = pd.DataFrame()
            df["sample_id"] = data["sample_id"]
            df["p"] = p
            df["l"] = label
            df['users'] = data['users']
            df['genders'] = data['genders']
            df['ages'] = data['ages']
            df['occs'] = data['occs']
            df = df.sort_values(by="p", ascending=False)
            df_group = df.groupby("sample_id")
            ages=[]
            genders=[]
            occs=[]
            for _ , group in df_group:
                ages.append(int(group['ages'].agg(np.mean)))
                genders.append(int(group['genders'].agg(np.mean)))
                occs.append(int(group['occs'].agg(np.mean)))
            evaluations['ages']= ages
            evaluations['genders']= genders
            evaluations['occs']= occs

            if metric.startswith("ndcg@"):
                ndcgs = []
                for uid, group in df_group:
                    ndcgs.append(ndcg_at_k(group["l"].tolist()[:k], k=k, method=1))
                evaluations[metric] = ndcgs
            elif metric.startswith("hit@"):
                hits = []
                for uid, group in df_group:
                    hits.append(int(np.sum(group["l"][:k]) > 0))
                evaluations[metric] = hits
            elif metric.startswith("precision@"):
                precisions = []
                for uid, group in df_group:
                    precisions.append(precision_at_k(group["l"].tolist()[:k], k=k))
                evaluations[metric] = precisions
            elif metric.startswith("recall@"):
                recalls = []
                for uid, group in df_group:
                    recalls.append(1.0 * np.sum(group["l"][:k]) / np.sum(group["l"]))
                evaluations[metric] = recalls
            elif metric.startswith("f1@"):
                f1 = []
                for uid, group in df_group:
                    num_overlap = 1.0 * np.sum(group["l"][:k])
                    f1.append(2 * num_overlap / (k + 1.0 * np.sum(group["l"])))
                evaluations[metric] = f1
    return evaluations


@torch.no_grad()
def eval_discriminator(model, labels, u_vectors, fair_disc_dict, num_disc):
    feature_info = model.data_processor_dict["train"].feature_info
    feature_eval_dict = {}
    for i in range(num_disc):
        discriminator = fair_disc_dict[i + 1]
        label = labels[:, i]
        feature_name = feature_info[i + 1].name
        discriminator.eval()
        if feature_info[i + 1].num_class == 2:
            prediction = discriminator.predict(u_vectors)["prediction"].squeeze()
        else:
            prediction = discriminator.predict(u_vectors)["output"]
        feature_eval_dict[feature_name] = {
            "label": label.cpu(),
            "prediction": prediction.detach().cpu(),
            "num_class": feature_info[i + 1].num_class,
        }
        discriminator.train()
    return feature_eval_dict


@torch.no_grad()
def evaluation_disc(model, fair_disc_dict, test_data, dp, no_filter):
    num_features = dp.num_features

    def eval_disc(labels, u_vectors, fair_disc_dict, mask):
        feature_info = dp.feature_info
        feature_eval_dict = {}
        for i, val in enumerate(mask):
            if val == 0:
                continue
            discriminator = fair_disc_dict[i + 1]
            label = labels[:, i]
            feature_name = feature_info[i + 1].name
            discriminator.eval()
            if feature_info[i + 1].num_class == 2:
                prediction = discriminator.predict(u_vectors)["prediction"].squeeze()
            else:
                prediction = discriminator.predict(u_vectors)["output"]
            feature_eval_dict[feature_name] = {
                "label": label.cpu(),
                "prediction": prediction.detach().cpu(),
                "num_class": feature_info[i + 1].num_class,
            }
            discriminator.train()
        return feature_eval_dict

    eval_dict = {}
    for batch in test_data:
        mask_list = [list(i) for i in it.product([0, 1], repeat=num_features)]
        mask_list.pop(0)

        batch = move_batch_to_gpu(batch)

        labels = batch["features"]
        uids = batch["X"] - 1

        for mask in mask_list:
            if no_filter:
                vectors = model.uid_embeddings(uids)
            else:
                vectors = model.apply_filter(model.uid_embeddings(uids), mask)
            batch_eval_dict = eval_disc(labels, vectors.detach(), fair_disc_dict, mask)

            for f_name in batch_eval_dict:
                if f_name not in eval_dict:
                    eval_dict[f_name] = batch_eval_dict[f_name]
                else:
                    new_label = batch_eval_dict[f_name]["label"]
                    current_label = eval_dict[f_name]["label"]
                    eval_dict[f_name]["label"] = torch.cat(
                        (current_label, new_label), dim=0
                    )

                    new_prediction = batch_eval_dict[f_name]["prediction"]
                    current_prediction = eval_dict[f_name]["prediction"]
                    eval_dict[f_name]["prediction"] = torch.cat(
                        (current_prediction, new_prediction), dim=0
                    )

    # generate discriminator evaluation scores
    d_score_dict = {}
    if eval_dict is not None:
        for f_name in eval_dict:
            l = eval_dict[f_name]["label"]
            pred = eval_dict[f_name]["prediction"]
            n_class = eval_dict[f_name]["num_class"]
            d_score_dict[f_name] = disc_eval_method(l, pred, n_class)

    output_dict = dict()
    output_dict["d_score"] = d_score_dict
    return output_dict
