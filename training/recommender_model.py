"""
Recommender System Models
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRecModel(nn.Module):
    """
    Base recommendation model.
    """

    def __init__(
        self,
        data_processor_dict,
        exp_out_dir,
        user_num,
        item_num,
        u_vector_size,
        i_vector_size,
        feature_columns,
        random_seed=42,
        dropout=0.2,
    ):
        """
        :param data_processor_dict:
        :param user_num:
        :param item_num:
        :param u_vector_size:
        :param i_vector_size:
        :param random_seed:
        :param dropout:
        :param model_path:
        'separate' -> one filter for one sensitive feature, do combination for complex case.
        """
        super(BaseRecModel, self).__init__()
        self.data_processor_dict = data_processor_dict
        self.user_num = user_num
        self.item_num = item_num
        self.u_vector_size = u_vector_size
        self.i_vector_size = i_vector_size
        self.dropout = dropout
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = os.path.join(exp_out_dir, "model/RecModel.pt")
        self.feature_columns = feature_columns

        self._init_nn()
        self._init_sensitive_filter()
        logging.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logging.info("Number of params: %d", self.total_parameters)
        self.optimizer = None

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_nn(self):
        """
        Initialize neural networks
        :return:
        """
        raise NotImplementedError

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim),
            )
            return sequential

        num_features = len(self.feature_columns)
        self.filter_num = 2**num_features
        self.num_features = num_features
        self.filter_dict = nn.ModuleDict(
            {
                str(i + 1): get_sensitive_filter(self.u_vector_size)
                for i in range(self.filter_num)
            }
        )

    def apply_filter(self, vectors, filter_mask):
        if np.sum(filter_mask) != 0:
            filter_mask = np.asarray(filter_mask)
            idx = filter_mask.dot(2 ** np.arange(filter_mask.size))
            sens_filter = self.filter_dict[str(idx)]
            result = sens_filter(vectors)
        else:
            result = vectors
        return result

    def count_variables(self):
        """
        Total number of parameters in the model
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def l2(self):
        """
        calc the summation of l2 of all parameters
        :return:
        """
        l2 = 0
        for p in self.parameters():
            l2 += (p**2).sum()
        return l2

    def predict(self, feed_dict, filter_mask):
        """
        prediction only without loss calculation
        :param feed_dict: input dictionary
        :param filter_mask: mask for filter selection
        :return: output dictionary，with keys (at least)
                "prediction": predicted values;
                "check": intermediate results to be checked and printed out
        """
        check_list = []
        x = self.x_bn(feed_dict["X"].float())
        x = torch.nn.Dropout(p=feed_dict["dropout"])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
        out_dict = {"prediction": prediction, "check": check_list}
        return out_dict

    def forward(self, feed_dict, filter_mask):
        out_dict = self.predict(feed_dict, filter_mask)
        batch_size = int(feed_dict["label"].shape[0] / 2)
        pos, neg = (
            out_dict["prediction"][:batch_size],
            out_dict["prediction"][batch_size:],
        )
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict["loss"] = loss
        return out_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info("Model saved to %s", model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info("Model loaded from %s", model_path)

    def freeze_model(self):
        self.eval()
        for params in self.parameters():
            params.requires_grad = False


class MLP(BaseRecModel):
    """
    MLP model
    """

    @staticmethod
    def parse_model_args(parser, model_name="MLP"):
        parser.add_argument(
            "--num_layers", type=int, default=3, help="Number of mlp layers."
        )
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(
        self,
        data_processor_dict,
        exp_out_dir,
        user_num,
        item_num,
        u_vector_size,
        i_vector_size,
        feature_columns,
        num_layers=3,
        random_seed=42,
        dropout=0.2,
    ):
        self.num_layers = num_layers
        self.factor_size = u_vector_size // (2 ** (self.num_layers - 1))
        BaseRecModel.__init__(
            self,
            data_processor_dict,
            exp_out_dir,
            user_num,
            item_num,
            u_vector_size,
            i_vector_size,
            feature_columns,
            random_seed=random_seed,
            dropout=dropout,
        )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity="sigmoid")
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_nn(self):
        # Init embeddings
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

        # Init MLP
        self.mlp = nn.ModuleList([])
        pre_size = self.factor_size * (2**self.num_layers)
        for _ in range(self.num_layers):
            self.mlp.append(nn.Dropout(p=self.dropout))
            self.mlp.append(nn.Linear(pre_size, pre_size // 2))
            self.mlp.append(nn.ReLU())
            pre_size = pre_size // 2
        self.mlp = nn.Sequential(*self.mlp)

        # Init predictive layer
        self.p_layer = nn.ModuleList([])
        assert pre_size == self.factor_size
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict["X"][:, 0] - 1
        i_ids = feed_dict["X"][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp_u_vectors = self.apply_filter(mlp_u_vectors, filter_mask)

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        prediction = self.prediction(mlp).view([-1])

        out_dict = {
            "prediction": prediction,
            "check": check_list,
            "u_vectors": mlp_u_vectors,
        }
        return out_dict


class PMF(BaseRecModel):
    """
    PMF model
    """

    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

    def predict(self, feed_dict, filter_mask):
        check_list = []
        u_ids = feed_dict["X"][:, 0] - 1
        i_ids = feed_dict["X"][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_vectors = self.apply_filter(pmf_u_vectors, filter_mask)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {
            "prediction": prediction,
            "check": check_list,
            "u_vectors": pmf_u_vectors,
        }
        return out_dict
