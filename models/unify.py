from torch import nn
from utils.sinkhorn_cal1 import *
from utils.euclidean import euc_sqdistance
from .hyperbolic import BaseH
from .euclidean import BaseE

import torch
import torch.nn.functional as F

from utils.euclidean import givens_rotations
from utils.hyperbolic import mobius_add, expmap0, project

# class RotE(BaseE):
#     """Euclidean 2x2 Givens rotations"""
#
#     def __init__(self, args):
#         super(RotE, self).__init__(args)
#         self.rel_diag = nn.Embedding(self.sizes[1]*2, self.rank)
#         self.sim = "dist"
#
#         with torch.no_grad():
#             nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
#
#     def get_queries(self, queries: torch.Tensor):
#         """Compute embedding and biases of queries."""
#         lhs_e = givens_rotations(self.rel_diag(queries[..., 1]), self.entity(queries[..., 0])) + self.rel(queries[..., 1])
#         lhs_biases = self.bh(queries[..., 0])
#         while lhs_e.dim() < 3:
#             lhs_e = lhs_e.unsqueeze(1)
#         while lhs_biases.dim() < 3:
#             lhs_biases = lhs_biases.unsqueeze(1)
#         return lhs_e, lhs_biases

class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases



# class RotH(BaseH):
#     """Hyperbolic 2x2 Givens rotations"""
#
#     def get_queries(self, queries):
#         """Compute embedding and biases of queries."""
#         c = F.softplus(self.c(queries[..., 1]))
#         head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
#         rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
#         rel1 = expmap0(rel1, c)   # hyperbolic
#         rel2 = expmap0(rel2, c)   # hyperbolic
#         lhs = project(mobius_add(head, rel1, c), c)   # hyperbolic
#         res1 = givens_rotations(self.rel_diag(queries[..., 1]), lhs)   # givens_rotation(Euclidean, hyperbolic)
#         res2 = mobius_add(res1, rel2, c)   # hyperbolic
#         lhs_biases = self.bh(queries[..., 0])
#         while res2.dim() < 3:
#             res2 = res2.unsqueeze(1)
#         while c.dim() < 3:
#             c = c.unsqueeze(1)
#         while lhs_biases.dim() < 3:
#             lhs_biases = lhs_biases.unsqueeze(1)
#         return (res2, c), lhs_biases

class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class UnifyKGModel(torch.nn.Module):
    def __init__(
            self, args,alpha=0.1,scale=10
    ):
        super(UnifyKGModel, self).__init__()

        self.model_e = RotE(args)
        self.model_h = RotH(args)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.scale = scale
        self.rank = args.rank
        self.bias = args.bias

    def cal_ot(self, mm_embeddings, st_embeddings, delta_ot):
        device = delta_ot.device
        # device = 'cpu'
        number = 10
        # print('mm shape:',mm_embeddings.shape)
        # print('st shape:', st_embeddings.shape)
        mm_dim = mm_embeddings.shape[-1]
        st_dim = st_embeddings.shape[-1]
        mm_dis = torch.ones_like(mm_embeddings[0, :])
        mm_dis = mm_dis / mm_dis.shape[-1]
        st_dis = torch.ones_like(st_embeddings[0, :])
        st_dis = st_dis / st_dis.shape[-1]
        matrix_temp = torch.zeros((number, mm_dim, st_dim))

        with torch.no_grad():
            for i in range(number):
                cost = (mm_embeddings[i, :].reshape(-1, mm_dim) - st_embeddings[i, :].reshape(st_dim,
                                                                                              -1)) ** 2 * self.scale
                # print(cost)
                matrix_temp[i, :, :] = sinkhorn(mm_dis, st_dis, cost.t())[0].t()

        ot_result = matrix_temp.mean(dim=0).to(device) * st_dim * self.scale + delta_ot
        return ot_result.double()

    def forward(self, queries, tails=None):
        # while queries.dim() < 3:
        #     queries = queries.unsqueeze(1)
        # if tails is not None:
        #     while tails.dim() < 2:
        #         tails = tails.unsqueeze(0)

        lhs_e, lhs_e_biases = self.model_e.get_queries(queries)
        lhs_h, lhs_h_biases = self.model_h.get_queries(queries)

        rhs_e, rhs_e_biases = self.model_e.get_rhs(queries)
        # rhs_e, rhs_e_biases = self.model_e.get_rhs(tails)
        # print('rhs_e_shape',rhs_e.shape)
        # rhs_h, rhs_h_biases = self.model_h.get_rhs(queries, eval_mode)

        emb_dimension = lhs_e.shape[-1]
        self.mats_img = nn.Parameter(torch.Tensor(emb_dimension, self.rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)

        device = queries.device

        # matrix_ot = self.cal_ot(self.lhs_e.to(device), self.lhs_h.to(device), self.mats_img.to(device))
        # lhs_h_0 = self.lhs_h[0]
        # lhs_h_1 = self.lhs_h[1]
        # print(self.lhs_h)
        # print('lhs_h shape:',lhs_h[0].shape)
        # print('lhs_e shape:',lhs_e.shape)
        # print('lhs_e_biases:',lhs_e_biases.shape)
        # print('lhs_h_biases:',lhs_h_biases.shape)
        # print('rhs_e_biases:',rhs_e_biases.shape)
        # print('rhs_e:',rhs_e.shape)
        # print('lhs_h[0] shape',lhs_h[0].shape)
        # print('lhs_e shape', lhs_e.shape)
        matrix_ot = self.cal_ot(lhs_h[0].squeeze(1), lhs_e.squeeze(1),self.mats_img)


        # print(matrix_ot)
        # print('matrix_ot:',matrix_ot)
        # print('matrix_ot_shape:',matrix_ot.shape)
        # matrix_ot = self.cal_ot(lhs_e, lhs_h, self.mats_img)
        # print('lhs_e type:',lhs_e.type())
        # print('matrix type:',matrix_ot.type())
        # hyper_embeddings = lhs_e.squeeze(1).to(device).mm(matrix_ot.to(device))
        hyper_embeddings = lhs_e.to(device).mm(matrix_ot.to(device))

        # embedding = (1 - self.alpha) *  lhs_e + self.alpha * hyper_embeddings.unsqueeze(1)
        embedding = (1 - self.alpha) * lhs_e + self.alpha * hyper_embeddings

        # print(embedding)
        # print('embedding shape:',embedding.shape)
        # lhs = embedding[(queries[:, 0])]
        # rhs = embedding[(queries[:, 2])]


        # print(lhs)
        # print(rhs)

        # print('lhs shape:',lhs.shape)
        # print('rhs shape:',rhs.shape)
        # print('lhs_e_biases',lhs_h_biases.shape)
        # print('rhs_e_biases',rhs_e_biases.shape)

        predictions = self.score((embedding, lhs_e_biases), (rhs_e, rhs_e_biases),eval_mode=False)
        # print(predictions)
        # print('prediction:',predictions)
        # print('prediction shape',predictions.shape)

        factors = self.model_e.get_factors(queries)
        # factors = self.get_factors(queries, tails)
        # print('fators:',factors)
        # print('factor',factors)
        # print('factor shape 0', factors[0].shape)
        # print('factor shape 1', factors[1].shape)
        # print('factor shape 2', factors[2].shape)
        return predictions, factors


    def score(self, lhs, rhs, eval_mode=False):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)

        if self.bias== 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score
        # if self.bias == 'constant':
        #     return self.gamma.item() + score
        # elif self.bias == 'learn':
        #     if eval_mode:
        #         return lhs_biases + rhs_biases.t() + score
        #     else:
        #         return lhs_biases + rhs_biases + score
        # else:
        # return score

    def similarity_score(self, lhs_e, rhs_e, eval_mode=False):
        """Compute similarity scores or queries against targets in embedding space."""
        score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

    def get_ranking(self, queries, filters, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.model_e.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.model_e.get_queries(these_queries)
                rhs = self.model_e.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks

    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.

        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.model_e.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at