import os
import cornac
from cornac.data import SentimentModality, Reader
from cornac.eval_methods import StratifiedSplit
from cornac.metrics import NDCG, RMSE, AUC
from cornac import Experiment


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="data/baby")
    parser.add_argument("-mu", "--min_user_freq", type=int, default=5)
    parser.add_argument("-k", "--num_factors", type=int, default=8)
    parser.add_argument("-nb", "--num_bpr_samples", type=int, default=1000)
    parser.add_argument("-na", "--num_aspect_ranking_samples", type=int, default=100)
    parser.add_argument("-no", "--num_opinion_ranking_samples", type=int, default=100)
    parser.add_argument("-ldreg", "--lambda_reg", type=float, default=0.1)
    parser.add_argument("-ldbpr", "--lambda_bpr", type=float, default=10)
    parser.add_argument("-ldp", "--lambda_p", type=float, default=10)
    parser.add_argument("-lda", "--lambda_a", type=float, default=10)
    parser.add_argument("-ldy", "--lambda_y", type=float, default=10)
    parser.add_argument("-ldz", "--lambda_z", type=float, default=10)
    parser.add_argument("-lds", "--lambda_s", type=float, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-e", "--max_iter", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--n_top_aspects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


args = parse_arguments()

rating = Reader(min_user_freq=args.min_user_freq).read(
    os.path.join(args.input_dir, "rating.txt"), fmt="UIRT", sep=","
)
sentiment = Reader().read(
    os.path.join(args.input_dir, "sentiment.txt"),
    fmt="UITup",
    sep=",",
    tup_sep=":",
)

md = SentimentModality(data=sentiment)

eval_method = StratifiedSplit(
    rating,
    group_by="user",
    chrono=True,
    sentiment=md,
    test_size=1,
    val_size=1,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

models = [
    cornac.models.Companion(
        n_user_factors=args.num_factors,
        n_item_factors=args.num_factors,
        n_aspect_factors=args.num_factors,
        n_opinion_factors=args.num_factors,
        n_bpr_samples=args.num_bpr_samples,
        n_aspect_ranking_samples=args.num_aspect_ranking_samples,
        n_opinion_ranking_samples=args.num_opinion_ranking_samples,
        n_top_aspects=args.n_top_aspects,
        lambda_reg=args.lambda_reg,
        lambda_bpr=args.lambda_bpr,
        lambda_p=args.lambda_p,
        lambda_a=args.lambda_a,
        lambda_y=args.lambda_y,
        lambda_z=args.lambda_z,
        max_iter=args.max_iter,
        lr=args.learning_rate,
        verbose=args.debug,
        seed=args.seed,
    )
]
# Instantiate and run an experiment
exp = Experiment(
    eval_method=eval_method,
    models=models,
    metrics=[RMSE(), NDCG(k=10), NDCG(k=20), NDCG(k=50), AUC()],
    save_dir=os.path.join(args.input_dir, "result"),
)
exp.run()
