from tqdm import tqdm
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F


def loss_function(y_true, y_pred, lamda, gamma):
    mae_loss = F.l1_loss(y_pred, y_true)
    mse_loss = F.mse_loss(y_pred, y_true)

    true_ranking = torch.argsort(y_true, descending=True)
    pred_ranking = torch.argsort(y_pred, descending=True)
    corr, _ = spearmanr(pred_ranking.cpu(), true_ranking.cpu())
    rank_loss = 1 - corr

    return (lamda*mae_loss+(1-lamda)*mse_loss+gamma*rank_loss)

def evaluate(model, loader, lamda, gamma, test):
    total_loss = []
    pred = []
    true = []
    pas = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            gene_loss = []
            for gene in batch:
                output = model(gene)
                gene_loss.append(loss_function(gene.y, output, lamda, gamma))
                pred.extend(output.tolist())
                true.extend(gene.y.tolist())
                pas.extend(gene.pas)
            loss = sum(gene_loss) / len(gene_loss)
            total_loss.append(loss.item())
    if test:
        with open(f'results.txt', 'w') as f:
            f.write('pas\ttrue\tpredicted\n')
            for i in range(len(pas)):
                f.write(f'{pas[i]}\t{true[i]}\t{pred[i]}\n')

    return sum(total_loss) / len(total_loss)