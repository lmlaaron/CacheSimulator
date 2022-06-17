import wandb

class WandbLogger:
    def __init__(self,
                 project,
                 config):
        wandb.init(project=project, config=config)
        self.dir = wandb.run.dir

    def get_dir(self):
        return self.dir

    def log(self,
            train_stats,
            eval_stats):
        
        # decode train stats:
        train_stats = stats_filter(train_stats, prefix='train')
        eval_stats = stats_filter(eval_stats, prefix='eval')

        train_stats.update(eval_stats)
        stats = train_stats
        wandb.log(stats)

def stats_filter(stats,
                prefix='train'):
    stats = stats.dict()
    res_dict = {}
    for k, v in stats.items():
        k = prefix+"_"+k
        res_dict[k] = v['mean']
    return res_dict