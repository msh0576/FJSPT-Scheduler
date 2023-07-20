import torch

from matnet_v2_models.TFJSPTrainer_matnet_v2 import TFJSPTrainer_matnet_v2

from dispatching_models.dispatchModel import dispatchModel

class dispatchTrainer(TFJSPTrainer_matnet_v2):
    def __init__(self,
                 env_paras,
                model_paras,
                train_paras,
                optimizer_paras,
                test_paras,
                change_paras,
                rule='spt'
                 ):
        super().__init__(
            env_paras,
            model_paras,
            train_paras,
            optimizer_paras,
            test_paras,
            change_paras,
        )
        
        self.model = dispatchModel(
            rule=rule,
            **model_paras
        )
    
    def _train_one_batch(self, batch_size, env):
        # ===== Preparation =====
        state = env.reset()
        self.model.init(state)
        
        # ===== Rollout =====
        done = False
        dones = env.done_batch
        while not done:
            action, _ = self.model.act(state)   # [3, B] | [B, 1]
            state, rewards, dones = env.step(action)
            done = dones.all()
        
        # === score ===
        # score = -rewards.mean()
        score = env.get_makespan().mean()
        
        
        return score.item(), 0
        
        