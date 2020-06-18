class Dqn(RLModel):

    def __init__(self):

        # 1. Replay Memory 초기화
        # 2. Aciton-Value Function Q 초기화
        # 3. Q hat 초기화
        # 4. exploration strategy 초기화
        pass
    # def reset_game

    def step(self):

        # def step
        # 1. exploration_strategy에 따라 action 선택
        # 2. action 실행, reward와 image 받아옴
        # 3. Replay Memory에 (s_t, a_t, r_t, s_t+1) 저장
        # 4. Replay Memory에서 minibatch 추출, Gradient Descent
        # 5. Q hat = Q (every _ step)
